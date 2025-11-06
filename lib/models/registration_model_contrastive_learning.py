import os
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from models.base_model import BaseModel
from models.networks import define_registration_model
from models.loss import ContrastiveLoss, RegistrationLoss
from utils.visualizer import get_visdom_kwargs


class RegistrationModel(BaseModel):
    def name(self):
        return 'registration_model_contrastive_learning'

    def initialize(self, opt):
        # 初始化父类
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        self.device = torch.device('cuda:0' if opt.gpu_ids else 'cpu')

        # 从配置获取网络参数
        input_nc = opt.input_nc
        output_nc = opt.output_nc
        ngf = opt.ngf
        netReg = opt.netReg
        norm = opt.norm
        init_type = opt.init_type
        init_gain = opt.init_gain
        gpu_ids = self.gpu_ids
        num_classes = opt.num_classes
        model_parallel = opt.model_parallel

        # 定义配准网络
        self.netReg = define_registration_model(
            input_nc, output_nc, ngf, netReg,
            norm, init_type, init_gain, gpu_ids,
            num_classes, model_parallel
        )

        # 打印网络信息
        total_params = sum(p.numel() for p in self.netReg.parameters()) / 1e6
        print(f"[Network Reg] 初始化完成，参数总数：{total_params:.6f} M")
        print(f"  输入通道：{input_nc} | 输出通道：{output_nc} | 网络类型：{netReg} | 归一化：{norm}")

        # 训练相关配置（推理时不生效）
        if self.isTrain:
            self.contrastive_loss = ContrastiveLoss(margin=1.0).to(self.device)
            self.registration_loss = RegistrationLoss(lambda_smooth=1e-4).to(self.device)
            self.optimizer_Reg = torch.optim.Adam(
                self.netReg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers = [self.optimizer_Reg]
            self.schedulers = [self.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        # 路径初始化
        self.folder_name = opt.name
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        # -------------------------- 核心：直接加载权重（无任何load_network调用） --------------------------
        if not self.isTrain:
            # 1. 权重文件路径
            weight_file = f"{opt.which_epoch}_net_Reg.pth"
            load_path = os.path.join(self.save_dir, weight_file)
            
            # 2. 检查文件是否存在
            if not os.path.exists(load_path):
                raise FileNotFoundError(
                    f"❌ 权重文件不存在！\n"
                    f"路径：{load_path}\n"
                    f"请确认 {self.save_dir} 目录下有 {weight_file} 文件"
                )
            
            # 3. 加载权重并处理多GPU前缀
            state_dict = torch.load(load_path, map_location=self.device)
            new_state_dict = {}
            for param_name, param_value in state_dict.items():
                # 移除训练时DataParallel添加的"module."前缀
                if param_name.startswith('module.'):
                    clean_name = param_name[len('module.'):]  # 截取"module."后的名称
                    new_state_dict[clean_name] = param_value
                else:
                    new_state_dict[param_name] = param_value
            
            # 4. 加载到网络
            self.netReg.load_state_dict(new_state_dict)
            print(f"✅ 成功加载权重：{load_path}")

    def get_scheduler(self, optimizer, opt):
        if opt.lr_policy == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        else:
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[opt.niter, opt.niter+opt.niter_decay], gamma=0.1)

    def set_input(self, data):
        self.input_A = data['A'].to(self.device)        # moving图像
        self.input_B = data['B'].to(self.device)        # fixed图像
        self.input_A_atlas = data['A_atlas'].to(self.device)  # moving标签
        self.input_B_atlas = data['B_atlas'].to(self.device)  # fixed标签
        self.image_paths = data['A_paths']              # 图像路径

    def forward(self):
        if self.isTrain:
            self.netReg.train()
            self.feat_moving, self.seg_result, self.flow, self.feat_fixed, self.feat_moving_proj, self.feat_fixed_proj, self.logits = self.netReg(
                self.input_A, self.input_B, self.input_B_atlas
            )

    def backward_Reg(self):
        if not self.isTrain:
            return
        loss_contrastive = self.contrastive_loss(self.feat_moving_proj, self.feat_fixed_proj, self.logits)
        loss_registration = self.registration_loss(self.seg_result, self.input_B_atlas, self.flow)
        self.loss_total = 0.3 * loss_contrastive + 0.7 * loss_registration
        self.loss_total.backward()

    def optimize_parameters(self):
        if not self.isTrain:
            return
        self.forward()
        self.optimizer_Reg.zero_grad()
        self.backward_Reg()
        self.optimizer_Reg.step()

    def test(self):
        self.netReg.eval()
        with torch.no_grad():
            # 模型推理
            _, self.seg_result, self.flow, _, _, _, _ = self.netReg(
                self.input_A, self.input_B, self.input_B_atlas
            )

            # 处理维度（移除batch）
            seg_np = self.seg_result.data.int().cpu().numpy().squeeze()
            fixed_atlas_np = self.input_B_atlas.data.int().cpu().numpy().squeeze()

            # 计算7个脑区Dice
            dsc_list = []
            brain_regions = ["FL", "PL", "OL", "TL", "CL", "Ptm", "Hpcp"]
            for class_id in range(1, 8):
                seg_mask = (seg_np == class_id).astype(np.float32)
                atlas_mask = (fixed_atlas_np == class_id).astype(np.float32)
                intersection = np.sum(seg_mask * atlas_mask)
                union = np.sum(seg_mask) + np.sum(atlas_mask)
                dsc = 2 * intersection / union if union != 0 else 0.0
                dsc_list.append(dsc)
                print(f"  脑区{class_id}（{brain_regions[class_id-1]}）Dice：{dsc:.4f}")

            # 总Dice
            total_dice = np.mean(dsc_list)
            moving_filename = os.path.basename(self.image_paths[0])
            print(f"处理样本 {moving_filename}，总Dice={total_dice:.4f}")

            # 保存结果
            output_dir = os.path.join(self.save_dir, f"output_{self.folder_name}")
            os.makedirs(output_dir, exist_ok=True)
            # 保存分割结果
            seg_sitk = sitk.GetImageFromArray(seg_np)
            sitk.WriteImage(seg_sitk, os.path.join(output_dir, f"seg_{moving_filename}"))
            # 保存形变场
            flow_np = np.transpose(self.flow.data.float().cpu().numpy().squeeze(), (1, 2, 3, 0))
            flow_sitk = sitk.GetImageFromArray(flow_np)
            sitk.WriteImage(flow_sitk, os.path.join(output_dir, f"flow_{moving_filename}"))

            return total_dice, dsc_list

    # 以下为训练时方法，推理时不影响
    def save_networks(self, which_epoch):
        save_path = os.path.join(self.save_dir, f"{which_epoch}_net_Reg.pth")
        torch.save(self.netReg.state_dict(), save_path)

    def get_current_losses(self):
        if not self.isTrain:
            return {}
        return {'total': self.loss_total.item(), 'contrastive': self.contrastive_loss.item(), 'registration': self.registration_loss.item()}

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print(f"学习率更新为：{lr:.6f}")

