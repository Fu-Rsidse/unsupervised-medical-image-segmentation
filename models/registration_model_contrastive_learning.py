import os
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from models.base_model import BaseModel
from .networks_contrastive_learning import define_registration_model
from .networks_contrastive_learning import contrastive_loss, recon_loss, smooth_loss
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
        input_nc = 1 
        output_nc = 3 
        ngf = 32
        netReg = 'unet_3d'
        norm = 'batch'
        init_type = 'normal'
        init_gain = 0.02
        gpu_ids = self.gpu_ids
        num_classes = 7
        model_parallel = False

        # 定义配准网络（修复括号不匹配问题）
        self.netReg = define_registration_model(
            gpu_ids=gpu_ids,
            is_training=self.isTrain,
            model_parallel=model_parallel,
            mode='lpba40',
            img_size=(72, 96, 72)
        )
        
        self.channel_align_conv = nn.Conv3d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=1,  # 1×1×1卷积，只改变通道数，不改变空间尺寸
            padding=0
        ).to(self.device)


        # 打印网络信息
        total_params = sum(p.numel() for p in self.netReg.parameters()) / 1e6
        print(f"[Network Reg] 初始化完成，参数总数：{total_params:.6f} M")
        print(f"  输入通道：{input_nc} | 输出通道：{output_nc} | 网络类型：{netReg} | 归一化：{norm}")

        # 训练相关配置（推理时不生效）
        if self.isTrain:
            self.contrastive_loss = contrastive_loss(batch_size=opt.batchSize).to(self.device)
            self.recon_loss = recon_loss().to(self.device)
            self.smooth_loss = smooth_loss().to(self.device)
            self.optimizer_Reg = torch.optim.Adam(
                self.netReg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers = [self.optimizer_Reg]
            self.schedulers = [self.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        # 路径初始化
        self.folder_name = opt.name
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        if not self.isTrain:
            # 1. 权重文件路径
            weight_file = f"{opt.which_epoch}_net_Reg.pth"
            load_path = os.path.join(self.save_dir, weight_file)
            
            # 2. 检查文件是否存在
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"❌ 权重文件不存在：{load_path}")
            
            # 3. 关键修复：加载字典并提取纯模型参数（model_state_dict）
            checkpoint = torch.load(load_path, map_location=self.device)
            if "model_state_dict" not in checkpoint:
                raise KeyError(f"❌ 权重文件中无 model_state_dict 键，请确认是训练生成的文件")
            state_dict = checkpoint["model_state_dict"]  # 只取纯模型参数
            
            new_state_dict = {}
            # 判断模型是否被DataParallel包装
            is_parallel = isinstance(self.netReg, torch.nn.DataParallel)
            
            for param_name, param_value in state_dict.items():
                if is_parallel:
                    new_state_dict[param_name] = param_value
                else:
                    if param_name.startswith("module."):
                        new_state_dict[param_name[len("module."):]] = param_value
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
            # 1. 模型前向传播（获取所有输出）
            self.feat_moving, self.seg_result, self.flow, self.feat_fixed, self.feat_moving_proj, self.feat_fixed_proj, self.logits = self.netReg(
                self.input_A, self.input_B, self.input_B_atlas
            )
            
            # 2. 赋值配准后图像
            self.warped_image = self.feat_moving
            print(f"warped_image 已定义，维度：{self.warped_image.shape}")
            
            # 3. 通道对齐、维度压缩
            if self.feat_moving_proj.dim() == 5:
                self.feat_moving_proj = self.channel_align_conv(self.feat_moving_proj)
                self.feat_moving_proj = torch.mean(self.feat_moving_proj, dim=[2, 3, 4])
            if self.feat_fixed_proj.dim() == 1:
                self.feat_fixed_proj = self.feat_fixed_proj.unsqueeze(0)
            
            # 验证最终特征维度
            print(f"处理后 feat_moving_proj：{self.feat_moving_proj.shape}")
            print(f"处理后 feat_fixed_proj：{self.feat_fixed_proj.shape}")


    def backward_Reg(self):
        if not self.isTrain:
            return
        # 1. 计算对比损失
        self.loss_contrastive = self.contrastive_loss(self.feat_moving_proj, self.feat_fixed_proj)
        
        # 2. 计算重建损失
        self.loss_recon = self.recon_loss(self.warped_image, self.input_B)
        
        # 3. 计算平滑损失
        self.loss_smooth = self.smooth_loss(self.flow)
        
        # 4. 计算总损失
        self.loss_total = 0.3 * self.loss_contrastive + 0.5 * self.loss_recon + 0.2 * self.loss_smooth
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
            print(f"Debug: 模型分割结果seg_np唯一值: {np.unique(seg_np)}")
            print(f"Debug: 模型分割结果各值数量: {[np.sum(seg_np == i) for i in range(8)]}")

            # -------------------------- 新增：分割结果映射 --------------------------
            # 与标签映射相同的规则，将模型输出的21/41等映射到1-7
            label_mapping = {
                21:1, 41:1,  # FL（额叶）
                22:2, 42:2,  # PL（顶叶）
                23:3, 43:3,  # OL（枕叶）
                24:4, 44:4,  # TL（颞叶）
                25:5, 45:5,  # CL（小脑）
                26:6, 46:6,  # Ptm（缘上回）
                27:7, 47:7   # Hpcp（海马旁回）
            }
            # 执行映射（将seg_np中的原始标签转换为1-7）
            mapped_seg = np.zeros_like(seg_np)
            for orig_label, new_label in label_mapping.items():
                mapped_seg[seg_np == orig_label] = new_label
            print(f"Debug: 映射后分割结果mapped_seg唯一值: {np.unique(mapped_seg)}")  # 应包含1-7
            # ------------------------------------------------------------------

            # 标签映射（已正确实现）
            fixed_atlas_np = self.input_B_atlas.data.int().cpu().numpy().squeeze()
            label_mapping = {
                21:1, 41:1, 22:2, 42:2, 23:3, 43:3, 
                24:4, 44:4, 25:5, 45:5, 26:6, 46:6, 27:7, 47:7
            }
            mapped_atlas = np.zeros_like(fixed_atlas_np)
            for orig_label, new_label in label_mapping.items():
                mapped_atlas[fixed_atlas_np == orig_label] = new_label
            print(f"Debug: 映射后标签值: {np.unique(mapped_atlas)}")

            # 计算Dice（使用映射后的分割结果mapped_seg和标签mapped_atlas）
            dsc_list = []
            brain_regions = ["FL", "PL", "OL", "TL", "CL", "Ptm", "Hpcp"]
            for class_id in range(1, 8):
                seg_mask = (mapped_seg == class_id).astype(np.float32)  # 使用映射后的分割结果
                atlas_mask = (mapped_atlas == class_id).astype(np.float32)
                intersection = np.sum(seg_mask * atlas_mask)
                union = np.sum(seg_mask) + np.sum(atlas_mask)
                dsc = 2 * intersection / union if union != 0 else 0.0
                dsc_list.append(dsc)
                print(f"  脑区{class_id}（{brain_regions[class_id-1]}）Dice：{dsc:.4f}")

            # 后续代码不变...
            total_dice = np.mean(dsc_list)
            moving_filename = os.path.basename(self.image_paths[0])
            print(f"处理样本 {moving_filename}，总Dice={total_dice:.4f}")
            return total_dice, dsc_list


            # 保存结果（不变）
            output_dir = os.path.join(self.save_dir, f"output_{self.folder_name}")
            os.makedirs(output_dir, exist_ok=True)
            seg_sitk = sitk.GetImageFromArray(seg_np)
            sitk.WriteImage(seg_sitk, os.path.join(output_dir, f"seg_{moving_filename}"))
            flow_np = np.transpose(self.flow.data.float().cpu().numpy().squeeze(), (1, 2, 3, 0))
            flow_sitk = sitk.GetImageFromArray(flow_np)
            sitk.WriteImage(flow_sitk, os.path.join(output_dir, f"flow_{moving_filename}"))

            return total_dice, dsc_list


    # 训练时方法，推理时不影响
    def save_networks(self, which_epoch):
        save_path = os.path.join(self.save_dir, f"{which_epoch}_net_Reg.pth")
        torch.save(self.netReg.state_dict(), save_path)

    def get_current_losses(self):
        if not self.isTrain:
            return {}
        return {
            'total': self.loss_total.item(),
            'contrastive': self.loss_contrastive.item(),
            'recon': self.loss_recon.item(),
            'smooth': self.loss_smooth.item()
        }


    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print(f"学习率更新为：{lr:.6f}")


    
