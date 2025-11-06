import torch
import numpy as np
from options.train_options import TrainOptions
from dataloaders import CreateDataLoader
from models import create_model

if __name__ == '__main__':
    # 1. 配置测试参数（与test_dice.py一致）
    opt = TrainOptions().parse()
    opt.phase = "val"  # 验证模式
    opt.isTrain = False
    opt.batchSize = 1
    opt.dataset_mode = "lpba40_contrastive_learning"
    opt.dataroot = "./datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small"
    opt.name = "lpba40_model"
    opt.gpu_ids = [0]
    opt.which_epoch = "latest"

    # 2. 加载验证集（只取前5个样本，快速检查）
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_iter = iter(dataset)  # 迭代器，取少量样本

    # 3. 加载模型
    model = create_model(opt)
    model.setup(opt)
    model.eval()  # 推理模式

    # 4. 检查前5个样本的模型输出
    print("="*60)
    print("检查模型输出有效性（前5个验证集样本）")
    print("="*60)
    for i in range(5):
        try:
            data = next(dataset_iter)
        except StopIteration:
            break  # 样本不足时退出

        # 模型推理
        model.set_input(data)
        with torch.no_grad():
            # 调用test方法，获取分割结果（需确保model.test()返回或保存分割结果）
            dice_total, dsc_list = model.test()
            # 从模型中获取分割结果（根据你的模型属性调整，若模型未保存，可从test返回值补充）
            # 若你的模型test()未返回分割结果，需先修改模型：在test()中return segmentation_result, dice_total, dsc_list
            seg_result = model.netReg.segmentation_result  # 假设模型将分割结果存在netReg的segmentation_result属性中
            seg_np = seg_result.data.cpu().numpy().squeeze()  # 转为numpy数组

        # 打印统计信息
        moving_path = data["A_paths"][0]
        print(f"\n【样本 {i+1}】{moving_path.split('/')[-1]}")
        print(f" - 分割结果值范围：{seg_np.min():.2f} ~ {seg_np.max():.2f}")
        print(f" - 非零像素占比：{(seg_np > 0).sum() / seg_np.size:.4f}")  # 非零占比>0.1才算有效
        print(f" - 当前样本Dice：{dice_total:.4f}")

    print("\n" + "="*60)
    print("结果判断：")
    print(" - 若'分割结果值范围'全为0 → 模型未输出有效结果")
    print(" - 若'非零像素占比'<0.01 → 模型输出几乎全是背景，无效")
    print(" - 若两者正常 → 问题在标签匹配或Dice计算逻辑")
    print("="*60)

