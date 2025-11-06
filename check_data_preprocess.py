import numpy as np
import SimpleITK as sitk
from dataloaders.lpba40_dataloader_contrastive_learning import LPBA40  # 导入你的LPBA40类
from options.base_options import BaseOptions

# 1. 构造参数
class Opt:
    phase = "val"
    dataroot = "./datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small"
    dataset_mode = "lpba40_contrastive_learning"

opt = Opt()

# 2. 初始化数据集，获取样本路径
dataset = LPBA40()
dataset.initialize(opt)

# 3. 检查前2个样本的预处理
print("="*60)
print("检查数据预处理合理性（归一化前后像素值）")
print("="*60)
for i in range(min(2, len(dataset.moving_path))):
    moving_path = dataset.moving_path[i]
    print(f"\n【样本 {i+1}】{moving_path.split('/')[-1]}")

    # 1. 读取原始图像（未预处理）
    raw_img = dataset.readVol(moving_path)
    print(f" - 原始图像像素值：")
    print(f"   范围：{raw_img.min():.2f} ~ {raw_img.max():.2f}")
    print(f"   均值：{raw_img.mean():.2f}，标准差：{raw_img.std():.2f}")

    # 2. 应用预处理（whitening函数）
    processed_img = dataset.whitening(raw_img)
    print(f" - 预处理后像素值：")
    print(f"   范围：{processed_img.min():.4f} ~ {processed_img.max():.4f}")
    print(f"   均值：{processed_img.mean():.4f}，标准差：{processed_img.std():.4f}")

    # 3. 检查是否有有效信息
    valid_pixels = processed_img[(processed_img > 0) & (processed_img < 1)]  # 0~1内的像素
    print(f" - 0~1范围内像素占比：{len(valid_pixels)/processed_img.size:.4f}")

print("\n" + "="*60)
print("结果判断：")
print(" - 若预处理后像素值全为0或全为1 → whitening函数错误（如clip范围不当）")
print(" - 若0~1像素占比<0.5 → 图像信息丢失过多，需修改归一化逻辑")
print(" - 若两者正常 → 预处理无问题")
print("="*60)

