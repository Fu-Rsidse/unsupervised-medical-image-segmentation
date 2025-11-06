import os
import numpy as np
import SimpleITK as sitk
from dataloaders.lpba40_dataloader_contrastive_learning import LPBA40  # 导入你的LPBA40类
from options.base_options import BaseOptions  # 若没有，可手动构造opt

# 1. 构造基础参数（模拟验证集模式）
class Opt:
    phase = "val"
    dataroot = "./datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small"
    dataset_mode = "lpba40_contrastive_learning"

opt = Opt()

# 2. 初始化LPBA40类，加载验证集标签
dataset = LPBA40()
dataset.initialize(opt)

# 3. 检查前3个样本的标签合并结果
print("="*60)
print("检查验证集标签合并正确性（前3个样本）")
print("="*60)
for i in range(min(3, len(dataset.moving_path))):
    moving_path = dataset.moving_path[i]
    fixed_path = dataset.moving_fixed[moving_path]

    # 读取原始标签（从label路径读取，与数据集结构匹配）
    # 注意：替换为你的标签文件路径（参考LPBA40类中的标签路径替换逻辑）
    moving_label_path = moving_path.replace(
        "LPBA40_rigidly_registered_pairs_histogram_standardization",
        "LPBA40_rigidly_registered_label_pairs"
    ).replace(".nii", ".hdr")
    fixed_label_path = fixed_path.replace(
        "LPBA40_rigidly_registered_pairs_histogram_standardization",
        "LPBA40_rigidly_registered_label_pairs"
    ).replace(".nii", ".hdr")

    # 读取标签并合并（复用LPBA40类的逻辑）
    def read_and_merge_label(label_path):
        if not os.path.exists(label_path):
            return None, "标签文件不存在"
        # 读取标签
        label_np = dataset.readVol(label_path)
        # 合并小区域（复用good_labels_list）
        good_labels_list = [
            [21,22,23,24,25,26,27,28,29,30,31,32,33,34],  # FL
            [41,42,43,44,45],                               # PL
            [61,62,63,64,65,66,67,68],                      # OL
            [81,82,83,84,85,86,87,88,89,90,91,92],          # TL
            [101,102,121,122],                               # CL
            [163,164],                                       # Ptm
            [165,166]                                        # Hpcp
        ]
        for class_idx, labels in enumerate(good_labels_list, 1):
            for lbl in labels:
                label_np[label_np == lbl] = class_idx
        label_np[label_np > 7] = 0  # 过滤无效标签
        return label_np, "成功"

    moving_label, status1 = read_and_merge_label(moving_label_path)
    fixed_label, status2 = read_and_merge_label(fixed_label_path)

    # 打印标签信息
    print(f"\n【样本 {i+1}】")
    print(f" - Moving标签路径：{moving_label_path.split('/')[-1]} → {status1}")
    if moving_label is not None:
        unique = np.unique(moving_label)
        print(f"   合并后标签类别：{sorted(unique)}（应包含1~7中的部分类别）")
        print(f"   标签值范围：{moving_label.min()} ~ {moving_label.max()}")
    
    print(f" - Fixed标签路径：{fixed_label_path.split('/')[-1]} → {status2}")
    if fixed_label is not None:
        unique = np.unique(fixed_label)
        print(f"   合并后标签类别：{sorted(unique)}（应包含1~7中的部分类别）")
        print(f"   标签值范围：{fixed_label.min()} ~ {fixed_label.max()}")

print("\n" + "="*60)
print("结果判断：")
print(" - 若标签类别只有[0] → 合并逻辑错误或标签文件路径错误")
print(" - 若标签类别包含1~7 → 标签合并正确，问题在其他地方")
print("="*60)

