import os
import shutil
import random

# 1. 定义路径
train_root = "./datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small"  # 原始训练集
val_root = "./datasets/LPBA40/LPBA40_validation"  # 验证集保存路径

# 2. 确保验证集目录存在
os.makedirs(val_root, exist_ok=True)

# 3. 获取所有配准对样本（筛选.nii文件）
all_reg_pairs = [f for f in os.listdir(train_root) if f.endswith(".nii") and "_to_" in f]
print(f"找到 {len(all_reg_pairs)} 个配准对样本")

# 4. 随机划分（20%作为验证集，固定种子确保复现）
random.seed(42)
random.shuffle(all_reg_pairs)
val_ratio = 0.2  # 验证集比例
val_count = int(len(all_reg_pairs) * val_ratio)
val_pairs = all_reg_pairs[:val_count]  # 验证集样本
train_pairs = all_reg_pairs[val_count:]  # 剩余训练集样本

# 5. 移动验证集样本到新目录
for pair_file in val_pairs:
    src_path = os.path.join(train_root, pair_file)
    dst_path = os.path.join(val_root, pair_file)
    shutil.move(src_path, dst_path)
    print(f"移动验证集样本：{pair_file}")

# 6. 输出划分结果
print(f"\n✅ 划分完成！")
print(f"训练集：{len(train_pairs)} 个配准对（路径：{train_root}）")
print(f"验证集：{len(val_pairs)} 个配准对（路径：{val_root}）")

