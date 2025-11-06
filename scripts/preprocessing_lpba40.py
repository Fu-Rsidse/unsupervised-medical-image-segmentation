import os
import numpy as np
import SimpleITK as sitk
import glob
from typing import Tuple


def calculate_landmarks(image_path: str, max_files: int = 10) -> np.ndarray:
    """计算直方图标准化的 landmarks"""
    image_files = glob.glob(os.path.join(image_path, "*.nii"))[:max_files]
    if not image_files:
        raise FileNotFoundError(f"在 {image_path} 目录中未找到 .nii 文件")

    landmarks_list = []
    for file in image_files:
        img_sitk = sitk.ReadImage(file)
        img_arr = sitk.GetArrayFromImage(img_sitk)
        valid_voxels = img_arr[img_arr > 0]
        if len(valid_voxels) == 0:
            continue
        landmarks = np.percentile(valid_voxels, np.linspace(0, 100, 13))
        landmarks_list.append(landmarks)

    if not landmarks_list:
        raise ValueError("无有效图像数据用于计算landmarks")
    return np.mean(landmarks_list, axis=0)


def histogram_standardization(img_arr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """执行直方图标准化"""
    img_std = img_arr.copy()
    mask = img_std > 0
    if np.sum(mask) == 0:
        return img_std

    for i in range(len(landmarks) - 1):
        lower = landmarks[i]
        upper = landmarks[i + 1]
        target_lower = 100 * i / (len(landmarks) - 1)
        target_upper = 100 * (i + 1) / (len(landmarks) - 1)
        img_std[mask & (img_arr >= lower) & (img_arr < upper)] = \
            (img_arr[mask & (img_arr >= lower) & (img_arr < upper)] - lower) * \
            (target_upper - target_lower) / (upper - lower) + target_lower

    img_std[mask & (img_arr >= landmarks[-1])] = 100.0
    return img_std


def resample_image(
        img_sitk: sitk.Image,
        target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        is_label: bool = False
) -> sitk.Image:
    """重采样图像/标签到目标间距"""
    original_spacing = img_sitk.GetSpacing()
    original_size = img_sitk.GetSize()
    original_direction = img_sitk.GetDirection()
    original_origin = img_sitk.GetOrigin()

    target_size = [
        int(np.round(original_size[0] * original_spacing[0] / target_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / target_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / target_spacing[2]))
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(original_direction)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(img_sitk)


def center_crop(
        img_arr: np.ndarray,
        target_size: Tuple[int, int, int] = (96, 96, 96)
) -> np.ndarray:
    """中心裁剪到目标尺寸"""
    z_start = max(0, (img_arr.shape[0] - target_size[0]) // 2)
    y_start = max(0, (img_arr.shape[1] - target_size[1]) // 2)
    x_start = max(0, (img_arr.shape[2] - target_size[2]) // 2)

    # 确保裁剪尺寸不超过原始尺寸
    z_end = min(z_start + target_size[0], img_arr.shape[0])
    y_end = min(y_start + target_size[1], img_arr.shape[1])
    x_end = min(x_start + target_size[2], img_arr.shape[2])

    return img_arr[z_start:z_end, y_start:y_end, x_start:x_end]


def histogram_stardardization_resample_center_crop(
        mapping: np.ndarray,
        img_input_path: str,
        label_input_path: str,
        output_path_hs_small: str,
        output_path_mask: str,
        batch_i: Tuple[int, int] = (1, 2),
        batch_j: Tuple[int, int] = (1, 10)
) -> None:
    """核心函数：生成small尺寸数据（修复尺寸不匹配+物理空间对齐）"""
    os.makedirs(output_path_hs_small, exist_ok=True)
    os.makedirs(output_path_mask, exist_ok=True)

    for i in range(batch_i[0], batch_i[1] + 1):
        for j in range(batch_j[0], batch_j[1] + 1):
            filename = f"l{i}_to_l{j}.nii"
            img_path = os.path.join(img_input_path, filename)
            mask_path = os.path.join(label_input_path, filename)

            if not os.path.exists(img_path):
                print(f"警告：图像文件 {filename} 不存在，跳过")
                continue
            if not os.path.exists(mask_path):
                print(f"警告：标签文件 {filename} 不存在，跳过")
                continue

            # -------------------------- 处理图像（修复物理空间对齐） --------------------------
            img_sitk = sitk.ReadImage(img_path)
            img_arr = sitk.GetArrayFromImage(img_sitk)
            img_std_arr = histogram_standardization(img_arr, mapping)

            # 1. 标准化图像：同步原始物理信息
            img_std_sitk = sitk.GetImageFromArray(img_std_arr)
            img_std_sitk.SetSpacing(img_sitk.GetSpacing())
            img_std_sitk.SetOrigin(img_sitk.GetOrigin())
            img_std_sitk.SetDirection(img_sitk.GetDirection())

            # 2. 重采样图像
            img_resampled = resample_image(img_std_sitk, is_label=False)
            img_resampled_arr = sitk.GetArrayFromImage(img_resampled)
            resampled_spacing = img_resampled.GetSpacing()  # 重采样后间距
            resampled_origin = img_resampled.GetOrigin()    # 重采样后原点

            # 3. 中心裁剪
            img_cropped_arr = center_crop(img_resampled_arr)

            # 4. 计算裁剪后的新原点（关键：解决尺寸不匹配问题）
            z_resampled, y_resampled, x_resampled = img_resampled_arr.shape
            z_cropped, y_cropped, x_cropped = img_cropped_arr.shape

            # 像素偏移量 → 物理偏移量（像素数 × 间距）
            z_offset_phys = ((z_resampled - z_cropped) // 2) * resampled_spacing[2]
            y_offset_phys = ((y_resampled - y_cropped) // 2) * resampled_spacing[1]
            x_offset_phys = ((x_resampled - x_cropped) // 2) * resampled_spacing[0]

            # 新原点 = 重采样原点 + 物理偏移量
            new_origin = (
                resampled_origin[0] + x_offset_phys,
                resampled_origin[1] + y_offset_phys,
                resampled_origin[2] + z_offset_phys
            )

            # 5. 生成裁剪后图像并设置物理信息
            img_cropped_sitk = sitk.GetImageFromArray(img_cropped_arr)
            img_cropped_sitk.SetSpacing(resampled_spacing)
            img_cropped_sitk.SetOrigin(new_origin)
            img_cropped_sitk.SetDirection(img_resampled.GetDirection())

            # -------------------------- 处理标签（与图像严格对齐） --------------------------
            mask_sitk = sitk.ReadImage(mask_path)

            # 1. 重采样标签（用最近邻插值，避免标签值失真）
            mask_resampled = resample_image(mask_sitk, is_label=True)
            mask_resampled_arr = sitk.GetArrayFromImage(mask_resampled)

            # 2. 中心裁剪（与图像用相同逻辑，确保尺寸一致）
            mask_cropped_arr = center_crop(mask_resampled_arr)

            # 3. 生成裁剪后标签并同步图像物理信息（确保对齐）
            mask_cropped_sitk = sitk.GetImageFromArray(mask_cropped_arr)
            mask_cropped_sitk.SetSpacing(resampled_spacing)  # 和图像间距一致
            mask_cropped_sitk.SetOrigin(new_origin)          # 和图像原点一致
            mask_cropped_sitk.SetDirection(img_resampled.GetDirection())  # 和图像方向一致

            # -------------------------- 保存数据 --------------------------
            output_img_path = os.path.join(output_path_hs_small, filename)
            output_mask_path = os.path.join(output_path_mask, filename)
            sitk.WriteImage(img_cropped_sitk, output_img_path, useCompression=False)
            sitk.WriteImage(mask_cropped_sitk, output_mask_path, useCompression=False)
            print(f"已处理：i={i}, j={j} → 保存到 {os.path.basename(output_img_path)}")


if __name__ == '__main__':
    # 路径配置（确保与你的数据集路径一致）
    img_input_dir = "../datasets/LPBA40/LPBA40_rigidly_registered_pairs"
    label_input_dir = "../datasets/LPBA40/LPBA40_rigidly_registered_label_pairs"
    output_small_img = "../datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small"
    output_small_mask = "../datasets/LPBA40/LPBA40_rigidly_registered_label_pairs_small"
    current_batch_i = (1, 40)  # 处理所有40个受试者
    current_batch_j = (1, 10)   # 每个受试者配对10个固定图像
    mapping_path = "../datasets/LPBA40/mapping.npy"

    # 加载或计算mapping
    if os.path.exists(mapping_path):
        mapping = np.load(mapping_path)
        print(f"✅ 已加载mapping：{mapping_path}")
    else:
        mapping = calculate_landmarks(img_input_dir, max_files=10)
        np.save(mapping_path, mapping)
        print(f"✅ 已计算并保存mapping：{mapping_path}")
    print("Mapping值：", np.round(mapping, 2))

    # 执行预处理
    print(f"\n🚀 开始处理批次：i={current_batch_i[0]}~{current_batch_i[1]}, j={current_batch_j[0]}~{current_batch_j[1]}")
    histogram_stardardization_resample_center_crop(
        mapping=mapping,
        img_input_path=img_input_dir,
        label_input_path=label_input_dir,
        output_path_hs_small=output_small_img,
        output_path_mask=output_small_mask,
        batch_i=current_batch_i,
        batch_j=current_batch_j
    )

    print(f"\n🎉 批次处理完成！")
    print(f"📁 small尺寸图像：{output_small_img}")
    print(f"📁 small尺寸标签：{output_small_mask}")

