import nibabel as nib
import os

# -------------------------- 路径配置（区分图像和标签） --------------------------
# 图像原始HDR/IMG文件夹
raw_img_dir = "../datasets/LPBA40/raw_img_hdr_img"
# 标签原始HDR/IMG文件夹
raw_label_dir = "../datasets/LPBA40/raw_label_hdr_img"
# 图像NII输出文件夹
output_img_dir = "../datasets/LPBA40/LPBA40_rigidly_registered_pairs"
# 标签NII输出文件夹
output_label_dir = "../datasets/LPBA40/LPBA40_rigidly_registered_label_pairs"

# 创建必要文件夹
os.makedirs(raw_img_dir, exist_ok=True)
os.makedirs(raw_label_dir, exist_ok=True)
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# -------------------------- 核心转换逻辑（分图像和标签） --------------------------
def convert_hdr_to_nii(input_dir, output_dir, is_image=True):
    """
    将HDR/IMG转为NII
    input_dir: 原始HDR/IMG所在文件夹
    output_dir: 转换后NII保存文件夹
    is_image: 是否为图像文件（仅用于打印区分）
    """
    for i in range(1, 41):  # LPBA40共40个受试者
        for j in range(1, 11):  # 每个受试者配对10个固定图像
            hdr_filename = f"l{i}_to_l{j}.hdr"
            img_filename = f"l{i}_to_l{j}.img"
            hdr_path = os.path.join(input_dir, hdr_filename)
            img_path = os.path.join(input_dir, img_filename)
            output_nii_path = os.path.join(output_dir, f"l{i}_to_l{j}.nii")

            # 检查文件是否存在
            if not os.path.exists(hdr_path):
                print(f"⚠️ {'图像' if is_image else '标签'}跳过：{hdr_filename} 不存在")
                continue
            if not os.path.exists(img_path):
                print(f"⚠️ {'图像' if is_image else '标签'}跳过：{img_filename} 不存在（与{hdr_filename}配套）")
                continue

            # 转换并保存
            try:
                nib_img = nib.load(hdr_path)
                nib.save(nib_img, output_nii_path)
                print(f"✅ {'图像' if is_image else '标签'}转换成功：{hdr_filename} → {os.path.basename(output_nii_path)}")
            except Exception as e:
                print(f"❌ {'图像' if is_image else '标签'}转换失败：{hdr_filename}，错误：{str(e)}")

# -------------------------- 执行转换（先图像，后标签） --------------------------
if __name__ == "__main__":
    # 1. 处理图像文件
    print("\n=== 开始转换图像文件（HDR→NII）===")
    convert_hdr_to_nii(raw_img_dir, output_img_dir, is_image=True)

    # 2. 处理标签文件
    print("\n=== 开始转换标签文件（HDR→NII）===")
    convert_hdr_to_nii(raw_label_dir, output_label_dir, is_image=False)

    print("\n🎉 所有转换完成！")
    print(f"图像NII文件路径：{output_img_dir}")
    print(f"标签NII文件路径：{output_label_dir}")

