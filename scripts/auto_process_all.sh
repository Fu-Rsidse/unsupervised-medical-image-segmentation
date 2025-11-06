#!/bin/bash

# --------------------------
# 配置参数（无需修改，已适配你的路径）
# --------------------------
# 项目根目录
PROJECT_ROOT="/home/anna/Documents/medical_image/unsupervised-medical-image-segmentation-master"
# 图像目录（.hdr/.nii存放位置）
IMAGE_DIR="${PROJECT_ROOT}/datasets/LPBA40/LPBA40_rigidly_registered_pairs"
# 转换脚本路径（之前创建的batch_convert_nii.py）
CONVERT_SCRIPT="${IMAGE_DIR}/batch_convert_nii.py"
# 预处理脚本路径
PREPROCESS_SCRIPT="${PROJECT_ROOT}/scripts/preprocessing_lpba40.py"
# 总批次配置：i从1到40，每批处理2个i（如1-2、3-4...）
START_I=1
END_I=40
BATCH_SIZE_I=2  # 每批处理2个i值（对应20个文件）
J_RANGE="1 2 3 4 5 6 7 8 9 10"  # j固定1~10


# --------------------------
# 函数：修改转换脚本的batch_i范围
# --------------------------
modify_convert_script() {
    local batch_i_start=$1
    local batch_i_end=$2
    # 生成batch_i列表（如[1,2]）
    local batch_i_list="[$batch_i_start, $batch_i_end]"
    # 修改转换脚本中的batch_i
    sed -i "s/batch_i = \[.*\]/batch_i = $batch_i_list/" "$CONVERT_SCRIPT"
    echo "✅ 转换脚本已修改为处理i=$batch_i_start~$batch_i_end"
}


# --------------------------
# 函数：修改预处理脚本的current_batch_i范围
# --------------------------
modify_preprocess_script() {
    local batch_i_start=$1
    local batch_i_end=$2
    # 修改预处理脚本中的current_batch_i
    sed -i "s/current_batch_i = (.*)/current_batch_i = ($batch_i_start, $batch_i_end)/" "$PREPROCESS_SCRIPT"
    echo "✅ 预处理脚本已修改为处理i=$batch_i_start~$batch_i_end"
}


# --------------------------
# 函数：删除当前批次的原始文件（.hdr/.nii）
# --------------------------
delete_raw_files() {
    local batch_i_start=$1
    local batch_i_end=$2
    echo -e "\n🗑️  开始删除i=$batch_i_start~$batch_i_end的原始文件..."
    for i in $(seq $batch_i_start $batch_i_end); do
        for j in $J_RANGE; do
            rm -f "${IMAGE_DIR}/l${i}_to_l${j}.hdr"
            rm -f "${IMAGE_DIR}/l${i}_to_l${j}.nii"
        done
    done
    echo "✅ 已删除i=$batch_i_start~$batch_i_end的原始文件"
}


# --------------------------
# 主流程：循环处理所有批次
# --------------------------
echo "🚀 开始自动批处理LPBA40数据集（共$(( (END_I - START_I + 1) / BATCH_SIZE_I ))批）"
for (( batch_i_start=START_I; batch_i_start<=END_I; batch_i_start+=BATCH_SIZE_I )); do
    # 计算当前批次的i结束值（如1→2，3→4...）
    batch_i_end=$(( batch_i_start + BATCH_SIZE_I - 1 ))
    # 避免最后一批i超过END_I（如40是偶数，无需处理）
    if [ $batch_i_end -gt $END_I ]; then
        batch_i_end=$END_I
    fi

    echo -e "\n=================================================="
    echo "📌 当前处理批次：i=$batch_i_start~$batch_i_end, j=1~10"
    echo "=================================================="

    # 1. 修改转换脚本并执行转换
    modify_convert_script $batch_i_start $batch_i_end
    echo -e "\n🔄 正在转换.hdr为.nii..."
    cd "$IMAGE_DIR" || exit
    python "$CONVERT_SCRIPT"

    # 2. 修改预处理脚本并执行预处理
    modify_preprocess_script $batch_i_start $batch_i_end
    echo -e "\n⚙️  正在执行预处理（生成small尺寸）..."
    cd "${PROJECT_ROOT}/scripts" || exit
    python "$PREPROCESS_SCRIPT"

    # 3. 删除当前批次的原始文件（释放空间）
    delete_raw_files $batch_i_start $batch_i_end

    echo -e "\n✅ 批次i=$batch_i_start~$batch_i_end处理完成！"
done


# --------------------------
# 处理完成提示
# --------------------------
echo -e "\n🎉 所有批次处理完成！"
echo "📁 small尺寸图像目录：${PROJECT_ROOT}/datasets/LPBA40/LPBA40_rigidly_registered_pairs_histogram_standardization_small"
echo "📁 small尺寸标签目录：${PROJECT_ROOT}/datasets/LPBA40/LPBA40_rigidly_registered_label_pairs_small"
echo "💡 可开始执行模型训练命令！"

