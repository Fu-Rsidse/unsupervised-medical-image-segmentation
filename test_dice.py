import time
import numpy as np
from options.train_options import TrainOptions
from dataloaders import CreateDataLoader
from models import create_model
from utils.visualizer import Visualizer
from utils.evaluation_metric import AverageMeter


if __name__ == '__main__':
    # 1. 解析参数（强制测试模式）
    opt = TrainOptions().parse()
    opt.isTrain = False  # 确保模型不加载训练相关组件（如优化器）
    opt.epoch = opt.which_epoch  # 记录加载的模型epoch，用于日志
    
    # 2. 加载测试数据（用训练集替代验证集，解决样本数为0问题）
    test_data_loader = CreateDataLoader(opt)
    test_dataset = test_data_loader.load_data()
    test_dataset_size = len(test_data_loader)
    print(f'#test images = {test_dataset_size}')
    if test_dataset_size == 0:
        raise ValueError("测试数据集为空，请检查--dataroot路径是否正确！")
    
    # 3. 初始化模型和可视化器
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    # 4. 初始化Dice指标记录器（7个脑区+总Dice）
    total_dice_meter = AverageMeter()
    dice_meters = [AverageMeter() for _ in range(7)]
    brain_regions = ["FL", "PL", "OL", "TL", "CL", "Ptm", "Hpcp"]

    # 5. 执行测试（单轮测试，无多epoch循环）
    for i, data in enumerate(test_dataset):
        # 打印样本信息，方便定位异常样本
        A_path = str(data["A_paths"][0]) if "A_paths" in data else "Unknown"
        B_path = str(data["B_paths"][0]) if "B_paths" in data else "Unknown"
        print(f'\n{"="*50}')
        print(f'Processing test sample {i+1}/{test_dataset_size}:')
        print(f'  Moving image: {A_path.split("/")[-1]}')
        print(f'  Fixed image: {B_path.split("/")[-1]}')

        # 模型输入赋值
        model.set_input(data)
        
        # 调试打印，检查输入数据有效性
        print(f'  Debug: input_A (moving) shape: {model.input_A.shape}, min: {model.input_A.min():.2f}, max: {model.input_A.max():.2f}')
        print(f'  Debug: input_B (fixed) shape: {model.input_B.shape}, min: {model.input_B.min():.2f}, max: {model.input_B.max():.2f}')
        print(f'  Debug: input_B_atlas (fixed标签) shape: {model.input_B_atlas.shape}, unique values: {np.unique(model.input_B_atlas.data.cpu().numpy())}')

        # 模型推理（获取Dice结果）
        total_dice, dsc_list = model.test()

        # 调试打印，检查Dice计算中间结果
        print(f'  Debug: 各脑区Dice列表: {[f"{d:.4f}" for d in dsc_list]}')
        print(f'  Debug: 总Dice: {total_dice:.4f}')

        # 更新Dice指标
        total_dice_meter.update(total_dice)
        for idx, (dice_val, meter) in enumerate(zip(dsc_list, dice_meters)):
            meter.update(float(dice_val))

        # 可视化Dice指标
        if opt.display_id > 0 and hasattr(visualizer, 'plot_current_losses'):
            metrics = {
                'Total Dice': total_dice,
                **{f'{brain_regions[idx]}_Dice': dice_val for idx, dice_val in enumerate(dsc_list)}
            }
            # 正确调用绘图方法（仅传递三个必要参数）
            visualizer.plot_current_losses(int(opt.epoch), i / test_dataset_size, metrics)

        # 保存可视化结果（按频率保存）
        total_steps += opt.batchSize
        save_result = total_steps % opt.update_html_freq == 0
        if hasattr(model, 'get_current_visuals'):
            visuals = model.get_current_visuals()  # 先获取可视化数据
            if visuals:  # 仅当字典非空时调用
                visualizer.display_current_results(visuals, opt.epoch, save_result)
            else:
                print("Warning: 模型未返回可视化数据，跳过显示")


    # 6. 打印最终结果
    print('\n' + '='*80)
    print('Final Test Dice Results (Average):')
    print(f'  Overall Total Dice: {total_dice_meter.avg:.4f}')
    for idx, (meter, region) in enumerate(zip(dice_meters, brain_regions)):
        print(f'  Brain Region {idx+1} ({region}): {meter.avg:.4f}')
    print('='*80)

