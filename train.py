import time
import torch
import os
from options.train_options import TrainOptions
from dataloaders import CreateDataLoader
from models import create_model
from utils.evaluation_metric import AverageMeter


if __name__ == '__main__':
    # 1. 解析训练参数
    opt = TrainOptions().parse()
    # 2. 创建训练数据加载器
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# The number of training images = %d' % dataset_size)

    # 3. 创建模型
    model = create_model(opt)
    total_iters = 0  # 总迭代次数初始化

    # -------------------------- 断点续训逻辑：加载历史训练状态 --------------------------
    start_epoch = opt.epoch_count  # 默认从1开始训练
    if opt.continue_train:
        # 定义最新模型保存路径
        latest_model_path = os.path.join(opt.checkpoints_dir, opt.name, 'latest_net_Reg.pth')
        if not os.path.exists(latest_model_path):
            print(f"警告：未找到最新模型文件 {latest_model_path}，将从第1轮开始训练")
        else:
            # 加载模型（自动适配GPU/CPU）
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(latest_model_path, map_location=device)
            
            # 加载模型权重和优化器状态
            model.netReg.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer_Reg.load_state_dict(checkpoint['optimizer_state_dict'])
            # 读取历史训练轮次和总迭代次数
            start_epoch = checkpoint['epoch'] + 1
            total_iters = checkpoint['total_iters']
            
            print(f'✅ 断点续训成功：从第 {start_epoch} 轮（总迭代 {total_iters} 次）开始训练')
    # -------------------------------------------------------------------------------------

    # 4. 训练循环：从start_epoch开始
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # 5. 单轮训练：遍历所有训练数据
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            
            # 初始化t_data为0（关键修复：避免未定义）
            t_data = 0.0  # 添加这行，确保变量始终存在
            # 打印数据加载时间（每print_freq次迭代）
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time  # 仅此时更新为实际值

            # 更新迭代次数
            total_iters += opt.batchSize
            epoch_iter += opt.batchSize

            # 模型前向+反向传播
            model.set_input(data)
            model.optimize_parameters()

            # 打印训练损失
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batchSize
                # 现在t_data一定已定义，不会报错
                print(f"[Epoch {epoch}/{opt.niter+opt.niter_decay}] [Iter {epoch_iter}/{dataset_size*opt.batchSize}] "
                      f"total_loss: {losses['total']:.4f} | recon_loss: {losses['recon']:.4f} "
                      f"| smooth_loss: {losses['smooth']:.4f} | contrastive_loss: {losses['contrastive']:.4f} "
                      f"| t_comp: {t_comp:.2f}s | t_data: {t_data:.2f}s")

            # 保存最新模型
            if total_iters % opt.save_latest_freq == 0:
                print(f'\n📌 保存最新模型（epoch {epoch}，总迭代 {total_iters} 次）...')
                save_dir = os.path.join(opt.checkpoints_dir, opt.name)
                os.makedirs(save_dir, exist_ok=True)
                save_filename = f'iter_{total_iters}_net_Reg.pth' if opt.save_by_iter else 'latest_net_Reg.pth'
                save_path = os.path.join(save_dir, save_filename)

                torch.save(
                    {
                        'epoch': epoch,
                        'total_iters': total_iters,
                        'model_state_dict': model.netReg.state_dict(),
                        'optimizer_state_dict': model.optimizer_Reg.state_dict(),
                        'loss': losses['total']
                    },
                    save_path
                )
                print(f"✅ 最新模型已保存至：{save_path}")

            iter_data_time = time.time()

        # 6. 按epoch保存模型
        if epoch % opt.save_epoch_freq == 0:
            print(f'\n📌 保存第 {epoch} 轮模型（总迭代 {total_iters} 次）...')
            save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{epoch}_net_Reg.pth')

            torch.save(
                {
                    'epoch': epoch,
                    'total_iters': total_iters,
                    'model_state_dict': model.netReg.state_dict(),
                    'optimizer_state_dict': model.optimizer_Reg.state_dict(),
                    'loss': model.get_current_losses()['total']
                },
                save_path
            )
            print(f"✅ 第 {epoch} 轮模型已保存至：{save_path}")

        # 7. 打印单轮训练总耗时
        epoch_total_time = time.time() - epoch_start_time
        print(f'\n🔚 第 {epoch} 轮训练结束 | 总耗时：{epoch_total_time:.0f} 秒\n')

        # 8. 更新学习率
        model.update_learning_rate()

