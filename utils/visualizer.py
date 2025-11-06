import numpy as np
import os
import sys
import torch
from visdom import Visdom
import time
from datetime import datetime


class Visualizer():
    """可视化工具类，用于训练过程中的损失和图像可视化"""
    def __init__(self, opt):
        self.opt = opt
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.env = opt.display_env
        self.server = opt.display_server
        self.ncols = opt.ncols
        
        # 初始化Visdom
        self.vis = Visdom(
            server=self.server,
            port=self.port,
            env=self.env,
            raise_exceptions=True
        )
        
        # 记录日志
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visuals, epoch, save_result=False):
        """显示当前结果图像"""
        if self.display_id > 0:
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: %dpx; height: %dpx; padding: 2px; outline: 1px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                
                for label, image in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image.transpose(2, 0, 1))  # 调整维度用于显示
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                
                white_image = np.ones_like(image.transpose(2, 0, 1)) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                
                try:
                    self.vis.images(
                        images,
                        nrow=ncols,
                        win=self.display_id + 1,
                        padding=2,
                        opts=dict(title=title + ' images')
                    )
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2, opts=dict(title=title + ' labels'))
                except Exception as e:
                    print(f"Visdom显示失败: {str(e)}")

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """绘制损失曲线或指标曲线"""
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' metrics over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'value'
                },
                win=self.display_id
            )
        except Exception as e:
            print(f"Visdom绘图失败: {str(e)}")

    def print_current_losses(self, epoch, i, losses, t, t_data):
        """打印当前损失并保存到日志"""
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


def get_visdom_kwargs(opt):
    """创建并返回Visualizer实例（修复导入错误的关键函数）"""
    if opt.display_id > 0:
        return Visualizer(opt)
    return None

