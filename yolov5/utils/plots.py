# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Plotting utils
"""

import math
import os
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.general import user_config_dir, is_ascii, is_chinese, xywh2xyxy, xyxy2xywh
from utils.metrics import fitness

# Settings
CONFIG_DIR = user_config_dir()  # Ultralytics settings dir
RANK = int(os.getenv('RANK', -1))
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


class Colors:
    """
    颜色类用于管理颜色调色板，基于 Ultralytics 颜色方案。

    方法：
        __call__(i, bgr=False): 获取指定索引的颜色，支持 BGR 格式。
        hex2rgb(h): 将十六进制颜色字符串转换为 RGB 元组。
    """

    def __init__(self):
        """
        初始化颜色调色板。
        使用 Ultralytics 颜色调色板的十六进制表示，并将其转换为 RGB 格式。
        """
        # Ultralytics 颜色调色板（十六进制形式）
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231',
               '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC',
               '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')

        # 将十六进制颜色转换为 RGB 格式，并存储在调色板中
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)  # 颜色数量

    def __call__(self, i, bgr=False):
        """
        根据索引获取颜色。

        参数：
            i (int): 颜色索引。
            bgr (bool): 是否返回 BGR 格式的颜色，默认为 False。

        返回：
            tuple: RGB 或 BGR 格式的颜色元组。
        """
        # 获取指定索引的颜色，使用取模以处理超出范围的索引
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c  # 根据需要返回 BGR 或 RGB

    @staticmethod
    def hex2rgb(h):
        """
        将十六进制颜色字符串转换为 RGB 元组。

        参数：
            h (str): 十六进制颜色字符串，格式为 '#RRGGBB'。

        返回：
            tuple: RGB 格式的颜色元组。
        """
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

# 创建 Colors 类的实例
colors = Colors()  # 用于在后续的绘图或可视化中调用颜色

def check_font(font='Arial.ttf', size=10):
    """
    返回一个 PIL 的 TrueType 字体。如果字体不存在，则从 CONFIG_DIR 下载必要的字体。

    参数:
        font (str): 字体文件名，默认为 'Arial.ttf'。
        size (int): 字体大小，默认为 10。

    返回:
        ImageFont: PIL 的 TrueType 字体对象。
    """
    font = Path(font)  # 将字体路径转换为 Path 对象
    font = font if font.exists() else (CONFIG_DIR / font.name)  # 检查字体是否存在，如果不存在，则构造 CONFIG_DIR 中的字体路径
    try:
        # 尝试加载 TrueType 字体
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception as e:  # 如果字体缺失，则下载
        url = "https://ultralytics.com/assets/" + font.name  # 构造字体下载 URL
        print(f'Downloading {url} to {font}...')  # 打印下载信息
        torch.hub.download_url_to_file(url, str(font), progress=False)  # 下载字体文件
        return ImageFont.truetype(str(font), size)  # 下载后加载字体


class Annotator:
    """
    YOLOv5 Annotator for training/validation mosaics and JPGs,
    as well as for detecting and annotating hub inference results.
    """
    if RANK in (-1, 0):
        check_font()  # Download TTF font if necessary

    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        """
        Initialize the Annotator with an image and optional parameters for annotation.

        参数:
            im (np.ndarray or PIL.Image): 输入图像。
            line_width (int): 线宽，默认为根据图像大小计算的值。
            font_size (int): 字体大小，默认为根据图像大小计算的值。
            font (str): 字体文件名，默认为 'Arial.ttf'。
            pil (bool): 是否使用 PIL 进行绘图，默认为 False。
            example (str): 示例文本，用于决定使用的字体。
        """
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.pil = pil or not is_ascii(example) or is_chinese(example)  # 确定是否使用 PIL

        if self.pil:  # 使用 PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)  # 确保图像为 PIL 格式
            self.draw = ImageDraw.Draw(self.im)  # 创建绘图对象
            # 根据是否为中文选择字体
            self.font = check_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
                                   size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # 使用 cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # 线宽

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """
        在图像中添加一个框和标签。

        参数:
            box (tuple): 边界框的坐标 (x1, y1, x2, y2)。
            label (str): 标签文本。
            color (tuple): 边框颜色。
            txt_color (tuple): 标签文本颜色。
        """
        if self.pil or not is_ascii(label):  # 如果使用 PIL 或标签不是 ASCII
            self.draw.rectangle(box, width=self.lw, outline=color)  # 绘制框
            if label:  # 如果有标签
                w, h = self.font.getbbox(label)[2:]  # 获取标签宽度和高度 (Pillow>=10)
                outside = box[1] - h >= 0  # 检查标签是否可以在框外显示
                self.draw.rectangle([box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1], fill=color)  # 绘制标签背景
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)  # 绘制标签
        else:  # 使用 cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))  # 边框的两个角点
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)  # 绘制边框
            if label:
                tf = max(self.lw - 1, 1)  # 字体厚度
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # 获取文本宽度和高度
                outside = p1[1] - h - 3 >= 0  # 检查标签是否可以在框外显示
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3  # 计算标签背景的位置
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # 绘制填充的标签背景
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)  # 绘制标签文本

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """
        在图像中添加矩形（仅适用于 PIL）。

        参数:
            xy (tuple): 矩形的坐标。
            fill (tuple): 填充颜色。
            outline (tuple): 边框颜色。
            width (int): 边框宽度。
        """
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        """
        在图像中添加文本（仅适用于 PIL）。

        参数:
            xy (tuple): 文本的坐标。
            text (str): 要添加的文本。
            txt_color (tuple): 文本颜色。
        """
        w, h = self.font.getbbox(text)[2:]  # 获取文本宽度和高度 (Pillow>=10)
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)  # 绘制文本

    def result(self):
        """
        返回注释后的图像作为数组。

        返回:
            np.ndarray: 注释后的图像数组。
        """
        return np.asarray(self.im)  # 转换为 NumPy 数组并返回



def hist2d(x, y, n=100):
    """
    Create a 2D histogram from two sets of data.

    参数:
        x (np.ndarray): 第一维数据，数组形式。
        y (np.ndarray): 第二维数据，数组形式。
        n (int): 直方图的分辨率（即边缘的数量），默认值为 100。

    返回:
        np.ndarray: 经过对数变换的 2D 直方图值，数组形状为 (n, n)。
    """
    # 生成 x 和 y 的边缘
    xedges = np.linspace(x.min(), x.max(), n)  # x 轴的边缘
    yedges = np.linspace(y.min(), y.max(), n)  # y 轴的边缘

    # 计算 2D 直方图
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))

    # 确定每个点所在的直方图单元的索引
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)  # x 轴索引
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)  # y 轴索引

    # 返回对应索引的直方图值，进行对数变换以增强可视化效果
    return np.log(hist[xidx, yidx] + 1)  # 加 1 避免对数零值的情况


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    """
    Apply a lowpass Butterworth filter to the input data using forward-backward filtering.

    参数:
        data (np.ndarray): 输入信号数据，需要进行滤波的数组。
        cutoff (float): 截止频率，单位为赫兹 (Hz)。默认值为 1500 Hz。
        fs (float): 采样频率，单位为赫兹 (Hz)。默认值为 50000 Hz。
        order (int): 滤波器的阶数。默认值为 5。

    返回:
        np.ndarray: 经过低通滤波后的数据数组。
    """
    from scipy.signal import butter, filtfilt

    # 创建低通Butterworth滤波器
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs  # 奈奎斯特频率
        normal_cutoff = cutoff / nyq  # 归一化截止频率
        return butter(order, normal_cutoff, btype='low', analog=False)

    # 计算滤波器系数
    b, a = butter_lowpass(cutoff, fs, order=order)

    # 应用前向-后向滤波以避免相位延迟
    return filtfilt(b, a, data)  # 经过滤波后的数据


def output_to_target(output):
    """
    Convert model output to the target format suitable for evaluation or further processing.

    参数:
        output (list): 模型的输出结果，通常为包含检测框、置信度和类别信息的张量。

    返回:
        np.ndarray: 转换后的目标格式数组，格式为 [batch_id, class_id, x, y, w, h, conf]。
    """
    targets = []  # 初始化目标列表

    for i, o in enumerate(output):  # 遍历每个输出，i 为批次索引
        for *box, conf, cls in o.cpu().numpy():  # 解构每个输出的边界框信息、置信度和类别
            # 将输出转换为目标格式并添加到列表中
            targets.append([
                i,  # 批次索引
                cls,  # 类别索引
                *list(*xyxy2xywh(np.array(box)[None])),  # 转换为 [x, y, w, h] 格式
                conf  # 置信度
            ])

    return np.array(targets)  # 返回目标数组


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=1920, max_subplots=16):
    """
    Plot a grid of images with bounding boxes and labels.

    参数:
        images (torch.Tensor or np.ndarray): 输入图像，形状为 [batch_size, channels, height, width]。
        targets (np.ndarray): 目标数组，包含每个图像的检测框信息，格式为 [batch_id, class_id, x, y, w, h, conf]。
        paths (list, optional): 图像路径，用于在每个子图上显示文件名。
        fname (str, optional): 输出图像文件名，默认为 'images.jpg'。
        names (list, optional): 类别名称列表，用于在图像上标注类别。
        max_size (int, optional): 图像的最大尺寸，默认为 1920。
        max_subplots (int, optional): 最大子图数量，默认为 16。
    """

    # 将输入转换为 NumPy 数组（如果是 Torch 张量）
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # 如果图像值范围在 [0, 1] 之间，则进行反归一化
    if np.max(images[0]) <= 1:
        images *= 255.0  # de-normalise (optional)

    bs, _, h, w = images.shape  # 提取批次大小、通道数、高度和宽度
    bs = min(bs, max_subplots)  # 限制绘制图像的数量
    ns = np.ceil(bs ** 0.5)  # 计算子图数量（近似为平方根）

    # 初始化马赛克图像
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # 创建白色背景

    # 填充马赛克图像
    for i, im in enumerate(images):
        if i == max_subplots:  # 如果最后一批次的图像少于预期数量
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # 计算当前图像的位置
        im = im.transpose(1, 2, 0)  # 转换通道顺序
        mosaic[y:y + h, x:x + w, :] = im  # 填充马赛克

    # 可选的图像调整大小
    scale = max_size / ns / max(h, w)  # 计算缩放比例
    if scale < 1:
        h = math.ceil(scale * h)  # 按比例调整高度
        w = math.ceil(scale * w)  # 按比例调整宽度
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))  # 调整马赛克图像大小

    # 注释设置
    fs = int((h + w) * ns * 0.01)  # 字体大小
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)  # 创建 Annotator 实例

    # 绘制边界框和标签
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # 计算当前图像位置
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # 绘制边框
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # 显示文件名

        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # 获取当前图像的目标
            boxes = xywh2xyxy(ti[:, 2:6]).T  # 转换为 [x1, y1, x2, y2] 格式
            classes = ti[:, 1].astype('int')  # 类别索引
            labels = ti.shape[1] == 6  # 检查是否包含置信度列
            conf = None if labels else ti[:, 6]  # 检查置信度的存在性

            # 处理边界框坐标
            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # 如果是归一化坐标
                    boxes[[0, 2]] *= w  # 转换为像素坐标
                    boxes[[1, 3]] *= h
                elif scale < 1:  # 如果图像被缩放，绝对坐标需要缩放
                    boxes *= scale

            # 更新边界框位置
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y

            # 绘制每个边界框
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]  # 当前类别
                color = colors(cls)  # 获取颜色
                cls = names[cls] if names else cls  # 获取类别名称
                if labels or conf[j] > 0.25:  # 0.25 置信度阈值
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'  # 生成标签
                    annotator.box_label(box, label, color=color)  # 绘制边界框及标签

    annotator.im.save(fname)  # 保存输出图像


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    """
    Plot the learning rate (LR) schedule over a specified number of epochs.

    参数:
        optimizer (torch.optim.Optimizer): 用于优化的优化器实例。
        scheduler (torch.optim.lr_scheduler): 学习率调度器实例。
        epochs (int, optional): 要模拟的训练轮数，默认为 300。
        save_dir (str, optional): 保存图像的目录，默认为当前目录。

    该函数通过模拟训练过程，绘制每个 epoch 的学习率变化情况。
    """

    # 复制优化器和调度器，以避免修改原始对象
    optimizer, scheduler = copy(optimizer), copy(scheduler)
    y = []  # 初始化学习率记录列表

    # 模拟训练过程
    for _ in range(epochs):
        scheduler.step()  # 更新学习率
        y.append(optimizer.param_groups[0]['lr'])  # 记录当前学习率

    # 绘制学习率曲线
    plt.plot(y, '.-', label='LR')  # 使用点线图展示学习率
    plt.xlabel('epoch')  # x轴标签
    plt.ylabel('LR')  # y轴标签
    plt.grid()  # 显示网格
    plt.xlim(0, epochs)  # 设置x轴范围
    plt.ylim(0)  # 设置y轴范围

    # 保存图像
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)  # 指定图像分辨率为200 DPI
    plt.close()  # 关闭当前图像


def plot_val_txt():
    """
    绘制 val.txt 中的坐标直方图。

    从 val.txt 文件中加载数据，转换为中心坐标格式，并绘制二维直方图以及两个一维直方图。

    流程：
    1. 加载 val.txt 文件中的数据。
    2. 将数据转换为 (中心 x, 中心 y) 格式。
    3. 绘制二维直方图，显示中心坐标分布。
    4. 绘制两个一维直方图，分别显示 x 和 y 方向的分布。
    """

    # 加载 val.txt 文件中的数据
    x = np.loadtxt('val.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])  # 将边界框格式转换为 (cx, cy, w, h)
    cx, cy = box[:, 0], box[:, 1]  # 提取中心坐标

    # 绘制二维直方图
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)  # 绘制 2D 直方图
    ax.set_aspect('equal')  # 设置坐标轴比例相等
    plt.savefig('hist2d.png', dpi=300)  # 保存二维直方图

    # 绘制一维直方图
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)  # 绘制 cx 的直方图
    ax[1].hist(cy, bins=600)  # 绘制 cy 的直方图
    plt.savefig('hist1d.png', dpi=200)  # 保存一维直方图


def plot_targets_txt():
    """
    绘制 targets.txt 中目标的直方图。

    从 targets.txt 文件加载目标数据，并绘制每个目标属性的直方图，包括 x 坐标、y 坐标、宽度和高度。

    流程：
    1. 加载 targets.txt 文件中的数据。
    2. 创建一个 2x2 的子图以容纳四个直方图。
    3. 对每个目标属性（x, y, width, height）绘制直方图，并添加均值和标准差的图例。
    4. 保存直方图为 targets.jpg。
    """

    # 加载 targets.txt 文件中的数据并转置
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']  # 目标属性标签

    # 创建 2x2 子图
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()  # 将二维数组展平，方便索引

    # 绘制每个目标属性的直方图
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))  # 直方图及均值/标准差
        ax[i].legend()  # 显示图例
        ax[i].set_title(s[i])  # 设置子图标题

    plt.savefig('targets.jpg', dpi=200)  # 保存直方图


def plot_val_study(file='', dir='', x=None):  # from utils.plots import *; plot_val_study()
    # 绘制由 val.py 生成的 study.txt 文件（或绘制目录中所有 study*.txt 文件）
    save_dir = Path(file).parent if file else Path(dir)  # 确定保存目录
    plot2 = False  # 是否绘制额外的结果
    if plot2:
        ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()  # 创建额外的子图

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)  # 创建主图
    # 遍历所有以 study 开头的文本文件
    for f in sorted(save_dir.glob('study*.txt')):
        # 从文件中加载数据，指定需要的列
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)  # 确定 x 轴数据
        if plot2:
            # 如果需要，绘制额外的结果
            s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_preprocess (ms/img)', 't_inference (ms/img)', 't_NMS (ms/img)']
            for i in range(7):
                ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)  # 绘制数据
                ax[i].set_title(s[i])  # 设置标题

        # 找到最佳 mAP@.5 的索引
        j = y[3].argmax() + 1
        ax2.plot(y[5, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))  # 绘制主图数据

    # 绘制 EfficientDet 的数据（示例线）
    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    # 设置图表样式
    ax2.grid(alpha=0.2)  # 网格
    ax2.set_yticks(np.arange(20, 60, 5))  # y 轴刻度
    ax2.set_xlim(0, 57)  # x 轴范围
    ax2.set_ylim(25, 55)  # y 轴范围
    ax2.set_xlabel('GPU Speed (ms/img)')  # x 轴标签
    ax2.set_ylabel('COCO AP val')  # y 轴标签
    ax2.legend(loc='lower right')  # 图例位置
    f = save_dir / 'study.png'  # 输出文件路径
    print(f'Saving {f}...')  # 打印保存信息
    plt.savefig(f, dpi=300)  # 保存图表



def plot_labels(labels, names=(), save_dir=Path('')):
    """
    绘制数据集标签的分布和相关性。

    参数：
    - labels: ndarray，形状为 (N, 5)，包含类标签和框坐标 [class, x_center, y_center, width, height]。
    - names: tuple，包含类名的列表（可选）。
    - save_dir: Path，保存绘图结果的目录。

    流程：
    1. 提取类标签和框坐标，并计算类的数量。
    2. 使用 Seaborn 绘制相关性图（correlogram）。
    3. 绘制类标签的直方图以及框的分布。
    4. 在画布上绘制前1000个框的可视化。
    5. 保存生成的图像到指定目录。

    注意：
    - 确保提供的 labels 数组具有正确的形状和数据类型。
    - 提供的 names 列表长度不应超过30，以便能够正常显示。

    """
    print('Plotting labels... ')
    c, b = labels[:, 0], labels[:, 1:].transpose()  # 类别，框
    nc = int(c.max() + 1)  # 类别数量
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # Seaborn 相关性图
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # Matplotlib 标签绘图
    matplotlib.use('svg')  # 提高速度
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')

    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')

    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # 矩形框绘制
    labels[:, 1:3] = 0.5  # 中心
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # 绘制

    ax[1].imshow(img)
    ax[1].axis('off')

    # 隐藏边框
    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()


def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    """
    绘制 iDetection 的每张图片的日志信息。

    参数：
    - start: int，开始绘制的索引，默认为 0。
    - stop: int，结束绘制的索引，默认为 0（表示绘制所有）。
    - labels: tuple，图例标签列表（可选）。
    - save_dir: str，保存结果的目录。

    流程：
    1. 创建一个包含多个子图的绘图区域。
    2. 遍历指定目录中所有符合条件的文本文件。
    3. 从文件中加载数据，并选择要绘制的时间范围。
    4. 将数据绘制到相应的子图上。
    5. 保存生成的图像。

    注意：
    - 确保提供的目录中包含格式正确的日志文件。
    - labels 列表的长度应与日志文件数量一致，以确保图例正确显示。

    """
    # 创建绘图区域
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']

    # 获取日志文件列表
    files = list(Path(save_dir).glob('frames*.txt'))

    # 遍历每个文件
    for fi, f in enumerate(files):
        try:
            # 加载数据并进行预处理
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]  # 剔除前后无关行
            n = results.shape[1]  # 数据行数
            x = np.arange(start, min(stop, n) if stop else n)  # 选择绘制范围
            results = results[:, x]
            t = (results[0] - results[0].min())  # 将时间调整为从 0 开始
            results[0] = x

            # 在每个子图中绘制数据
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])  # 设置子图标题
                    a.set_xlabel('time (s)')  # 设置 x 轴标签

                    # 可选：设置 y 轴下限
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)

                    # 隐藏顶部和右侧边框
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()  # 如果没有数据，则移除子图
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    ax[1].legend()  # 添加图例
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)  # 保存图像


def plot_evolve(evolve_csv='path/to/evolve.csv'):  # from utils.plots import *; plot_evolve()
    # 绘制 evolve.csv 中的超参数演化结果
    evolve_csv = Path(evolve_csv)  # 将 evolve_csv 转换为 Path 对象
    data = pd.read_csv(evolve_csv)  # 读取 CSV 文件
    keys = [x.strip() for x in data.columns]  # 获取列名并去除多余空格
    x = data.values  # 获取数据值
    f = fitness(x)  # 计算适应度
    j = np.argmax(f)  # 找到最大适应度的索引
    plt.figure(figsize=(10, 12), tight_layout=True)  # 创建图形
    matplotlib.rc('font', **{'size': 8})  # 设置字体大小

    # 遍历每个超参数，绘制散点图
    for i, k in enumerate(keys[7:]):  # 从第8列开始（超参数）
        v = x[:, 7 + i]  # 获取当前超参数的值
        mu = v[j]  # 获取最佳单一结果
        plt.subplot(6, 5, i + 1)  # 创建子图
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')  # 绘制散点图
        plt.plot(mu, f.max(), 'k+', markersize=15)  # 绘制最佳结果的标记
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # 设置标题，限制字符数为40
        if i % 5 != 0:
            plt.yticks([])  # 隐藏 y 轴刻度
        print('%15s: %.3g' % (k, mu))  # 打印超参数及其最佳值

    f = evolve_csv.with_suffix('.png')  # 设置保存的文件名
    plt.savefig(f, dpi=200)  # 保存图像
    plt.close()  # 关闭图形
    print(f'Saved {f}')  # 打印保存信息


def plot_results(file='path/to/results.csv', dir=''):
    """
    绘制训练结果的 CSV 文件。

    参数：
    - file: str，结果文件的路径（默认为 'path/to/results.csv'）。
    - dir: str，保存结果的目录。

    使用示例：
    from utils.plots import *; plot_results('path/to/results.csv')

    流程：
    1. 确定保存目录。
    2. 创建绘图区域，并准备子图。
    3. 遍历所有以 results 开头的 CSV 文件。
    4. 从文件中加载数据并绘制到子图上。
    5. 保存生成的图像。

    注意：
    - 确保指定目录中存在符合条件的 CSV 文件。
    - CSV 文件的第一列应为 x 轴数据，其他列为 y 轴数据。

    """
    # 确定保存目录
    save_dir = Path(file).parent if file else Path(dir)

    # 创建绘图区域
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()

    # 获取结果文件列表
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'

    # 遍历每个结果文件
    for fi, f in enumerate(files):
        try:
            data = pd.read_csv(f)  # 加载数据
            s = [x.strip() for x in data.columns]  # 列名
            x = data.values[:, 0]  # x 轴数据

            # 绘制数据
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j]  # y 轴数据
                # y[y == 0] = np.nan  # 可选：不显示零值
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)
                ax[i].set_title(s[j], fontsize=12)  # 设置标题

                # 可选：共享训练和验证损失的 y 轴
                # if j in [8, 9, 10]:
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')  # 错误处理

    ax[1].legend()  # 添加图例
    fig.savefig(save_dir / 'results.png', dpi=200)  # 保存图像
    plt.close()  # 关闭图形窗口


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    可视化特征图。

    参数：
    - x: 需要可视化的特征图，形状为 (batch, channels, height, width)。
    - module_type: 模块类型，用于区分可视化的层。
    - stage: 模块在模型中的阶段。
    - n: 最大可绘制的特征图数量，默认值为 32。
    - save_dir: 保存结果的目录，默认为 'runs/detect/exp'。

    功能：
    该函数将输入的特征图中的特定模块的特征进行可视化，并保存为图像文件。

    流程：
    1. 检查模块类型是否为 'Detect'，若不是，继续执行。
    2. 获取输入特征图的形状信息。
    3. 根据通道数和最大绘制数量确定绘制特征图的数量。
    4. 使用 matplotlib 创建子图，并绘制特征图。
    5. 保存生成的图像文件。

    """
    if 'Detect' not in module_type:  # 如果模块类型不包含 'Detect'
        batch, channels, height, width = x.shape  # 获取输入特征图的维度

        if height > 1 and width > 1:  # 确保特征图的高度和宽度大于 1
            f = f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # 构造文件名

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # 选择第一个 batch，按通道分块
            n = min(n, channels)  # 确定绘制的数量，不能超过通道数
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 创建子图
            ax = ax.ravel()  # 将二维数组展平
            plt.subplots_adjust(wspace=0.05, hspace=0.05)  # 调整子图间距

            for i in range(n):  # 循环绘制特征图
                ax[i].imshow(blocks[i].squeeze(), cmap='gray')  # 显示特征图
                ax[i].axis('off')  # 关闭坐标轴

            print(f'Saving {save_dir / f}... ({n}/{channels})')  # 打印保存信息
            plt.savefig(save_dir / f, dpi=300, bbox_inches='tight')  # 保存图像
            plt.close()  # 关闭图形窗口