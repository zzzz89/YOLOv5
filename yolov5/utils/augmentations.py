# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

import logging
import math
import random

import cv2
import numpy as np

from utils.general import colorstr, segment2box, resample_segments, check_version
from utils.metrics import bbox_ioa


class Albumentations:
    # YOLOv5 的 Albumentations 类（可选，仅在安装了该包时使用）
    def __init__(self):
        self.transform = None  # 初始化变换为空
        try:
            import albumentations as A  # 导入 albumentations 库
            check_version(A.__version__, '1.0.3')  # 检查版本是否满足要求

            # 定义图像增强的变换组合
            self.transform = A.Compose([
                A.Blur(p=0.01),  # 模糊处理
                A.MedianBlur(p=0.01),  # 中值模糊处理
                A.ToGray(p=0.01),  # 转为灰度图像
                A.CLAHE(p=0.01),  # 对比度限制的自适应直方图均衡化
                A.RandomBrightnessContrast(p=0.0),  # 随机亮度对比度调整
                A.RandomGamma(p=0.0),  # 随机伽马校正
                A.ImageCompression(quality_lower=75, p=0.0)  # 图像压缩，质量下限为75
            ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))  # 设置边界框参数，格式为 YOLO，标签字段为 'class_labels'

            logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))  # 记录应用的变换
        except ImportError:  # 如果包未安装，则跳过
            pass
        except Exception as e:  # 捕获其他异常
            logging.info(colorstr('albumentations: ') + f'{e}')  # 记录异常信息

    def __call__(self, im, labels, p=1.0):
        # 调用该类时执行图像增强
        if self.transform and random.random() < p:  # 如果变换存在且随机数小于 p
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # 进行变换
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])  # 更新图像和标签
        return im, labels  # 返回变换后的图像和标签


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV 颜色空间增强
    if hgain or sgain or vgain:  # 如果有色调、饱和度或明度增益
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # 随机增益
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))  # 将图像从 BGR 转换为 HSV 并分离通道
        dtype = im.dtype  # 获取图像的数据类型，通常为 uint8

        x = np.arange(0, 256, dtype=r.dtype)  # 创建 0 到 255 的数组
        lut_hue = ((x * r[0]) % 180).astype(dtype)  # 色调查找表（LUT），确保值在 0-180 范围内
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 饱和度查找表，确保值在 0-255 范围内
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 明度查找表，确保值在 0-255 范围内

        # 使用查找表对 HSV 通道进行增强
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # 将增强后的 HSV 图像转换回 BGR，并直接更新原图像（不需要返回）



def hist_equalize(im, clahe=True, bgr=False):
    # 对 BGR 图像 'im' 进行直方图均衡化，im.shape(n,m,3)，像素值范围为 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)  # 将图像转换为 YUV 颜色空间
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 创建 CLAHE 对象
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])  # 应用 CLAHE 增强 Y 通道
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # 对 Y 通道进行直方图均衡化
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # 将 YUV 图像转换回 BGR 或 RGB



def replicate(im, labels):
    # 复制标签
    h, w = im.shape[:2]  # 获取图像的高度和宽度
    boxes = labels[:, 1:].astype(int)  # 获取框的坐标并转换为整数
    x1, y1, x2, y2 = boxes.T  # 拆分框的坐标
    s = ((x2 - x1) + (y2 - y1)) / 2  # 计算边长（像素）

    # 选择边长最小的前 50% 的框进行复制
    for i in s.argsort()[:round(s.size * 0.5)]:  # 根据边长排序并选择最小的索引
        x1b, y1b, x2b, y2b = boxes[i]  # 原框坐标
        bh, bw = y2b - y1b, x2b - x1b  # 计算框的高度和宽度
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # 随机偏移 x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]  # 新框的坐标
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # 在新位置复制原框的内容
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)  # 添加新标签
    return im, labels  # 返回增强后的图像和标签



def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # 调整图像大小并填充，同时满足步幅倍数约束
    shape = im.shape[:2]  # 当前形状 [高度, 宽度]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)  # 如果 new_shape 是整数，转换为元组

    # 计算缩放比例（新 / 旧）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 高度和宽度的缩放比例
    if not scaleup:  # 只缩小，不放大（以提高验证 mAP）
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽、高比率
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 新的未填充尺寸
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽、高填充
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 根据步幅调整填充
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])  # 新的未填充尺寸
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽、高比率

    dw /= 2  # 将填充分配到两侧
    dh /= 2

    if shape[::-1] != new_unpad:  # 如果需要调整大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)  # 调整图像大小
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 上下填充
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 左右填充
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return im, ratio, (dw, dh)  # 返回调整后的图像、缩放比率和填充量


def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # 随机透视变换图像
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # 形状(h, w, c)
    width = im.shape[1] + border[1] * 2

    # 中心
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x 平移（像素）
    C[1, 2] = -im.shape[0] / 2  # y 平移（像素）

    # 透视变换
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x 透视（关于 y）
    P[2, 1] = random.uniform(-perspective, perspective)  # y 透视（关于 x）

    # 旋转和缩放
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)  # 随机角度
    s = random.uniform(1 - scale, 1 + scale)  # 随机缩放因子
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)  # 计算旋转矩阵

    # 剪切
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x 剪切（度）
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y 剪切（度）

    # 平移
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x 平移（像素）
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y 平移（像素）

    # 合并变换矩阵
    M = T @ S @ R @ P @ C  # 变换顺序（从右到左）非常重要
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # 图像是否已更改
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))  # 透视变换
        else:  # 仿射变换
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # 可视化（可选）
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # 原图
    # ax[1].imshow(im2[:, :, ::-1])  # 变换后的图像

    # 转换标签坐标
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)  # 检查是否使用分段
        new = np.zeros((n, 4))  # 新的边界框
        if use_segments:  # 变换分段
            segments = resample_segments(segments)  # 上采样
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # 变换
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # 透视缩放或仿射

                # 裁剪
                new[i] = segment2box(xy, width, height)

        else:  # 变换边界框
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # 变换
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # 透视缩放或仿射

            # 创建新的边界框
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # 裁剪
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # 过滤候选框
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets  # 返回变换后的图像和更新后的目标


def copy_paste(im, labels, segments, p=0.5):
    # 实现 Copy-Paste 数据增强 https://arxiv.org/abs/2012.07177，标签为 nx5 的 np.array(cls, xyxy)
    n = len(segments)  # 获取分段数量
    if p and n:  # 如果概率 p 和分段数量 n 都有效
        h, w, c = im.shape  # 获取图像的高度、宽度和通道数
        im_new = np.zeros(im.shape, np.uint8)  # 创建新图像用于存储增强效果
        for j in random.sample(range(n), k=round(p * n)):  # 随机选择部分分段进行增强
            l, s = labels[j], segments[j]  # 获取当前标签和分段
            box = w - l[3], l[2], w - l[1], l[4]  # 计算新的框
            ioa = bbox_ioa(box, labels[:, 1:5])  # 计算与现有标签的重叠面积比例
            if (ioa < 0.30).all():  # 允许现有标签被遮挡不超过 30%
                labels = np.concatenate((labels, [[l[0], *box]]), 0)  # 将新标签添加到标签列表
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))  # 更新分段
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)  # 绘制填充的轮廓

        result = cv2.bitwise_and(src1=im, src2=im_new)  # 获取与新图像的按位与
        result = cv2.flip(result, 1)  # 翻转图像（左右翻转）
        i = result > 0  # 获取需要替换的像素
        # i[:, :] = result.max(2).reshape(h, w, 1)  # 对通道进行处理（注释掉）
        im[i] = result[i]  # 替换原图像中的像素
    return im, labels, segments  # 返回增强后的图像、标签和分段


def cutout(im, labels, p=0.5):
    # 应用图像 Cutout 数据增强 https://arxiv.org/abs/1708.04552
    if random.random() < p:  # 根据概率 p 决定是否应用 Cutout
        h, w = im.shape[:2]  # 获取图像的高度和宽度
        # 定义不同尺度的遮罩比例
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # 图像尺寸比例
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # 随机生成遮罩的高度
            mask_w = random.randint(1, int(w * s))  # 随机生成遮罩的宽度

            # 计算遮罩框的坐标
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # 应用随机颜色遮罩
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # 返回未被遮挡的标签
            if len(labels) and s > 0.03:  # 只处理较大的遮罩
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)  # 创建遮罩框
                ioa = bbox_ioa(box, labels[:, 1:5])  # 计算与标签的重叠面积比例
                labels = labels[ioa < 0.60]  # 移除被遮挡超过 60% 的标签

    return labels  # 返回未被遮挡的标签


def mixup(im, labels, im2, labels2):
    # 应用 MixUp 数据增强 https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup 比例，alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)  # 根据比例混合两张图像
    labels = np.concatenate((labels, labels2), 0)  # 将两个标签合并
    return im, labels  # 返回混合后的图像和标签


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # 计算候选框：box1 为增强前的框，box2 为增强后的框
    # wh_thr (像素阈值)，ar_thr (宽高比阈值)，area_thr (面积比例阈值)

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]  # box1 的宽和高
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]  # box2 的宽和高

    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # 计算宽高比

    # 返回符合条件的候选框
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)
