# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Auto-anchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import colorstr


def check_anchor_order(m):
    # 检查 YOLOv5 Detect() 模块 m 中的锚框顺序是否与步幅顺序一致，并在必要时进行修正
    a = m.anchors.prod(-1).view(-1)  # 计算锚框的面积
    da = a[-1] - a[0]  # 计算锚框面积的差值
    ds = m.stride[-1] - m.stride[0]  # 计算步幅的差值
    if da.sign() != ds.sign():  # 如果面积和步幅的顺序不一致
        print('Reversing anchor order')  # 打印提示信息
        m.anchors[:] = m.anchors.flip(0)  # 反转锚框顺序


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # 检查锚点是否适合数据，如有必要则重新计算
    prefix = colorstr('autoanchor: ')
    print(f'\n{prefix}Analyzing anchors... ', end='')

    # 获取模型的检测层
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()

    # 计算图像尺寸的比例
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)

    # 随机缩放因子
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale

    # 计算宽高
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # 计算指标
        r = wh[:, None] / k[None]  # 计算比例
        x = torch.min(r, 1. / r).min(2)[0]  # 比例指标
        best = x.max(1)[0]  # 最佳比例
        aat = (x > 1. / thr).float().sum(1).mean()  # 超过阈值的锚点数量
        bpr = (best > 1. / thr).float().mean()  # 最佳可能召回率
        return bpr, aat

    # 获取当前锚点并考虑模型的步幅
    anchors = m.anchors.clone() * m.stride.to(m.anchors.device).view(-1, 1, 1)  # 当前锚点
    bpr, aat = metric(anchors.cpu().view(-1, 2))  # 计算指标
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')

    # 如果最佳可能召回率低于阈值，则尝试改善锚点
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to improve anchors, please wait...')
        na = m.anchors.numel() // 2  # 锚点数量
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            print(f'{prefix}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # 如果新锚点更好，则替换
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)  # 检查锚点顺序
            print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')

    print('')  # 换行


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ 创建经过kmeans进化的锚点

        参数:
            dataset: 数据集的路径或已加载的数据集
            n: 锚点的数量
            img_size: 用于训练的图像尺寸
            thr: 用于训练的锚点-标签宽高比阈值，默认为4.0
            gen: 使用遗传算法进化锚点的代数
            verbose: 是否打印所有结果

        返回:
            k: kmeans进化后的锚点

        用法:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans  # 导入kmeans函数

    thr = 1. / thr  # 将阈值反转，以便于后续比较
    prefix = colorstr('autoanchor: ')  # 设置打印前缀

    def metric(k, wh):  # 计算指标
        r = wh[:, None] / k[None]  # 计算宽高比
        x = torch.min(r, 1. / r).min(2)[0]  # 获取比例指标
        return x, x.max(1)[0]  # 返回比例和最佳比例

    def anchor_fitness(k):  # 计算锚点的适应度
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)  # 计算当前锚点的指标
        return (best * (best > thr).float()).mean()  # 计算适应度，只有满足阈值的才计入

    def print_results(k):  # 打印结果
        k = k[np.argsort(k.prod(1))]  # 按面积从小到大排序锚点
        x, best = metric(k, wh0)  # 计算当前锚点的指标
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # 计算最佳可能召回率
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')  # 打印召回率
        print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')  # 打印各类指标
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # 输出锚点的尺寸
        return k  # 返回锚点

    if isinstance(dataset, str):  # 如果输入的是文件路径
        with open(dataset, errors='ignore') as f:  # 以忽略错误的方式打开文件
            data_dict = yaml.safe_load(f)  # 读取数据字典
        from utils.datasets import LoadImagesAndLabels  # 导入数据加载工具
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)  # 加载训练数据集

    # 获取标签的宽高
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)  # 计算每个图像的形状比例
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # 合并所有标签的宽高

    # 过滤极小物体
    i = (wh0 < 3.0).any(1).sum()  # 统计小于3像素的标签数量
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')  # 警告信息
    wh = wh0[(wh0 >= 2.0).any(1)]  # 过滤掉小于2像素的标签

    # Kmeans计算
    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')  # 开始Kmeans计算
    s = wh.std(0)  # 计算宽高的标准差
    k, dist = kmeans(wh / s, n, iter=30)  # 执行Kmeans聚类
    assert len(k) == n, f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'  # 确保返回的锚点数量正确
    k *= s  # 还原锚点
    wh = torch.tensor(wh, dtype=torch.float32)  # 转换为张量
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # 原始数据转张量
    k = print_results(k)  # 打印锚点结果

    # 进化锚点
    npr = np.random  # 引用随机数生成器
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # 初始化适应度和参数
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')  # 进度条
    for _ in pbar:
        v = np.ones(sh)  # 初始化变异向量
        while (v == 1).all():  # 变异直到有变化
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)  # 生成变异
        kg = (k.copy() * v).clip(min=2.0)  # 生成新锚点
        fg = anchor_fitness(kg)  # 计算新锚点的适应度
        if fg > f:  # 如果新适应度更高
            f, k = fg, kg.copy()  # 更新适应度和锚点
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'  # 更新进度条描述
            if verbose:
                print_results(k)  # 打印新锚点结果

    return print_results(k)  # 返回最终的锚点结果
