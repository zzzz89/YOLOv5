# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import logging
import math
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import colorstr, increment_path, make_divisible, non_max_suppression, save_one_box, \
    scale_coords, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync

LOGGER = logging.getLogger(__name__)


def autopad(k, p=None):  # kernel, padding
    # 如果未提供 `p` 参数，则进行自动填充，使输出尺寸与输入相同 ('same' 填充)
    if p is None:
        # 如果 `k` 是整数，计算填充为 `k // 2`；如果 `k` 是列表或其他可迭代对象，计算每个元素的填充为 `x // 2`
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动填充
    return p  # 返回计算的填充值


class Conv(nn.Module):
    # 标准卷积模块
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # 输入通道数, 输出通道数, 卷积核大小, 步幅, 填充, 组数, 激活函数
        super().__init__()  # 调用父类构造函数进行初始化
        # 定义卷积层，自动计算填充，禁用偏置以配合批归一化层
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # 定义批归一化层，归一化输出
        # 定义激活函数，默认为 SiLU，如果 `act` 为 `False`，则使用 `nn.Identity()`，否则使用用户提供的模块
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # 标准前向传播：输入数据通过卷积、批归一化和激活函数
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # 用于推理时的前向传播：输入数据直接通过卷积和激活函数，省略批归一化以提高速度
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # 标准瓶颈层，常用于减少参数并提高模型的计算效率
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()  # 调用父类构造函数进行初始化
        c_ = int(c2 * e)  # 计算中间层的通道数，通过扩展系数 `e` 控制
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一个 1x1 卷积，用于降维
        self.cv2 = Conv(c_, c2, 3, 1, g=g)  # 第二个 3x3 卷积，用于特征提取
        # 判断是否启用捷径连接（shortcut），仅当 `c1` 等于 `c2` 且 `shortcut` 为 True 时启用
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 如果 `self.add` 为 True，则返回输入 `x` 与经过两个卷积层后的输出的相加结果（残差连接）
        # 否则，仅返回卷积后的输出
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP（Cross Stage Partial）瓶颈结构，带有 3 个卷积层
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # 输入通道数, 输出通道数, 层数, 是否使用捷径连接, 组数, 扩展系数
        super().__init__()  # 调用父类构造函数进行初始化
        c_ = int(c2 * e)  # 计算中间层的通道数，通过扩展系数 `e` 控制
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一分支的 1x1 卷积层，用于降维
        self.cv2 = Conv(c1, c_, 1, 1)  # 第二分支的 1x1 卷积层，用于降维
        self.cv3 = Conv(2 * c_, c2, 1)  # 最后的 1x1 卷积层，用于合并输出通道
        # 创建 `n` 个 `Bottleneck` 层，使用 nn.Sequential 进行顺序连接
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])  # 可替换实现

    def forward(self, x):
        # 将经过 `self.cv1` 和 `self.m` 的输出，以及 `self.cv2` 的输出在通道维度上拼接，最终通过 `self.cv3` 输出
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # 空间金字塔池化 (SPP) 层，用于捕获不同尺度的上下文信息
    # 参考论文: https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):  # 输入通道数, 输出通道数, 最大池化核尺寸
        super().__init__()  # 调用父类构造函数进行初始化
        c_ = c1 // 2  # 设置隐藏层的通道数为输入通道数的一半
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1 卷积用于降维
        # 定义一个卷积层，将池化后的特征拼接后输出为指定的 `c2` 通道数
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        # 定义一个模块列表，包含多个不同核尺寸的最大池化层
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)  # 输入通过第一个 1x1 卷积降维
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # 忽略 torch 1.9.0 版本的 max_pool2d() 警告
            # 将降维后的输入和不同池化结果在通道维度上拼接，再通过第二个 1x1 卷积输出
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        """
        初始化SPPF层。

        参数:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        k (int): 最大池化的核大小，默认为5，代表使用的核为k=(5, 9, 13)。
        """
        super().__init__()  # 调用父类的初始化方法
        c_ = c1 // 2  # 计算隐藏通道数，通常取输入通道数的一半
        self.cv1 = Conv(c1, c_, 1, 1)  # 1x1卷积层，将输入通道数c1转换为隐藏通道数c_
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 1x1卷积层，将拼接后的通道数转换为输出通道数c2
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # 定义最大池化层，核大小为k，步长为1，填充为k的一半

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (Tensor): 输入的特征图。

        返回:
        Tensor: 经过SPPF层处理后的输出特征图。
        """
        x = self.cv1(x)  # 通过第一层卷积处理输入特征图

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # 抑制torch 1.9.0中的max_pool2d()警告
            y1 = self.m(x)  # 对处理后的特征图进行第一次最大池化
            y2 = self.m(y1)  # 对第一次池化的结果进行第二次最大池化

            # 将原始特征图和两次池化的结果进行拼接，然后通过第二层卷积进行处理
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class Focus(nn.Module):
    # 将宽高 (wh) 信息集中到通道 (c) 维度上
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # 输入通道数, 输出通道数, 卷积核大小, 步幅, 填充, 组数, 激活函数
        super().__init__()  # 调用父类构造函数进行初始化
        # 定义一个卷积层，将输入的通道数扩展到 4 倍（拼接后），再转变为输出的 `c2` 通道数
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)  # 可选的收缩层，用于将输入收缩以减少尺寸

    def forward(self, x):  # 输入 x 的维度为 (batch_size, channels, width, height) -> 输出 y 的维度为 (batch_size, 4 * channels, width/2, height/2)
        # 将输入 x 分割为四个部分，分别取步长为 2 的不同起点上的像素，并在通道维度拼接
        return self.conv(torch.cat([x[..., ::2, ::2],    # 左上角像素
                                    x[..., 1::2, ::2],   # 右上角像素
                                    x[..., ::2, 1::2],   # 左下角像素
                                    x[..., 1::2, 1::2]], # 右下角像素
                                   1))  # 在通道维度上拼接
        # return self.conv(self.contract(x))  # 可选的实现方式，通过 Contract 层进行操作

class Concat(nn.Module):
    # 该类实现了沿指定维度连接多个张量的功能
    def __init__(self, dimension=1):
        # 初始化方法，指定连接的维度
        super().__init__()
        self.d = dimension  # 保存要连接的维度，默认为1（通常是按列连接）

    def forward(self, x):
        # 前向传播方法，执行张量连接操作
        return torch.cat(x, self.d)  # 使用torch.cat沿self.d维度连接输入的张量列表x



class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x

class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)




class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_) for _ in range(n)])



class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)

class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)



class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class
    multi_label = False  # NMS multiple labels per box
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def autoshape(self):
        LOGGER.info('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model.model[-1]  # Detect()
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes,
                                    multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
