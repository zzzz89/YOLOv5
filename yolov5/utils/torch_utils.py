# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
PyTorch utils
"""

import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    装饰器，用于使所有分布式训练的进程等待每个 local_master 完成某些操作。
    """

    # 如果当前进程的 local_rank 不是 -1 或 0
    if local_rank not in [-1, 0]:
        # 在指定的设备上同步所有进程，确保它们在执行后续操作之前等待
        dist.barrier(device_ids=[local_rank])

    yield  # 暂停函数执行，等待后续代码执行

    # 如果当前进程是 local_master（local_rank 为 0）
    if local_rank == 0:
        # 同步本地 master 进程，确保其完成后续操作后，其他进程才能继续执行
        dist.barrier(device_ids=[0])



def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository


def select_device(device='', batch_size=None):
    # 选择可用的计算设备，device 可以是 'cpu'、'0'（表示第一块 GPU）或 '0,1,2,3'（表示多块 GPU）

    s = f'YOLOv5 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # 创建字符串，包含 YOLOv5 版本、git 描述或修改日期以及 PyTorch 版本
    device = str(device).strip().lower().replace('cuda:', '')  # 将 device 转换为字符串，并格式化为 '0' 形式，去掉 'cuda:' 前缀
    cpu = device == 'cpu'  # 检查是否请求使用 CPU
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制设置环境变量，使 torch.cuda.is_available() 返回 False，禁用 GPU
    elif device:  # 如果请求使用非 CPU 的设备
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # 设置环境变量，以指定可用的 CUDA 设备
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # 检查 CUDA 是否可用，如果不可用则抛出异常

    cuda = not cpu and torch.cuda.is_available()  # 如果不是 CPU 并且 CUDA 可用，则 cuda 为 True
    if cuda:
        devices = device.split(',') if device else '0'  # 将设备字符串分割为列表（例如 '0,1,2'），如果没有指定则默认为 '0'
        n = len(devices)  # 计算请求的设备数量
        if n > 1 and batch_size:  # 如果请求了多个设备且提供了 batch_size
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'  # 检查 batch_size 是否能被设备数量整除

        space = ' ' * (len(s) + 1)  # 计算需要的空格，确保设备信息的对齐
        for i, d in enumerate(devices):  # 遍历每个设备
            p = torch.cuda.get_device_properties(i)  # 获取设备的属性信息
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # 添加设备名称和总内存信息，单位为 MB
    else:
        s += 'CPU\n'  # 如果没有 CUDA 设备可用，则添加 CPU 信息

    # 记录设备信息，处理 Windows 系统中的 emoji 问题
    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)

    return torch.device('cuda:0' if cuda else 'cpu')  # 返回选择的设备（CUDA 设备或 CPU）


def time_sync():
    # PyTorch 精确的时间同步函数
    # 检查是否有可用的 CUDA（即 GPU）设备
    if torch.cuda.is_available():
        # 如果有可用的 CUDA 设备，则同步当前 GPU 的状态
        # 这将确保在调用此函数之前，所有先前的 CUDA 操作都已完成
        torch.cuda.synchronize()

        # 返回当前的系统时间（以秒为单位）
    return time.time()  # 使用 time.time() 获取当前时间戳


def profile(input, ops, n=10, device=None):
    """
    YOLOv5 速度/内存/FLOPs 分析器。

    用法示例：
    - input = torch.randn(16, 3, 640, 640)  # 随机输入
    - m1 = lambda x: x * torch.sigmoid(x)  # 自定义操作
    - m2 = nn.SiLU()  # PyTorch 内置激活函数
    - profile(input, [m1, m2], n=100)  # 对 100 次迭代进行分析

    参数：
    - input: 输入张量或张量列表。
    - ops: 要分析的操作或操作列表。
    - n: 每个操作分析的迭代次数，默认为 10。
    - device: 运行分析的设备，默认为 None，将自动选择。

    返回：
    - 结果列表，每个操作的参数数量、FLOPs、内存使用、前向时间、反向时间和输入输出形状。
    """
    results = []  # 用于存储分析结果
    logging.basicConfig(format="%(message)s", level=logging.INFO)  # 配置日志格式
    device = device or select_device()  # 选择设备
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")  # 打印表头

    # 确保输入是张量列表
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)  # 将输入转移到指定设备
        x.requires_grad = True  # 允许计算梯度

        # 确保操作是操作列表
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m  # 转移操作到指定设备
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0., 0., [0., 0., 0.]  # 初始化前向、反向时间

            try:
                # 计算 FLOPs
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # 转为 GFLOPs
            except:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()  # 开始时间
                    y = m(x)  # 前向传播
                    t[1] = time_sync()  # 结束前向时间
                    try:
                        _ = (sum([yi.sum() for yi in y]) if isinstance(y, list) else y).sum().backward()  # 反向传播
                        t[2] = time_sync()  # 结束反向时间
                    except Exception as e:  # 如果没有反向传播方法
                        print(e)
                        t[2] = float('nan')  # 记录为 NaN
                    tf += (t[1] - t[0]) * 1000 / n  # 每次前向操作的时间
                    tb += (t[2] - t[1]) * 1000 / n  # 每次反向操作的时间

                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # 获取 GPU 内存使用（GB）
                s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'  # 输入形状
                s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'  # 输出形状
                p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # 计算参数总数

                # 打印结果
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])  # 将结果添加到列表
            except Exception as e:
                print(e)
                results.append(None)  # 记录错误
            torch.cuda.empty_cache()  # 清空缓存
    return results  # 返回分析结果


def is_parallel(model):
    """
    检查模型是否为数据并行（DP）或分布式数据并行（DDP）类型。

    参数：
    - model: 要检查的模型实例。

    返回：
    - 如果模型是 DataParallel 或 DistributedDataParallel 类型，则返回 True；否则返回 False。
    """
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    """
    去除模型的并行化：如果模型是 DP 或 DDP 类型，则返回单 GPU 模型。

    参数：
    - model: 要去并行化的模型实例。

    返回：
    - 返回去并行化后的模型。如果模型不是并行化类型，则返回原模型。
    """
    return model.module if is_parallel(model) else model



def intersect_dicts(da, db, exclude=()):
    # 获取两个字典中匹配键和形状的交集，省略 'exclude' 键，使用 da 的值
    return {k: v for k, v in da.items()  # 遍历字典 da 的键值对
            if k in db  # 仅保留在字典 db 中存在的键
            and not any(x in k for x in exclude)  # 排除包含任何 exclude 中元素的键
            and v.shape == db[k].shape}  # 仅保留形状与字典 db 中相应值相同的键值对



def initialize_weights(model):
    """
    初始化模型的权重。

    参数：
    - model: 要初始化的模型实例。

    该函数对模型的各个模块进行初始化：
    - 对于卷积层（Conv2d），可以选择使用 Kaiming 正态分布初始化。
    - 对于批归一化层（BatchNorm2d），设置 eps 和 momentum。
    - 对于激活函数（如 Hardswish、LeakyReLU、ReLU、ReLU6），启用 inplace 计算以节省内存。
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # 可以取消注释以使用 Kaiming 初始化
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    """
    查找与指定模块类匹配的层索引。

    参数：
    - model: 要查找的模型实例。
    - mclass: 需要匹配的模块类，默认为 nn.Conv2d。

    返回：
    - 返回与 mclass 匹配的层索引列表。
    """
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]



def sparsity(model):
    """
    计算模型的全局稀疏度。

    参数：
    - model: 要计算稀疏度的模型实例。

    返回：
    - 返回模型的全局稀疏度，计算方式为零权重的数量与总权重数量的比率。
    """
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()  # 总权重数量
        b += (p == 0).sum()  # 零权重的数量
    return b / a  # 返回稀疏度


def prune(model, amount=0.3):
    """
    对模型进行剪枝以达到请求的全局稀疏度。

    参数：
    - model: 要进行剪枝的模型实例。
    - amount: 请求的剪枝比例，默认为 0.3（即 30% 的权重将被剪枝）。

    该函数遍历模型中的所有卷积层，执行 L1 非结构化剪枝，并移除剪枝掩码以使更改永久生效。
    """
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # 执行剪枝
            prune.remove(m, 'weight')  # 移除剪枝掩码，使其永久生效
    print(' %.3g global sparsity' % sparsity(model))  # 打印剪枝后的全局稀疏度


def fuse_conv_and_bn(conv, bn):
    """
    将卷积层和批归一化层融合为一个卷积层。

    参数：
    - conv: 待融合的卷积层（nn.Conv2d 实例）。
    - bn: 待融合的批归一化层（nn.BatchNorm2d 实例）。

    返回：
    - 返回融合后的卷积层（nn.Conv2d 实例），其中包含了批归一化的影响。

    融合过程参考：
    - https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    """
    # 创建新的卷积层，设置为无梯度
    fusedconv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)

    # 准备卷积层的权重
    w_conv = conv.weight.clone().view(conv.out_channels, -1)  # 展平卷积权重
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))  # 计算批归一化权重的对角矩阵
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))  # 融合权重

    # 准备空间偏置
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))  # 计算批归一化后的偏置
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)  # 融合偏置

    return fusedconv  # 返回融合后的卷积层



def model_info(model, verbose=False, img_size=640):
    """
    输出模型信息，包括参数数量、可训练参数数量和FLOPs。

    参数：
    - model: 待分析的模型（nn.Module 实例）。
    - verbose: 是否详细输出每层的信息（布尔值）。
    - img_size: 输入图像的尺寸，可以是整数或列表（例如，640 或 [640, 320]）。

    返回：
    - None: 直接打印模型的摘要信息。
    """
    n_p = sum(x.numel() for x in model.parameters())  # 总参数数量
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # 可训练参数数量

    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')  # 去除模块列表前缀
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))  # 打印层信息

    try:  # 计算FLOPs
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # 确定步幅
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # 创建输入张量
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # 计算GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # 扩展图像尺寸
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 计算640x640的GFLOPs
    except (ImportError, Exception):
        fs = ''  # 如果发生异常，设置为默认值

    LOGGER.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")



def load_classifier(name='resnet101', n=2):
    """
    加载一个预训练的分类模型，并将输出层调整为 n 类输出。

    参数：
    - name: 要加载的模型名称（字符串），默认为 'resnet101'。
    - n: 输出类别的数量（整数），默认为 2。

    返回：
    - model: 调整后的模型（torchvision.models 的实例）。
    """
    model = torchvision.models.__dict__[name](pretrained=True)  # 加载预训练模型

    # ResNet 模型的属性
    # input_size = [3, 224, 224]  # 输入大小
    # input_space = 'RGB'  # 输入空间
    # input_range = [0, 1]  # 输入范围
    # mean = [0.485, 0.456, 0.406]  # 均值
    # std = [0.229, 0.224, 0.225]  # 标准差

    # 将输出层调整为 n 个类别
    filters = model.fc.weight.shape[1]  # 获取原输出层的特征数
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)  # 创建新的偏置
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)  # 创建新的权重
    model.fc.out_features = n  # 设置输出特征数量
    return model  # 返回调整后的模型



def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """
    按照给定的比例缩放图像，并确保图像尺寸为 gs 的倍数。

    参数：
    - img: 输入图像张量，形状为 (batch_size, channels, height, width)。
    - ratio: 缩放比例，默认为 1.0（不缩放）。
    - same_shape: 是否保持输入和输出图像的形状一致，默认为 False。
    - gs: 图像尺寸的基数，默认为 32，缩放后的尺寸会被调整为 gs 的倍数。

    返回：
    - 处理后的图像张量。
    """
    if ratio == 1.0:
        return img  # 如果比例为 1.0，直接返回原图像
    else:
        h, w = img.shape[2:]  # 获取原图像的高度和宽度
        s = (int(h * ratio), int(w * ratio))  # 计算新的尺寸
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # 缩放图像
        if not same_shape:  # 如果不保持形状一致，进行填充/裁剪
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]  # 调整为 gs 的倍数
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # 填充图像，使用 imagenet 均值



def copy_attr(a, b, include=(), exclude=()):
    """
    从对象 b 复制属性到对象 a。

    参数：
    - a: 目标对象，将接收属性。
    - b: 源对象，将提供属性。
    - include: 仅复制这些属性的名称（可选）。
    - exclude: 排除这些属性的名称（可选）。
    """
    for k, v in b.__dict__.items():
        # 检查是否仅包含指定属性，是否以 '_' 开头，或是否在排除列表中
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue  # 跳过不满足条件的属性
        else:
            setattr(a, k, v)  # 设置目标对象的属性



class EarlyStopping:
    # YOLOv5 简单的早停机制
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # 最佳适应度，例如mAP
        self.best_epoch = 0  # 最佳epoch
        self.patience = patience or float('inf')  # 在适应度停止提升后，等待的epoch数
        self.possible_stop = False  # 可能在下一个epoch停止

    def __call__(self, epoch, fitness):
        # 判断是否需要早停
        if fitness >= self.best_fitness:  # >= 0 允许在训练的早期阶段适应度为零
            self.best_epoch = epoch  # 更新最佳epoch
            self.best_fitness = fitness  # 更新最佳适应度

        delta = epoch - self.best_epoch  # 计算没有改进的epoch数
        self.possible_stop = delta >= (self.patience - 1)  # 可能在下一个epoch停止
        stop = delta >= self.patience  # 如果超过耐心值，则停止训练

        if stop:
            LOGGER.info(f'EarlyStopping patience {self.patience} exceeded, stopping training.')  # 记录停止信息

        return stop  # 返回是否需要停止


class ModelEMA:
    """
    Model Exponential Moving Average (EMA) 类，参考自 https://github.com/rwightman/pytorch-image-models。

    该类保持模型 state_dict（参数和缓冲区）的指数移动平均。
    这旨在实现类似于 TensorFlow 的 https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage 的功能。
    平滑版本的权重对于某些训练方案的良好表现是必要的。
    此类对初始化顺序敏感，包括模型初始化、GPU 分配和分布式训练包装器。
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        初始化 ModelEMA 实例。

        参数:
        model (torch.nn.Module): 需要计算 EMA 的模型。
        decay (float): 指数衰减率，默认为 0.9999。
        updates (int): EMA 更新的次数，默认为 0。

        说明:
        - 创建 EMA 的深拷贝并将其设置为评估模式。
        - 将更新次数和衰减函数初始化为指定的值。
        - 将 EMA 的所有参数设置为不需要梯度计算。
        """
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # 更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # 衰减指数曲线（帮助早期训练阶段）
        for p in self.ema.parameters():
            p.requires_grad_(False)  # 禁用梯度计算

    def update(self, model):
        """
        更新 EMA 参数。

        参数:
        model (torch.nn.Module): 当前模型实例，用于更新 EMA 参数。

        说明:
        - 计算当前的衰减值，并使用它更新 EMA 参数。
        """
        with torch.no_grad():
            self.updates += 1  # 更新次数加 1
            d = self.decay(self.updates)  # 计算当前衰减值

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # 获取模型的 state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # 如果参数是浮点型
                    v *= d  # 按衰减值缩放 EMA 权重
                    v += (1. - d) * msd[k].detach()  # 更新 EMA 权重

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        更新 EMA 属性。

        参数:
        model (torch.nn.Module): 当前模型实例。
        include (tuple): 要包含的属性名，默认为空元组。
        exclude (tuple): 要排除的属性名，默认为 ('process_group', 'reducer')。

        说明:
        - 使用 copy_attr 函数更新 EMA 的属性。
        """
        copy_attr(self.ema, model, include, exclude)  # 更新 EMA 属性

