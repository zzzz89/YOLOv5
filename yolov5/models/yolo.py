# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import check_yaml, make_divisible, print_args, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
    select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)


class Detect(nn.Module):
    stride = None  # 在构建过程中计算的 strides（步幅）
    onnx_dynamic = False  # ONNX 导出参数，表示是否使用动态输入

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # 检测层的初始化
        super().__init__()  # 调用父类的初始化方法
        self.nc = nc  # 检测的类别数量（默认80类）
        self.no = nc + 5  # 每个锚点的输出数量（类别数 + 5个额外信息：x, y, w, h, 置信度）
        self.nl = len(anchors)  # 检测层的数量（通常是3）
        self.na = len(anchors[0]) // 2  # 每层的锚点数量（每个锚点由两个值组成：宽度和高度）
        self.grid = [torch.zeros(1)] * self.nl  # 初始化网格，用于推理时生成网格
        self.anchor_grid = [torch.zeros(1)] * self.nl  # 初始化锚点网格，用于推理时计算锚点的形状
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # 注册锚点为 buffer，形状为 (nl, na, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 定义每个检测层的输出卷积层
        self.inplace = inplace  # 是否使用原地操作（例如切片赋值）

    def forward(self, x):
        z = []  # 用于存储推理输出的列表
        for i in range(self.nl):  # 遍历每个检测层
            x[i] = self.m[i](x[i])  # 使用卷积层处理输入
            bs, _, ny, nx = x[i].shape  # 获取输出的形状：bs=batch_size, ny, nx = 网格的高度和宽度
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # 调整形状为 (batch_size, anchors, grid_y, grid_x, outputs)

            if not self.training:  # 仅在推理阶段进行处理
                # 检查网格的形状是否匹配，或者是否需要动态调整（例如ONNX导出）
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)  # 生成新的网格和锚点网格

                y = x[i].sigmoid()  # 对输出进行sigmoid激活，得到范围在[0, 1]之间的预测值

                if self.inplace:  # 使用原地操作进行更新
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # 更新 x, y 坐标
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # 更新 w, h 维度
                else:  # 如果不使用原地操作，采用常规操作
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # 更新 x, y 坐标
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # 更新 w, h 维度
                    y = torch.cat((xy, wh, y[..., 4:]), -1)  # 将 x, y, w, h 和其他预测信息连接起来

                z.append(y.view(bs, -1, self.no))  # 将每层的输出展平，并添加到推理结果列表 z 中

        # 如果是推理阶段，返回拼接的预测结果和原始输入，否则返回训练时的结果
        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        # 获取当前使用的设备（例如CPU或GPU）
        d = self.anchors[i].device

        # 生成 x 和 y 方向上的网格坐标
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])

        # 将 x 和 y 坐标堆叠成一个形状为 (ny, nx, 2) 的张量，并扩展维度
        # 扩展后的形状为 (1, num_anchors, ny, nx, 2)
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()

        # 根据锚点的尺寸和 stride（步幅）调整锚点的大小
        # 锚点按当前 stride 放大，并扩展到与 grid 相同的形状
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()

        # 返回网格和锚点网格
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # 初始化模型，输入通道数，类别数，锚点
        super().__init__()  # 调用父类的初始化方法
        if isinstance(cfg, dict):
            self.yaml = cfg  # 如果配置是字典，直接使用它作为模型配置
        else:  # 如果配置是一个 *.yaml 文件
            import yaml  # 导入yaml库，用于加载yaml配置文件
            self.yaml_file = Path(cfg).name  # 获取文件名
            with open(cfg, errors='ignore') as f:  # 打开yaml文件
                self.yaml = yaml.safe_load(f)  # 读取并解析yaml文件为字典

        # 设置模型配置
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 获取输入通道数，如果yaml没有提供，则使用默认值ch
        if nc and nc != self.yaml['nc']:  # 如果提供了类别数且与yaml文件中的类别数不匹配，则进行覆盖
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # 覆盖yaml中的类别数
        if anchors:  # 如果提供了锚点，覆盖yaml中的锚点
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # 覆盖yaml中的锚点配置

        # 解析模型架构，生成模型和保存列表
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # 深拷贝yaml配置并解析模型
        self.names = [str(i) for i in range(self.yaml['nc'])]  # 默认的类别名称（字符串类型）
        self.inplace = self.yaml.get('inplace', True)  # 是否使用原地操作，默认为True

        # 构建步幅（stride）和锚点
        m = self.model[-1]  # 获取模型的最后一层（通常是检测层）
        if isinstance(m, Detect):  # 如果是 Detect 层（YOLO 的检测层）
            s = 256  # 默认步幅为256
            m.inplace = self.inplace  # 设置是否使用原地操作
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # 计算步幅
            m.anchors /= m.stride.view(-1, 1, 1)  # 调整锚点比例
            check_anchor_order(m)  # 检查锚点顺序
            self.stride = m.stride  # 保存步幅
            self._initialize_biases()  # 初始化偏置，只执行一次

        # 初始化权重和偏置
        initialize_weights(self)
        self.info()  # 输出模型信息
        LOGGER.info('')  # 打印空行，用于日志格式

    def forward(self, x, augment=False, profile=False, visualize=False):
        # 如果启用了数据增强，使用增强推理方法
        if augment:
            return self._forward_augment(x)  # augmented inference, None

        # 否则执行标准的单尺度推理（通常用于训练过程）
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # 获取输入图像的高度和宽度 (height, width)
        s = [1, 0.83, 0.67]  # 定义三个不同的尺度（用于数据增强）
        f = [None, 3, None]  # 定义翻转方式 (2: 上下翻转，3: 左右翻转)

        y = []  # 存储每个尺度和翻转下的输出

        for si, fi in zip(s, f):
            # 如果需要翻转，则先翻转图像，再进行缩放
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # 对当前图像进行前向传播，得到预测输出

            # 保存增强后的图像（此行已注释掉）
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # 保存图片

            # 将预测输出从增强的尺度和翻转转换回原始图像的尺度
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)  # 添加当前尺度和翻转下的输出到结果列表

        y = self._clip_augmented(y)  # 对增强后的结果进行剪切处理，去掉不必要的部分（例如填充）

        # 将所有增强结果在通道维度上拼接，并返回
        return torch.cat(y, 1), None  # 返回增强后的推理结果和None（可能用于其他用途）

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # 初始化输出列表和性能分析数据列表

        # 遍历模型中的所有层
        for m in self.model:
            # 如果该层的前驱层索引（m.f）不是-1，表示该层的输入来自于之前的层
            if m.f != -1:
                # 如果 m.f 是整数，表示输入来自于某一层的输出；如果 m.f 是列表，表示输入来自多个层的输出
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 从前面的层获取输入

            # 如果开启了性能分析，调用 `_profile_one_layer` 方法分析当前层的性能
            if profile:
                self._profile_one_layer(m, x, dt)

            # 将输入 `x` 传入当前层 `m`，进行前向传播（计算输出）
            x = m(x)  # run

            # 如果该层的索引 `m.i` 在 `self.save` 中，表示需要保存该层的输出，否则保存 None
            y.append(x if m.i in self.save else None)

            # 如果开启了可视化，调用 `feature_visualization` 方法对特征图进行可视化并保存
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        # 返回最终的输出
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # 逆操作：对经过增强推理后的预测结果进行反缩放处理
        if self.inplace:  # 如果使用原地操作
            p[..., :4] /= scale  # 对前四个元素（xywh）进行反缩放处理
            if flips == 2:  # 如果进行了上下翻转（ud flip）
                p[..., 1] = img_size[0] - p[..., 1]  # 逆操作：将y坐标反转
            elif flips == 3:  # 如果进行了左右翻转（lr flip）
                p[..., 0] = img_size[1] - p[..., 0]  # 逆操作：将x坐标反转
        else:  # 如果不使用原地操作
            # 将xywh部分按缩放因子进行反缩放
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # 对xy和wh进行反缩放
            if flips == 2:  # 如果进行了上下翻转
                y = img_size[0] - y  # 逆操作：将y坐标反转
            elif flips == 3:  # 如果进行了左右翻转
                x = img_size[1] - x  # 逆操作：将x坐标反转
            # 拼接反缩放后的x, y, wh和其他部分（例如类别、置信度等）
            p = torch.cat((x, y, wh, p[..., 4:]), -1)

        return p  # 返回反缩放后的预测结果

    def _clip_augmented(self, y):
        # 对YOLOv5增强推理后的尾部进行裁剪
        nl = self.model[-1].nl  # 获取检测层数（例如P3-P5）
        g = sum(4 ** x for x in range(nl))  # 计算网格点的数量（所有层的4的幂次和）
        e = 1  # 设置排除层数（这里为1）

        # 计算大尺度输出的裁剪索引
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # 计算大尺度裁剪索引
        y[0] = y[0][:, :-i]  # 裁剪大尺度层的多余部分

        # 计算小尺度输出的裁剪索引
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 计算小尺度裁剪索引
        y[-1] = y[-1][:, i:]  # 裁剪小尺度层的前半部分

        return y  # 返回裁剪后的结果

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # 判断是否是最后一层（Detect层），如果是，输入需要复制以修正inplace操作
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[
                0] / 1E9 * 2 if thop else 0  # 计算FLOPs（每秒浮点运算次数），单位为GFLOPs
        t = time_sync()  # 记录当前时间，用于性能计时
        for _ in range(10):  # 测试10次模型推理时间
            m(x.copy() if c else x)  # 运行模型，复制输入x以防止inplace修改
        dt.append((time_sync() - t) * 100)  # 记录本次推理时间（单位ms）

        if m == self.model[0]:  # 如果是第一个模块（通常为输入模块）
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")  # 打印表头信息

        # 打印当前层的推理时间、FLOPs和参数量
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

        if c:  # 如果是最后一层（Detect层）
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")  # 打印总的推理时间

    def _initialize_biases(self, cf=None):  # 初始化Detect()中的偏置项，cf为类别频率（可选）
        # 参考论文：https://arxiv.org/abs/1708.02002 第3.3节
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # 获取Detect()模块（模型最后一层）

        # 遍历每个检测层的卷积层，s为该层的stride（步幅）
        for mi, s in zip(m.m, m.stride):  # 遍历Detect模块中的每个卷积层
            b = mi.bias.view(m.na, -1)  # 将卷积偏置（255）重塑为(3, 85)，对应每个anchor的偏置

            # 调整物体置信度的偏置（假设每640px的图像有8个物体）
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # 物体置信度的偏置，调整为根据图像大小和步幅计算

            # 调整类别偏置，如果没有传入cf（类别频率），则使用默认值0.6进行初始化
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # 类别偏置

            # 更新卷积层的偏置参数
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # 获取模型的最后一层Detect()模块
        for mi in m.m:  # 遍历Detect模块中的每个卷积层
            # 获取每个卷积层的偏置（原为255维），并将其重塑为(3, 85)的形状，转置后便于查看
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3, 85)，去除梯度信息

            # 打印卷积层的偏置，显示前5个值的均值（用于物体检测）以及类别部分的均值
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # 将模型中的Conv2d()与BatchNorm2d()层融合
        LOGGER.info('Fusing layers... ')  # 输出日志，表示正在进行层融合

        # 遍历模型中的每一个模块
        for m in self.model.modules():
            # 如果模块是卷积层（Conv或DWConv），并且具有BatchNorm层（bn）
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                # 调用fuse_conv_and_bn函数将卷积层与BatchNorm层进行融合
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积层
                delattr(m, 'bn')  # 删除BatchNorm层（已融合，移除不再需要）
                m.forward = m.forward_fuse  # 更新前向传播方法为融合后的forward方法

        # 打印模型信息
        self.info()
        return self  # 返回当前模型对象，支持链式调用

    def autoshape(self):  # 添加AutoShape模块
        LOGGER.info('Adding AutoShape... ')  # 输出日志，表示正在添加AutoShape模块

        # 创建AutoShape实例，将当前模型作为参数传入，包装模型
        m = AutoShape(self)  # wrap model

        # 将当前模型的一些属性复制到AutoShape模块中，确保AutoShape能够继承这些属性
        # 复制属性：'yaml', 'nc', 'hyp', 'names', 'stride'
        # 不复制的属性为空元组
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())

        return m  # 返回带有AutoShape功能的模型

    def info(self, verbose=False, img_size=640):  # 打印模型信息
        model_info(self, verbose, img_size)  # 调用model_info函数打印详细的模型信息

    def _apply(self, fn):
        # 将fn应用到模型的张量上，除了参数和已注册的缓冲区
        self = super()._apply(fn)  # 调用父类的_apply方法

        m = self.model[-1]  # 获取模型的最后一个模块（通常是Detect层）

        # 如果最后一个模块是Detect层，更新相关属性
        if isinstance(m, Detect):
            m.stride = fn(m.stride)  # 应用fn到stride（步长）
            m.grid = list(map(fn, m.grid))  # 应用fn到网格（grid）

            # 如果anchor_grid是列表，应用fn到每个元素
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))

        return self  # 返回修改后的模型


def parse_model(d, ch):  # 解析模型字典，输入通道数（默认3）
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))  # 打印表头
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']  # 提取模型配置中的锚框、类数、深度和宽度倍数
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 计算锚框数量
    no = na * (nc + 5)  # 输出数量 = 锚框数 * (类别数 + 5)，5是包括x, y, w, h, confidence

    layers, save, c2 = [], [], ch[-1]  # 初始化层列表、保存列表、输出通道数（初始值为最后一层的通道数）

    # 遍历模型字典中的backbone和head部分
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # f：来自层的索引，n：重复次数，m：模块类型，args：模块的参数
        m = eval(m) if isinstance(m, str) else m  # 如果模块是字符串，评估其为模块
        for j, a in enumerate(args):  # 遍历模块参数
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # 如果参数是字符串，评估其值
            except NameError:  # 如果出现未定义名称的错误，跳过该参数
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # 根据深度倍数计算每个模块的重复次数，至少为1
        # 如果模块是常见的卷积或瓶颈结构，处理输入输出通道数
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]  # 获取输入和输出通道数
            if c2 != no:  # 如果输出通道数不是目标输出通道数
                c2 = make_divisible(c2 * gw, 8)  # 对输出通道数进行宽度倍数调整

            args = [c1, c2, *args[1:]]  # 更新模块的参数
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:  # 如果模块是某些特殊结构，插入重复次数
                args.insert(2, n)  # 插入重复次数参数
                n = 1  # 重复次数为1
        elif m is nn.BatchNorm2d:  # 如果模块是BatchNorm2d
            args = [ch[f]]  # 只需要输入通道数
        elif m is Concat:  # 如果模块是Concat
            c2 = sum([ch[x] for x in f])  # 将输入的通道数相加
        elif m is Detect:  # 如果模块是Detect（通常用于最后的检测层）
            args.append([ch[x] for x in f])  # 添加输入通道数
            if isinstance(args[1], int):  # 如果锚框数是整数，转换为具体的锚框
                args[1] = [list(range(args[1] * 2))] * len(f)  # 每个输入特征图的锚框
        elif m is Contract:  # 如果模块是Contract
            c2 = ch[f] * args[0] ** 2  # 扩展通道数
        elif m is Expand:  # 如果模块是Expand
            c2 = ch[f] // args[0] ** 2  # 收缩通道数
        else:
            c2 = ch[f]  # 默认情况下，输出通道数等于输入通道数

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # 如果n > 1，则重复n次该模块；否则直接创建单个模块
        t = str(m)[8:-2].replace('__main__.', '')  # 获取模块类型的名称，并去除 '__main__.' 前缀
        np = sum([x.numel() for x in m_.parameters()])  # 计算模块所有参数的总元素数，即参数数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 为模块附加索引（i）、来源索引（f）、类型（t）和参数数量（np）
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # 打印模块的索引、来源、重复次数、参数数量、类型和模块的参数
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 将模块的来源索引添加到保存列表（忽略-1）
        layers.append(m_)  # 将模块添加到层列表中
        if i == 0:  # 如果是第一个模块
            ch = []  # 初始化通道数列表
        ch.append(c2)  # 将当前模块的输出通道数添加到通道数列表中
    return nn.Sequential(*layers), sorted(save)  # 返回一个由所有模块组成的顺序容器(nn.Sequential)，以及按升序排序的保存列表


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建 ArgumentParser 对象，用于命令行参数解析
    parser.add_argument('--cfg', type=str, default='yolov5l.yaml', help='model.yaml')  # 添加参数 --cfg，指定模型配置文件
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # 添加参数 --device，指定设备（GPU或CPU）
    parser.add_argument('--profile', action='store_true', help='profile model speed')  # 添加参数 --profile，开启后会进行模型性能分析
    opt = parser.parse_args()  # 解析命令行参数，并返回命令行参数对象 opt
    opt.cfg = check_yaml(opt.cfg)  # 检查指定的 YAML 配置文件是否合法
    print_args(FILE.stem, opt)  # 打印配置文件和解析后的命令行参数
    set_logging()  # 设置日志配置
    device = select_device(opt.device)  # 选择设备（GPU/CPU）

    # 创建模型
    model = Model(opt.cfg).to(device)  # 初始化模型并将其转移到指定设备
    model.train()  # 设置模型为训练模式

    # 性能分析
    if opt.profile:  # 如果启用了性能分析
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(
            device)  # 创建一个随机输入图像（8张或者1张，取决于是否有可用的GPU）
        y = model(img, profile=True)  # 执行一次模型推理，并启用性能分析

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
