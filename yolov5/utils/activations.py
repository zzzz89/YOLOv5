# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Activation functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):  # SiLU激活函数的导出友好版本
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)  # SiLU (Sigmoid-weighted Linear Unit)


class Hardswish(nn.Module):  # Hardswish激活函数的导出友好版本
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # Hardswish激活


# Mish激活函数
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()  # Mish激活


# 记忆效率高的Mish实现
class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)  # 保存输入以备后续反向传播使用
            return x.mul(torch.tanh(F.softplus(x)))  # 记忆效率高的Mish

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]  # 取出保存的输入
            sx = torch.sigmoid(x)  # 计算sigmoid值
            fx = F.softplus(x).tanh()  # 计算softplus的tanh值
            return grad_output * (fx + x * sx * (1 - fx * fx))  # 反向传播

    def forward(self, x):
        return self.F.apply(x)


# FReLU激活函数
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # c1: 输入通道数, k: 卷积核大小
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)  # 深度卷积
        self.bn = nn.BatchNorm2d(c1)  # 批归一化

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))  # FReLU激活


# ACON https://arxiv.org/pdf/2009.04759.pdf ----------------------------------------------------------------------------
# ACON激活函数
class AconC(nn.Module):
    r""" ACON激活函数（启用或不启用）。
    AconC: (p1*x - p2*x) * sigmoid(beta * (p1*x - p2*x)) + p2*x,
    其中beta是一个可学习的参数。
    """
    def __init__(self, c1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))  # 可学习参数
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))  # 可学习的beta

    def forward(self, x):
        dpx = (self.p1 - self.p2) * x  # 计算差值
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x  # ACON前向传递


# MetaAconC激活函数
class MetaAconC(nn.Module):
    r""" Meta ACON激活函数（启用或不启用）。
    beta是通过一个小网络生成的。
    """
    def __init__(self, c1, k=1, s=1, r=16):  # ch_in, kernel, stride, r
        super().__init__()
        c2 = max(r, c1 // r)  # 中间通道数
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))  # 可学习参数
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)  # 第一个卷积
        self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)  # 第二个卷积

    def forward(self, x):
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)  # 全局平均池化
        beta = torch.sigmoid(self.fc2(self.fc1(y)))  # 学习beta
        dpx = (self.p1 - self.p2) * x  # 计算差值
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x  # Meta ACON前向传递
