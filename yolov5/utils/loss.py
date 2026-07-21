# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # 参考链接: https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # 返回平滑的正负标签，适用于二元交叉熵（BCE）损失
    return 1.0 - 0.5 * eps, 0.5 * eps  # 正标签和负标签的平滑值



class BCEBlurWithLogitsLoss(nn.Module):
    # 自定义的二元交叉熵损失，结合了 logits 和标签平滑处理，减少缺失标签的影响
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # 必须使用 nn.BCEWithLogitsLoss()
        self.alpha = alpha  # 平滑因子

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)  # 计算损失
        pred = torch.sigmoid(pred)  # 从 logits 转换为概率
        dx = pred - true  # 计算预测值与真实值之间的差异
        # dx = (pred - true).abs()  # 计算绝对差异，可选：减少缺失标签和错误标签的影响
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))  # 计算 alpha 因子
        loss *= alpha_factor  # 加权损失
        return loss.mean()  # 返回平均损失



class FocalLoss(nn.Module):
    # 包装焦点损失，用于在现有损失函数上应用焦点损失，例：criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # 必须是 nn.BCEWithLogitsLoss()
        self.gamma = gamma  # 焦点因子
        self.alpha = alpha  # 权重因子
        self.reduction = loss_fcn.reduction  # 保存原始的归约方式
        self.loss_fcn.reduction = 'none'  # 需要对每个元素应用焦点损失

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)  # 计算基础损失
        # p_t = torch.exp(-loss)  # 计算预测概率，注释掉的实现方式
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # 使用非零幂以保持梯度稳定

        # TensorFlow 实现参考 https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # 从 logits 转换为概率
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)  # 计算 p_t
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # 计算 alpha 因子
        modulating_factor = (1.0 - p_t) ** self.gamma  # 计算调制因子
        loss *= alpha_factor * modulating_factor  # 应用焦点损失

        if self.reduction == 'mean':
            return loss.mean()  # 返回平均损失
        elif self.reduction == 'sum':
            return loss.sum()  # 返回总损失
        else:  # 'none'
            return loss  # 返回未归约损失


class QFocalLoss(nn.Module):
    # 包装质量焦点损失，用于在现有损失函数上应用质量焦点损失，例：criteria = QFocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # 必须是 nn.BCEWithLogitsLoss()
        self.gamma = gamma  # 焦点因子
        self.alpha = alpha  # 权重因子
        self.reduction = loss_fcn.reduction  # 保存原始的归约方式
        self.loss_fcn.reduction = 'none'  # 需要对每个元素应用焦点损失

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)  # 计算基础损失

        pred_prob = torch.sigmoid(pred)  # 从 logits 转换为概率
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # 计算 alpha 因子
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma  # 计算调制因子
        loss *= alpha_factor * modulating_factor  # 应用质量焦点损失

        if self.reduction == 'mean':
            return loss.mean()  # 返回平均损失
        elif self.reduction == 'sum':
            return loss.sum()  # 返回总损失
        else:  # 'none'
            return loss  # 返回未归约损失


class ComputeLoss:
    # 计算损失的类
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False  # 是否对目标IoU进行排序
        device = next(model.parameters()).device  # 获取模型设备
        h = model.hyp  # 获取超参数

        # 定义损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))  # 类别损失
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))  # 目标损失

        # 类别标签平滑处理
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # 正负BCE目标

        # 焦点损失
        g = h['fl_gamma']  # 焦点损失的gamma值
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)  # 应用焦点损失

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # 获取检测模块
        # 设定损失平衡因子，根据检测层数（P3-P7）
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride为16的索引
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance

        # 将检测模块的参数赋值给计算损失类的属性
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device  # 获取目标的设备
        # 初始化损失值
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        # 构建目标
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # 计算损失
        for i, pi in enumerate(p):  # 遍历每个层的预测
            b, a, gj, gi = indices[i]  # 获取图像、锚框、网格y坐标和网格x坐标
            tobj = torch.zeros_like(pi[..., 0], device=device)  # 初始化目标对象

            n = b.shape[0]  # 获取目标数量
            if n:  # 如果有目标
                ps = pi[b, a, gj, gi]  # 获取与目标对应的预测子集

                # 回归损失
                pxy = ps[:, :2].sigmoid() * 2. - 0.5  # 预测的中心点
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # 预测的宽高
                pbox = torch.cat((pxy, pwh), 1)  # 组合为预测框
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # 计算预测框与目标框的IoU
                lbox += (1.0 - iou).mean()  # 累加IoU损失

                # 目标置信度损失
                score_iou = iou.detach().clamp(0).type(tobj.dtype)  # 处理IoU得分
                if self.sort_obj_iou:  # 如果需要排序
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # 计算目标置信度

                # 分类损失
                if self.nc > 1:  # 如果有多个类别
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # 初始化目标
                    t[range(n), tcls[i]] = self.cp  # 设置正样本
                    lcls += self.BCEcls(ps[:, 5:], t)  # 计算分类的BCE损失

                # 目标记录（注释掉的部分）
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            # 计算目标置信度损失
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # 累加对象损失
            if self.autobalance:  # 自动平衡损失
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        # 自动平衡损失
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        # 根据超参数缩放损失
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # 获取批大小

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()  # 返回总损失和各类损失

    def build_targets(self, p, targets):
        # 为compute_loss()构建目标，输入格式为targets(image, class, x, y, w, h)
        na, nt = self.na, targets.shape[0]  # 锚框数量和目标数量
        tcls, tbox, indices, anch = [], [], [], []  # 初始化目标分类、边界框、索引和锚框列表
        gain = torch.ones(7, device=targets.device)  # 用于归一化到网格空间的增益
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # 创建锚框索引
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # 追加锚框索引到目标

        g = 0.5  # 偏移量
        # 定义偏移数组
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g  # 偏移量

        for i in range(self.nl):  # 遍历每个层
            anchors = self.anchors[i]  # 获取当前层的锚框
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # 获取xyxy增益

            # 将目标与锚框匹配
            t = targets * gain  # 归一化目标
            if nt:  # 如果有目标
                # 计算宽高比
                r = t[:, :, 4:6] / anchors[:, None]  # 计算宽高比
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # 比较宽高比
                t = t[j]  # 筛选匹配的目标

                # 计算偏移量
                gxy = t[:, 2:4]  # 网格xy坐标
                gxi = gain[[2, 3]] - gxy  # 计算逆偏移
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T  # 检查网格坐标
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T  # 检查逆网格坐标
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # 合并条件
                t = t.repeat((5, 1, 1))[j]  # 重复并筛选目标
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # 计算偏移量
            else:
                t = targets[0]  # 如果没有目标，直接获取
                offsets = 0  # 无偏移

            # 定义目标
            b, c = t[:, :2].long().T  # 提取图像和类别
            gxy = t[:, 2:4]  # 网格xy坐标
            gwh = t[:, 4:6]  # 网格宽高
            gij = (gxy - offsets).long()  # 计算网格索引
            gi, gj = gij.T  # 网格xy索引

            # 追加目标信息
            a = t[:, 6].long()  # 锚框索引
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # 保存图像、锚框和网格索引
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # 保存边界框信息
            anch.append(anchors[a])  # 保存锚框
            tcls.append(c)  # 保存类别

        return tcls, tbox, indices, anch  # 返回目标分类、边界框、索引和锚框
