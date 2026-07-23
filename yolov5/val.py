# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_img_size, check_requirements, \
    check_suffix, check_yaml, box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, \
    increment_path, colorstr, print_args
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks


def save_one_txt(predn, save_conf, shape, file):
    # 保存单个预测结果到文本文件
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # 计算归一化增益（宽高的归一化因子）

    # 遍历预测结果
    for *xyxy, conf, cls in predn.tolist():
        # 将坐标从 (xmin, ymin, xmax, ymax) 转换为 (x, y, w, h) 并归一化
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化的 xywh

        # 根据是否保存置信度选择输出格式
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式

        # 以追加模式打开文件并写入一行
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')  # 写入结果

def save_one_json(predn, jdict, path, class_map):
    # 保存单个预测结果为 JSON 格式
    # 格式: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}

    # 获取图像 ID，使用文件名作为 ID，若文件名是数字则转换为整数
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem

    # 将预测的边界框从 (xmin, ymin, xmax, ymax) 转换为 (x, y, w, h)
    box = xyxy2xywh(predn[:, :4])  # 取得前四列作为边界框
    box[:, :2] -= box[:, 2:] / 2  # 将中心坐标转换为左上角坐标

    # 遍历预测结果和转换后的边界框
    for p, b in zip(predn.tolist(), box.tolist()):
        # 将每个预测结果格式化并添加到 JSON 字典
        jdict.append({
            'image_id': image_id,  # 图像 ID
            'category_id': class_map[int(p[5])],  # 类别 ID
            'bbox': [round(x, 3) for x in b],  # 边界框，保留三位小数
            'score': round(p[4], 5)  # 置信度，保留五位小数
        })


def process_batch(detections, labels, iouv):
    """
    处理检测结果和标签，返回正确预测的矩阵。两个框集均采用 (x1, y1, x2, y2) 格式。

    参数:
        detections (Array[N, 6]): 检测结果，包含 x1, y1, x2, y2, 置信度, 类别
        labels (Array[M, 5]): 标签，包含 类别, x1, y1, x2, y2
        iouv (Array): IoU 阈值

    返回:
        correct (Array[N, 10]): 10 个 IoU 水平的正确预测矩阵
    """

    # 初始化一个形状为 (N, 10) 的布尔矩阵，用于存储每个检测框是否为正确预测
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)

    # 计算每个标签与检测框之间的 IoU
    iou = box_iou(labels[:, 1:], detections[:, :4])

    # 找到 IoU 大于阈值且类别匹配的检测框
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))

    # 如果找到匹配的框
    if x[0].shape[0]:
        # 组合标签索引、检测索引和 IoU 值，形成匹配数组
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]

        # 如果匹配框数量大于 1，按 IoU 值降序排序并去重
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]  # 按 IoU 降序排序
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 按检测框去重
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 按标签去重

        # 转换为张量，并移动到与 iouv 相同的设备
        matches = torch.Tensor(matches).to(iouv.device)

        # 更新正确预测矩阵
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv

    return correct


@torch.no_grad()
def run(data,
        weights=None,  # 模型路径（model.pt）
        batch_size=32,  # 批次大小
        imgsz=640,  # 推理图像尺寸（像素）
        conf_thres=0.001,  # 置信度阈值
        iou_thres=0.6,  # NMS（非极大值抑制）IoU阈值
        task='val',  # 任务类型：train（训练）、val（验证）、test（测试）、speed（速度测试）或 study（研究）
        device='',  # CUDA设备，例如：0、0,1,2,3 或 cpu
        single_cls=False,  # 将数据集视为单类数据集
        augment=False,  # 是否进行增强推理
        verbose=False,  # 是否输出详细信息
        save_txt=False,  # 是否将结果保存为 *.txt 文件
        save_hybrid=False,  # 是否保存标签与预测的混合结果到 *.txt 文件
        save_conf=False,  # 是否在 --save-txt 标签中保存置信度
        save_json=False,  # 是否保存为 COCO-JSON 结果文件
        project=ROOT / 'runs/val',  # 结果保存路径
        name='exp',  # 保存的实验名称
        exist_ok=False,  # 是否允许存在的项目/名称，若存在则不递增
        half=True,  # 是否使用 FP16 半精度推理
        model=None,  # 加载的模型
        dataloader=None,  # 数据加载器
        save_dir=Path(''),  # 保存目录
        plots=True,  # 是否生成图表
        callbacks=Callbacks(),  # 回调函数
        compute_loss=None,  # 计算损失函数
        ):

    # 初始化/加载模型并设置设备
    training = model is not None  # 判断是否在训练中
    if training:  # 由 train.py 调用
        device = next(model.parameters()).device  # 获取模型所在设备

    else:  # 直接调用
        device = select_device(device, batch_size=batch_size)  # 选择设备

        # 创建目录
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增加运行次数
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

        # 加载模型
        check_suffix(weights, '.pt')  # 检查权重文件后缀
        model = attempt_load(weights, map_location=device)  # 加载 FP32 模型
        gs = max(int(model.stride.max()), 32)  # 网格大小（最大步幅）
        imgsz = check_img_size(imgsz, s=gs)  # 检查图像尺寸

        # 多 GPU 不支持，因 .half() 不兼容
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # 数据
        data = check_dataset(data)  # 检查数据集

    # 半精度
    half &= device.type != 'cpu'  # 半精度仅支持 CUDA
    model.half() if half else model.float()  # 根据条件设置模型为半精度或单精度

    # Configure
    model.eval()  # 设置模型为评估模式
    # 检查数据集是否为 COCO 格式，验证集路径以 'coco/val2017.txt' 结尾
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')
    nc = 1 if single_cls else int(data['nc'])  # 类别数量，单类数据集则为 1
    # 创建一个 IoU 向量，用于计算 mAP@0.5:0.95
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()  # 获取 IoU 的数量

    # Dataloader
    if not training:  # 如果不是训练模式
        if device.type != 'cpu':
            # 在设备上运行一次模型，以确保模型已正确加载
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        pad = 0.0 if task == 'speed' else 0.5  # 根据任务类型设置填充值
        # 确保任务类型是有效的，如果无效则默认使用 'val'
        task = task if task in ('train', 'val', 'test') else 'val'
        # 创建数据加载器，获取指定任务的数据集
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=pad, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    # 初始化计数器和混淆矩阵
    seen = 0  # 记录已处理的图像数量
    confusion_matrix = ConfusionMatrix(nc=nc)  # 创建混淆矩阵实例
    # 获取模型的类名
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # 设置类别映射，如果是COCO数据集则使用COCO特定的映射
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 打印结果的格式
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # 初始化各种性能指标
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)  # 初始化损失值
    jdict, stats, ap, ap_class = [], [], [], []  # 初始化结果存储列表

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # 遍历数据加载器中的每个批次，并显示进度条
        t1 = time_sync()  # 记录开始时间
        img = img.to(device, non_blocking=True)  # 将图像数据移动到指定设备（GPU或CPU）
        img = img.half() if half else img.float()  # 将图像转换为半精度（FP16）或单精度（FP32）
        img /= 255.0  # 将图像数据从0-255范围归一化到0.0-1.0
        targets = targets.to(device)  # 将目标数据移动到指定设备
        nb, _, height, width = img.shape  # 获取当前批次的大小（nb），通道数（_），高度和宽度
        t2 = time_sync()  # 记录结束时间
        dt[0] += t2 - t1  # 计算并累加处理时间

        # Run model
        out, train_out = model(img, augment=augment)  # 进行推理，获取模型输出和训练输出
        dt[1] += time_sync() - t2  # 记录模型推理所需时间

        # Compute loss
        if compute_loss:
            # 如果指定计算损失，则计算并累加损失
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # 将目标框转换为像素坐标
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # 为自动标注准备标签
        t3 = time_sync()  # 记录开始时间
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        # 运行非极大值抑制，过滤掉重叠框
        dt[2] += time_sync() - t3  # 记录NMS所需时间

        # Statistics per image
        for si, pred in enumerate(out):  # 遍历每张图像的预测结果
            labels = targets[targets[:, 0] == si, 1:]  # 获取当前图像的真实标签
            nl = len(labels)  # 标签数量
            tcls = labels[:, 0].tolist() if nl else []  # 目标类别
            path, shape = Path(paths[si]), shapes[si][0]  # 当前图像路径和形状
            seen += 1  # 统计已处理的图像数量

            if len(pred) == 0:  # 如果没有预测框
                if nl:  # 如果有真实标签
                    stats.append(
                        (torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))  # 添加空的统计信息
                continue  # 继续处理下一张图像

            # Predictions
            if single_cls:  # 如果为单类检测，将所有预测的类别设为0
                pred[:, 5] = 0
            predn = pred.clone()  # 克隆预测结果
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # 将预测框缩放到原始图像空间

            # Evaluate
            if nl:  # 如果有真实标签
                tbox = xywh2xyxy(labels[:, 1:5])  # 将标签框从xywh格式转换为xyxy格式
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # 缩放标签框到原始图像空间
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # 将标签合并为一张表
                correct = process_batch(predn, labelsn, iouv)  # 计算正确的预测
                if plots:  # 如果需要绘图
                    confusion_matrix.process_batch(predn, labelsn)  # 更新混淆矩阵
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)  # 如果没有标签，则初始化为全零
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # 添加统计信息

            # Save/log
            if save_txt:  # 如果需要保存txt格式的结果
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:  # 如果需要保存为COCO-JSON格式
                save_one_json(predn, jdict, path, class_map)  # 将结果添加到COCO-JSON字典中
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])  # 运行回调函数

        # Plot images
        if plots and batch_i < 3:  # 如果需要绘图且当前批次小于3
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # 保存真实标签图像
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # 保存预测结果图像
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    # 计算统计数据
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # 将每个统计结果连接为numpy数组
    if len(stats) and stats[0].any():  # 检查是否有有效的统计数据
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        # 计算每类的精确率p、召回率r、平均精度ap和F1分数f1
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5和AP@0.5:0.95的平均精度
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()  # 计算各项指标的均值
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # 计算每个类别的目标数量
    else:
        nt = torch.zeros(1)  # 如果没有有效数据，返回一个零的张量

    # Print results
    # 打印结果
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # 打印格式
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))  # 打印整体统计结果

    # Print results per class
    # 打印每类的结果
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))  # 打印每个类别的统计信息

    # Print speeds
    # 打印速度信息
    t = tuple(x / seen * 1E3 for x in dt)  # 每张图片的速度
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    # 绘图
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))  # 绘制混淆矩阵
        callbacks.run('on_val_end')  # 执行验证结束的回调

    # 保存 JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # 权重文件名
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # 注释 JSON 文件
        pred_json = str(save_dir / f"{w}_predictions.json")  # 预测 JSON 文件
        print(f'\n正在评估 pycocotools mAP... 保存 {pred_json}...')

        with open(pred_json, 'w') as f:
            json.dump(jdict, f)  # 将预测结果写入 JSON 文件

        try:
            # 检查是否安装 pycocotools
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # 初始化注释 API
            pred = anno.loadRes(pred_json)  # 初始化预测 API
            eval = COCOeval(anno, pred, 'bbox')  # 创建 COCO 评估对象

            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # 要评估的图像 ID
            eval.evaluate()  # 进行评估
            eval.accumulate()  # 汇总评估结果
            eval.summarize()  # 输出评估摘要
            map, map50 = eval.stats[:2]  # 更新结果 (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools 无法运行: {e}')  # 错误处理

    # 返回结果
    model.float()  # 转换模型为浮点数模式以进行训练
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"结果已保存到 {colorstr('bold', save_dir)}{s}")

    maps = np.zeros(nc) + map  # 初始化 mAP 数组
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]  # 将每个类别的平均精度存入 maps

    # 返回包括指标和损失的元组
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/VOC-hat.yaml', help='dataset.yaml path')  # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp/weights/best.pt', help='model.pt path(s)')   #  模型的权重文件地址 weights
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')   # 前向传播的批次大小 默认32
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=608, help='inference size (pixels)')  #  输入网络的图片分辨率 默认640
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')  # object置信度阈值 默认0.25
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')  # 进行NMS时IOU的阈值 默认0.6
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')   # 设置测试的类型 有train, val, test, speed or study几种 默认val
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # 测试的设备
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')  # 数据集是否只用一个类别 默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')   # 是否使用数据增强进行推理，默认为False
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')   # 是否打印出每个类别的mAP 默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')  #  是否以txt文件的形式保存模型预测框的坐标 默认False
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')  # 是否save label+prediction hybrid results to *.txt  默认False 是否将gt_label+pre_label一起输入nms
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')   # save-conf: 是否保存预测每个目标的置信度到预测txt文件中 默认False
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')    # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认False
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')  # 测试保存的源文件 默认runs/val
    parser.add_argument('--name', default='exp', help='save to project/name')   # name: 当前测试结果放在runs/val下的文件名  默认是exp
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')   # -exist-ok: 是否覆盖已有结果，默认为 False
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')   # half: 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
    opt = parser.parse_args()  # 解析上述参数
    opt.data = check_yaml(opt.data)   # 解析并检查参数文件（通常是 YAML 格式）
    opt.save_json |= opt.data.endswith('coco.yaml')  # 如果 opt.data 以 'coco.yaml' 结尾，则设置 save_json 为 True
    opt.save_txt |= opt.save_hybrid   # 如果 save_hybrid 为 True，则设置 save_txt 为 True
    print_args(FILE.stem, opt)   # 打印参数信息
    return opt


def main(opt):
    # 设置日志记录
    set_logging()
    # 检查依赖项，排除 'tensorboard' 和 'thop'
    check_requirements(exclude=('tensorboard', 'thop'))

    # 根据任务类型执行相应的操作
    if opt.task in ('train', 'val', 'test'):  # 正常运行
        run(**vars(opt))  # 运行训练、验证或测试

    elif opt.task == 'speed':  # 进行速度基准测试
        # 例如：python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                device=opt.device, save_json=False, plots=False)  # 运行速度测试

    elif opt.task == 'study':  # 在一系列设置上运行并保存/绘制结果
        # 例如：python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
        x = list(range(256, 1536 + 128, 128))  # x 轴（图像尺寸范围）
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # 保存的文件名
            y = []  # y 轴（结果列表）
            for i in x:  # 对每个图像尺寸进行运行
                print(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, save_json=opt.save_json, plots=False)
                y.append(r + t)  # 将结果和时间添加到 y 轴列表中
            np.savetxt(f, y, fmt='%10.4g')  # 保存结果到文件
        os.system('zip -r study.zip study_*.txt')  # 压缩保存的结果文件
        plot_val_study(x=x)  # 绘制结果图

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
