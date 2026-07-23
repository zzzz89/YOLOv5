# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_args, print_mutation, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp,  # path/to/hyp.yaml 或者超参数字典
          opt,
          device,
          callbacks
          ):
    # 设置训练相关的目录和参数
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # 创建保存模型权重的目录
    w = save_dir / 'weights'  # 权重保存目录
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # 如果需要演化，则创建父目录，否则创建权重目录
    last, best = w / 'last.pt', w / 'best.pt'  # 定义最后和最好的模型文件路径

    # 加载超参数
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 从 YAML 文件中加载超参数字典
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))  # 记录超参数信息

    # 保存运行设置
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)  # 保存超参数到 YAML 文件
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)  # 保存训练选项到 YAML 文件
    data_dict = None  # 初始化数据字典

    # 初始化日志记录器
    if RANK in [-1, 0]:  # 仅在主进程中执行
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # 创建日志记录器实例
        if loggers.wandb:  # 如果使用 wandb 进行实验追踪
            data_dict = loggers.wandb.data_dict  # 获取 wandb 的数据字典
            if resume:  # 如果是恢复训练
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # 更新权重和超参数

        # 注册回调函数
        for k in methods(loggers):  # 遍历日志记录器的方法
            callbacks.register_action(k, callback=getattr(loggers, k))  # 将日志记录器的方法注册为回调

    # 配置
    plots = not evolve  # 是否创建绘图，演化模式下不创建
    cuda = device.type != 'cpu'  # 检查是否使用 CUDA（GPU）
    init_seeds(1 + RANK)  # 初始化随机种子，确保每个进程的种子不同

    # 在分布式训练的主进程中执行以下操作
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # 检查数据集，如果数据字典为 None，则加载数据集

    # 获取训练和验证数据集的路径
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # 获取类别数量，单类别情况下数量为 1
    # 获取类别名称，如果是单类别且名称列表长度不为 1，则设为 ['item']
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # 检查类别名称的数量是否与 nc 匹配
    is_coco = data.endswith('coco.yaml') and nc == 80  # 检查数据集是否为 COCO 数据集，且类别数量是否为 80

    # Model
    check_suffix(weights, '.pt')  # 检查权重文件的后缀是否为 .pt
    pretrained = weights.endswith('.pt')  # 判断权重文件是否为预训练模型

    if pretrained:
        # 如果是预训练模型，则尝试下载它
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # 如果本地找不到权重文件，则下载

        ckpt = torch.load(weights, map_location=device, weights_only=False)  # 加载检查点
        # 创建模型，cfg 为配置文件，ch 为输入通道数（一般为3），nc 为类别数，anchors 为锚框
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 创建模型实例

        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # 定义需要排除的键
        csd = ckpt['model'].float().state_dict()  # 获取检查点的 state_dict，转为 FP32 格式
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # 交集，获取匹配的参数
        model.load_state_dict(csd, strict=False)  # 加载参数
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # 输出转移的参数数量
    else:
        # 如果不是预训练模型，则使用给定的 cfg 创建新模型
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 创建模型实例

    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # 定义需要冻结的层
    for k, v in model.named_parameters():
        v.requires_grad = True  # 默认所有层均可训练
        if any(x in k for x in freeze):  # 检查当前层是否在冻结列表中
            print(f'freezing {k}')  # 打印冻结层的信息
            v.requires_grad = False  # 冻结该层的参数

    # 优化器
    nbs = 64  # 规定的批量大小
    accumulate = max(round(nbs / batch_size), 1)  # 在优化之前累积损失
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # 按照批量大小缩放 weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")  # 记录缩放后的 weight_decay

    g0, g1, g2 = [], [], []  # 定义优化器参数组

    # 遍历模型的所有模块
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # 如果模块有偏置
            g2.append(v.bias)  # 将偏置添加到 g2
        if isinstance(v, nn.BatchNorm2d):  # 如果模块是 BatchNorm2d
            g0.append(v.weight)  # 将权重添加到 g0（不使用权重衰减）
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # 如果模块有权重
            g1.append(v.weight)  # 将权重添加到 g1（使用权重衰减）

    # 根据选择的优化器类型创建优化器
    if opt.adam:
        # 使用 Adam 优化器，调整 beta1 为动量
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        # 使用 SGD 优化器
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # 添加参数组 g1（使用 weight_decay）和 g2（偏置）
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': g2})  # 添加偏置 g2
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")  # 记录优化器信息

    # 清理不再需要的参数组
    del g0, g1, g2

    # 学习率调度器
    if opt.linear_lr:
        # 如果选择线性学习率调度，定义学习率函数 lf
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # 线性调度
    else:
        # 否则使用余弦调度，创建学习率函数 lf
        lf = one_cycle(1, hyp['lrf'], epochs)  # 从 1 到 hyp['lrf'] 的余弦调度

    # 创建学习率调度器，将优化器和学习率函数 lf 传入
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # 可视化学习率调度器 (可选)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0  # 初始化开始的轮次和最佳适应度
    if pretrained:  # 如果使用预训练模型
        # Optimizer
        if ckpt['optimizer'] is not None:  # 如果检查点中包含优化器状态
            optimizer.load_state_dict(ckpt['optimizer'])  # 加载优化器的状态字典
            best_fitness = ckpt['best_fitness']  # 更新最佳适应度

        # EMA
        if ema and ckpt.get('ema'):  # 如果启用 EMA 且检查点中包含 EMA 状态
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # 加载 EMA 的状态字典
            ema.updates = ckpt['updates']  # 更新 EMA 的次数

        # Epochs
        start_epoch = ckpt['epoch'] + 1  # 设置开始的轮次为检查点的轮次加 1
        if resume:  # 如果选择了恢复训练
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'  # 确保可以恢复
        if epochs < start_epoch:  # 如果设置的轮次小于恢复的轮次
            LOGGER.info(
                f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")  # 日志记录
            epochs += ckpt['epoch']  # 继续训练更多轮次

        del ckpt, csd  # 清理检查点和其他变量以释放内存

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # 获取模型的最大步幅作为网格大小，确保至少为 32
    nl = model.model[-1].nl  # 获取检测层的数量（用于缩放 hyp['obj'] 超参数）
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # 验证图像大小是否是网格大小的倍数，且不小于 gs 的两倍

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        # 如果在多 GPU 环境下且未使用分布式训练，发出警告
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)  # 使用数据并行（DataParallel）

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        # 如果启用了同步批归一化且处于分布式训练模式，转换模型为同步批归一化
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')  # 记录使用同步批归一化的信息

    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
        hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect,
        rank=LOCAL_RANK, workers=workers, image_weights=opt.image_weights,
        quad=opt.quad, prefix=colorstr('train: ')
    )  # 创建训练数据加载器和数据集

    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # 找到数据集中最大标签类
    nb = len(train_loader)  # 计算批次的数量

    # 检查最大标签类是否小于类别总数
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # 处理过程 0
    if RANK in [-1, 0]:
        # 创建验证数据加载器，批大小是原来的两倍
        val_loader = create_dataloader(
            val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
            hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
            workers=workers, pad=0.5,
            prefix=colorstr('val: ')
        )[0]

        # 如果不是恢复训练
        if not resume:
            labels = np.concatenate(dataset.labels, 0)  # 合并所有标签
            # c = torch.tensor(labels[:, 0])  # 提取类别
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # 统计频率
            # model._initialize_biases(cf.to(device))  # 初始化偏置

            if plots:
                plot_labels(labels, names, save_dir)  # 绘制标签分布

            # 锚框检查
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

            model.half().float()  # 预先减少锚框精度

        callbacks.run('on_pretrain_routine_end')  # 运行训练前例程结束的回调

    # DDP模式
    if cuda and RANK != -1:
        # 使用分布式数据并行（DDP）包装模型
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # 模型参数
    hyp['box'] *= 3. / nl  # 将框的超参数缩放到检测层数量
    hyp['cls'] *= nc / 80. * 3. / nl  # 将类别超参数缩放到类别数量和检测层数量
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # 将目标超参数缩放到图像尺寸和检测层数量
    hyp['label_smoothing'] = opt.label_smoothing  # 设置标签平滑参数
    # 将类别数量附加到模型
    model.nc = nc  # attach number of classes to model
    # 将超参数附加到模型
    model.hyp = hyp  # attach hyperparameters to model
    # 计算并附加类别权重到模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # 将类别名称附加到模型
    model.names = names  # attach class names to model

    # 开始训练
    t0 = time.time()  # 记录开始时间
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # 计算预热迭代次数，最小为1000次（相当于3个epoch）
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # 限制预热时间小于总训练时间的一半

    last_opt_step = -1  # 最后一次优化步骤
    maps = np.zeros(nc)  # 每个类别的mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    scheduler.last_epoch = start_epoch - 1  # 设置调度器的最后epoch为当前epoch之前
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)  # 初始化混合精度训练的梯度缩放器
    stopper = EarlyStopping(patience=opt.patience)  # 初始化早停机制，设定耐心值
    compute_loss = ComputeLoss(model)  # 初始化损失计算类

    # 记录训练信息
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')  # 打印训练开始信息

    for epoch in range(start_epoch, epochs):  # 训练周期循环
        model.train()  # 设置模型为训练模式

        # 可选：更新图像权重（仅适用于单GPU）
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # 计算类别权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # 计算图像权重
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # 随机加权索引

        # 可选：更新马赛克边框
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # 设置高度和宽度边框

        mloss = torch.zeros(3, device=device)  # 初始化平均损失
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)  # 设置训练加载器的当前周期
        pbar = enumerate(train_loader)  # 遍历训练数据加载器
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))  # 日志信息
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # 显示进度条
        optimizer.zero_grad()  # 优化器梯度清零
        for i, (imgs, targets, paths, _) in pbar:  # 批处理循环
            ni = i + nb * epoch  # 计算自训练开始以来的集成批次数
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # 将uint8类型转换为float32并归一化

            # Warmup阶段
            if ni <= nw:
                xi = [0, nw]  # 线性插值范围
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # IOU损失比率（obj_loss = 1.0或IOU）
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())  # 计算累积步骤
                for j, x in enumerate(optimizer.param_groups):  # 遍历优化器参数组
                    # 更新学习率：偏置学习率从0.1降到lr0，其他学习率从0.0升到lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])  # 更新动量

            # 多尺度训练
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # 随机选择大小
                sf = sz / max(imgs.shape[2:])  # 计算缩放因子
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # 计算新形状（调整为gs的倍数）
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)  # 重新调整图像大小

            # 前向传播
            with torch.amp.autocast('cuda', enabled=cuda):  # 使用混合精度训练
                pred = model(imgs)  # 前向传播得到预测
                loss, loss_items = compute_loss(pred, targets.to(device))  # 计算损失
                if RANK != -1:
                    loss *= WORLD_SIZE  # 在DDP模式下进行梯度平均
                if opt.quad:
                    loss *= 4.  # 如果使用四元组，则损失乘以4

            # 反向传播
            scaler.scale(loss).backward()  # 反向传播并缩放

            # 优化
            if ni - last_opt_step >= accumulate:  # 如果达到累积步骤
                scaler.step(optimizer)  # 更新优化器
                scaler.update()  # 更新缩放器
                optimizer.zero_grad()  # 清零梯度
                if ema:  # 如果使用EMA（指数移动平均）
                    ema.update(model)  # 更新EMA
                last_opt_step = ni  # 更新上一次优化步骤

            # 日志记录
            if RANK in [-1, 0]:  # 如果是主进程
                mloss = (mloss * i + loss_items) / (i + 1)  # 更新平均损失
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # 获取GPU内存使用情况
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))  # 更新进度条描述
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)  # 调用回调函数
        # 结束批处理循环

        # 学习率调度器
        lr = [x['lr'] for x in optimizer.param_groups]  # 记录当前学习率
        scheduler.step()  # 更新学习率

        if RANK in [-1, 0]:  # 如果是主进程
            # 计算mAP（平均精度均值）
            callbacks.run('on_train_epoch_end', epoch=epoch)  # 运行回调函数，记录训练周期结束
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])  # 更新EMA模型属性
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop  # 检查是否是最后一个周期

            if not noval or final_epoch:  # 如果不进行验证或是最后一个周期
                results, maps, _ = val.run(data_dict,  # 验证模型
                                           batch_size=batch_size // WORLD_SIZE * 2,  # 设置验证批次大小
                                           imgsz=imgsz,  # 图像尺寸
                                           model=ema.ema,  # 使用EMA模型进行验证
                                           single_cls=single_cls,  # 是否为单类别
                                           dataloader=val_loader,  # 验证数据加载器
                                           save_dir=save_dir,  # 保存路径
                                           plots=False,  # 是否绘制图
                                           callbacks=callbacks,  # 回调函数
                                           compute_loss=compute_loss)  # 计算损失

            # 更新最佳mAP
            fi = fitness(np.array(results).reshape(1, -1))  # 计算适应度（加权组合[精度, 召回率, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:  # 如果当前适应度大于最佳适应度
                best_fitness = fi  # 更新最佳适应度

            log_vals = list(mloss) + list(results) + lr  # 合并损失、结果和学习率
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)  # 运行适应度记录回调

            # 保存模型
            if (not nosave) or (final_epoch and not evolve):  # 如果需要保存模型
                ckpt = {'epoch': epoch,  # 当前周期
                        'best_fitness': best_fitness,  # 最佳适应度
                        'model': deepcopy(de_parallel(model)).half(),  # 深拷贝模型并转换为半精度
                        'ema': deepcopy(ema.ema).half(),  # 深拷贝EMA模型并转换为半精度
                        'updates': ema.updates,  # EMA更新次数
                        'optimizer': optimizer.state_dict(),  # 优化器状态
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}  # wandb ID

                # 保存最后模型和最佳模型，并根据周期删除
                torch.save(ckpt, last)  # 保存最后的模型
                if best_fitness == fi:  # 如果当前适应度为最佳
                    torch.save(ckpt, best)  # 保存最佳模型
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):  # 根据周期保存模型
                    torch.save(ckpt, w / f'epoch{epoch}.pt')  # 保存指定周期的模型
                del ckpt  # 删除检查点以释放内存
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)  # 运行模型保存回调

            # 停止单GPU训练
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):  # 如果是单GPU且满足停止条件
                break  # 结束训练
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:  # 如果是主进程
        # 记录已完成的周期和耗时
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')

        # 对于最后一个和最佳模型进行处理
        for f in last, best:
            if f.exists():  # 如果文件存在
                strip_optimizer(f)  # 去除优化器状态以减小模型文件大小

                if f is best:  # 如果是最佳模型
                    LOGGER.info(f'\nValidating {f}...')  # 记录验证信息
                    results, _, _ = val.run(data_dict,  # 验证模型
                                            batch_size=batch_size // WORLD_SIZE * 2,  # 设置批次大小
                                            imgsz=imgsz,  # 图像尺寸
                                            model=attempt_load(f, device).half(),  # 加载模型并转换为半精度
                                            iou_thres=0.65 if is_coco else 0.60,  # 设置IOU阈值（针对COCO数据集）
                                            single_cls=single_cls,  # 是否为单类别
                                            dataloader=val_loader,  # 验证数据加载器
                                            save_dir=save_dir,  # 保存路径
                                            save_json=is_coco,  # 是否保存为JSON（针对COCO数据集）
                                            verbose=True,  # 是否输出详细信息
                                            plots=True,  # 是否绘制图
                                            callbacks=callbacks,  # 回调函数
                                            compute_loss=compute_loss)  # 计算损失

        callbacks.run('on_train_end', last, best, plots, epoch)  # 运行训练结束的回调
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")  # 记录结果保存路径

    torch.cuda.empty_cache()  # 清空CUDA缓存
    return results  # 返回结果


def parse_opt(known=False):
    """
            函数功能：设置opt参数
    """
    parser = argparse.ArgumentParser()
    # --------------------------------------------------- 常用参数 ---------------------------------------------
    parser.add_argument('--weights', type=str, default=ROOT / 'weights/yolov5x.pt', help='initial weights path')  # weights: 权重文件
    parser.add_argument('--cfg', type=str, default='models/yolov5x.yaml', help='model.yaml path')  # cfg: 网络模型配置文件 包括nc、depth_multiple、width_multiple、anchors、backbone、head等
    parser.add_argument('--data', type=str, default=ROOT / 'data/VOC-hat.yaml', help='dataset.yaml path')  # data: 实现数据集配置文件 包括path、train、val、test、nc、names等
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')  # hyp: 训练时的超参文件
    parser.add_argument('--epochs', type=int, default=60)  # epochs: 训练轮次
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')  # batch-size: 训练批次大小
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=608, help='train, val image size (pixels)')  # imgsz: 输入网络的图片分辨率大小
    parser.add_argument('--rect', action='store_true', help='rectangular training')  # rect: 是否采用Rectangular training/inference，一张图片为长方形，我们在将其送入模型前需要将其resize到要求的尺寸，所以我们需要通过补灰padding来变为正方形的图。
    parser.add_argument('--resume', nargs='?', const=True, default="", help='resume most recent training')  # resume: 断点续训, 从上次打断的训练结果处接着训练  默认False
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')  # nosave: 不保存模型  默认保存  store_true: only test final epoch
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')   # noval: 只在最后一次进行测试，默认False
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')   # noautoanchor: 不自动调整anchor 默认False(自动调整anchor)
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')  # evolve: 是否进行超参进化，使得数值更好 默认False
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')   # bucket: 谷歌云盘bucket 一般用不到
    parser.add_argument('--cache', type=str, nargs='?', const='ram', default="True", help='--cache images in "ram" (default) or "disk"')  # cache:是否提前缓存图片到内存，以加快训练速度
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')  #  image-weights: 对于那些训练不好的图片，会在下一轮中增加一些权重
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # device: 训练的设备
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')  # multi-scale: 是否使用多尺度训练 默认False，要被32整除。
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')  # single-cls: 数据集是否只有一个类别 默认False
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')  # adam: 是否使用adam优化器
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')  # sync-bn: 是否使用跨卡同步bn操作,再DDP中使用  默认False
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')  # workers: dataloader中的最大work数（线程个数）
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')  # project: 训练结果保存的根目录 默认是runs/train
    parser.add_argument('--name', default='exp', help='save to project/name')  # name: 训练结果保存的目录 默认是exp
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')  # exist_ok: 是否重新创建日志文件, False时重新创建文件(默认文件都是不存在的)
    parser.add_argument('--quad', action='store_true', help='quad dataloader')  # quad: dataloader取数据时, 是否使用collate_fn4代替collate_fn  默认False
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')  # linear-lr：用于对学习速率进行调整，默认为 False，（通过余弦函数来降低学习率）
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')  # label-smoothing: 标签平滑增强 默认0.0不增强  要增强一般就设为0.1
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')  # 早停机制，训练到一定的epoch，如果模型效果未提升，就让模型提前停止训练。
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')  # freeze: 使用预训练模型的规定固定权重不进行调整  --freeze 10  :意思从第0层到到第10层不训练
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')  # 设置多少个epoch保存一次模型
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')  # local_rank: rank为进程编号  -1且gpu=1时不进行分布式  -1且多块gpu使用DataParallel模式

    # --------------------------------------------------- W&B(wandb)参数 ---------------------------------------------
    parser.add_argument('--entity', default=None, help='W&B: Entity')  #wandb entity 默认None
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')  # 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')  # 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt, callbacks=Callbacks()):
    # 设置日志记录
    set_logging(RANK)

    # 主进程检查
    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)  # 打印运行参数
        check_git_status()  # 检查Git仓库状态（确保代码是最新版本）
        check_requirements(exclude=['thop'])  # 检查依赖包，排除'thop'包

    # 恢复中断的运行
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # 检查是否从中断位置恢复
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # 获取指定或最近的检查点路径
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'  # 检查检查点文件是否存在
        # 从指定检查点目录加载训练配置
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # 加载训练配置到opt变量
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # 设置配置文件路径和权重文件路径
        LOGGER.info(f'Resuming training from {ckpt}')  # 打印恢复信息
    else:
        # 校验文件和路径的配置
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(
                opt.project)  # 检查配置文件路径
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'  # 确保cfg或weights参数至少一个存在
        if opt.evolve:  # 演化模式
            opt.project = str(ROOT / 'runs/evolve')  # 设置演化运行保存路径
            opt.exist_ok, opt.resume = opt.resume, False  # 设置路径是否覆盖并禁用恢复
        # 设置保存目录并递增路径
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP模式（分布式数据并行）
    device = select_device(opt.device, batch_size=opt.batch_size)  # 选择计算设备
    if LOCAL_RANK != -1:  # 如果启用DDP
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'  # 检查CUDA设备数量是否足够
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'  # 确保batch size是设备数的倍数
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'  # 确保未启用图像权重
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'  # 确保未启用演化模式
        torch.cuda.set_device(LOCAL_RANK)  # 设置CUDA设备
        device = torch.device('cuda', LOCAL_RANK)  # 指定设备
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")  # 初始化进程组，选择nccl或gloo作为通信后端

    # 训练模型
    if not opt.evolve:  # 如果不是演化模式，进行训练
        train(opt.hyp, opt, device, callbacks)  # 调用train函数进行训练
        if WORLD_SIZE > 1 and RANK == 0:  # 在多GPU模式下，销毁进程组
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()  # 销毁DDP进程组

    # 进行超参数演化（可选）
    else:
        # 超参数演化元数据（变异规模 0-1，最小值，上限）
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # 初始学习率 (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # 最终 OneCycleLR 学习率 (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD 动量/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # 优化器的权重衰减
            'warmup_epochs': (1, 0.0, 5.0),  # 预热轮数（允许使用小数）
            'warmup_momentum': (1, 0.0, 0.95),  # 预热初始动量
            'warmup_bias_lr': (1, 0.0, 0.2),  # 预热初始偏置学习率
            'box': (1, 0.02, 0.2),  # 边框损失增益
            'cls': (1, 0.2, 4.0),  # 分类损失增益
            'cls_pw': (1, 0.5, 2.0),  # 分类 BCELoss 正权重
            'obj': (1, 0.2, 4.0),  # 目标损失增益（与像素成比例）
            'obj_pw': (1, 0.5, 2.0),  # 目标 BCELoss 正权重
            'iou_t': (0, 0.1, 0.7),  # IoU 训练阈值
            'anchor_t': (1, 2.0, 8.0),  # 锚点倍数阈值
            'anchors': (2, 2.0, 10.0),  # 每个输出网格的锚点数量（0 为忽略）
            'fl_gamma': (0, 0.0, 2.0),  # 聚焦损失伽马（efficientDet 默认伽马=1.5）
            'hsv_h': (1, 0.0, 0.1),  # 图像 HSV-色相增强（比例）
            'hsv_s': (1, 0.0, 0.9),  # 图像 HSV-饱和度增强（比例）
            'hsv_v': (1, 0.0, 0.9),  # 图像 HSV-亮度增强（比例）
            'degrees': (1, 0.0, 45.0),  # 图像旋转 (+/- 度)
            'translate': (1, 0.0, 0.9),  # 图像平移 (+/- 比例)
            'scale': (1, 0.0, 0.9),  # 图像缩放 (+/- 增益)
            'shear': (1, 0.0, 10.0),  # 图像剪切 (+/- 度)
            'perspective': (0, 0.0, 0.001),  # 图像透视 (+/- 比例)，范围 0-0.001
            'flipud': (1, 0.0, 1.0),  # 图像上下翻转（概率）
            'fliplr': (0, 0.0, 1.0),  # 图像左右翻转（概率）
            'mosaic': (1, 0.0, 1.0),  # 图像混合（概率）
            'mixup': (1, 0.0, 1.0),  # 图像混合（概率）
            'copy_paste': (1, 0.0, 1.0)  # 段落复制粘贴（概率）
        }

        # 打开超参数文件并加载超参数字典
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # 使用 YAML 加载超参数字典
            if 'anchors' not in hyp:  # 如果超参数中没有 anchors（可能被注释掉）
                hyp['anchors'] = 3  # 设置默认的 anchors 数量

        # 设置选项，指示只进行验证和保存最终的训练结果
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # 只验证和保存最终轮次的模型

        # 定义演化文件路径
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:  # 如果指定了云存储桶
            # 下载已有的 evolve.csv 文件
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # 下载 evolve.csv（如果存在）

        # 进行指定轮数的超参数演化
        for _ in range(opt.evolve):  # 迭代演化的代数
            if evolve_csv.exists():  # 如果 evolve.csv 存在，选择最佳超参数并进行变异
                # 选择父代
                parent = 'single'  # 父代选择方法：'single' 或 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)  # 加载演化结果
                n = min(5, len(x))  # 考虑的上一个结果的数量
                x = x[np.argsort(-fitness(x))][:n]  # 按适应度排序，选择前 n 个变异
                w = fitness(x) - fitness(x).min() + 1E-6  # 计算权重（确保和大于0）

                # 根据选择方法选取父代
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # 随机选择
                    x = x[random.choices(range(n), weights=w)[0]]  # 基于权重选择
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 加权组合

                # 进行变异
                mp, s = 0.8, 0.2  # 变异概率，标准差
                npr = np.random
                npr.seed(int(time.time()))  # 设置随机种子
                g = np.array([meta[k][0] for k in hyp.keys()])  # 获取增益，范围 0-1
                ng = len(meta)  # 元数据中的超参数数量
                v = np.ones(ng)  # 初始化变异量

                # 确保变异发生，避免重复
                while all(v == 1):  # 在没有变化时继续变异
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)

                # 应用变异
                for i, k in enumerate(hyp.keys()):  # 遍历超参数
                    hyp[k] = float(x[i + 7] * v[i])  # 变异超参数

            # 限制超参数在预设范围内
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # 限制下限
                hyp[k] = min(hyp[k], v[2])  # 限制上限
                hyp[k] = round(hyp[k], 5)  # 保留五位有效数字

            # 训练变异后的模型
            results = train(hyp.copy(), opt, device, callbacks)

            # 写入变异结果
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # 绘制结果图表
        plot_evolve(evolve_csv)
        print(f'Hyperparameter evolution finished\n'
              f"Results saved to {colorstr('bold', save_dir)}\n"
              f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')

def run(**kwargs):
    # 用法示例: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)  # 解析命令行参数并返回选项对象
    for k, v in kwargs.items():  # 遍历关键字参数
        setattr(opt, k, v)  # 将每个参数设置到选项对象中
    main(opt)  # 调用主函数，传入选项对象

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
