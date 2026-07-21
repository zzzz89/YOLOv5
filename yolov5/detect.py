# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # 模型路径
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # 推理图像大小（像素）
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # 非极大值抑制（NMS）IOU 阈值
        max_det=1000,  # 每张图像的最大检测数
        device='',  # CUDA 设备，例如 0 或 0,1,2,3 或 CPU
        view_img=False,  # 显示结果
        save_txt=False,  # 将结果保存到 *.txt
        save_conf=False,  # 在保存的标签中包含置信度
        save_crop=False,  # 保存裁剪后的预测框
        nosave=False,  # 不保存图像/视频
        classes=None,  # 按类别过滤：--class 0 或 --class 0 2 3
        agnostic_nms=False,  # 类别无关的 NMS
        augment=False,  # 增强推理
        visualize=False,  # 可视化特征
        update=False,  # 更新所有模型
        project=ROOT / 'runs/detect',  # 保存结果的项目路径
        name='exp',  # 保存结果的项目名称
        exist_ok=False,  # 允许现有的项目名称，不递增
        line_thickness=3,  # 边界框厚度（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用 FP16 半精度推理
        dnn=False,  # 使用 OpenCV DNN 进行 ONNX 推理
        ):
    # ===================================== 1、初始化一些配置 =====================================
    # 是否保存预测后的图片 默认nosave=False 所以只要传入的文件地址不是以.txt结尾 就都是要保存预测后的图片的
    source = str(source)  # 将输入的 source 转换为字符串，确保兼容性，方便后续处理
    save_img = not nosave and not source.endswith('.txt')  # 判断是否保存推理后的图像。条件为 nosave 为 False 且 source 不以 '.txt' 结尾
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(  # 判断输入 source 是否为网络流或数字摄像头
        ('rtsp://', 'rtmp://', 'http://', 'https://'))  # 如果 source 是数字、以 .txt 结尾，或者是以指定协议开头的网络流，则将 webcam 设置为 True

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 为保存目录创建递增路径，例如 runs/exp, runs/exp2 等
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # 创建保存目录，如果 save_txt 为 True，则在保存目录下创建 'labels' 子目录
    # Initialize
    set_logging()  # 初始化日志配置
    device = select_device(device)  # 选择计算设备（CUDA 或 CPU），如果系统支持 GPU，则优先使用 GPU
    half &= device.type != 'cpu'  # 如果设备不是 CPU 且支持 CUDA，则启用半精度浮点数处理（加速推理）

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)  # 将权重转换为字符串，支持单个或列表输入
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']  # 初始化分类标志和权重后缀
    check_suffix(w, suffixes)  # 检查权重文件后缀是否有效
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # 确定模型类型的布尔值
    stride, names = 64, [f'class{i}' for i in range(1000)]  # 设置默认步幅和类名称

    # 如果加载的是 PyTorch 模型
    if pt:
        # 如果是 torchscript 模型，则通过 torch.jit.load 加载；否则使用 attempt_load 函数加载权重
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        # 获取模型的步长（stride），通常用于调整输入图像尺寸
        stride = int(model.stride.max())  # model stride
        # 获取模型的类别名称，适配分布式训练的情况
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        # 如果使用半精度 (FP16) 推理，则将模型转换为 FP16
        if half:
            model.half()  # to FP16
        # 如果启用分类，则加载第二阶段分类器（如 resnet50）
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device, weights_only=False)['model']).to(device).eval()

    # 如果加载的是 ONNX 模型
    elif onnx:
        if dnn:
            # 使用 OpenCV DNN 模块加载 ONNX 模型
            # check_requirements(('opencv-python>=4.5.4',))  # 可选的版本检查
            net = cv2.dnn.readNetFromONNX(w)
        else:
            # 检查是否安装了 ONNX 和 ONNX Runtime
            check_requirements(('onnx', 'onnxruntime'))
            import onnxruntime
            # 使用 ONNX Runtime 加载 ONNX 模型
            session = onnxruntime.InferenceSession(w, None)

    # 如果加载的是 TensorFlow 模型
    else:  # TensorFlow models
        # 检查是否安装了 TensorFlow 2.4.1 或更高版本
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        # 如果是 .pb 格式的冻结图模型
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            # 定义一个包装冻结图的函数
            def wrap_frozen_graph(gd, inputs, outputs):
                # 使用 tf.compat.v1.wrap_function 导入冻结图，并返回经过修剪的函数
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            # 创建 TensorFlow 图定义对象并加载 .pb 文件
            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            # 使用包装函数加载冻结图
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        # 如果是保存的 TensorFlow 模型
        elif saved_model:
            # 使用 tf.keras 加载保存的模型
            model = tf.keras.models.load_model(w)
        # 如果是 TensorFlow Lite 模型
        elif tflite:
            # 加载 TensorFlow Lite 模型
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            # 分配张量空间
            interpreter.allocate_tensors()  # allocate
            # 获取模型的输入信息
            input_details = interpreter.get_input_details()  # inputs
            # 获取模型的输出信息
            output_details = interpreter.get_output_details()  # outputs
            # 判断是否是量化的 uint8 模型
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model

    # 检查输入图像的尺寸是否符合模型要求，调整为 stride 的倍数
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # 数据加载器部分
    # 如果输入来源是摄像头
    if webcam:
        # 检查系统是否支持 OpenCV 的图像显示功能
        view_img = check_imshow()
        # 如果输入图像尺寸是固定的，设置 cudnn.benchmark 为 True 以加速推理
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # 使用 LoadStreams 加载摄像头输入流作为数据集
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        # 获取摄像头输入流的批量大小（即摄像头数量）
        bs = len(dataset)  # batch_size
    else:
        # 如果输入来源是图片或视频文件，则使用 LoadImages 加载数据
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        # 单张图像或视频的批量大小为 1
        bs = 1  # batch_size

    # 初始化视频路径和视频写入器列表，长度与批量大小一致
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 推理过程
    # 如果使用的是 PyTorch 模型并且设备不是 CPU
    if pt and device.type != 'cpu':
        # 模型预热：传入一个零张量（形状为 [1, 3, imgsz[0], imgsz[1]]），模拟一次前向传播
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    # 初始化计时变量和已处理样本计数
    dt, seen = [0.0, 0.0, 0.0], 0

    # 遍历数据集中的每一帧数据
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()  # 记录开始时间
        # 如果使用 ONNX 模型
        if onnx:
            img = img.astype('float32')  # 将图像数据类型转换为 float32
        else:
            # 如果使用 PyTorch 模型，将图像数据从 numpy 转换为 torch 张量，并加载到设备
            img = torch.from_numpy(img).to(device)
            # 根据 half 参数选择数据类型（FP16 或 FP32）
            img = img.half() if half else img.float()  # uint8 to fp16/32
        # 归一化图像数据，将像素值从 [0, 255] 缩放到 [0.0, 1.0]
        img = img / 255.0
        # 如果图像是三维的（即没有 batch 维度），则添加 batch 维度
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()  # 记录预处理结束时间
        dt[0] += t2 - t1  # 累计预处理时间

        # 模型推理
        if pt:  # 如果使用的是 PyTorch 模型
            # 如果启用了可视化，递增保存路径
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 执行前向传播，获取预测结果
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:  # 如果使用的是 ONNX 模型
            if dnn:  # 如果使用 OpenCV 的 DNN 模块
                net.setInput(img)  # 设置输入
                pred = torch.tensor(net.forward())  # 获取预测结果并转换为 torch 张量
            else:  # 使用 ONNX Runtime 进行推理
                pred = torch.tensor(session.run([session.get_outputs()[0].name],
                                                {session.get_inputs()[0].name: img}))  # 获取预测结果
        else:  # 如果使用的是 TensorFlow 模型（包括 tflite、pb、saved_model）
            # 将图像从 torch 格式转换为 NumPy 格式，并调整维度顺序
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:  # 如果是 .pb 格式的冻结图模型
                pred = frozen_func(x=tf.constant(imn)).numpy()  # 使用冻结图函数进行推理
            elif saved_model:  # 如果是保存的 TensorFlow 模型
                pred = model(imn, training=False).numpy()  # 执行前向传播
            elif tflite:  # 如果是 TensorFlow Lite 模型
                if int8:  # 如果是量化的 int8 模型
                    # 获取量化参数，将图像数据反量化
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)  # 设置输入张量
                interpreter.invoke()  # 执行推理
                pred = interpreter.get_tensor(output_details[0]['index'])  # 获取输出张量
                if int8:  # 如果输出也是量化模型
                    # 获取量化参数，将预测结果重新量化到实际范围
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            # 将预测框的归一化坐标转换为实际图像尺寸
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            # 将预测结果转换为 torch 张量
            pred = torch.tensor(pred)
        t3 = time_sync()  # 记录推理结束时间
        dt[1] += t3 - t2  # 累计推理时间

        # NMS
        # 非极大值抑制（NMS）处理
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # 对模型的预测结果执行非极大值抑制操作，去除冗余的检测框
        # - pred: 模型的原始预测结果
        # - conf_thres: 置信度阈值，仅保留置信度大于此值的检测框
        # - iou_thres: IOU 阈值，仅保留 IOU 小于此值的框（避免重叠检测框）
        # - classes: 用于过滤检测类别，如果为 None 则保留所有类别
        # - agnostic_nms: 是否类别无关的 NMS，启用后忽略类别信息
        # - max_det: 最大检测框数量，限制返回的检测框总数
        dt[2] += time_sync() - t3  # 记录 NMS 所花费的时间

        # 二阶段分类器（可选步骤）
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
            # 如果启用了 classify 选项，使用二阶段分类器对检测结果进行进一步分类
            # - pred: NMS 处理后的检测框结果
            # - modelc: 二阶段分类器模型
            # - img: 模型输入图片（处理后）
            # - im0s: 原始输入图片，用于二阶段分类器的输入

        # 处理预测结果
        for i, det in enumerate(pred):  # 遍历每张图片的预测结果
            seen += 1  # 统计已处理的图片数量

            # 处理视频流的输入
            if webcam:  # 如果是通过摄像头输入（batch_size >= 1）
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:  # 单张图片或视频文件的输入
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            # 设置保存路径
            p = Path(p)  # 将路径转换为 Path 对象
            save_path = str(save_dir / p.name)  # 保存的图片路径（如 img.jpg）
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # 保存的标签路径（如 img.txt）
            s += '%gx%g ' % img.shape[2:]  # 图片尺寸信息，添加到打印字符串中
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化比例，用于将坐标从 img_size 转换到原图尺寸
            imc = im0.copy() if save_crop else im0  # 如果需要裁剪保存目标框，则复制图片
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # 初始化标注工具

            if len(det):  # 如果有检测结果
                # 将预测框的坐标从 img_size 转换为原图尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印每个类别的检测数量
                for c in det[:, -1].unique():  # 遍历所有预测框的类别
                    n = (det[:, -1] == c).sum()  # 统计该类别的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 将类别和数量信息添加到打印字符串中

                # 写入检测结果
                for *xyxy, conf, cls in reversed(det):  # 遍历每个检测框
                    if save_txt:  # 如果需要保存为文本文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                            -1).tolist()  # 将坐标格式从 xyxy 转为 xywh，并归一化
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式，是否包含置信度
                        with open(txt_path + '.txt', 'a') as f:  # 将标签写入文件
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # 如果需要保存图片或裁剪目标框
                        c = int(cls)  # 转换类别为整数
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 标签内容
                        annotator.box_label(xyxy, label, color=colors(c, True))  # 在图片上标注框和标签
                        if save_crop:  # 如果需要裁剪目标框
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 打印推理时间（仅推理）
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            # 将标注后的结果提取出来
            im0 = annotator.result()

            # 显示检测结果
            if view_img:  # 如果设置为显示图片
                cv2.imshow(str(p), im0)  # 在窗口显示带检测框的图片
                # 图片：等按键再关；视频/摄像头：1ms 刷新
                cv2.waitKey(0 if dataset.mode == 'image' else 1)

            # 保存检测结果（带有检测框的图片或视频）
            if save_img:
                if dataset.mode == 'image':  # 如果是单张图片
                    cv2.imwrite(save_path, im0)  # 保存结果图片到指定路径
                else:  # 如果是视频或流媒体输入
                    if vid_path[i] != save_path:  # 如果保存路径改变，说明是新视频
                        vid_path[i] = save_path  # 更新视频保存路径
                        if isinstance(vid_writer[i], cv2.VideoWriter):  # 如果之前的视频写入器存在
                            vid_writer[i].release()  # 释放之前的视频写入器资源
                        if vid_cap:  # 如果是视频输入
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
                        else:  # 如果是流媒体输入
                            fps, w, h = 30, im0.shape[1], im0.shape[0]  # 默认帧率为 30，宽高为图片尺寸
                            save_path += '.mp4'  # 为保存路径添加扩展名
                        # 初始化视频写入器
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
                        )
                    vid_writer[i].write(im0)  # 将检测结果帧写入视频

    # 打印处理速度
    t = tuple(x / seen * 1E3 for x in dt)  # 每张图片的平均处理时间（单位：毫秒）
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    # 如果保存了结果（图片或文本标签），打印保存路径信息
    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt else ''
        )  # 如果保存了标签，打印标签数量和保存路径
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    # 如果设置了更新模型，执行优化更新
    if update:
        strip_optimizer(weights)  # 更新模型以移除优化器（例如，为模型文件瘦身）


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp/weights/best.pt', help='model path(s)')  # weights: 模型的权重地址 默认 weights/best.pt
    parser.add_argument('--source', type=str, default=ROOT / 'source_files/construction-safety.jpg', help='file/dir/URL/glob, 0 for webcam')  # source: 测试数据文件(图片或视频)的保存路径 默认data/images
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[608], help='inference size h,w')  # imgsz: 网络输入图片的大小 默认640
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')  # conf-thres: object置信度阈值 默认0.25
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')  # iou-thres: 做nms的iou阈值 默认0.45
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')  # max-det: 每张图片最大的目标个数 默认1000
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # device: 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
    parser.add_argument('--view-img', action='store_false', help='show results')   # view-img: 是否展示预测之后的图片或视频 默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')  # save-txt: 是否将预测的框坐标以txt文件格式保存 默认False 会在runs/detect/expn/labels下生成每张图片预测的txt文件
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')  # save-conf: 是否保存预测每个目标的置信度到预测tx文件中 默认False
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')  # save-crop: 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面  默认False
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')    # nosave: 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')   # classes: 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留, default=[0,6,1,8,9, 7]
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')  # agnostic-nms: 进行nms是否也除去不同类别之间的框 默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')  # 是否使用数据增强进行推理，默认为False
    parser.add_argument('--visualize', action='store_true', help='visualize features')  #  -visualize:是否可视化特征图，默认为 False
    parser.add_argument('--update', action='store_true', help='update all models')  # -update: 如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')   # project: 当前测试结果放在哪个主文件夹下 默认runs/detect
    parser.add_argument('--name', default='exp', help='save results to project/name')  # name: 当前测试结果放在run/detect下的文件名  默认是exp
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')  # -exist-ok: 是否覆盖已有结果，默认为 False
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')   # -line-thickness:画 bounding box 时的线条宽度，默认为 3
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')   # -hide-labels:是否隐藏标签信息，默认为 False
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')   # -hide-conf:是否隐藏置信度信息，默认为 False
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')   # half: 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')  # -dnn:是否使用 OpenCV DNN 进行 ONNX 推理，默认为 False
    opt = parser.parse_args()   # 解析命令行参数，并将结果存储在opt对象中
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 如果imgsz参数的长度为1，则将其值乘以2；否则保持不变
    print_args(FILE.stem, opt)   #  打印解析后的参数，FILE.stem是文件的名称（不含扩展名）
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))  # 检查项目所需的依赖项，排除 'tensorboard' 和 'thop' 这两个库
    run(**vars(opt))   # 使用命令行参数的字典形式调用 run 函数


if __name__ == "__main__":
    # 这是 Python 中的一个惯用语法，
    # 它确保以下的代码块只有在当前脚本作为主程序运行时才会被执行，而不是作为模块被导入时执行。
    opt = parse_opt()
    main(opt)
