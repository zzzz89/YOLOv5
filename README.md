# YOLOv5 安全帽 / 反光衣检测

基于 YOLOv5 的目标检测项目（安全帽、未戴帽、反光衣）。

## 环境

- Python 3.11
- PyTorch（CUDA 版，可选 GPU）
- 依赖见 `yolov5/requirements.txt`

推荐解释器：`D:\p119\python.exe`（已验证 GPU 可用）

```bash
cd yolov5
pip install -r requirements.txt
# CUDA Torch 示例：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 目录说明

| 路径 | 说明 |
|------|------|
| `yolov5/train.py` | 训练 |
| `yolov5/detect.py` | 检测推理 |
| `yolov5/val.py` | 验证 |
| `yolov5/Dataset_partitioning.py` | VOC 数据集划分 |
| `yolov5/data/VOC-hat.yaml` | 数据配置 |
| `yolov5/weights/` | 预训练权重（本地，未上传） |
| `yolov5/VOCdevkit/` | 数据集（本地，未上传） |
| `yolov5/runs/` | 训练/检测输出（本地，未上传） |

> 大数据集、权重、`runs` 已加入 `.gitignore`，不会上传到 GitHub。

## 快速检测

将图片放到 `yolov5/source_files/`，并准备好权重（如 `runs/train/exp/weights/best.pt`）：

```bash
cd yolov5
python detect.py --weights runs/train/exp/weights/best.pt --source source_files --device 0
```

## 训练

```bash
cd yolov5
python train.py --data data/VOC-hat.yaml --weights weights/yolov5s.pt --device 0
```

## 兼容性修改说明

本仓库相对原版做了适配新环境的修改，主要包括：

- PyTorch 2.6：`torch.load(..., weights_only=False)`
- NumPy 2：`np.int` → `np.int32`，`np.trapz` → `np.trapezoid`
- Pillow 10+：`font.getsize` → `font.getbbox`
- Windows 缓存文件重命名修复
- 检测单张图片时窗口等待按键关闭（不再一闪而过）
