# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
General utils
"""

import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

from utils.downloads import gsutil_getsize
from utils.metrics import box_iou, fitness

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
# 设置 PyTorch 输出选项：
# linewidth: 设置每行的最大宽度为 320 个字符
# precision: 设置浮点数的精度为 5 位
# profile: 设置为 'long'，以便获得更详细的输出格式

np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
# 设置 NumPy 输出选项：
# linewidth: 设置每行的最大宽度为 320 个字符
# formatter: 为浮点数设置输出格式，使用短格式（最多 5 位有效数字）

pd.options.display.max_columns = 10
# 设置 Pandas 显示选项：
# max_columns: 限制在 DataFrame 中最多显示 10 列

cv2.setNumThreads(0)
# 设置 OpenCV 的线程数为 0，防止其使用多线程
# 这避免了与 PyTorch DataLoader 发生不兼容问题

os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))
# 设置 NumExpr 的最大线程数：
# 根据可用的 CPU 核心数（最多为 8）来限制线程数

FILE = Path(__file__).resolve()
# 获取当前文件的绝对路径，并返回一个 Path 对象

ROOT = FILE.parents[1]
# 设置根目录为当前文件的父目录的父目录，即 YOLOv5 的根目录


class Profile(contextlib.ContextDecorator):
    """
    Profile class用于性能分析，既可以作为装饰器使用，也可以作为上下文管理器。

    Usage:
        @Profile()  # 用作装饰器
        or
        with Profile():  # 用作上下文管理器
    """

    def __enter__(self):
        """
        上下文管理器的入口方法。
        在进入上下文时记录当前时间。
        """
        self.start = time.time()  # 记录当前时间戳

    def __exit__(self, type, value, traceback):
        """
        上下文管理器的出口方法。
        在退出上下文时计算并打印耗时。

        Arguments:
            type: 异常类型（如果有异常发生）
            value: 异常值
            traceback: 异常追踪对象
        """
        # 计算耗时并打印结果，保留5位小数
        print(f'Profile results: {time.time() - self.start:.5f}s')


class Timeout(contextlib.ContextDecorator):
    """
    Timeout 类用于设置超时机制，既可以作为装饰器使用，也可以作为上下文管理器。

    Usage:
        @Timeout(seconds)  # 用作装饰器
        or
        with Timeout(seconds):  # 用作上下文管理器
    """

    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        """
        初始化 Timeout 类的实例。

        Arguments:
            seconds: 超时时间（秒）
            timeout_msg: 超时后抛出的消息（可选）
            suppress_timeout_errors: 是否抑制 TimeoutError（默认为 True）
        """
        self.seconds = int(seconds)  # 将超时秒数转换为整数
        self.timeout_message = timeout_msg  # 设置超时消息
        self.suppress = bool(suppress_timeout_errors)  # 设置是否抑制超时错误

    def _timeout_handler(self, signum, frame):
        """
        超时信号处理函数，抛出 TimeoutError。

        Arguments:
            signum: 信号编号
            frame: 当前的栈帧
        """
        raise TimeoutError(self.timeout_message)  # 抛出超时异常

    def __enter__(self):
        """
        上下文管理器的入口方法。
        在进入上下文时设置超时处理程序并启动计时器。
        """
        signal.signal(signal.SIGALRM, self._timeout_handler)  # 设置 SIGALRM 的处理函数
        signal.alarm(self.seconds)  # 启动 SIGALRM 计时器

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器的出口方法。
        在退出上下文时取消计时器，如果抑制了超时错误，则返回 True。

        Arguments:
            exc_type: 异常类型（如果有异常发生）
            exc_val: 异常值
            exc_tb: 异常追踪对象
        """
        signal.alarm(0)  # 取消计划的 SIGALRM
        if self.suppress and exc_type is TimeoutError:  # 如果抑制超时错误
            return True  # 阻止异常传播


def try_except(func):
    """
    try_except 装饰器用于捕获并处理函数执行中的异常。

    Usage:
        @try_except
        def my_function():
            # 函数体
    """

    def handler(*args, **kwargs):
        """
        装饰器内部的处理函数，负责执行被装饰的函数并捕获异常。

        Arguments:
            *args: 位置参数
            **kwargs: 关键字参数
        """
        try:
            func(*args, **kwargs)  # 执行被装饰的函数
        except Exception as e:
            print(e)  # 捕获并打印异常

    return handler  # 返回处理函数



def methods(instance):
    """
    获取给定类实例的所有可调用方法（不包括特殊方法）。

    Arguments:
        instance: 任何类的实例。

    Returns:
        list: 包含实例方法名称的列表。
    """

    # 使用 dir() 获取实例的所有属性和方法名称
    return [f for f in dir(instance)
            if callable(getattr(instance, f))  # 检查属性是否可调用
            and not f.startswith("__")]  # 过滤掉特殊方法（以双下划线开头）



def set_logging(rank=-1, verbose=True):
    # 设置日志记录的基本配置
    logging.basicConfig(
        format="%(message)s",  # 日志输出格式，只输出消息内容
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN  # 根据 verbose 和 rank 的值确定日志级别
        # 如果 verbose 为 True 且 rank 是 -1 或 0，则设置为 INFO 级别（显示详细信息）
        # 否则设置为 WARN 级别（只显示警告及以上级别的信息）
    )



def print_args(name, opt):
    # 打印解析的命令行参数
    # 使用 colorstr 函数给输出内容添加颜色，方便辨识
    # f'{name}: '表示打印传入的name变量（一般为文件名），之后用逗号分隔打印所有参数及其对应的值
    print(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))



def init_seeds(seed=0):
    """
    初始化随机数生成器（RNG）种子。

    该函数设置 Python、NumPy 和 PyTorch 的随机种子，以确保实验的可重复性。
    使用种子 0 时，cudnn 的设置更慢但更可重复；其他种子则更快但可重复性较差。

    Arguments:
        seed (int): 要设置的随机种子，默认为 0。

    参考文献:
        - PyTorch 随机性说明: https://pytorch.org/docs/stable/notes/randomness.html
    """

    import torch.backends.cudnn as cudnn
    import random
    import numpy as np

    # 设置随机种子
    random.seed(seed)  # Python 内置随机模块
    np.random.seed(seed)  # NumPy 随机数生成
    torch.manual_seed(seed)  # PyTorch 随机数生成

    # 设置 cuDNN 的随机性参数
    # seed 为 0 时，cudnn 设置为慢但可重复；否则快速但可重复性较差
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)



def get_latest_run(search_dir='.'):
    # 返回指定目录中最近的 'last.pt' 文件的路径，用于从最近的检查点继续训练
    # 在 search_dir 目录下递归查找所有符合 'last*.pt' 模式的文件
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    # 如果找到符合条件的文件，返回最近修改的文件路径；若没有找到则返回空字符串
    return max(last_list, key=os.path.getctime) if last_list else ''


def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    """
    返回用户配置目录的路径。如果存在环境变量，则优先使用环境变量。如果需要，创建该目录。

    Arguments:
        dir (str): 要创建的子目录名称，默认为 'Ultralytics'。
        env_var (str): 指定的环境变量名称，默认为 'YOLOV5_CONFIG_DIR'。

    Returns:
        Path: 用户配置目录的路径。
    """

    # 获取环境变量的值
    env = os.getenv(env_var)

    if env:
        # 如果环境变量存在，则使用该路径
        path = Path(env)
    else:
        # 定义不同操作系统下的配置目录
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3个操作系统目录
        # 获取当前用户的主目录，并拼接操作系统特定的配置目录
        path = Path.home() / cfg.get(platform.system(), '')  # 根据操作系统返回相应的配置目录

        # 如果该路径不可写，则使用 '/tmp' 目录
        path = (path if is_writeable(path) else Path('/tmp')) / dir  # GCP 和 AWS lambda 的修复，只有 /tmp 可写

    # 创建目录（如果需要）
    path.mkdir(exist_ok=True)

    return path


def is_writeable(dir, test=False):
    """
    检查目录是否具有写入权限。如果 test=True，将尝试以写权限打开一个文件进行测试。

    Arguments:
        dir (str or Path): 要检查的目录路径。
        test (bool): 是否进行写权限测试，默认为 False。

    Returns:
        bool: 如果目录可写，返回 True；否则返回 False。
    """

    if test:  # 如果需要测试写权限
        # 创建临时文件的路径
        file = Path(dir) / 'tmp.txt'
        try:
            # 尝试以写权限打开文件
            with open(file, 'w'):
                pass  # 创建文件并立即关闭
            file.unlink()  # 删除临时文件
            return True  # 如果成功，返回 True
        except IOError:
            return False  # 如果出现 IOError，返回 False
    else:  # 不进行测试，直接检查读取权限
        # 使用 os.access 检查目录的读取权限
        return os.access(dir, os.R_OK)  # 可能在 Windows 上有问题


def is_docker():
    """
    检查当前环境是否为 Docker 容器。

    Returns:
        bool: 如果当前环境是 Docker 容器，返回 True；否则返回 False。
    """

    # 检查 '/workspace' 目录是否存在
    return Path('/workspace').exists()  # 或者检查 '/.dockerenv' 目录是否存在


def is_colab():
    """
    检查当前环境是否为 Google Colab 实例。

    Returns:
        bool: 如果当前环境是 Google Colab，返回 True；否则返回 False。
    """

    try:
        import google.colab  # 尝试导入 google.colab 模块
        return True  # 成功导入，说明是在 Colab 环境中
    except ImportError:
        return False  # 导入失败，说明不是 Colab 环境


def is_pip():
    """
    检查当前文件是否在 pip 包中。

    Returns:
        bool: 如果当前文件位于 pip 安装的 site-packages 目录中，返回 True；否则返回 False。
    """

    # 检查当前文件路径的各个部分是否包含 'site-packages'
    return 'site-packages' in Path(__file__).resolve().parts


def is_ascii(s=''):
    """
    检查字符串是否由所有 ASCII 字符组成（不包含 UTF 字符）。

    Args:
        s (str): 要检查的字符串，默认为空字符串。

    Returns:
        bool: 如果字符串完全由 ASCII 字符组成，返回 True；否则返回 False。
    """

    # 将输入转换为字符串，以处理列表、元组、None 等类型
    s = str(s)

    # 编码为 ASCII 并解码，忽略非 ASCII 字符，然后比较长度
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    """
    检查字符串是否包含任何中文字符。

    Args:
        s (str): 要检查的字符串，默认为 '人工智能'。

    Returns:
        bool: 如果字符串中包含中文字符，则返回 True；否则返回 False。
    """
    return re.search('[\u4e00-\u9fff]', s) is not None



def emojis(str=''):
    """
    返回平台相关的、安全的表情符号字符串版本。

    Args:
        str (str): 要处理的字符串，默认为空字符串。

    Returns:
        str: 处理后的字符串，适合在不同平台上使用。
    """
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str



def file_size(path):
    """
    返回文件或目录的大小（以 MB 为单位）。

    Args:
        path (str): 文件或目录的路径。

    Returns:
        float: 文件或目录的大小（MB）。如果路径不存在，则返回 0.0。
    """
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1E6  # 返回文件大小（MB）
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1E6  # 返回目录大小（MB）
    else:
        return 0.0  # 如果路径不存在，则返回 0.0



def check_online():
    """
    检查互联网连接状态。

    Returns:
        bool: 如果能够成功连接到互联网，返回 True；否则返回 False。
    """
    import socket
    try:
        # 尝试连接到 1.1.1.1（Cloudflare DNS）上的 443 端口，以检查主机可访问性
        socket.create_connection(("1.1.1.1", 443), 5)
        return True  # 如果连接成功，返回 True
    except OSError:
        return False  # 如果出现 OSError，返回 False



@try_except
def check_git_status():
    # 检查Git仓库状态，如果代码不是最新版本，建议用户执行`git pull`更新
    msg = ', for updates see https://github.com/ultralytics/yolov5'  # 提示消息链接
    print(colorstr('github: '), end='')  # 打印带颜色的“github”标签，便于在控制台识别输出来源

    # 确保当前目录是Git仓库，否则跳过检查
    assert Path('.git').exists(), 'skipping check (not a git repository)' + msg
    # 如果在Docker容器中运行，跳过检查
    assert not is_docker(), 'skipping check (Docker image)' + msg
    # 检查是否在线，离线则跳过检查
    assert check_online(), 'skipping check (offline)' + msg

    # 获取当前仓库的远程URL和分支状态
    cmd = 'git fetch && git config --get remote.origin.url'  # 同步远程仓库并获取URL
    url = check_output(cmd, shell=True, timeout=5).decode().strip().rstrip('.git')  # 获取仓库URL并去掉“.git”后缀
    branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # 获取当前分支名称
    n = int(check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # 获取本地分支相对远程主分支的落后提交数

    # 如果落后提交数大于0，提示更新命令，否则确认仓库已最新
    if n > 0:
        s = f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."
    else:
        s = f'up to date with {url} ✅'
    print(emojis(s))  # 使用emoji风格打印消息，支持在不同控制台中安全显示



def check_python(minimum='3.6.2'):
    # 检查当前的Python版本是否符合要求的最低版本
    check_version(platform.python_version(), minimum, name='Python ')


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False):
    # 检查当前版本与要求的版本
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))  # 解析当前和最低版本
    result = (current == minimum) if pinned else (current >= minimum)  # 比较当前版本与最低版本
    assert result, f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'
    # 如果当前版本不符合要求，抛出异常并给出提示



@try_except
def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True):
    # 检查已安装的依赖是否符合要求（支持传入 *.txt 文件或包列表）
    prefix = colorstr('red', 'bold', 'requirements:')  # 设置前缀，便于错误输出时识别
    check_python()  # 检查 Python 版本是否符合要求

    # 检查 requirements 参数是否为路径（即 requirements.txt 文件）
    if isinstance(requirements, (str, Path)):
        file = Path(requirements)  # 将路径字符串转为 Path 对象
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."  # 检查文件是否存在
        # 从文件中读取并解析包的名称和版本要求，并排除 exclude 列表中的包
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:
        # 若 requirements 是列表或元组，则直接排除 exclude 中的包
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # 记录自动更新的包数量
    for r in requirements:
        try:
            pkg.require(r)  # 尝试导入和检查包是否符合版本要求
        except Exception as e:  # 如果未找到包或版本冲突则捕获异常
            s = f"{prefix} {r} not found and is required by YOLOv5"  # 显示缺少的包信息
            if install:  # 若允许自动安装
                print(f"{s}, attempting auto-update...")
                try:
                    # 检查是否在线，若在线则自动安装
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    print(check_output(f"pip install '{r}'", shell=True).decode())  # 安装缺失的包
                    n += 1
                except Exception as e:
                    print(f'{prefix} {e}')  # 打印安装失败的错误信息
            else:
                print(f'{s}. Please install and rerun your command.')  # 提示用户手动安装

    # 如果更新了包，提示用户重启或重新运行命令以应用更新
    if n:
        source = file.resolve() if 'file' in locals() else requirements  # 更新源信息（文件或列表）
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        print(emojis(s))  # 使用 emoji 打印带颜色的提示信息


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    # 验证图像尺寸在每个维度上是否为步幅 s 的倍数
    if isinstance(imgsz, int):  # 如果 img_size 是整数，例如 img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)  # 将图像尺寸调整为 s 的倍数，并不小于 floor
    else:  # 如果 img_size 是列表，例如 img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]  # 对每个维度进行相同的处理
    # 如果调整后的尺寸与原始尺寸不一致
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
        # 输出警告，表明原始尺寸必须是最大步幅 s 的倍数，并显示更新后的尺寸
    return new_size  # 返回经过验证和调整的图像尺寸



def check_imshow():
    # Check if environment supports image displays
    try:
        # 检查当前环境是否为 Docker，如果是，则 cv2.imshow() 不支持
        assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        # 检查当前环境是否为 Google Colab，如果是，则 cv2.imshow() 不支持
        assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        # 测试显示一幅空白图像，确保 cv2.imshow() 可用
        cv2.imshow('test', np.zeros((1, 1, 3)))  # 创建一个 1x1 的黑色图像并显示
        cv2.waitKey(1)  # 等待 1 毫秒，以便图像可以显示
        cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
        cv2.waitKey(1)  # 再次等待 1 毫秒，以确保窗口关闭
        return True  # 如果没有异常，返回 True，表示支持图像显示
    except Exception as e:
        # 捕获异常并输出警告，指明不支持图像显示
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False  # 返回 False，表示不支持图像显示


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffixes
    # 检查文件（或文件列表）是否具有可接受的后缀
    if file and suffix:  # 确保文件和后缀不为空
        if isinstance(suffix, str):  # 如果后缀是字符串类型
            suffix = [suffix]  # 将其转换为列表形式，以便后续处理
        # 遍历输入的文件（支持单个文件或文件列表）
        for f in file if isinstance(file, (list, tuple)) else [file]:
            # 检查文件的后缀是否在接受的后缀列表中
            assert Path(f).suffix.lower() in suffix, f"{msg}{f} acceptable suffix is {suffix}"
            # 如果后缀不在接受的后缀列表中，抛出 AssertionError，显示错误信息

def check_yaml(file, suffix=('.yaml', '.yml')):
    # 检查指定的文件是否为 YAML 文件，必要时进行下载，并返回文件路径
    # 传递 file 和后缀 suffix 到 check_file 函数，验证文件后缀是否为 '.yaml' 或 '.yml'
    return check_file(file, suffix)



def check_file(file, suffix=''):
    # 检查文件路径的有效性，若文件不存在则下载或搜索，最终返回文件路径
    check_suffix(file, suffix)  # 可选：检查文件后缀是否符合指定格式
    file = str(file)  # 确保 file 是字符串格式
    if Path(file).is_file() or file == '':  # 检查文件是否已存在或路径为空
        return file
    elif file.startswith(('http:/', 'https:/')):  # 若路径为URL，下载文件
        url = str(Path(file)).replace(':/', '://')  # 修复Path对象格式化URL时的':'问题
        file = Path(urllib.parse.unquote(file).split('?')[0]).name  # 获取文件名并移除URL中的参数
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, file)  # 使用PyTorch工具下载文件
        # 确保文件成功下载且非空
        assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'
        return file
    else:  # 若路径不是URL且文件不存在，开始搜索文件
        files = []
        for d in 'data', 'models', 'utils':  # 在特定的目录中搜索文件
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True))  # 递归搜索匹配的文件
        assert len(files), f'File not found: {file}'  # 如果找不到文件则报错
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # 若找到多个文件则报错
        return files[0]  # 返回唯一匹配的文件路径



def check_dataset(data, autodownload=True):
    """
    检查数据集是否存在。如果数据集在本地未找到，则下载并/或解压数据集。
    使用示例: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip

    参数:
        data (str or Path): 数据集的路径或 URL，支持 .zip 文件。
        autodownload (bool): 如果为 True，尝试自动下载数据集。

    返回:
        dict: 包含数据集路径及其他信息的字典。
    """

    # 下载（可选）
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):
        # 如果 data 是一个 zip 文件路径，则下载并解压
        download(data, dir='../datasets', unzip=True, delete=False, curl=False, threads=1)
        # 获取解压后的 yaml 文件路径
        data = next((Path('../datasets') / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False  # 更新提取目录并禁用自动下载

    # 读取 yaml 文件（可选）
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # 将 yaml 文件加载为字典

    # 解析 yaml 内容
    path = extract_dir or Path(data.get('path') or '')  # 可选的 'path' 默认设置为当前目录
    for k in 'train', 'val', 'test':
        if data.get(k):  # 如果存在相应路径
            # 预pend path
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    assert 'nc' in data, "数据集缺少 'nc' 键。"  # 检查 'nc' 键是否存在
    if 'names' not in data:
        # 如果 'names' 键缺失，则为每个类分配默认名称
        data['names'] = [f'class{i}' for i in range(data['nc'])]
    train, val, test, s = [data.get(x) for x in ('train', 'val', 'test', 'download')]

    # 检查验证集路径是否存在
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # 获取验证集路径
        if not all(x.exists() for x in val):  # 检查所有路径是否存在
            print('\nWARNING: 数据集未找到，缺失路径: %s' % [str(x) for x in val if not x.exists()])
            if s and autodownload:  # 如果存在下载脚本并允许自动下载
                root = path.parent if 'path' in data else '..'  # 解压目录 i.e. '../'
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # 文件名
                    print(f'正在下载 {s} 到 {f}...')
                    torch.hub.download_url_to_file(s, f)  # 下载文件
                    Path(root).mkdir(parents=True, exist_ok=True)  # 创建根目录
                    ZipFile(f).extractall(path=root)  # 解压文件
                    Path(f).unlink()  # 删除 zip 文件
                    r = None  # 下载成功标志
                elif s.startswith('bash '):  # bash 脚本
                    print(f'正在运行 {s} ...')
                    r = os.system(s)  # 执行 bash 脚本
                else:  # python 脚本
                    r = exec(s, {'yaml': data})  # 执行 python 脚本
                print(f"数据集自动下载 {f'success, saved to {root}' if r in (0, None) else 'failure'}\n")
            else:
                raise Exception('数据集未找到。')  # 抛出异常

    return data  # 返回包含数据集信息的字典


def url2file(url):
    # 将URL转换为文件名，例如将 https://url.com/file.txt?auth 转换为 file.txt
    url = str(Path(url)).replace(':/', '://')  # 将路径中的 :/ 替换为 ://，以避免Pathlib的处理问题
    file = Path(urllib.parse.unquote(url)).name.split('?')[0]  # 将URL解码，获取文件名，并去掉查询参数部分
    return file  # 返回提取的文件名



def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
    # 多线程文件下载和解压函数，用于 data.yaml 中的自动下载
    def download_one(url, dir):
        # 下载单个文件
        f = dir / Path(url).name  # 获取文件名
        if Path(url).is_file():  # 检查文件是否存在于当前路径
            Path(url).rename(f)  # 将文件移动到指定目录
        elif not f.exists():
            print(f'Downloading {url} to {f}...')  # 开始下载
            if curl:
                # 使用 curl 进行下载，支持重试和断点续传
                os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")
            else:
                # 使用 torch 下载文件
                torch.hub.download_url_to_file(url, f, progress=True)
        # 如果需要解压并且文件是压缩格式
        if unzip and f.suffix in ('.zip', '.gz'):
            print(f'Unzipping {f}...')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)  # 解压 zip 文件
            elif f.suffix == '.gz':
                os.system(f'tar xfz {f} --directory {f.parent}')  # 解压 gz 文件
            if delete:
                f.unlink()  # 删除压缩文件

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # 创建目录
    if threads > 1:
        pool = ThreadPool(threads)
        # 使用多线程下载
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        # 如果没有使用多线程，逐个下载
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)



def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    # 返回 x 被 divisor 整除的最小值
    return math.ceil(x / divisor) * divisor  # 先计算 x 除以 divisor 的结果，再向上取整，最后乘以 divisor 得到可整除的值



def clean_str(s):
    # 清理字符串，通过将特殊字符替换为下划线 _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)



def one_cycle(y1=0.0, y2=1.0, steps=100):
    """
    创建一个从 y1 到 y2 的正弦波形学习率调度函数。

    参数:
    y1 (float): 学习率调度的起始值，默认为 0.0。
    y2 (float): 学习率调度的结束值，默认为 1.0。
    steps (int): 学习率变化的步骤数量，默认为 100。

    返回:
    function: 一个接受单个参数 x 的 lambda 函数，该参数表示当前步骤，返回在该步骤的学习率值。

    参考文献:
    https://arxiv.org/pdf/1812.01187.pdf
    """
    # 返回一个 lambda 函数，该函数计算从 y1 到 y2 的正弦波形调度
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # 给字符串着色，参考：https://en.wikipedia.org/wiki/ANSI_escape_code，例如：colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # 颜色参数，字符串
    colors = {
        'black': '\033[30m',  # 基本颜色
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # 明亮颜色
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # 其他
        'bold': '\033[1m',
        'underline': '\033[4m'
    }
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']



def labels_to_class_weights(labels, nc=80):
    # 从训练标签获取类别权重（逆频率）
    if labels[0] is None:  # 如果没有加载标签
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) 对于 COCO 数据集
    classes = labels[:, 0].astype(np.int32)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # 每个类别的出现次数

    # 在前面添加网格点计数（用于 uCE 训练）
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # 每张图片的网格点
    # weights = np.hstack([gpi * len(labels) - weights.sum() * 9, weights * 9]) ** 0.5  # 在开始时添加网格点

    weights[weights == 0] = 1  # 将空的类别权重替换为 1
    weights = 1 / weights  # 每个类别的目标数量
    weights /= weights.sum()  # 归一化
    return torch.from_numpy(weights)



def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # 根据类权重和图像内容生成图像权重
    # labels: 每个图像的标签，形状为(n, 5)，其中 n 是目标数量，5 包含 [class, x, y, w, h]
    # nc: 类的数量，默认为80
    # class_weights: 每个类的权重，默认为全1的数组

    # 计算每个图像中每个类的目标数量
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int32), minlength=nc) for x in labels])

    # 计算每个图像的权重，权重为类权重与目标数量的乘积的总和
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)

    # 返回每个图像的权重
    return image_weights


def coco80_to_coco91_class():  # 将COCO数据集的80类索引转换为91类索引
    # 参考链接：https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')  # 加载COCO类名
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')  # 加载论文中类名
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # 从darknet到COCO的映射
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # 从COCO到darknet的映射

    # 直接定义从80类到91类的索引映射
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x  # 返回转换后的索引列表


def xyxy2xywh(x):
    # 将 nx4 的边界框从 [x1, y1, x2, y2] 转换为 [x, y, w, h]，其中 xy1=左上角，xy2=右下角
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)  # 根据输入类型创建副本
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # 计算 x 中心
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # 计算 y 中心
    y[:, 2] = x[:, 2] - x[:, 0]  # 计算宽度
    y[:, 3] = x[:, 3] - x[:, 1]  # 计算高度
    return y



def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    # 将nx4格式的框从[x, y, w, h]转换为[x1, y1, x2, y2]格式
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)  # 创建输入x的副本
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # 计算左上角的x坐标
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # 计算左上角的y坐标
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # 计算右下角的x坐标
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # 计算右下角的y坐标
    return y  # 返回转换后的坐标



def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # 将 nx4 的边界框从 [x, y, w, h]（归一化）转换为 [x1, y1, x2, y2]，其中 xy1=左上角，xy2=右下角
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)  # 根据输入类型创建副本
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # 计算左上角 x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # 计算左上角 y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # 计算右下角 x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # 计算右下角 y
    return y



def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # 将 nx4 的边界框从 [x1, y1, x2, y2] 转换为 [x, y, w, h]（归一化），其中 xy1=左上角，xy2=右下角
    if clip:
        clip_coords(x, (h - eps, w - eps))  # 警告：就地裁剪
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)  # 根据输入类型创建副本
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # 计算中心 x（归一化）
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # 计算中心 y（归一化）
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # 计算宽度（归一化）
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # 计算高度（归一化）
    return y



def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # 将归一化的坐标转换为像素坐标，形状为 (n, 2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)  # 根据输入类型创建副本
    y[:, 0] = w * x[:, 0] + padw  # 计算左上角 x（像素）
    y[:, 1] = h * x[:, 1] + padh  # 计算左上角 y（像素）
    return y



def segment2box(segment, width=640, height=640):
    # 将一个分段标签转换为一个框标签，应用图像内约束，即将 (xy1, xy2, ...) 转换为 (xyxy)
    x, y = segment.T  # 提取分段的 x 和 y 坐标
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)  # 检查坐标是否在图像内
    x, y = x[inside], y[inside]  # 仅保留在图像内的坐标
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # 返回框坐标 (xyxy) 或零数组



def segments2boxes(segments):
    # 将分段标签转换为框标签，即将 (cls, xy1, xy2, ...) 转换为 (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # 提取分段的 x 和 y 坐标
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # 计算框的最小和最大 x、y 坐标，形成 (xyxy) 格式
    return xyxy2xywh(np.array(boxes))  # 将框转换为 (cls, xywh) 格式



def resample_segments(segments, n=1000):
    # 对 (n,2) 分段进行上采样
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)  # 创建均匀分布的 n 个点
        xp = np.arange(len(s))  # 原始索引
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # 通过插值生成新的分段坐标
    return segments  # 返回上采样后的分段



def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # 如果没有提供比例填充，则从img0_shape计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # 计算缩放比
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # 计算填充
    else:
        gain = ratio_pad[0][0]  # 从给定的比例填充中获取缩放比
        pad = ratio_pad[1]  # 获取填充值

    coords[:, [0, 2]] -= pad[0]  # 对x坐标进行填充调整
    coords[:, [1, 3]] -= pad[1]  # 对y坐标进行填充调整
    coords[:, :4] /= gain  # 根据缩放比调整坐标
    clip_coords(coords, img0_shape)  # 将坐标限制在原始图像边界内
    return coords  # 返回调整后的坐标



def clip_coords(boxes, shape):
    # 将边界框 (xyxy) 限制在图像形状内 (高度, 宽度)
    if isinstance(boxes, torch.Tensor):  # 如果是 PyTorch 张量（单独处理速度更快）
        boxes[:, 0].clamp_(0, shape[1])  # 限制 x1
        boxes[:, 1].clamp_(0, shape[0])  # 限制 y1
        boxes[:, 2].clamp_(0, shape[1])  # 限制 x2
        boxes[:, 3].clamp_(0, shape[0])  # 限制 y2
    else:  # 如果是 numpy 数组（批量处理速度更快）
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # 限制 x1 和 x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # 限制 y1 和 y2



def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """对推理结果执行非极大值抑制（NMS）

    参数：
        prediction: 模型的预测结果
        conf_thres: 置信度阈值，低于此阈值的检测将被过滤
        iou_thres: IOU（Intersection over Union）阈值，用于判断重叠框的合并
        classes: 要考虑的类别列表
        agnostic: 如果为True，则类别之间不进行区分
        multi_label: 如果为True，则每个框可以具有多个标签
        labels: 真实标签，用于自标注
        max_det: 每张图像最大检测框数

    返回：
         每张图像的检测结果列表，每个检测结果为(n, 6)的张量 [xyxy, conf, cls]
    """

    # nc: 类别数
    nc = prediction.shape[2] - 5  # 预测结果中的类别数
    xc = prediction[..., 4] > conf_thres  # 符合置信度阈值的候选框

    # 检查阈值有效性
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # 设置参数
    min_wh, max_wh = 2, 4096  # (像素) 最小和最大框宽高
    max_nms = 30000  # 传入 torchvision.ops.nms() 的最大框数
    time_limit = 10.0  # 超过此时间后退出
    redundant = True  # 是否需要冗余检测
    multi_label &= nc > 1  # 如果类别数大于1，启用多标签（增加处理时间）
    merge = False  # 是否使用合并 NMS

    t = time.time()  # 记录开始时间
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]  # 初始化输出
    for xi, x in enumerate(prediction):  # 遍历每张图像的预测结果
        # 应用约束条件
        x = x[xc[xi]]  # 仅保留符合置信度阈值的框

        # 如果存在真实标签，则将其合并到预测结果中
        if labels and len(labels[xi]):
            l = labels[xi]  # 真实标签
            v = torch.zeros((len(l), nc + 5), device=x.device)  # 初始化与真实标签相同形状的张量
            v[:, :4] = l[:, 1:5]  # 提取真实框的坐标
            v[:, 4] = 1.0  # 置信度设为1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # 设置类别
            x = torch.cat((x, v), 0)  # 合并预测框和真实框

        # 如果没有符合条件的框，则处理下一个图像
        if not x.shape[0]:
            continue

        # 计算置信度
        x[:, 5:] *= x[:, 4:5]  # 置信度 = 目标置信度 * 类别置信度

        # 将框从 (中心x, 中心y, 宽, 高) 转换为 (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # 创建检测矩阵 nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T  # 确定哪些框符合多标签条件
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)  # 合并框信息
        else:  # 仅保留最佳类别
            conf, j = x[:, 5:].max(1, keepdim=True)  # 找到置信度最高的类别
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]  # 合并框和置信度

        # 根据类别过滤框
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]  # 仅保留指定类别的框

        # 检查框的数量
        n = x.shape[0]  # 当前框的数量
        if not n:  # 如果没有框
            continue
        elif n > max_nms:  # 如果框的数量超过最大限制
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 按置信度排序并保留前 max_nms 个框

        # 批量处理 NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 处理类信息
        boxes, scores = x[:, :4] + c, x[:, 4]  # 添加类别偏移，获取框和置信度
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # 执行 NMS
        if i.shape[0] > max_det:  # 如果检测到的框数量超过限制
            i = i[:max_det]  # 仅保留最大检测数

        # 可选的合并 NMS（使用加权均值合并框）
        if merge and (1 < n < 3E3):  # 如果需要合并且检测框数量合理
            iou = box_iou(boxes[i], boxes) > iou_thres  # 计算 IOU 矩阵
            weights = iou * scores[None]  # 计算框的权重
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # 更新框为加权均值
            if redundant:
                i = i[iou.sum(1) > 1]  # 如果需要冗余，保留符合条件的框

        output[xi] = x[i]  # 将结果存储到输出中
        if (time.time() - t) > time_limit:  # 如果超出时间限制
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # 超时退出

    return output  # 返回每张图像的检测结果


def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # 从文件 'f' 中去除优化器信息，以完成训练，并可选择性地保存为 's'
    x = torch.load(f, map_location=torch.device('cpu'), weights_only=False)  # 加载模型文件
    if x.get('ema'):  # 如果存在 EMA (Exponential Moving Average) 模型
        x['model'] = x['ema']  # 用 EMA 模型替换原模型
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # 移除指定的键
        x[k] = None
    x['epoch'] = -1  # 设置为-1，表示训练已结束
    x['model'].half()  # 将模型转换为 FP16 精度
    for p in x['model'].parameters():  # 设置模型参数为不需要梯度
        p.requires_grad = False
    torch.save(x, s or f)  # 保存模型，优先保存为 's'，否则保存为 'f'
    mb = os.path.getsize(s or f) / 1E6  # 获取文件大小（MB）
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")  # 打印信息



def print_mutation(results, hyp, save_dir, bucket):
    # 定义输出文件路径
    evolve_csv = save_dir / 'evolve.csv'  # 演化结果文件
    results_csv = save_dir / 'results.csv'  # 结果文件（未使用）
    evolve_yaml = save_dir / 'hyp_evolve.yaml'  # 超参数演化 YAML 文件

    # 定义要记录的键，包括评估指标和超参数
    keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            'val/box_loss', 'val/obj_loss', 'val/cls_loss') + tuple(hyp.keys())  # [结果 + 超参数]
    keys = tuple(x.strip() for x in keys)  # 去掉多余的空格
    vals = results + tuple(hyp.values())  # 组合结果和超参数值
    n = len(keys)  # 键的数量

    # 可选：下载 evolve.csv 文件
    if bucket:
        url = f'gs://{bucket}/evolve.csv'  # 云存储中的文件 URL
        # 如果云端文件大于本地文件，则下载
        if gsutil_getsize(url) > (os.path.getsize(evolve_csv) if os.path.exists(evolve_csv) else 0):
            os.system(f'gsutil cp {url} {save_dir}')  # 下载 evolve.csv

    # 记录到 evolve.csv 文件
    # 如果文件不存在，则添加标题
    s = '' if evolve_csv.exists() else (('%20s,' * n % keys).rstrip(',') + '\n')  # 添加表头
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')  # 写入数据

    # 打印到屏幕
    print(colorstr('evolve: ') + ', '.join(f'{x.strip():>20s}' for x in keys))  # 打印键
    print(colorstr('evolve: ') + ', '.join(f'{x:20.5g}' for x in vals), end='\n\n\n')  # 打印值

    # 保存超参数演化结果为 YAML 文件
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)  # 读取 evolve.csv
        data = data.rename(columns=lambda x: x.strip())  # 去掉列名的空格
        i = np.argmax(fitness(data.values[:, :7]))  # 找到最佳结果的索引
        # 写入文件头和最佳结果信息
        f.write('# YOLOv5 Hyperparameter Evolution Results\n' +
                f'# Best generation: {i}\n' +
                f'# Last generation: {len(data)}\n' +
                '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:7]) + '\n' +
                '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :7]) + '\n\n')
        yaml.safe_dump(hyp, f, sort_keys=False)  # 保存超参数到 YAML 文件

    # 可选：上传 evolve.csv 和 YAML 文件到云存储
    if bucket:
        os.system(f'gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}')  # 上传



def apply_classifier(x, model, img, im0):
    # 对 YOLO 输出应用第二阶段分类器
    im0 = [im0] if isinstance(im0, np.ndarray) else im0  # 如果 im0 是 ndarray 类型，则将其放入列表中
    for i, d in enumerate(x):  # 遍历每个图像的检测结果
        if d is not None and len(d):
            d = d.clone()  # 克隆检测结果，以免修改原始数据

            # 重塑和填充切割区域
            b = xyxy2xywh(d[:, :4])  # 将检测框坐标从 (x1, y1, x2, y2) 转换为 (x_center, y_center, width, height)
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # 将矩形框调整为正方形
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # 在正方形的基础上增加填充，30为填充的像素
            d[:, :4] = xywh2xyxy(b).long()  # 将正方形框转换回 (x1, y1, x2, y2) 格式并转为整型

            # 将检测框从 img_size 重新缩放到 im0 大小
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)  # 根据原始图像的尺寸缩放检测框

            # 获取类别信息
            pred_cls1 = d[:, 5].long()  # 提取预测的类别
            ims = []  # 用于存储处理后的图像
            for j, a in enumerate(d):  # 遍历每个检测结果
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]  # 从原图像中裁剪出检测区域
                im = cv2.resize(cutout, (224, 224))  # 将裁剪后的图像调整为 224x224 尺寸 (BGR格式)
                # cv2.imwrite('example%i.jpg' % j, cutout)  # （可选）保存裁剪图像

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR 转 RGB，并调整维度为 3x224x224
                im = np.ascontiguousarray(im, dtype=np.float32)  # 将数据类型转换为 float32
                im /= 255.0  # 将像素值从 0-255 范围缩放到 0.0-1.0
                ims.append(im)  # 将处理后的图像添加到列表中

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # 使用分类器进行预测，得到类别索引
            x[i] = x[i][pred_cls1 == pred_cls2]  # 保留类别匹配的检测结果

    return x  # 返回经过分类器处理后的检测结果


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    # 将图像裁剪保存为 {file}，裁剪大小为原始大小的 {gain} 倍，并加 {pad} 像素的边距。可选择保存或返回裁剪结果
    xyxy = torch.tensor(xyxy).view(-1, 4)  # 将输入的坐标转换为张量并调整形状
    b = xyxy2xywh(xyxy)  # 将坐标从 [x1, y1, x2, y2] 转换为 [x, y, w, h]

    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # 如果需要，将矩形裁剪框调整为正方形

    b[:, 2:] = b[:, 2:] * gain + pad  # 调整裁剪框的宽高，乘以增益并加上边距
    xyxy = xywh2xyxy(b).long()  # 将调整后的宽高框转换回 [x1, y1, x2, y2] 格式
    clip_coords(xyxy, im.shape)  # 确保裁剪框不超出图像边界
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    # 根据坐标裁剪图像，并根据 BGR 标志选择通道顺序

    if save:
        cv2.imwrite(str(increment_path(file, mkdir=True).with_suffix('.jpg')), crop)  # 保存裁剪图像

    return crop  # 返回裁剪后的图像


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    # 递增文件或目录路径，例如：runs/exp --> runs/exp2, runs/exp3 等
    path = Path(path)  # 将路径转换为 Path 对象，确保跨平台兼容性（不依赖操作系统）
    if path.exists() and not exist_ok:  # 如果路径已经存在并且 exist_ok 为 False，则开始递增路径名
        suffix = path.suffix  # 获取文件的后缀（如果有）
        path = path.with_suffix('')  # 去掉文件后缀，便于递增操作
        dirs = glob.glob(f"{path}{sep}*")  # 查找与当前路径相似的所有路径
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]  # 使用正则表达式匹配路径中的数字（递增部分）
        i = [int(m.groups()[0]) for m in matches if m]  # 提取路径中匹配到的数字（递增的数字部分）
        n = max(i) + 1 if i else 2  # 确定递增的数字，如果之前没有，则从 2 开始
        path = Path(f"{path}{sep}{n}{suffix}")  # 生成递增后的新路径，保持原始后缀
    dir = path if path.suffix == '' else path.parent  # 如果是文件路径，使用其父目录；如果是目录，直接使用该目录
    if not dir.exists() and mkdir:  # 如果目录不存在并且 mkdir 为 True，则创建目录
        dir.mkdir(parents=True, exist_ok=True)  # 创建目录，确保父目录存在（parents=True）
    return path  # 返回递增后的路径

