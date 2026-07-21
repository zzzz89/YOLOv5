# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Download utils
"""

import os
import platform
import subprocess
import time
import urllib
from pathlib import Path
from zipfile import ZipFile

import requests
import torch


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file))
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            print(f"ERROR: {assert_msg}\n{error_msg}")
        print('')


def attempt_download(file, repo='ultralytics/yolov5'):  # 从 utils.downloads 导入 *; attempt_download()
    # 如果文件不存在，尝试下载该文件
    file = Path(str(file).strip().replace("'", ''))  # 清理文件路径

    if not file.exists():  # 检查文件是否存在
        # URL 指定的情况
        name = Path(urllib.parse.unquote(str(file))).name  # 解码文件名，例如 '%2F' 解码为 '/'
        if str(file).startswith(('http:/', 'https:/')):  # 如果是 HTTP/HTTPS 下载
            url = str(file).replace(':/', '://')  # 处理路径格式，确保 URL 格式正确
            name = name.split('?')[0]  # 解析身份验证，例如 'https://url.com/file.txt?auth...'
            safe_download(file=name, url=url, min_bytes=1E5)  # 安全下载文件
            return name  # 返回下载的文件名

        # 从 GitHub 获取资源
        file.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录（如果需要的话）
        try:
            # 获取最新版本的 GitHub 资源
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # GitHub API
            assets = [x['name'] for x in response['assets']]  # 获取发布资产，如 ['yolov5s.pt', 'yolov5m.pt', ...]
            tag = response['tag_name']  # 获取标签名，如 'v1.0'
        except:  # 回退计划
            assets = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
                      'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']  # 默认资产列表
            try:
                # 获取最新 Git 标签
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except:
                tag = 'v5.0'  # 如果无法获取，默认使用当前发布版本

        # 如果文件名在资产列表中
        if name in assets:
            safe_download(file,
                          url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                          # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # 备份 URL（可选）
                          min_bytes=1E5,
                          error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')

    return str(file)  # 返回文件的字符串路径


def gdrive_download(id='16TiPfZj7htmTyhntwcZyEEAejOUxuT6m', file='tmp.zip'):
    # 从 Google Drive 下载文件。用法示例: from yolov5.utils.downloads import *; gdrive_download()
    t = time.time()  # 记录开始时间
    file = Path(file)  # 将文件名转换为 Path 对象
    cookie = Path('cookie')  # Google Drive 的 cookie 文件
    print(f'Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ', end='')

    file.unlink(missing_ok=True)  # 删除已存在的文件
    cookie.unlink(missing_ok=True)  # 删除已存在的 cookie 文件

    # 尝试下载文件
    out = "NUL" if platform.system() == "Windows" else "/dev/null"  # 根据系统设置输出
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')  # 初始请求以处理大文件下载
    if os.path.exists('cookie'):  # 如果存在 cookie，说明是大文件
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'  # 带确认令牌下载
    else:  # 小文件直接下载
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)  # 执行下载命令，捕获返回值
    cookie.unlink(missing_ok=True)  # 删除 cookie 文件

    # 错误检查
    if r != 0:
        file.unlink(missing_ok=True)  # 删除部分下载的文件
        print('Download error ')  # 提示下载错误
        return r  # 返回错误代码

    # 如果是压缩文件，则解压
    if file.suffix == '.zip':
        print('unzipping... ', end='')
        ZipFile(file).extractall(path=file.parent)  # 解压缩
        file.unlink()  # 删除 zip 文件

    print(f'Done ({time.time() - t:.1f}s)')  # 打印下载完成信息和耗时
    return r  # 返回下载结果


def get_token(cookie="./cookie"):
    """ 从 cookie 文件中提取 Google Drive 下载确认 token
    Arguments:
        cookie:  Cookie 文件的路径，默认为 './cookie'
    Returns:
        str: 下载确认 token，如果未找到则返回空字符串
    """
    with open(cookie) as f:  # 打开指定的 cookie 文件
        for line in f:  # 遍历文件的每一行
            if "download" in line:  # 检查行中是否包含 "download"
                return line.split()[-1]  # 返回行的最后一个单词（token）
    return ""  # 如果未找到 token，返回空字符串

