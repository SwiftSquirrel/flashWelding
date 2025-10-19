# copy the U75VH data

import os
import shutil
from pathlib import Path


def copy_subfolders(src_dir: str, dest_dir: str):
    """
    将源文件夹中的所有子文件夹复制到目标文件夹。
    如果目标文件夹不存在，则创建它。
    
    :param src_dir: 源文件夹路径
    :param dest_dir: 目标文件夹路径
    """
    src = Path(src_dir)
    dest = Path(dest_dir)

    # 检查源文件夹是否存在且是目录
    if not src.exists():
        raise FileNotFoundError(f"源文件夹不存在: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"源路径不是文件夹: {src}")

    # 创建目标文件夹（如果不存在）
    dest.mkdir(parents=True, exist_ok=True)

    # 遍历源文件夹中的所有项目
    copied_count = 0
    for item in src.iterdir():
        if item.is_dir():  # 只处理子文件夹
            dest_item = dest / item.name
            if dest_item.exists():
                print(f"跳过: {dest_item.name} 已存在于目标目录")
                continue
            shutil.copytree(item, dest_item)
            print(f"已复制子文件夹: {item.name}")
            copied_count += 1

    print(f"完成！共复制 {copied_count} 个子文件夹。")


# copy_subfolders('data_all/', 'data_all/')


# current_path = os.getcwd()
# print("当前工作路径:", current_path)


# copy the 