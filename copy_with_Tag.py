

# 定义变量
SRC_DIR="test1/2N"   # 源目录
DEST_DIR="验证N" # 目标目录
SOURCE_TAG="processed_2N"               # 来源标签（可自定义）

import os
import shutil
from pathlib import Path


def copy_csv_with_source_tag(src_dir: str, dest_dir: str, source_tag: str):
    """
    遍历源文件夹中的 .csv 或 .CSV 文件，复制到目标文件夹，
    并在文件名后添加源文件夹的名称作为标签。

    :param src_dir: 源文件夹路径
    :param dest_dir: 目标文件夹路径
    """
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    # 检查源文件夹是否存在
    if not src_path.exists():
        raise FileNotFoundError(f"源文件夹不存在: {src_dir}")
    if not src_path.is_dir():
        raise NotADirectoryError(f"源路径不是文件夹: {src_dir}")

    # 获取源文件夹的名称（用于标记）
    # source_tag = src_path.name

    # 创建目标文件夹（如果不存在）
    dest_path.mkdir(parents=True, exist_ok=True)

    # 查找所有 .csv 或 .CSV 文件（不递归子目录）
    csv_files = []
    csv_files.extend(src_path.glob("*.csv"))
    csv_files.extend(src_path.glob("*.CSV"))

    # 去重并只保留文件（排除目录）
    files_to_copy = [f for f in set(csv_files) if f.is_file()]

    if not files_to_copy:
        print(f"在 '{src_dir}' 中未找到 .csv 或 .CSV 文件。")
        return

    print(f"找到 {len(files_to_copy)} 个 CSV 文件，开始复制...")

    for file_path in files_to_copy:
        # 分离文件名、扩展名（保留原大小写）
        stem = file_path.stem          # 如 'data'
        suffix = file_path.suffix      # 如 '.csv' 或 '.CSV'

        # 构造新文件名：原名_源文件夹名.扩展名
        new_filename = f"{stem}_{source_tag}{suffix}"
        dest_file = dest_path / new_filename

        # 检查目标文件是否已存在
        if dest_file.exists():
            print(f"跳过: {dest_file.name} 已存在")
            continue

        # 复制文件
        shutil.copy2(file_path, dest_file)  # copy2 保留元数据
        print(f"已复制: {file_path.name} → {new_filename}")

    print(f"复制完成！共处理 {len(files_to_copy)} 个文件。")

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 请根据实际情况修改以下路径
    source_directory = "data_all/test2/2P"   # 修改为你的源文件夹
    destination_directory = "data_all/验证P"  # 修改为你的目标文件夹
    source_tag = '924_2P'

    try:
        copy_csv_with_source_tag(
            source_directory, destination_directory, source_tag)
    except Exception as e:
        print(f"错误: {e}")