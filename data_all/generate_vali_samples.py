import os
import shutil

# move 

# 定义源文件夹和目标文件夹
test1_dir = 'data_all/test1/P'
test2_dir = 'data_all/test2/P'
dest_dir = 'data_all/验证P'

# 确保目标文件夹存在
os.makedirs(dest_dir, exist_ok=True)

def copy_and_rename(source_dir, suffix):
    for filename in os.listdir(source_dir):
        if (filename.endswith('.csv') or filename.endswith('.CSV')):
            name, ext = os.path.splitext(filename)
            new_name = f"{name}{suffix}{ext}"
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(dest_dir, new_name)
            shutil.copy2(src_path, dst_path)
            print(f"已复制: {src_path} -> {dst_path}")

# 处理 test1 和 test2
copy_and_rename(test1_dir, '_processed')
copy_and_rename(test2_dir, '_924')