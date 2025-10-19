import os
import pandas as pd

# 定义目标文件夹路径
src_folder = "2025.9.24"
des_folder = "data_all/924"

# 初始化存储结果的列表
data = []

name_map = {
    '二线': '2P',
    '二线不合格': '2N',
    '一线': '1P',
    '一线不合格': '1N'
}

# 遍历 train 文件夹中的所有子文件夹和文件
# for folder_name in os.listdir(src_folder):
for folder_name in ['二线', '二线不合格', '一线', '一线不合格']:
    folder_path = os.path.join(src_folder, folder_name)
    if os.path.isdir(folder_path):  # 确保是文件夹
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):  # 确保是文件
                try:
                    data = pd.read_csv(file_path, encoding='gbk', header=None)
                    data = data.transpose()
                    data.columns = ['TIME', 'PRESSURE', 'CURRENT', 'DISPLACEMENT']
                    data = data[1:].reset_index(drop=True)
                    new_folder_path = os.path.join(
                        des_folder, name_map[folder_name])
                    new_file_name = file_name.replace(".CSV", "-transpose.csv")
                    new_file_path = os.path.join(new_folder_path, new_file_name)
                    data.to_csv(new_file_path, index=False, encoding='utf-8')
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")


print('completed !')

