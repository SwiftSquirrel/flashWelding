# -*- coding: utf-8 -*-
# @Author: zzt
# @Date:   2022-10-21 20:14:23
# @Last Modified by:   zzt
# @Last Modified time: 2022-11-28 11:49:22
import numpy as np
from sklearn.model_selection import train_test_split
import rdkit.Chem as Chem
from rdkit.Chem.EState import Fingerprinter
from sqlalchemy import column, true
#from gcn.gcn_model import GCN_class
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
import pandas as pd
#import deepchem as dc


'''
model_params = torch.load("..\gcn\model_saved\gcn_models_params.pkl", map_location=torch.device('cuda:0'))
model = GCN_class(30,512,77,0.2)
model.load_state_dict(model_params)

model.cuda()'''

import os
import pandas as pd
from torch.utils.data import Dataset



import os
import pandas as pd

def read_csv_files(folder_path):
    # 存储所有文件的数据
    all_data = {}

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # print(f"正在读取文件: {filename}")

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 检查是否包含所需的列
            required_columns = ['PRESSURE', 'CURRENT', 'DISPLACEMENT']
            if all(col in df.columns for col in required_columns):
                # 提取三列数据并存储为数组
                pressure = df['PRESSURE'].tolist()
                current = df['CURRENT'].tolist()
                displacement = df['DISPLACEMENT'].tolist()

                # 将数据按文件名存储
                all_data[filename] = {
                    'PRESSURE': pressure,
                    'CURRENT': current,
                    'DISPLACEMENT': displacement
                }
            else:
                print(f"文件 {filename} 缺少必要的列：{required_columns}")

    return all_data

#零值填充，统一长度
def pad_and_split(data, target_length=16000, segment_length=100):
    """
    将输入的数组填充到指定长度（target_length），并按指定长度（segment_length）切割成多个子序列。

    参数:
        data (list): 输入的数组。
        target_length (int): 填充后的目标长度，默认为 20。
        segment_length (int): 每个子序列的长度，默认为 3。

    返回:
        list: 包含所有子序列的列表，每个子序列形状为 (segment_length,)
    """

    # 填充数据到目标长度
    padded_data = data + [0] * (target_length - len(data))

    # 按照 segment_length 切割
    sequences = []
    for i in range(0, target_length, segment_length):
        end = i + segment_length
        if end > target_length:
            break
        sequences.append(padded_data[i:end])

    return sequences

#滤波，消除噪声
def moving_average_filter(data, window_size=100):
    """
    使用移动平均滤波器对输入数据进行平滑处理。

    参数:
        data (list): 输入的原始数据列表。
        window_size (int): 滑动窗口大小，默认为 3。

    返回:
        list: 滤波后的数据列表。
    """

    if window_size < 1:
        raise ValueError("窗口大小必须大于等于 1")

    filtered_data = []
    for i in range(len(data)):
        # 计算窗口范围
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)

        # 取出窗口内的数据
        window = data[start:end]

        # 计算窗口内数据的平均值
        avg = sum(window) / len(window)

        filtered_data.append(avg)

    return filtered_data


def get_dataset (folder_path,mode ='P'):
    all_data = read_csv_files(folder_path)
    filenames = list(all_data.keys())
    data_list = []
    for file in filenames:    
        
        item_data = all_data[file]
        pressure = torch.tensor(pad_and_split(list(item_data['PRESSURE'])))#moving_average_filter()
        current =  torch.tensor(pad_and_split(list(item_data['CURRENT'])))#moving_average_filter()
        displacement = torch.tensor(pad_and_split(list(item_data['DISPLACEMENT'])))#moving_average_filter()
        matrix = [np.array(pressure).flatten().tolist(), np.array(current).flatten().tolist(), np.array(displacement).flatten().tolist()]
        matrix = torch.tensor(matrix)
        data= [pressure,current,displacement,matrix]
        if mode == 'P':
            y = 1
        else:
            y = 0
        data_list.append(data+[torch.tensor(y)])
    return data_list
