

from collections import Counter
import numpy as np
import json
from utils import get_labeled_file_paths, process_file_for_prediction


def prepare_datasets(dataset_mapping=None):

    labeled_file_paths = get_labeled_file_paths(dataset_mapping)

    # 初始化训练集和测试集
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    shapes = []


    for path, label, dataset_type in labeled_file_paths:
        # 1. 特征提取
        features, add_features = process_file_for_prediction(path)
        if features is not None:
            # # 获取特征的值
            feature_values = features.values.flatten()
            shapes.append(feature_values.shape[0])
            # 根据 dataset_type 和 label 分类
            if dataset_type == 'train':
                X_train.append(feature_values)
                y_train.append(label)
            elif dataset_type == 'validation':  # 或 'test'，根据实际情况调整
                X_test.append(feature_values)
                y_test.append(label)

    # 找到 shapes 中频次最高的形状
    shape_counts = Counter(shapes)
    most_common_shape, _ = shape_counts.most_common(1)[0]  # 获取最高频次的形状

    # 筛选出与最高频次形状一致的数据
    filtered_X_train = []
    filtered_y_train = []
    filtered_X_test = []
    filtered_y_test = []


    num = 0
    for i, feature_values in enumerate(X_train):
        if feature_values.shape[0] == most_common_shape:
            filtered_X_train.append(feature_values)
            filtered_y_train.append(y_train[i])
        else:
            num += 1

    for i, feature_values in enumerate(X_test):
        if feature_values.shape[0] == most_common_shape:
            filtered_X_test.append(feature_values)
            filtered_y_test.append(y_test[i])
        # else:

    print(f'this is the problem {num}')

    # 转换为 NumPy 数组
    X_train = np.array(filtered_X_train)
    y_train = np.array(filtered_y_train)
    X_test = np.array(filtered_X_test)
    y_test = np.array(filtered_y_test)

    return X_train, y_train, X_test, y_test

    # # 打印数据集大小
    # print(f"训练集大小: {X_train.shape}, 标签大小: {y_train.shape}")
    # print(f"测试集大小: {X_test.shape}, 标签大小: {y_test.shape}")

