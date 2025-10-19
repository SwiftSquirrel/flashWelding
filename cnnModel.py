import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_labeled_file_paths, shuffle_data
import numpy as np
import pandas as pd
import joblib
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import random
import os


import torch
import torch.nn as nn
import numpy as np


def compute_mmd_loss(x_src, x_tgt, kernel_type='linear', kernel_num=5):
    """
    改进版 MMD 损失：基于数据尺度自适应带宽
    """
    if x_src.size(0) != x_tgt.size(0):
        min_batch = min(x_src.size(0), x_tgt.size(0))
        x_src = x_src[:min_batch]
        x_tgt = x_tgt[:min_batch]

    x_concat = torch.cat([x_src, x_tgt], 0)

    if kernel_type == 'linear':
        kernels = torch.matmul(x_concat, x_concat.t())
    elif kernel_type == 'rbf':
        device = x_concat.device
        xx = torch.mm(x_concat, x_concat.t())
        x_sq = torch.diag(xx).unsqueeze(1)
        xy = -2.0 * xx
        y_sq = x_sq.t()
        pairwise_dists = x_sq + xy + y_sq + 1e-8  # 防止为0

        with torch.no_grad():
            # 基于距离中位数选择带宽
            dists = torch.pdist(x_concat, p=2)
            median_dist = torch.median(dists)
            base_scales = torch.logspace(-1, 1, kernel_num).to(device)  # 更安全范围
            scales = (median_dist ** 2) * base_scales
            scales = scales.view(1, 1, -1)

        pairwise_dists_exp = pairwise_dists.unsqueeze(2)
        kernels = torch.exp(-pairwise_dists_exp / (2 * scales)).sum(2)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    num_src = x_src.size(0)
    xx = kernels[:num_src, :num_src].mean()
    yy = kernels[num_src:, num_src:].mean()
    xy = kernels[:num_src, num_src:].mean()
    mmd_loss = xx + yy - 2 * xy
    return mmd_loss



def set_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 返回一个 worker_init_fn
    def worker_init_fn(worker_id):
        # 每个 worker 得到不同的但可预测的种子
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn

# 设置全局种子
worker_init_fn = set_deterministic_seed(42)



def global_standardize_and_convert_to_tensor(dataframes, scaler=None):
    """
    对所有样本的特征进行全局标准化，并转换为 torch.Tensor。

    :param dataframes: 包含多个 DataFrame 的列表，每个 DataFrame 的形状为 (4, n)。
    :return: 标准化后的 torch.Tensor 列表。
    """
    # 合并所有样本的特征数据
    all_data = []
    all_labels = []
    for df, label in dataframes:
        # if df.shape[0] != 4:
        #     raise ValueError(f"DataFrame 的行数应为 4，但得到 {df.shape[0]}")
        all_data.append(df)  # 转置为 n*4
        all_labels.append(label)

    # 将所有样本合并为一个大的 DataFrame
    combined_data = pd.concat(all_data, axis=0)

    # 使用 StandardScaler 进行全局标准化
    if not scaler:
        scaler = StandardScaler()
        standardized_combined_data = scaler.fit_transform(combined_data)

        # 保存 scaler 到本地文件
        scaler_file = "scaler.pkl"
        joblib.dump(scaler, scaler_file)

        print(f"Scaler 已保存到 {scaler_file}")
    else:
        # 使用已加载的 scaler 进行标准化
        standardized_combined_data = scaler.transform(combined_data)

    # 将标准化后的数据拆分回原始样本
    tensors = []
    start_idx = 0
    for df, label in dataframes:
        n = df.shape[0]  # 当前样本的行数（n）
        standardized_data = standardized_combined_data[start_idx:start_idx + n].T  # 提取并转置回 4*n
        tensor = torch.tensor(standardized_data, dtype=torch.float32)
        tensors.append((tensor, label))
        start_idx += n

    return tensors



class FlashWeldingCNN_Simple(nn.Module):
    """
    简化版闪光焊CNN模型（适用于小样本，约500条）
    - 更小的通道数
    - 更简单的分类头
    - 减少过拟合风险
    """

    def __init__(self, input_channels=4, num_classes=2):
        super(FlashWeldingCNN_Simple, self).__init__()

        # 轻量化特征提取器（减少通道数）
        self.conv_layers = nn.Sequential(
            # 第一层：捕捉短期模式
            nn.Conv1d(input_channels, 32, kernel_size=5,
                      padding=2),  # 64 -> 32
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 第二层：捕捉中期模式
            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # 128 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 第三层：捕捉长期模式（可选保留）
            nn.Conv1d(64, 64, kernel_size=3, padding=1),  # 256 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),      # 全局平均池化，适应变长序列
            nn.Flatten(),                 # 展平为 (batch_size, 64)
        )

        # 极简分类头：全局池化 + Dropout + 单层全连接
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)    # 直接输出类别 logits
        )

    def forward(self, x):
        feat = self.conv_layers(x) # [B, 64, T']
        # 全局平均池化得到固定长度特征
        x = self.classifier(feat) # 全局池化 + 分类
        return x, feat



# 数据准备示例
def prepare_welding_data(batch_data, max_len=None, device='cuda:0'):
    """
    准备焊接数据批次
    Args:
        batch_data: list of (sequence, label), 
                   sequence形状: (4, seq_len) - [电流, 压力, 位移, 时间]
    Returns:
        padded_sequences: 填充后的序列 (batch, 4, max_len)
        labels: 标签 (batch,)
        lengths: 原始序列长度
    """
    # 分离序列和标签
    sequences, labels = zip(*batch_data)
    
    # 找到最大序列长度
    if not max_len:
        # 如果有max_len，说明是测试集，则使用训练集的max_len
        max_len = max(seq.shape[1] for seq in sequences)
    
    # 填充序列到相同长度
    padded_sequences = []
    for seq in sequences:
        pad_size = max_len - seq.shape[1]
        if pad_size > 0:
            # 在时间维度末尾填充0
            padded_seq = F.pad(seq, (0, pad_size), 'constant', 0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    # 堆叠成批次
    batch_x = torch.stack(padded_sequences).to(device)
    batch_y = torch.tensor(labels).to(device)
    
    return batch_x, batch_y, max_len


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()





def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001, lambda_mmd = 2):
    """
    训练模型并打印损失和准确率。

    :param model: 要训练的模型。
    :param train_data: 训练数据 (batch_x_train, batch_y_train)。
    :param test_data: 测试数据 (batch_x_test, batch_y_test)。
    :param num_epochs: 训练的轮数。
    :param learning_rate: 学习率。
    """

    # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    # 使用
    criterion = FocalLoss(alpha=2, gamma=2)  # alpha 可调，增强对少数类关注
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 将 test_loader 转为迭代器，循环取 batch
    test_iter = iter(test_loader)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        total_loss = 0
        total_cls_loss = 0
        total_mmd_loss = 0
        all_train_preds = []
        all_train_labels = []

        # Mini-batch 训练
        for batch_x_src, batch_y_src in train_loader:
            # 前向传播
            try:
                batch_x_tgt, _ = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                batch_x_tgt, _ = next(test_iter)

            # 前向传播
            logits_src, features_src = model(batch_x_src)  # 源域
            _, features_tgt = model(batch_x_tgt)           # 目标域（只取特征）

            # 分类损失（仅源域）
            cls_loss = criterion(logits_src, batch_y_src)

            # MMD 损失（对齐特征分布）
            mmd_loss = compute_mmd_loss(
                features_src, features_tgt, kernel_type='rbf')

            # 总损失
            loss = cls_loss + lambda_mmd * mmd_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_cls_loss += cls_loss.item()
            total_mmd_loss += mmd_loss.item()
            total_loss += loss.item()

            # 记录预测结果
            _, predicted = torch.max(logits_src, 1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(batch_y_src.cpu().numpy())

        # 计算平均指标
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_mmd_loss = total_mmd_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)


        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {total_loss/len(train_loader):.4f}, "
              f"Cls Loss: {avg_cls_loss:.4f}, "
              f"MMD Loss: {avg_mmd_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}")


        avg_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # 测试模型
        test_accuracy = test_model(model, test_loader)
        print(f"Test Accuracy: {test_accuracy:.4f}")
    


def test_model(model, test_loader, print_report=True):
    """
    评估模型在测试集上的表现。

    :param model: 训练好的模型
    :param test_loader: 测试数据的 DataLoader
    :param print_report: 是否打印详细的分类报告
    :return: 准确率（accuracy）
    """
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 不需要梯度
        for batch_x, batch_y in test_loader:
            outputs, _ = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    # 打印分类报告（每类的 precision, recall, f1-score）
    if print_report:
        print("\nClassification Report on Test Set:")
        print(classification_report(all_labels, all_preds, zero_division=0))

    return accuracy


# 使用示例
if __name__ == "__main__":
    device = 'cuda:0'
    # base_dir='U75VH/model_train/data'
    base_dir = 'data_all'
    labeled_file_paths = get_labeled_file_paths(base_dir)
    data_train = []
    data_test = []

    for path, label, dataset_type in labeled_file_paths:
        # 1. 特征提取
        df = pd.read_csv(path)
        time = df.tail(1)['TIME'].values[0]

        # 确保所有列都是数值类型
        df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
        df['PRESSURE'] = pd.to_numeric(df['PRESSURE'], errors='coerce')
        df['CURRENT'] = pd.to_numeric(df['CURRENT'], errors='coerce')
        df['DISPLACEMENT'] = pd.to_numeric(df['DISPLACEMENT'], errors='coerce')

        # 删除包含NaN值的行
        df = df.dropna()
        if dataset_type == 'train':
            data_train.append([df, label])
        else:
            data_test.append([df, label])


    tensors_train = global_standardize_and_convert_to_tensor(data_train)
    # 加载保存的 scaler
    scaler_file = "scaler.pkl"
    scaler = joblib.load(scaler_file)
    print(f"Scaler 已从 {scaler_file} 加载")
    tensors_test = global_standardize_and_convert_to_tensor(data_test, scaler=scaler)

    # # 打乱数据
    # tensors_train = shuffle_data(tensors_train, seed=42)
    # tensors_test = shuffle_data(tensors_test, seed=42)


    # 创建模型
    model = FlashWeldingCNN_Simple(input_channels=4, num_classes=2).to(device)
    # 准备数据
    x_train, y_train, max_len = prepare_welding_data(tensors_train, device='cuda:0')
    print(f"批次数据形状: {x_train.shape}")  # torch.Size([3, 4, 100])
    x_test, y_test, _ = prepare_welding_data(tensors_test, max_len, device='cuda:0')
    print(f"批次数据形状: {x_test.shape}")  # torch.Size([3, 4, 100])



    # 创建 Dataset 和 DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=96, shuffle=True, drop_last=False,
        worker_init_fn=worker_init_fn
        )  # shuffle 每轮打乱

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=96, shuffle=False, drop_last=False,
        worker_init_fn=worker_init_fn
        )


    # 调用训练函数
    train_model(model, train_loader, test_loader, num_epochs=100,
                learning_rate=0.001, lambda_mmd=1)





