from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_labeled_file_paths, shuffle_data, process_file_for_prediction
import numpy as np
import pandas as pd
import joblib
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import random
import os
import torch.autograd as autograd
import math
from collections import Counter


# 创建日志目录
log_dir = "runs/flash_welding_adapt_" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

# 梯度反转层（Gradient Reverse Layer）
class GradientReverseFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReverseLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.lambda_)

# 领域判别器（简单 MLP）


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)  # 输出 logits
        )

    def forward(self, features):
        return self.net(features)



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
        scaler_file = "scaler_seq.pkl"
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



class DualInputFlashWeldingModel(nn.Module):
    def __init__(self, input_channels=4, seq_len=None, static_dim=5, num_classes=2, feature_dim=64):
        super(DualInputFlashWeldingModel, self).__init__()

        # === 分支1: CNN 处理时间序列 x1 (B, C, T) ===
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()  # -> (B, 64)
        )

        # === 分支2: MLP 处理静态特征 x2 (B, k) ===
        self.mlp_branch = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # === 融合后总特征维度 ===
        self.fusion_dim = 64 + 64  # cnn_feat(64) + mlp_feat(64)

        # === 分类头 ===
        self.classifier = nn.Linear(self.fusion_dim, num_classes)

        # === 特征融合方式可选：拼接、相加、注意力等 ===
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_seq, x_static):
        """
        Args:
            x_seq: (B, C, T) 时序信号
            x_static: (B, k)   静态特征
        Returns:
            logits: (B, num_classes)
            fused_features: (B, fusion_dim) 用于 domain discriminator
        """
        feat_cnn = self.cnn_branch(x_seq)           # (B, 64)
        feat_mlp = self.mlp_branch(x_static)        # (B, 64)
        fused_features = torch.cat([feat_cnn, feat_mlp], dim=1)  # (B, 128)

        logits = self.classifier(fused_features)
        return logits, fused_features

    def get_features(self, x_seq, x_static):
        """ 提取融合特征，用于对抗训练 """
        return self.forward(x_seq, x_static)[1]  # 返回 fused_features


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



def train_model_adversarial_dual(
    model, train_loader, test_loader,
    num_epochs=10, learning_rate=0.001,
    lambda_max=0.5, writer=None, use_linear_schedule=False
):
    global_step = 0
    total_steps = num_epochs * len(train_loader)

    criterion_cls = FocalLoss(alpha=2, gamma=2)
    domain_discriminator = DomainDiscriminator(
        feature_dim=model.fusion_dim).to(device)
    grl = GradientReverseLayer()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(domain_discriminator.parameters()),
        lr=learning_rate
    )

    test_iter = iter(test_loader)

    for epoch in range(num_epochs):
        model.train()
        domain_discriminator.train()

        total_cls_loss = 0
        total_domain_loss = 0
        all_train_preds = []
        all_train_labels = []
        domain_preds, domain_labels = [], []

        for batch_x_seq_src, batch_x_static_src, batch_y_src in train_loader:
            p = global_step / total_steps
            if use_linear_schedule:
                lambda_adv = lambda_max * p
            else:
                lambda_adv = lambda_max * (2 / (1 + math.exp(-10 * p)) - 1)

            try:
                batch_x_seq_tgt, batch_x_static_tgt, _ = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                batch_x_seq_tgt, batch_x_static_tgt, _ = next(test_iter)

            # 移动到设备
            batch_x_seq_src = batch_x_seq_src.to(device)
            batch_x_static_src = batch_x_static_src.to(device)
            batch_y_src = batch_y_src.to(device)

            batch_x_seq_tgt = batch_x_seq_tgt.to(device)
            batch_x_static_tgt = batch_x_static_tgt.to(device)

            # 拼接源域和目标域输入
            x_seq_concat = torch.cat([batch_x_seq_src, batch_x_seq_tgt], 0)
            x_static_concat = torch.cat(
                [batch_x_static_src, batch_x_static_tgt], 0)
            domain_labels_true = torch.cat([
                torch.zeros(batch_x_seq_src.size(0)),
                torch.ones(batch_x_seq_tgt.size(0))
            ], 0).to(device)

            # 提取融合特征
            features = model.get_features(x_seq_concat, x_static_concat)

            # === 域判别损失（冻结主干）===
            domain_logits = domain_discriminator(features.detach())
            loss_domain = F.binary_cross_entropy_with_logits(
                domain_logits.squeeze(), domain_labels_true
            )
            total_domain_loss += loss_domain.item()

            # === 分类损失（仅源域）===
            logits_src, _ = model(batch_x_seq_src, batch_x_static_src)
            loss_cls = criterion_cls(logits_src, batch_y_src)
            total_cls_loss += loss_cls.item()

            # === 对抗损失（梯度反转）===
            domain_logits_adv = domain_discriminator(grl(features))
            loss_adv = F.binary_cross_entropy_with_logits(
                domain_logits_adv.squeeze(), domain_labels_true
            )

            loss = loss_cls + lambda_adv * loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录指标
            _, predicted = torch.max(logits_src, 1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(batch_y_src.cpu().numpy())

            dom_pred = (torch.sigmoid(domain_logits) > 0.5).float()
            domain_preds.extend(dom_pred.cpu().numpy())
            domain_labels.extend(domain_labels_true.cpu().numpy())

            global_step += 1

        # 计算平均损失与准确率
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_domain_loss = total_domain_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        domain_acc = accuracy_score(domain_labels, domain_preds)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Cls Loss: {avg_cls_loss:.4f}, "
              f"Domain Loss: {avg_domain_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Domain Acc: {domain_acc:.4f}")

        test_acc = test_model_dual(model, test_loader)  # 见下方定义
        print(f"Test Accuracy: {test_acc:.4f}")

        # 写入 TensorBoard
        current_lambda = lambda_max * \
            (2 / (1 + math.exp(-10 * (global_step / total_steps))) - 1)
        writer.add_scalar('Loss/Classification', avg_cls_loss, epoch)
        writer.add_scalar('Loss/Domain', avg_domain_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('Accuracy/Domain', domain_acc, epoch)
        writer.add_scalar('current_lambda', current_lambda, epoch)

    # 测试模型
    test_model_dual(model, test_loader, record_report=True, writer=writer)



def test_model_dual(model, test_loader, record_report=False, writer=None):
    """
    在测试集上评估双输入模型的准确率，并可选记录分类报告。

    Args:
        model: 训练好的双输入模型（DualInputFlashWeldingModel）
        test_loader: 测试数据加载器，返回 (x_seq, x_static, labels)
        record_report (bool): 是否生成并记录 classification_report
        writer: TensorBoard SummaryWriter（如果 record_report=True，则需要）
        epoch (int): 当前 epoch，用于 TensorBoard 记录

    Returns:
        accuracy (float): 测试准确率
    """
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 关闭梯度计算
        for batch_x_seq, batch_x_static, batch_y in test_loader:
            batch_x_seq = batch_x_seq.to(device)
            batch_x_static = batch_x_static.to(device)
            batch_y = batch_y.to(device)

            # 前向传播：获取 logits（忽略 features）
            logits, _ = model(batch_x_seq, batch_x_static)  # [B, num_classes]

            # 预测类别
            _, predicted = torch.max(logits, 1)

            # 收集结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    # 可选：记录详细的分类报告到 TensorBoard
    if record_report:
        report = classification_report(all_labels, all_preds, zero_division=0)
        writer.add_text('Test Classification Report', str(report), 0)
        print(f"Test Report:\n{report}")

    return accuracy



# 使用示例
if __name__ == "__main__":
    device = 'cuda:0'

    dataset_mapping = {
        '/home/dawn/Documents/HJ/data_all/U75VH/N': (1, 'train'),
        '/home/dawn/Documents/HJ/data_all/U75VH/P': (0, 'train'),
        '/home/dawn/Documents/HJ/data_all/U75VH/valid_N': (1, 'train'),
        '/home/dawn/Documents/HJ/data_all/U75VH/valid_P': (0, 'train'),

        # '/home/dawn/Documents/HJ/data_all/U75VH_sampled/N': (1, 'train'),
        # '/home/dawn/Documents/HJ/data_all/U75VH_sampled/P': (0, 'train'),
        # '/home/dawn/Documents/HJ/data_all/U75VH_sampled/valid_N': (1, 'train'),
        # '/home/dawn/Documents/HJ/data_all/U75VH_sampled/valid_P': (0, 'train'),

        '/home/dawn/Documents/HJ/data_all/processed test/1N': (1, 'validation'),
        '/home/dawn/Documents/HJ/data_all/processed test/1P': (0, 'validation'),
        '/home/dawn/Documents/HJ/data_all/924/1N': (1, 'validation'),
        '/home/dawn/Documents/HJ/data_all/924/1P': (0, 'validation'),

        # '/home/dawn/Documents/HJ/data_all/processed test/2N': (1, 'validation'),
        # '/home/dawn/Documents/HJ/data_all/processed test/2P': (0, 'validation'),
        # '/home/dawn/Documents/HJ/data_all/924/2N': (1, 'validation'),
        # '/home/dawn/Documents/HJ/data_all/924/2P': (0, 'validation')
    }

    labeled_file_paths = get_labeled_file_paths(dataset_mapping)
    data_train_seq = []
    data_test_seq = []
    data_train_static = []
    data_test_static = []
    shape_train_static = []
    shape_test_static = []

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

        # 1. 特征提取
        features, _ = process_file_for_prediction(path)
        if features is not None:
            # # 获取特征的值
            feature_values = features.values.flatten()
            if dataset_type == 'train':
                shape_train_static.append(feature_values.shape[0])
                data_train_seq.append(
                    [df, label])
                data_train_static.append(feature_values)
            else:
                shape_test_static.append(feature_values.shape[0])
                data_test_seq.append(
                    [df, label])
                data_test_static.append(feature_values)


    # 找到 shapes 中频次最高的形状
    shape_counts = Counter(shape_train_static + shape_test_static)
    most_common_shape, _ = shape_counts.most_common(1)[0]  # 获取最高频次的形状

    # 筛选出与最高频次形状一致的数据
    filtered_data_train_seq = []
    filtered_data_test_seq = []
    filtered_data_train_static = []
    filtered_data_test_static = []

    for data_train_seq_, data_train_static_, shape_train_static_ in zip(data_train_seq, data_train_static, shape_train_static):
        if shape_train_static_ == most_common_shape:
            filtered_data_train_seq.append(data_train_seq_)
            filtered_data_train_static.append(data_train_static_)

    for data_test_seq_, data_test_static_, shape_test_static_ in zip(data_test_seq, data_test_static, shape_test_static):
        if shape_test_static_ == most_common_shape:
            filtered_data_test_seq.append(data_test_seq_)
            filtered_data_test_static.append(data_test_static_)


    # for seq data
    tensors_train_seq = global_standardize_and_convert_to_tensor(data_train_seq)
    # 加载保存的 scaler
    scaler_file = "scaler_seq.pkl"
    scaler_seq = joblib.load(scaler_file)
    print(f"Scaler seq 已从 {scaler_file} 加载")

    tensors_test_seq = global_standardize_and_convert_to_tensor(
        data_test_seq, scaler=scaler_seq)
    sequences_train, _ = zip(*tensors_train_seq)
    sequences_test, _ = zip(*tensors_test_seq)

    # 找到最大序列长度
    # 如果有max_len，说明是测试集，则使用训练集的max_len
    max_len = max(seq.shape[1] for seq in sequences_train + sequences_test)

    x_train_seq, y_train, max_len = prepare_welding_data(
        tensors_train_seq, device='cuda:0', max_len=max_len)
    print(f"批次数据形状: {x_train_seq.shape}")  # torch.Size([3, 4, 100])
    x_test_seq, y_test, _ = prepare_welding_data(
        tensors_test_seq, max_len, device='cuda:0')
    print(f"批次数据形状: {x_test_seq.shape}")  # torch.Size([3, 4, 100])


    # standard scaler
    # 1. 创建并拟合标准化器（仅在训练集上 fit！）
    scaler_static = StandardScaler()
    filtered_data_train_static = scaler_static.fit_transform(
        filtered_data_train_static)
    # 2. 使用相同的 scaler 转换测试集（不能 fit！防止数据泄露）
    filtered_data_test_static = scaler_static.transform(
        filtered_data_test_static)

    # for static data
    x_train_static = torch.tensor(
        filtered_data_train_static, dtype=torch.float32).to(device)
    x_test_static = torch.tensor(
        filtered_data_test_static, dtype=torch.float32).to(device)

    # 创建模型
    model = DualInputFlashWeldingModel(
        input_channels=4, num_classes=2, static_dim=shape_train_static_).to(device)
    # 创建 Dataset 和 DataLoader
    train_dataset = TensorDataset(x_train_seq, x_train_static, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=False,
        worker_init_fn=worker_init_fn
        )  # shuffle 每轮打乱

    test_dataset = TensorDataset(x_test_seq, x_test_static, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, drop_last=False,
        worker_init_fn=worker_init_fn
        )

    lambda_max = 1
    num_epochs = 400

    # 初始化 SummaryWriter
    log_dir = "try_runs/flash_welding_adapt_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志已保存至: {log_dir}")
    writer.add_text('info',
                    'CNN提取特征+统计和物理特征，使用对抗，使用AdaptiveAvgPool1d来解决维度不一样',
                    0
                    )
    writer.add_text('data',
                    str(dataset_mapping),
                    0
                    )
    writer.add_text('data',
                    str(dataset_mapping),
                    0
                    )
    writer.add_text('num_epochs',
                    str(num_epochs),
                    0
                    )
    writer.add_text('lambda_max',
                    str(lambda_max),
                    0
                    )
    writer.add_text('X_seq_shape',
                    str(x_train_seq.shape),
                    0
                    )
    writer.add_text('X_static_shape',
                    str(x_train_static.shape),
                    0
                    )

    # # 添加模型结构到 TensorBoard
    # data_iter = iter(train_loader)
    # batch_x, _ = next(data_iter)
    # writer.add_graph(model, batch_x.to(device))

    train_model_adversarial_dual(model, train_loader, test_loader,
                            num_epochs=num_epochs, learning_rate=1e-2,
                            lambda_max=lambda_max, writer=writer)

    # === 训练结束后关闭 writer ===
    writer.close()
    print(f"TensorBoard 日志已保存完毕。使用命令启动：\ntensorboard --logdir {log_dir}")
