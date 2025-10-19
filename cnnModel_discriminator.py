from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
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
import torch.autograd as autograd
import math


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
        logits, features = self.forward_with_features(x)
        return logits

    def forward_with_features(self, x):
        features = self.conv_layers(x)  # [B, 64]
        logits = self.classifier(features)
        return logits, features

    def get_features(self, x):
        return self.conv_layers(x)  # 只提取特征 [B, 64]



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




def train_model_adversarial(model, train_loader, test_loader,
                            num_epochs=10, learning_rate=0.001,
                            lambda_max=0.5, writer=None, use_linear_schedule=False):
    """
    使用对抗训练进行领域自适应
    """

    global_step = 0
    total_steps = num_epochs * len(train_loader)  # 总优化步数

    # 定义损失函数和优化器
    criterion_cls = FocalLoss(alpha=2, gamma=2)  # 分类损失
    domain_discriminator = DomainDiscriminator(feature_dim=64).to(device)
    grl = GradientReverseLayer()

    # 优化器：联合优化 model 和 domain_discriminator
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

        for batch_x_src, batch_y_src in train_loader:

            # ✅ 动态计算 lambda_adv
            p = global_step / total_steps  # 当前训练进度 [0, 1]
            if use_linear_schedule:
                lambda_adv = lambda_max * p  # 线性增长
            else:
                # 可选：更平滑的调度（来自 DANN 论文）
                lambda_adv = lambda_max * (2 / (1 + math.exp(-10 * p)) - 1)


            try:
                batch_x_tgt, _ = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                batch_x_tgt, _ = next(test_iter)

            # 移动到设备
            batch_x_src = batch_x_src.to(device)
            batch_y_src = batch_y_src.to(device)
            batch_x_tgt = batch_x_tgt.to(device)

            # 拼接 src 和 tgt
            x_concat = torch.cat([batch_x_src, batch_x_tgt], 0)
            domain_labels_true = torch.cat([
                torch.zeros(batch_x_src.size(0)),  # src -> 0
                torch.ones(batch_x_tgt.size(0))    # tgt -> 1
            ], 0).to(device)

            # 提取特征
            features = model.get_features(x_concat)  # [B_src + B_tgt, 64]

            # === 领域判别损失（判别器要能分清 src/tgt）===
            domain_logits = domain_discriminator(
                features.detach())  # detach：不更新 feature extractor
            loss_domain = F.binary_cross_entropy_with_logits(
                domain_logits.squeeze(), domain_labels_true
            )
            total_domain_loss += loss_domain.item()

            # === 分类损失（仅源域）===
            logits_src, _ = model.forward_with_features(batch_x_src)
            loss_cls = criterion_cls(logits_src, batch_y_src)
            total_cls_loss += loss_cls.item()

            # === 对抗损失（特征提取器要骗过判别器）===
            # 使用 GRL：让特征提取器“最小化”域判别损失，但梯度反向
            domain_logits_adv = domain_discriminator(grl(features))
            loss_adv = F.binary_cross_entropy_with_logits(
                domain_logits_adv.squeeze(),
                1 - domain_labels_true  # 对抗目标：让判别器输出相反
                # 或直接用 domain_labels_true，因为 grl 已反转梯度
            )
            # 更简单写法：直接用 domain_labels_true，GRL 负梯度自动对抗
            # loss_adv = F.binary_cross_entropy_with_logits(domain_logits_adv.squeeze(), domain_labels_true)


            # 总损失
            loss = loss_cls + lambda_adv * loss_adv

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # 记录
            _, predicted = torch.max(logits_src, 1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(batch_y_src.cpu().numpy())

            # 记录 domain 判别结果（用于监控）
            dom_pred = (torch.sigmoid(domain_logits) > 0.5).float()
            domain_preds.extend(dom_pred.cpu().numpy())
            domain_labels.extend(domain_labels_true.cpu().numpy())

            global_step += 1



        # 计算指标
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_domain_loss = total_domain_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        domain_acc = accuracy_score(domain_labels, domain_preds)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Cls Loss: {avg_cls_loss:.4f}, "
              f"Domain Loss: {avg_domain_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Domain Acc: {domain_acc:.4f}")

        # 测试模型
        test_acc = test_model(model, test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")

        # ✅ 记录当前 lambda_adv
        current_lambda = lambda_max * (global_step / total_steps)
        if not use_linear_schedule:
            current_lambda = lambda_max * \
                (2 / (1 + math.exp(-10 * (global_step / total_steps))) - 1)

        # 记录到 TensorBoard
        writer.add_scalar('Loss/Classification', avg_cls_loss, epoch)
        writer.add_scalar('Loss/Domain', avg_domain_loss, epoch)
        writer.add_scalar('Loss/Total', avg_cls_loss +
                          lambda_adv * avg_domain_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('Accuracy/Domain', domain_acc, epoch)
        writer.add_scalar(
            'Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar(
            'current_lambda', current_lambda, epoch)

    # 测试模型
    test_model(model, test_loader, record_report=True, writer=writer)



def test_model(model, test_loader, record_report=False, writer=None):
    """
    在测试集上评估模型准确率。
    
    Args:
        model: 训练好的模型（FlashWeldingCNN_Simple）
        test_loader: 测试数据加载器（目标域）

    Returns:
        accuracy: 测试准确率（float）
    """
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 关闭梯度计算
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 前向传播：只取 logits
            logits = model(batch_x)  # shape: [B, num_classes]

            # 预测类别
            _, predicted = torch.max(logits, 1)  # 取最大概率的类别

            # 收集结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    if record_report:
        report = classification_report(all_labels, all_preds, zero_division=0)
        writer.add_text('test report',
                        str(report),
                        0
                )


    return accuracy





# 使用示例
if __name__ == "__main__":
    device = 'cuda:0'

    dataset_mapping = {
        '/home/dawn/Documents/HJ/data_all/U75VH/N': (1, 'train'),
        '/home/dawn/Documents/HJ/data_all/U75VH/P': (0, 'train'),
        '/home/dawn/Documents/HJ/data_all/U75VH/valid_N': (1, 'train'),
        '/home/dawn/Documents/HJ/data_all/U75VH/valid_P': (0, 'train'),

        '/home/dawn/Documents/HJ/data_all/U75VH_sampled/N': (1, 'train'),
        '/home/dawn/Documents/HJ/data_all/U75VH_sampled/P': (0, 'train'),
        '/home/dawn/Documents/HJ/data_all/U75VH_sampled/valid_N': (1, 'train'),
        '/home/dawn/Documents/HJ/data_all/U75VH_sampled/valid_P': (0, 'train'),

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

    sequences_train, _ = zip(*tensors_train)
    sequences_test, _ = zip(*tensors_test)

    # 找到最大序列长度
    # 如果有max_len，说明是测试集，则使用训练集的max_len
    max_len = max(seq.shape[1] for seq in sequences_train + sequences_test)

    # 创建模型
    model = FlashWeldingCNN_Simple(input_channels=4, num_classes=2).to(device)
    # 准备数据
    x_train, y_train, max_len = prepare_welding_data(
        tensors_train, device='cuda:0', max_len=max_len)
    print(f"批次数据形状: {x_train.shape}")  # torch.Size([3, 4, 100])
    x_test, y_test, _ = prepare_welding_data(tensors_test, max_len, device='cuda:0')
    print(f"批次数据形状: {x_test.shape}")  # torch.Size([3, 4, 100])



    # 创建 Dataset 和 DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=False,
        worker_init_fn=worker_init_fn
        )  # shuffle 每轮打乱

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, drop_last=False,
        worker_init_fn=worker_init_fn
        )

    lambda_max = 1
    num_epochs = 400

    # 初始化 SummaryWriter
    log_dir = "runs/flash_welding_adapt_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志已保存至: {log_dir}")
    writer.add_text('info',
                    'CNN提取特征，使用对抗，使用AdaptiveAvgPool1d来解决维度不一样, used sampled data',
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
    writer.add_text('X_train_shape',
                    str(x_train.shape),
                    0
                    )
    writer.add_text('X_test_shape',
                    str(x_test.shape),
                    0
                    )

    # 添加模型结构到 TensorBoard
    data_iter = iter(train_loader)
    batch_x, _ = next(data_iter)
    writer.add_graph(model, batch_x.to(device))

    train_model_adversarial(model, train_loader, test_loader, 
                            num_epochs=num_epochs, learning_rate=1e-2,
                            lambda_max=lambda_max, writer=writer)

    # === 训练结束后关闭 writer ===
    writer.close()
    print(f"TensorBoard 日志已保存完毕。使用命令启动：\ntensorboard --logdir {log_dir}")
