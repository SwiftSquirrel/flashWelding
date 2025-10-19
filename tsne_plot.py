import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from feature_generate import prepare_datasets



X_train, y_train, X_test_924, y_test_924 = prepare_datasets()
X_test_processed, y_test_processed, _, _ = prepare_datasets(base_dir="data_all/test1")

# 1. 合并数据，并标记来源（train/test）
X_combined = np.vstack([X_train, X_test_processed, X_test_924])
y_combined = np.hstack([y_train, y_test_processed, y_test_924])

# X_combined = np.vstack([X_train])
# y_combined = np.hstack([y_train])
source = np.hstack([
    np.full(X_train.shape[0], 'train'),
    np.full(X_test_processed.shape[0], 'processed'),
    np.full(X_test_924.shape[0], '924'),
])


# 2. 标准化（t-SNE 对尺度敏感！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)  # 注意：用全部数据拟合 scaler（仅用于可视化）

# 3. t-SNE 降维到 2D
tsne = TSNE(
    n_components=2,
    perplexity=30,        # 通常 5~50，数据量大可调高
    max_iter=1000,
    n_iter_without_progress=300,  # 默认值，避免过早停止
    random_state=42,
    init='pca',           # 更稳定
    learning_rate='auto'
)
X_tsne = tsne.fit_transform(X_scaled)

# 4. 可视化
plt.figure(figsize=(10, 8))

# 定义颜色和标记
colors = {0: 'red', 1: 'blue'}  # 假设标签是 0/1
markers = {'train': 'o', 'processed': 'x', '924': '^'}

for label in np.unique(y_combined):
    for src in ['train', 'processed', '924']:
        mask = (y_combined == label) & (source == src)
        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=colors[label],
            marker=markers[src],
            label=f'{"Pos" if label else "Neg"} ({src})',
            alpha=0.6,
            s=20
        )

plt.title('t-SNE: Distribution')
plt.legend()
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
# 保存图片
plt.savefig("TSNE.png", dpi=300)  # 保存为高分辨率 PNG 文件
plt.show()
