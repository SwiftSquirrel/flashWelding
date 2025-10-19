import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold



def train_ensemble_models_disjoint_valid(X_train, y_train, n_splits=5):
    """
    训练 n_splits 个 XGBoost 模型，每个模型使用不同的、不重叠的验证集
    验证集彼此无交集，确保独立性
    """
    # 使用 StratifiedKFold 划分数据
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    models = []
    valid_indices = []  # 用于记录每个 fold 的验证集索引（可选）

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n[训练第 {fold+1} 个模型]")

        # 获取划分后的数据
        X_train_part = X_train[train_idx]
        y_train_part = y_train[train_idx]
        X_valid = X_train[valid_idx]
        y_valid = y_train[valid_idx]

        # 保存验证集索引（可选，用于检查）
        valid_indices.append(valid_idx)

        # 创建并训练模型
        model = xgb.XGBClassifier(
            booster='gbtree',
            max_depth=3,
            learning_rate=0.05,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.7,
            colsample_bylevel=0.7,
            scale_pos_weight=scale_pos_weight,
            gamma=0.1,
            reg_alpha=1.0,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )

        # 训练 + 早停
        model.fit(
            X_train_part, y_train_part,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=50,
            verbose=50
        )

        # 输出验证集表现
        y_valid_pred = model.predict(X_valid)
        # print(f"[模型 {fold+1}] 验证集分类报告:")
        # print(classification_report(y_valid, y_valid_pred))

        models.append(model)

    all_valid_indices = np.concatenate(valid_indices)
    assert len(all_valid_indices) == len(np.unique(all_valid_indices)), \
        "错误：验证集存在重叠！"


    return models


def ensemble_predict_majority_vote(models, X_test):
    """
    对测试集 X_test 使用模型列表进行多数投票预测
    
    参数:
        models: 训练好的模型列表（例如 [model1, model2, ..., model5]）
        X_test: 测试数据 (numpy array 或 pandas DataFrame)
    
    返回:
        final_predictions: 投票后的最终整数标签 (0 或 1)
        all_predictions: 每个模型的预测结果 (可选，用于分析)
    """
    # 存储每个模型的预测结果
    predictions = []

    for i, model in enumerate(models):
        try:
            pred = model.predict(X_test)
            predictions.append(pred)
            print(f"模型 {i+1} 预测完成")
        except Exception as e:
            print(f"模型 {i+1} 预测失败: {e}")
            continue  # 可选择跳过失败模型

    # 转为 numpy 数组：shape = (n_models, n_samples)
    predictions = np.array(predictions)

    # 多数投票：对每一列（每个样本）统计 0 和 1 的数量，取最多者
    # axis=0 表示沿模型维度投票
    final_predictions = []
    for i in range(predictions.shape[1]):  # 遍历每个样本
        vote = int(np.bincount(predictions[:, i]).argmax())
        final_predictions.append(vote)

    final_predictions = np.array(final_predictions)

    return final_predictions


if __name__=='__main__':
    # from lgbModel import lgb_model, ensemble_lgb_model
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
    from feature_generate import prepare_datasets


    # use U75VH to train and test
    # note negative means the sample is bad, we use 1 to represent bad
    dataset_mapping = {

        # '/home/dawn/Documents/HJ/data_all/U75VH/N': (1, 'train'),
        # '/home/dawn/Documents/HJ/data_all/U75VH/P': (0, 'train'),
        # '/home/dawn/Documents/HJ/data_all/U75VH/valid_N': (1, 'train'),
        # '/home/dawn/Documents/HJ/data_all/U75VH/valid_P': (0, 'train'),

        # '/home/dawn/Documents/HJ/data_all/U75VH_sampled/N': (1, 'train'),
        # '/home/dawn/Documents/HJ/data_all/U75VH_sampled/P': (0, 'train'),
        # '/home/dawn/Documents/HJ/data_all/U75VH_sampled/valid_N': (1, 'train'),
        # '/home/dawn/Documents/HJ/data_all/U75VH_sampled/valid_P': (0, 'train'),


        # '/home/dawn/Documents/HJ/data_all/processed test/1N': (1, 'train'),
        # '/home/dawn/Documents/HJ/data_all/processed test/1P': (0, 'train'),
        '/home/dawn/Documents/HJ/data_all/924/1N': (1, 'train'),
        '/home/dawn/Documents/HJ/data_all/924/1P': (0, 'train'),


        # '/home/dawn/Documents/HJ/data_all/processed test/2N': (1, 'validation'),
        # '/home/dawn/Documents/HJ/data_all/processed test/2P': (0, 'validation'),
        # '/home/dawn/Documents/HJ/data_all/924/2N': (1, 'validation'),
        # '/home/dawn/Documents/HJ/data_all/924/2P': (0, 'validation')

    }
    # base_dir = '/home/dawn/Documents/HJ/HJ/data_all/U75VH'
    X_train, y_train, X_test, y_test = prepare_datasets(
        # base_dir=base_dir,
        dataset_mapping=dataset_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=0.3,      # 测试集占 20%
        random_state=42,    # 随机种子，保证可复现
        shuffle=True        # 是否打乱数据（默认为 True）
    )
    models = train_ensemble_models_disjoint_valid(X_train, y_train)

    # models = ensemble_lgb_model(X_train, y_train)

    # 在测试集上进行预测
    y_pred_train = ensemble_predict_majority_vote(models, X_train)

    # 在测试集上进行预测
    y_pred = ensemble_predict_majority_vote(models, X_test)

    # 计算评估指标
    accuracy_train = accuracy_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    recall = recall_score(y_train, y_pred_train)
    precision = precision_score(y_train, y_pred_train)

    # 生成分类报告
    report = classification_report(
        y_train, y_pred_train, target_names=['good', 'bad'])

    # 打印分类报告
    print("train 分类报告:")
    print(report)



    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # 生成分类报告
    report = classification_report(y_test, y_pred, target_names=['good', 'bad'])

    # 打印分类报告
    print("分类报告:")
    print(report)

    print('X train shape')
    print(X_train.shape)
    print('X test shape')
    print(X_test.shape)
