import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb


# 如果您想直接使用增强模型（不进行对比）
def lgb_model(X_train, y_train):
    """简化的增强训练版本"""
    # 使用调整后的参数
    model = lgb.LGBMClassifier(
        num_leaves=127,
        max_depth=-1,
        learning_rate=0.03,
        # min_child_samples=5,
        min_split_gain=0.1,
        reg_alpha=0.3,
        reg_lambda=0.3,
        # subsample=0.8,
        # colsample_bytree=0.8,
        n_estimators=100,
        random_state=42,
        is_unbalance=True
    )
    
    model.fit(X_train, y_train)


    return model


# 集成学习版本
def ensemble_lgb_model(X_train, y_train, n_models=10):
    """
    构建集成学习模型 - 训练多个LightGBM模型
    
    参数:
    - X_train: 训练特征
    - y_train: 训练标签
    - n_models: 基模型数量，默认为10
    
    返回:
    - ensemble_models: 训练好的模型列表，包含n_models个模型
    """
    
    ensemble_models = []
    
    for i in range(n_models):
        # 使用不同的随机种子创造模型差异性
        model = lgb.LGBMClassifier(
            num_leaves=127,
            max_depth=-1,
            learning_rate=0.05,
            min_child_samples=20,
            min_split_gain=0.01,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=1000,
            random_state=42 + i  # 关键：不同的随机种子
        )
        
        model.fit(X_train, y_train)
        ensemble_models.append(model)
    
    return ensemble_models


if __name__=='__main__':
    from lgbModel import lgb_model, ensemble_lgb_model
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
    from feature_generate import prepare_datasets


    # use U75VH to train and test
    # note negative means the sample is bad, we use 1 to represent bad
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
    # base_dir = '/home/dawn/Documents/HJ/HJ/data_all/U75VH'
    X_train, y_train, X_test, y_test = prepare_datasets(
        # base_dir=base_dir,
        dataset_mapping=dataset_mapping)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_train, y_train,
    #     test_size=0.3,      # 测试集占 20%
    #     random_state=42,    # 随机种子，保证可复现
    #     shuffle=True        # 是否打乱数据（默认为 True）
    # )
    model = lgb_model(X_train, y_train)
    # models = ensemble_lgb_model(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)


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
