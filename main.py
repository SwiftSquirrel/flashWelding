# not used for now


from lgbModel import lgb_model, ensemble_lgb_model
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from feature_generate import prepare_datasets



X_train, y_train, X_test, y_test = prepare_datasets()





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
report = classification_report(y_test, y_pred, target_names=['N', 'P'])

# 打印分类报告
print("分类报告:")
print(report)


print('train shape', X_train.shape)
print('test shape', X_test.shape)

