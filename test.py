import pandas as pd
import numpy as np
import os
import sys

import tsfel

from sklearn.preprocessing import StandardScaler


# ==================== 特征提取函数（与训练时完全一致）====================
def calculate_displacement_slope_and_detect_phases(time, displacement):
    """计算位移斜率并按规则划分阶段."""
    slope = np.gradient(displacement, time)
    abs_slope = np.abs(slope)

    phases = {}

    # 规则 1: 在10-30s内，第一个斜率数值超过3的点
    range_10_30 = (time >= 10) & (time <= 30)
    point1_index = np.argmax((slope > 3) & range_10_30)
    phases['Phase 1'] = time[point1_index] if slope[point1_index] > 3 else None

    # 规则 2: 在100s之前，最后一个斜率绝对值超过5的点
    range_before_100 = time < 100
    valid_indices_before_100 = np.where((abs_slope > 5) & range_before_100)[0]
    point2_index = valid_indices_before_100[-1] if len(valid_indices_before_100) > 0 else None
    phases['Phase 2'] = time[point2_index] if point2_index is not None else None

    # 规则 3: 在100s之后，第一个斜率绝对值超过40的点
    range_after_100 = time > 100
    valid_indices_after_100 = np.where((abs_slope > 40) & range_after_100)[0]
    point3_index = valid_indices_after_100[0] if len(valid_indices_after_100) > 0 else None
    phases['Phase 3'] = time[point3_index] if point3_index is not None else None

    return phases

def process_file_for_prediction(file_path):
    """为预测处理单个文件（与训练时的process_file函数一致，但不需要label）"""
    try:
        df = pd.read_csv(file_path)
        time = df.tail(1)['TIME'].values[0]

        print('文件路径: ', file_path)

        fs = np.round(df.shape[0]/time)

        print('采样频率', fs)

        # 确保所有列都是数值类型
        df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
        df['PRESSURE'] = pd.to_numeric(df['PRESSURE'], errors='coerce')
        df['CURRENT'] = pd.to_numeric(df['CURRENT'], errors='coerce')
        df['DISPLACEMENT'] = pd.to_numeric(df['DISPLACEMENT'], errors='coerce')

        # 删除包含NaN值的行
        df = df.dropna()

        if df.empty:
            return None

        # 使用位移的斜率划分阶段
        phases = calculate_displacement_slope_and_detect_phases(df['TIME'].values, df['DISPLACEMENT'].values)

        # 自动时间分割逻辑（结合阶段划分）
        segments = {
            'stage_1': df[(df['TIME'] <= phases['Phase 1'])] if phases['Phase 1'] else pd.DataFrame(),
            'stage_2': df[(df['TIME'] > phases['Phase 1']) & (df['TIME'] <= phases['Phase 2'])] if phases['Phase 2'] else pd.DataFrame(),
            'stage_3': df[(df['TIME'] > phases['Phase 2']) & (df['TIME'] <= phases['Phase 3'])] if phases['Phase 3'] else pd.DataFrame(),
            'stage_4': df[(df['TIME'] > phases['Phase 3'])] if phases['Phase 3'] else pd.DataFrame()
        }

        all_features = []

        for stage_name, segment in segments.items():
            if segment.empty:
                continue

            file_name = file_path.split('/')[-1].split('.')[0]

            # 提取特征
            cfg = tsfel.get_features_by_domain()
            features = tsfel.time_series_features_extractor(cfg, segment[['PRESSURE', 'CURRENT', 'DISPLACEMENT']], verbose=0)
            features.to_csv(f'features_{file_name}_with_data.csv')
            # features_new = tsfel.time_series_features_extractor(cfg, segment[['PRESSURE', 'CURRENT', 'DISPLACEMENT']], fs=fs, verbose=0)
            # 为特征添加前缀以区分不同阶段的特征
            features.columns = [f"{stage_name}_{col}" for col in features.columns]
            all_features.append(features)

        if all_features:
            # 合并所有阶段的特征
            combined_features = pd.concat(all_features, axis=1)
            return combined_features
        else:
            return None

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None



def align_features(self, extracted_features):
    """特征对齐"""
    if self.feature_names is None:
        return extracted_features.fillna(0)

    # 按训练时的特征顺序构建对齐的特征
    aligned_features = pd.DataFrame(index=extracted_features.index)

    for feature_name in self.feature_names:
        if feature_name in extracted_features.columns:
            aligned_features[feature_name] = extracted_features[feature_name]
        else:
            aligned_features[feature_name] = 0

    return aligned_features.fillna(0)




import os

def get_file_paths(dir_path):
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

# 示例用法
dir_path = '/Users/dawn/Desktop/成都材料院/钢轨焊接工艺参数分析/processed test/processed test/2N'
csv_file_paths = get_file_paths(dir_path)

print(csv_file_paths)




results = []
scaler = StandardScaler()

for i, file_path in enumerate(csv_file_paths):

    # 1. 特征提取
    features = process_file_for_prediction(file_path)
    if features is None:
        result = {
            'filename': os.path.basename(file_path),
            'prediction': None,
            'probability': None,
            'confidence': None,
            'status': 'feature_extraction_failed'
        }
    # else:
    #     # 2. 特征对齐
    #     aligned_features = align_features(features)

    #     # 3. 数据标准化
    #     features_scaled = scaler.transform(aligned_features.values)

    #     # # 4. 模型预测
    #     # prediction_prob = model.predict(features_scaled, verbose=0)[0][0]

    #     # # 5. 二分类预测
    #     # prediction = 1 if prediction_prob > self.threshold else 0

    #     # # 6. 计算置信度
    #     # confidence = max(prediction_prob, 1 - prediction_prob)

    #     # # 7. 预测标签
    #     # prediction_label = "好焊接" if prediction == 1 else "差焊接"

    #     # result = {
    #     #     'filename': os.path.basename(file_path),
    #     #     'prediction': prediction,
    #     #     'prediction_label': prediction_label,
    #     #     'probability': float(prediction_prob),
    #     #     'confidence': float(confidence),
    #     #     'status': 'success'
    #     # }

    # results.append(result)


