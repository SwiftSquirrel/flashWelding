import numpy as np
import pandas as pd
import os
import tsfel
from scipy.integrate import simpson
from scipy.signal import savgol_filter, argrelextrema
from scipy import signal
from typing import Tuple, Optional
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.interpolate import interp1d
from scipy.stats import linregress


def time_series_to_image(ts_data, target_length=5000):
    """
    将变长时间序列转换为固定长度 5000 x 3 的“图像”表示
    
    Args:
        ts_data: numpy array of shape (N, 4), columns = [time, pressure, displacement, current]
        target_length: int, 输出长度，默认 5000
    
    Returns:
        image: numpy array of shape (target_length, 3), 对应 [pressure, displacement, current]
    """
    # 提取各列
    t = ts_data[:, 0]  # 时间
    p = ts_data[:, 1]  # 压力
    d = ts_data[:, 2]  # 位移
    c = ts_data[:, 3]  # 电流

    # 归一化时间轴到 [0, 1] 区间（避免数值过大导致插值问题）
    t_normalized = (t - t.min()) / (t.max() - t.min() + 1e-8)

    # 目标时间点：均匀分布在 [0, 1] 上的 5000 个点
    t_new = np.linspace(0, 1, target_length)

    # 插值函数（线性插值即可，也可用更高阶）
    interp_func_p = interp1d(
        t_normalized, p, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_func_d = interp1d(
        t_normalized, d, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_func_c = interp1d(
        t_normalized, c, kind='linear', bounds_error=False, fill_value='extrapolate')

    # 插值得到新序列
    p_new = interp_func_p(t_new)
    d_new = interp_func_d(t_new)
    c_new = interp_func_c(t_new)

    # 合并为 (5000, 3)
    image = np.stack([p_new, d_new, c_new], axis=1)  # shape: (5000, 3)

    return image


def convert_dict_to_dataframe(add_feature_dict):
    """
    将特征字典转换为 pandas.DataFrame，并对列进行排序。

    :param add_feature_dict: 包含特征的字典。
    :return: 转换后的 pandas.DataFrame。
    """
    # 将字典转换为 DataFrame
    df = pd.DataFrame([add_feature_dict])

    # 对列名进行排序（按字母顺序）
    sorted_columns = sorted(df.columns)
    df = df[sorted_columns]

    return df


def get_labeled_file_paths(dataset_mapping):
    """
    读取路径下的文件夹中的文件路径，并根据文件夹名称分配标签。
    - 训练集: N 文件夹中的文件为 0，Y 文件夹中的文件为 1。
    - 验证集: 验证N 文件夹中的文件为 0，验证Y 文件夹中的文件为 1。
    
    :param base_dir: 基础路径
    :return: 一个包含 (文件路径, 标签, 数据集类型) 的列表
    """
    labeled_paths = []
    
    for folder_path, (label, dataset_type) in dataset_mapping.items():
        if not os.path.exists(folder_path):
            continue
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                labeled_paths.append((file_path, label, dataset_type))
    
    return labeled_paths


# # ==================== 特征提取函数（与训练时完全一致）====================
# def calculate_displacement_slope_and_detect_phases(time, displacement):
#     """计算位移斜率并按规则划分阶段."""
#     slope = np.gradient(displacement, time)
#     abs_slope = np.abs(slope)

#     phases = {}

#     # 规则 1: 在10-30s内，第一个斜率数值超过3的点
#     range_10_30 = (time >= 10) & (time <= 30)
#     point1_index = np.argmax((slope > 3) & range_10_30)
#     phases['Phase 1'] = time[point1_index] if slope[point1_index] > 3 else None

#     # 规则 2: 在100s之前，最后一个斜率绝对值超过5的点
#     range_before_100 = time < 100
#     valid_indices_before_100 = np.where((abs_slope > 5) & range_before_100)[0]
#     point2_index = valid_indices_before_100[-1] if len(valid_indices_before_100) > 0 else None
#     phases['Phase 2'] = time[point2_index] if point2_index is not None else None

#     # 规则 3: 在100s之后，第一个斜率绝对值超过40的点
#     range_after_100 = time > 100
#     valid_indices_after_100 = np.where((abs_slope > 40) & range_after_100)[0]
#     point3_index = valid_indices_after_100[0] if len(valid_indices_after_100) > 0 else None
#     phases['Phase 3'] = time[point3_index] if point3_index is not None else None

#     return phases



def calculate_displacement_slope_and_detect_phases(time, displacement):
    """直接在原始数据上计算位移斜率并按规则划分阶段."""
    # 使用滑动窗口计算斜率
    def calculate_slope_with_window(time, displacement, window_size):
        slopes = np.zeros_like(time)
        half_window = window_size // 2

        for i in range(len(time)):
            start = max(0, i - half_window)
            end = min(len(time), i + half_window + 1)

            if end - start > 1:
                # 使用中心差分法计算斜率
                delta_time = time[end - 1] - time[start]
                delta_displacement = displacement[end - 1] - displacement[start]
                slopes[i] = delta_displacement / delta_time if delta_time != 0 else 0
            else:
                slopes[i] = 0

        return slopes

    # # 对 time 进行线性化
    # linearized_time = np.linspace(time[0], time[-1], len(time))
    # time = linearized_time  # 替代原始 time

    window_size = 11  # 滑动窗口大小
    slope = calculate_slope_with_window(time, displacement, window_size)
    abs_slope = np.abs(slope)


    phases = {}

    # 规则 1: 在10-30s内，第一个斜率数值超过3的点
    range_10_30 = (time >= 10) & (time <= 30)
    point1_index = np.argmax((slope > 3) & range_10_30)
    phases['Phase 1'] = time[point1_index] if slope[point1_index] > 3 else None

    # 规则 2: 在100s之前，最后一个斜率绝对值超过3的点
    range_before_100 = time < 92.5
    valid_indices_before_100 = np.where((abs_slope > 3) & range_before_100)[0]
    point2_index = valid_indices_before_100[-1] if len(valid_indices_before_100) > 0 else None
    phases['Phase 2'] = time[point2_index] if point2_index is not None else None

    # 规则 3: 在90s之后，找到梯度最大的点
    range_after_90 = time > 90
    valid_indices_after_90 = np.where(range_after_90)[0]

    if len(valid_indices_after_90) > 0:
        max_slope_index = valid_indices_after_90[np.argmax(abs_slope[valid_indices_after_90])]
        phases['Phase 3'] = time[max_slope_index]
    else:
        phases['Phase 3'] = None


    # 检查 Phase 3 和 Phase 2 的时间差
    if phases['Phase 2'] is not None and phases['Phase 3'] is not None:
        if abs(phases['Phase 3'] - phases['Phase 2']) < 5:
            # 如果时间差小于 5，删除 Phase 2
            phases['Phase 2'] = None


    return phases



def refine_feature_cfg(cfg):
    if 'spectral' in cfg:
        # 只保留关键频域特征，去掉冗余的
        spectral_keep = [
            'Spectral entropy',
            'Dominant frequency',
            'Spectral roll-off',
            'Power bandwidth',
            'FFT coefficient'  # 可限制 coeffs 数量
        ]
        cfg['spectral'] = {
            k: v for k, v in cfg['spectral'].items()
            if k in spectral_keep
        }
        # 可进一步限制 FFT 系数数量（避免爆炸）
        if 'FFT coefficient' in cfg['spectral']:
            cfg['spectral']['FFT coefficient'] = {
                'coeffs': [0, 1, 2, 3, 4]}  # 只取前5个

    if 'temporal' in cfg:
        # 只保留对焊接有意义的时域特征
        temporal_keep = [
            'Number of peaks',
            'Zero crossings',
            'Slope',
            'Abs energy',
            'Auto-correlation',
            'Crest factor'
        ]
        cfg['temporal'] = {
            k: v for k, v in cfg['temporal'].items()
            if k in temporal_keep
        }

    if 'statistical' in cfg:
        # 保留基础统计量
        statistical_keep = [
            'Mean', 'Standard deviation', 'Variance',
            'Skewness', 'Kurtosis', 'Root mean square',
            'Median', 'Interquartile range'
        ]
        cfg['statistical'] = {
            k: v for k, v in cfg['statistical'].items()
            if k in statistical_keep
        }

    if 'wavelet' in cfg:
        # 小波特征对瞬态敏感，推荐保留
        wavelet_keep = [
            'Wavelet entropy',
            'Sparsity',
            'Wavelet packet spectrum'
        ]
        cfg['wavelet'] = {
            k: v for k, v in cfg['wavelet'].items()
            if k in wavelet_keep
        }
    return cfg


def process_file_for_prediction(file_path):
    """为预测处理单个文件（与训练时的process_file函数一致，但不需要label）"""
    try:
        df = pd.read_csv(file_path)
        if ('202606/bad' in file_path) or ('202606/good' in file_path):
            df = df.rename(
                columns={'时间(s)': 'TIME', '压力': 'PRESSURE', '电流': 'CURRENT', '位移(mm)': 'DISPLACEMENT'})

        time = df.tail(1)['TIME'].values[0]
        # 确保所有列都是数值类型
        df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce').round(1)
        df['PRESSURE'] = pd.to_numeric(df['PRESSURE'], errors='coerce').round(1)
        df['CURRENT'] = pd.to_numeric(df['CURRENT'], errors='coerce').round(1)
        df['DISPLACEMENT'] = pd.to_numeric(
            df['DISPLACEMENT'], errors='coerce').round(1)
        # print(file_path)
        # 删除包含NaN值的行

        df = df.dropna()

        if df.empty:
            return None

        time = df['TIME'].values
        # 对 time 进行线性化
        linearized_time = np.linspace(time[0], time[-1], len(time))
        df['TIME'] = linearized_time  # 替代原始 time

        # 使用位移的斜率划分阶段
        phases = calculate_displacement_slope_and_detect_phases(df['TIME'].values, df['DISPLACEMENT'].values)


        # 对 PRESSURE, CURRENT, DISPLACEMENT 列进行 Z-Score 标准化
        columns_to_normalize = ['PRESSURE', 'CURRENT', 'DISPLACEMENT']

        for col in columns_to_normalize:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:  # 避免除以 0
                    df[col] = (df[col] - mean_val) / std_val
                else:
                    df[col] = 0  # 如果标准差为 0，直接设置为 0


        # 自动时间分割逻辑（结合阶段划分）
        segments = {
            'stage_1': df[(df['TIME'] <= phases['Phase 1'])] if phases['Phase 1'] else pd.DataFrame(),
            'stage_2': df[(df['TIME'] > phases['Phase 1']) & (df['TIME'] <= phases['Phase 2'])] if phases['Phase 2'] else pd.DataFrame(),
            'stage_3': df[(df['TIME'] > phases['Phase 2']) & (df['TIME'] <= phases['Phase 3'])] if phases['Phase 3'] else pd.DataFrame(),
            'stage_4': df[(df['TIME'] > phases['Phase 3'])] if phases['Phase 3'] else pd.DataFrame()
        }
        for phase, phase_value in phases.items():
            if not phase_value:
                return None, None


        all_features = []

        add_features = feature_gene(segments)

        for stage_name, segment in segments.items():
            if segment.empty:
                continue

            file_name = file_path.split('/')[-1].split('.')[0]

            # 提取特征
            cfg = tsfel.get_features_by_domain()
            cfg = refine_feature_cfg(cfg)
            features = tsfel.time_series_features_extractor(cfg, segment[['PRESSURE', 'CURRENT', 'DISPLACEMENT']], verbose=0)
            features.columns = [f"{stage_name}_{col}" for col in features.columns]
            all_features.append(features)

        add_features = convert_dict_to_dataframe(add_features)
        all_features.append(add_features)
        # all_features = [add_features]


        if all_features:
            # 合并所有阶段的特征
            combined_features = pd.concat(all_features, axis=1)
            return combined_features, add_features
        else:
            return None, None

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None, None



import numpy as np

def analyze_current_quantiles(current_data, quantiles=None, stage=None):
    """
    统计电流关键分位数
    
    参数:
    current_data -- 电流信号数据 (列表或numpy数组)
    quantiles -- 要计算的分位数列表，默认为常用分位数
    
    返回:
    dict -- 包含分位数统计结果的字典
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    current_array = np.array(current_data)
    
    # 计算分位数
    quantile_values = np.quantile(current_array, quantiles)
    
    # 构建结果字典
    result = {}
    for q, value in zip(quantiles, quantile_values):
        result[f'q_{int(q*100)}'] = float(value)
    
    iqr = result['q_75'] - result['q_25']
    # 添加基本统计信息
    result.update({
        stage+'_iqr': float(result['q_75'] - result['q_25']),
        stage+'_lower_fence': float(result['q_25'] - 1.5 * iqr),
        stage+'_upper_fence': float(result['q_75'] + 1.5 * iqr)
    })

    return result



def calculate_flash_burst_frequency_and_amplitude(current, time):
    """
    计算闪光爆破的频率和幅度。

    :param current: 电流数据 (numpy array 或 pandas Series)
    :param time: 时间数据 (numpy array 或 pandas Series)
    :return: 闪光爆破频率 (Hz), 平均幅度
    """
    # 计算电流的一阶差分
    current_diff = np.diff(current)

    # 设置阈值，检测快速变化的点（可以根据数据分布调整阈值）
    threshold = np.std(current_diff) * 2  # 使用标准差的倍数作为阈值
    burst_indices = np.where(np.abs(current_diff) > threshold)[0]

    # 计算闪光爆破的频率
    total_time = time[-1] - time[0]
    burst_count = len(burst_indices)
    frequency = burst_count / total_time if total_time > 0 else 0

    # 计算每次闪光爆破的幅度
    amplitudes = []
    for idx in burst_indices:
        # 找到每次爆破的局部最大值和最小值
        start_idx = max(0, idx - 1)
        end_idx = min(len(current) - 1, idx + 1)
        local_max = np.max(current[start_idx:end_idx + 1])
        local_min = np.min(current[start_idx:end_idx + 1])
        amplitudes.append(local_max - local_min)

    # 计算平均幅度
    average_amplitude = np.mean(amplitudes) if amplitudes else 0

    return frequency, average_amplitude



def find_high_stages(current, time, threshold=50):
    """
    找到数据中各高阶段的时间范围。

    :param current: 电流数据 (numpy array 或 pandas Series)
    :param time: 时间数据 (numpy array 或 pandas Series)
    :param threshold: 高阶段的电流阈值
    :return: 高阶段的时间范围列表，每个元素为 (start_time, end_time)
    """
    # 标记高阶段：电流值大于阈值
    high_mask = current > threshold

    # 找到高阶段的开始和结束索引
    high_stages = []
    start_idx = None

    for i in range(len(high_mask)):
        if high_mask[i] and start_idx is None:
            # 高阶段开始
            start_idx = i
        elif not high_mask[i] and start_idx is not None:
            # 高阶段结束
            end_idx = i - 1
            high_stages.append((start_idx, end_idx))
            start_idx = None

    # 如果高阶段持续到最后
    if start_idx is not None:
        high_stages.append((start_idx, len(time)-1))

    return high_stages


def find_minima_pairs(pressure_smooth, time_values, time_diff_threshold=1.5):
    """
    找到局部极小值，并返回 selected_minima 和对应的 minima_pair。

    :param pressure_smooth: 平滑后的压力数据 (numpy array)
    :param time_values: 时间数据 (numpy array)
    :param time_diff_threshold: 选中极小值点之间的最小时间差 (默认值为 1.5 秒)
    :return: minima_pair 列表，包含 (next_minima, previous_minima) 的索引对
    """
    # 计算压力的斜率
    pressure_slope = np.gradient(pressure_smooth, time_values)

    # 找到局部极小值的索引
    local_minima = argrelextrema(pressure_slope, np.less)[0]

    # 找到局部极小值的值和对应的时间
    local_minima_values = pressure_slope[local_minima]
    local_minima_times = time_values[local_minima]

    # 按极小值从小到大排序
    sorted_indices = np.argsort(local_minima_values)
    sorted_minima = local_minima[sorted_indices]
    sorted_minima_times = local_minima_times[sorted_indices]

    # 选择满足时间差条件的三个点
    selected_minima = []
    selected_times = []

    for idx, time in zip(sorted_minima, sorted_minima_times):
        if not selected_times or all(abs(time - t) > time_diff_threshold for t in selected_times):
            selected_minima.append(idx)
            selected_times.append(time)
        if len(selected_minima) == 3:  # 找到三个点后停止
            break

    selected_minima = sorted(selected_minima)

    # 找出在每个 selected_minima 之后的第一个 local_minima 点
    next_minima = []
    for selected_idx in selected_minima:
        # 找到所有在 selected_idx 之后的 local_minima
        subsequent_minima = [idx for idx in local_minima if idx > selected_idx]
        # 如果存在后续的 local_minima，选择第一个
        if subsequent_minima:
            next_minima.append(subsequent_minima[0])
        else:
            next_minima.append(None)  # 如果没有后续的 local_minima，标记为 None

    # 找出在每个 selected_minima 之前的第一个 local_minima 点
    previous_minima = []
    for selected_idx in selected_minima:
        # 找到所有在 selected_idx 之前的 local_minima
        preceding_minima = [idx for idx in local_minima if idx < selected_idx]
        # 如果存在前续的 local_minima，选择最后一个
        if preceding_minima:
            previous_minima.append(preceding_minima[-1])
        else:
            previous_minima.append(None)  # 如果没有前续的 local_minima，标记为 None

    # 构造 minima_pair
    minima_pair = [
        (next_minima[0], previous_minima[1]),
        (next_minima[1], previous_minima[2]),
        (next_minima[2], len(time_values) - 1)
    ]

    return minima_pair


def calculate_stage1_features(seg1):
    """
    计算闪平阶段的特征。
    # ● 闪平阶段（捕捉初始清理是否充分）
    #   ○ 位移变化、电流的分位数（90分位）、电流变异系数（标准差/均值）、闪光爆破的频率和幅度
    :param seg1: 第一阶段的数据 (DataFrame)，包含 'TIME', 'DISPLACEMENT', 'CURRENT' 列。
    :return: 包含闪平阶段特征的字典 feature_dict。
    """
    feature_dict = {}

    # 计算闪平阶段的时间
    stage1_time = seg1['TIME'].values[-1] - seg1['TIME'].values[0]
    feature_dict['stage1_time'] = stage1_time

    # 计算位移变化
    feature_dict['stage1_displacement'] = seg1['DISPLACEMENT'].values[-1] - seg1['DISPLACEMENT'].values[0]

    # 计算电流的标准差和均值
    current_std = seg1['CURRENT'].std()  # 标准差
    current_mean = seg1['CURRENT'].mean()  # 均值

    # 计算电流的变异系数（标准差 / 均值）
    current_variation_coefficient = current_std / current_mean if current_mean != 0 else None  # 避免除以 0
    feature_dict['stage1_current_variation_coefficient'] = current_variation_coefficient

    # 计算闪光爆破的频率和幅度
    stage1_burst_frequency, stage1_burst_amplitude = calculate_flash_burst_frequency_and_amplitude(
        seg1['CURRENT'].values, seg1['TIME'].values
    )
    feature_dict['stage1_burst_frequency'] = stage1_burst_frequency
    feature_dict['stage1_burst_amplitude'] = stage1_burst_amplitude

    # 计算闪光率
    sampling_rate = np.round(1/(seg1['TIME'].values[1]-seg1['TIME'].values[0]))
    flash_rate = calculate_flash_rate(seg1['CURRENT'].values, sampling_rate)
    feature_dict['stage1_flash_rate'] = flash_rate

    # 计算爆破率相关
    burst_rates, _ = calculate_burst_rate(seg1['CURRENT'].values, sampling_rate)
    feature_dict['stage1_burst_rate_mean'] = np.mean(burst_rates)
    feature_dict['stage1_burst_rate_std'] = np.std(burst_rates)
    feature_dict['stage1_burst_rate_min'] = min(burst_rates)
    feature_dict['stage1_burst_rate_max'] = max(burst_rates)

    # 计算电流的分位数
    current_stats = analyze_current_quantiles(current_data=seg1['CURRENT'].values, stage='stage1')
    feature_dict.update(current_stats)

    return feature_dict


def remove_smallest_diff_tuples(intervals, k):
    # 按差值 (end - start) 升序排序，保留原始索引用于稳定排序
    sorted_intervals = sorted(intervals, key=lambda x: x[1] - x[0])
    # 取前 k 个要删除的
    to_remove = set(sorted_intervals[:k])
    # 从原列表中删除这些元组（保持剩余顺序）
    result = [interval for interval in intervals if interval not in to_remove]
    return result



def calculate_speeds_from_high_stages(coords, displacements, times, high_stages):
    """
    计算 high_stages 中每个阶段的速度。

    :param coords: 所有坐标的索引列表。
    :param displacements: 对应的位移值列表。
    :param times: 对应的时间值列表。
    :param high_stages: 高阶段的时间范围列表，每个元素为 (start, end)。
    :return: 每个 high_stage 的速度列表。
    """
    speeds = []

    for high_stage in high_stages:
        start, end = high_stage

        # 找到 coords 中小于 start 的坐标
        valid_coords = [i for i in coords if i < start]
        if not valid_coords:
            continue  # 如果没有有效坐标，跳过

        # 找到对应的最小位移及其索引
        min_coord = min(valid_coords, key=lambda i: displacements[i])
        min_displacement = displacements[min_coord]
        min_time = times[min_coord]

        # 获取 start 的位移和时间
        start_displacement = displacements[start]
        start_time = times[start]

        # 计算位移和时间差
        displacement_diff = start_displacement - min_displacement
        time_diff = start_time - min_time

        # 计算速度
        if time_diff > 0:
            speed = displacement_diff / time_diff
            speeds.append(speed)

        # 剔除 coords 中小于 end 的坐标
        coords = [i for i in coords if i > end]

    return speeds


def calculate_flash_rate(current_signal, sampling_rate, lowcut=5.0, highcut=200.0, threshold_std=1.5):
    """
    计算闪光焊过程中的闪光率（简化版）
    
    参数:
    current_signal -- 电流信号数据 (1D array)
    sampling_rate -- 采样率 (Hz)
    lowcut -- 低频截止频率 (Hz)
    highcut -- 高频截止频率 (Hz) 
    threshold_std -- 阈值标准差的倍数
    
    返回:
    flash_rate -- 平均闪光率 (Hz)
    """

    # 1. 带通滤波
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    # 对 low 和 high 进行裁剪，确保它们在 (0, 1) 范围内
    low = max(0.001, min(low, 0.999))  # 确保 low 不小于 0.001 且不大于 0.999
    high = max(0.001, min(high, 0.999))  # 确保 high 不小于 0.001 且不大于 0.999

    # 确保 low < high
    if low >= high:
        raise ValueError(f"Invalid cutoff frequencies after clipping: low={low}, high={high}. Ensure low < high.")

    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, current_signal)

    # 2. 计算统计量
    mean_val = np.mean(filtered_signal)
    std_val = np.std(filtered_signal)

    # 3. 设置阈值
    upper_threshold = mean_val + threshold_std * std_val
    lower_threshold = mean_val - threshold_std * std_val

    # 4. 计数闪光事件 - 阈值交叉法
    flash_count = 0
    below_lower = False

    for i in range(1, len(filtered_signal)):
        # 检测从下阈值下方上升到上方
        if filtered_signal[i-1] < lower_threshold and filtered_signal[i] >= lower_threshold:
            below_lower = True
        # 当在下阈值上方时，检测上升到上阈值
        elif below_lower and filtered_signal[i-1] < upper_threshold and filtered_signal[i] >= upper_threshold:
            flash_count += 1
            below_lower = False

    # 5. 计算闪光率
    total_time = len(current_signal) / sampling_rate  # 总时间（秒）
    flash_rate = flash_count / total_time  # 闪光率（Hz）

    return flash_rate



def calculate_burst_rate(
    data: np.ndarray,
    sampling_rate: float,
    signal_type: str = "current",
    window_duration: float = 1.0,
    step_size: float = 0.2,
    min_burst_interval: float = 0.03,  # 最小爆破间隔（秒），避免重复检测
    height_percentile: float = 95,     # 阈值百分位（默认前5%为爆破）
    bandpass_freq: Optional[Tuple[float, float]] = None,
    return_events: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算闪光焊过程中的爆破率（Burst Rate），单位：Hz。

    Parameters
    ----------
    data : np.ndarray
        原始信号（如电流、电压、光强），一维数组。
    sampling_rate : float
        采样率（Hz），如 125。
    signal_type : str, optional
        信号类型，用于自动设置滤波参数（"current", "voltage", "light"），默认 "current"。
    window_duration : float, optional
        滑动窗口长度（秒），默认 1.0 秒。
    step_size : float, optional
        滑动步长（秒），默认 0.2 秒。
    min_burst_interval : float, optional
        最小爆破时间间隔（秒），防止一个爆破被多次检测，默认 0.03（30ms）。
    height_percentile : float, optional
        峰值检测阈值（百分位数），默认 95（即只保留最强的5%变化）。
    bandpass_freq : tuple or None, optional
        自定义带通滤波范围 (low, high) in Hz。若为 None，根据 signal_type 自动设置。
    return_events : bool, optional
        是否返回爆破事件时间戳（秒）。

    Returns
    -------
    burst_rates : np.ndarray
        瞬时爆破率（Hz），长度 = 时间点数量。
    time_centers : np.ndarray
        对应的时间中心点（秒）。
    (optional) burst_times : np.ndarray
        若 return_events=True，额外返回爆破发生时刻（秒）。
    """
    
    # 输入校验
    if len(data) == 0:
        raise ValueError("输入数据为空")
    if sampling_rate <= 0:
        raise ValueError("采样率必须为正数")
    
    min_burst_interval = 2/sampling_rate
    n_samples = len(data)
    time_original = np.arange(n_samples) / sampling_rate

    # 1. 去除直流/趋势（使用高通或 detrend）
    data_detrended = signal.detrend(data)

    # 2. 自动设置带通滤波范围（针对爆破瞬态）
    if bandpass_freq is None:
        if signal_type in ["current", "voltage"]:
            # 闪光爆破主要能量在 5–60 Hz，抑制工频（50/60 Hz）可选
            low, high = 5, 60
        elif signal_type == "light":
            low, high = 1, 50
        else:
            low, high = 1, 50
    else:
        low, high = bandpass_freq

    # 确保滤波器有效
    nyquist = sampling_rate / 2
    low = max(low, 1)
    high = min(high, nyquist - 1)
    if low >= high:
        raise ValueError(f"无效的带通范围: ({low}, {high})，采样率={sampling_rate} Hz")

    # 3. 带通滤波
    b, a = signal.butter(4, [low, high], btype='band', fs=sampling_rate)
    data_filtered = signal.filtfilt(b, a, data_detrended)

    # 4. 使用一阶导数增强瞬态（爆破表现为电流快速上升）
    derivative = np.abs(np.diff(data_filtered, prepend=data_filtered[0]))
    
    # 5. 自适应阈值：取导数的 height_percentile 分位数
    threshold = np.percentile(derivative, height_percentile)
    min_distance = max(2, int(min_burst_interval * sampling_rate)) 

    # 6. 峰值检测
    peaks, _ = signal.find_peaks(
        derivative,
        height=threshold,
        distance=min_distance
    )

    # 转换为时间（秒）
    burst_times = peaks / sampling_rate

    # 7. 滑动窗口计算爆破率
    total_time = n_samples / sampling_rate
    time_centers = []
    burst_rates = []

    win_samples = int(window_duration * sampling_rate)
    step_samples = int(step_size * sampling_rate)

    for start_idx in range(0, n_samples - win_samples + 1, step_samples):
        end_idx = start_idx + win_samples
        t_center = (start_idx + win_samples / 2) / sampling_rate

        # 统计该窗口内爆破次数
        count = np.sum((peaks >= start_idx) & (peaks < end_idx))
        rate = count / window_duration  # Hz

        time_centers.append(t_center)
        burst_rates.append(rate)

    time_centers = np.array(time_centers)
    burst_rates = np.array(burst_rates)

    if return_events:
        return burst_rates, time_centers, burst_times
    else:
        return burst_rates, time_centers



def calculate_stage2_features(seg2):
    """
    计算预热阶段的特征。
    # ● 预热阶段 (评估预热是否稳定、均匀地提高了工件温度)：
    #   ○ 时间：预热的持续时间和标准差
    #   ○ 电流：接触时电流的均值和标准差
    #   ○ 压力/位移：接触时的压力均值和标准差、位移的均值和标准差
    #   ○ 功：压力*距离
    #   

    :param seg2: 第二阶段的数据 (DataFrame)，包含 'TIME', 'DISPLACEMENT', 'CURRENT', 'PRESSURE' 列。
    :return: 包含预热阶段特征的字典 feature_dict。
    """
    feature_dict = {}

    # 计算预热阶段的时间
    stage2_time = seg2['TIME'].values[-1] - seg2['TIME'].values[0]
    feature_dict['stage2_time'] = stage2_time

    # 计算功（压力对位移的积分）
    work_done = simpson(seg2['PRESSURE'].values, seg2['DISPLACEMENT'].values)
    feature_dict['stage2_work_done'] = work_done

    # 找到高阶段（电流超过阈值的阶段）
    high_stages = find_high_stages(seg2['CURRENT'].values, seg2['TIME'].values)
    num_cycle = len(high_stages)

    # 计算预热阶段，每次推进时候的位移速度
    coords = [i for i in range(seg2['TIME'].values.shape[0])]
    speeds = np.median(calculate_speeds_from_high_stages(coords, seg2['DISPLACEMENT'].values, seg2['TIME'].values, high_stages))
    feature_dict['stage2_displacement_speed'] = speeds
    feature_dict['stage2_num_cycles'] = 10

    if num_cycle > 10:
        high_stages = remove_smallest_diff_tuples(high_stages, num_cycle-10)
        num_cycle = 10
    
    # 时间分辨率
    time_resolution = seg2['TIME'].values[1] - seg2['TIME'].values[0]

    # 计算每个高阶段的时间
    cycle_time = [time_resolution * (high_stage[1] - high_stage[0]) for high_stage in high_stages]
    feature_dict['stage2_cycle_time_mean'] = np.mean(cycle_time)
    feature_dict['stage2_cycle_time_std'] = np.std(cycle_time)

    # 短路/断路 时间
    feature_dict['cycle_time_ratio'] = sum(
        cycle_time)/(stage2_time-sum(cycle_time))


    # 初始化循环特征
    cycle_currents = []
    cycle_pressures = []
    cycle_displacements = []

    # 遍历每个高阶段，计算特征
    for high_stage in high_stages:
        start_idx, end_idx = high_stage
        cycle_currents.append(np.mean(seg2['CURRENT'].values[start_idx:end_idx + 1]))
        cycle_pressures.append(np.mean(seg2['PRESSURE'].values[start_idx:end_idx + 1]))
        cycle_displacements.append(
            np.mean(seg2['DISPLACEMENT'].values[start_idx:end_idx + 1]) - seg2['DISPLACEMENT'].values[0]
        )

    # 计算循环特征的均值和标准差
    feature_dict['stage2_cycle_current_mean'] = np.mean(cycle_currents)
    feature_dict['stage2_cycle_current_std'] = np.std(cycle_currents)
    feature_dict['stage2_cycle_pressure_mean'] = np.mean(cycle_pressures)
    feature_dict['stage2_cycle_pressure_std'] = np.std(cycle_pressures)
    feature_dict['stage2_cycle_displacement_mean'] = np.mean(cycle_displacements)
    feature_dict['stage2_cycle_displacement_std'] = np.std(cycle_displacements)

    return feature_dict


def calculate_stage3_features(seg3):
    """
    计算烧化阶段的特征。
    # ● 烧化阶段
    #   ○ 变异系数（标准差/均值）、闪光爆破的频率和幅度、位移速度的均值和标准差（评估稳定性）、变异系数、平均电流
    :param seg3: 第三阶段的数据 (DataFrame)，包含 'TIME', 'DISPLACEMENT', 'CURRENT' 列。
    :return: 包含烧化阶段特征的字典 feature_dict。
    """
    feature_dict = {}

    # 计算烧化阶段的时间
    stage3_time = seg3['TIME'].values[-1] - seg3['TIME'].values[0]
    feature_dict['stage3_time'] = stage3_time

    # 计算位移变化
    feature_dict['stage3_displacement'] = seg3['DISPLACEMENT'].values[-1] - seg3['DISPLACEMENT'].values[0]

    # 计算电流的标准差和均值
    current_std = seg3['CURRENT'].std()  # 标准差
    current_mean = seg3['CURRENT'].mean()  # 均值
    feature_dict['stage3_current_mean'] = current_mean

    # 计算电流的变异系数（标准差 / 均值）
    current_variation_coefficient = current_std / current_mean if current_mean != 0 else None  # 避免除以 0
    feature_dict['stage3_current_variation_coefficient'] = current_variation_coefficient

    # 计算闪光爆破的频率和幅度
    stage3_burst_frequency, stage3_burst_amplitude = calculate_flash_burst_frequency_and_amplitude(
        seg3['CURRENT'].values, seg3['TIME'].values
    )
    feature_dict['stage3_burst_frequency'] = stage3_burst_frequency
    feature_dict['stage3_burst_amplitude'] = stage3_burst_amplitude

    # 计算闪光率
    sampling_rate = np.round(1/(seg3['TIME'].values[1]-seg3['TIME'].values[0]))
    flash_rate = calculate_flash_rate(seg3['CURRENT'].values, sampling_rate)
    feature_dict['stage3_flash_rate'] = flash_rate

    # 计算爆破率相关
    burst_rates, _ = calculate_burst_rate(seg3['CURRENT'].values, sampling_rate)
    feature_dict['stage3_burst_rate_mean'] = np.mean(burst_rates)
    feature_dict['stage3_burst_rate_std'] = np.std(burst_rates)
    feature_dict['stage3_burst_rate_min'] = min(burst_rates)
    feature_dict['stage3_burst_rate_max'] = max(burst_rates)

    # 计算位移速度（位移的一阶差分除以时间的一阶差分）
    displacement_diff = np.diff(seg3['DISPLACEMENT'].values)
    time_diff = np.diff(seg3['TIME'].values)
    velocity = displacement_diff / time_diff

    # 计算位移速度的均值
    velocity_mean = np.mean(velocity)
    # 计算位移速度的标准差
    velocity_std = np.std(velocity)
    # 计算变异系数（标准差 / 均值）
    velocity_variation_coefficient = velocity_std / velocity_mean if velocity_mean != 0 else None
    feature_dict['stage3_velocity_mean'] = velocity_mean
    feature_dict['stage3_velocity_std'] = velocity_std
    feature_dict['stage3_velocity_variation_coefficient'] = velocity_variation_coefficient

    # 计算电流的分位数
    current_stats = analyze_current_quantiles(current_data=seg3['CURRENT'].values, stage='stage3')
    feature_dict.update(current_stats)

    return feature_dict


def calculate_upset_features(seg4, minima_pair):
    """
    # ● 顶锻
    #   ○ 顶锻留量（顶锻开始到结束的总位移）、顶锻速度（位移曲线在顶锻瞬间的斜率，越高越好）。
    #   ○ 顶锻力峰值
    基于 minima_pair[0][0] 划分顶锻和保压阶段，并计算顶锻阶段的特征。

    :param seg4: 第四阶段的数据 (DataFrame)，包含 'TIME', 'DISPLACEMENT', 'PRESSURE' 列。
    :param minima_pair: 包含划分点的索引对列表。
    :return: 顶锻阶段的特征字典。
    """
    # 获取顶锻阶段的结束索引
    upset_end_idx = minima_pair[0][0]

    # 顶锻阶段数据
    upset_time = seg4['TIME'].values[:upset_end_idx + 1]
    upset_displacement = seg4['DISPLACEMENT'].values[:upset_end_idx + 1]
    upset_pressure = seg4['PRESSURE'].values[:upset_end_idx + 1]
    # 计算顶锻阶段的压力峰值
    pressure_peak = np.max(upset_pressure)

    # 计算顶锻阶段的位移
    total_displacement = upset_displacement[-1] - upset_displacement[0]

    # 计算顶锻阶段的位移速度
    displacement_diff = np.diff(upset_displacement)
    time_diff = np.diff(upset_time)
    velocity = displacement_diff / time_diff
    velocity_mean = np.mean(velocity)
    velocity_std = np.std(velocity)
    velocity_variation_coefficient = velocity_std / velocity_mean if velocity_mean != 0 else None  # 避免除以 0



    # 使用 simpson 替代 np.trapz
    work_done = simpson(upset_pressure, upset_displacement)

    # 返回计算结果
    return {
        'upset_time': upset_time[-1] - upset_time[0],
        'upset_total_displacement': total_displacement,
        'upset_velocity_mean': velocity_mean,
        'upset_velocity_std': velocity_std,
        'upset_velocity_variation_coefficient': velocity_variation_coefficient,
        'upset_pressure_peak':pressure_peak,
        'upset_work_done': work_done
    }



# from scipy.integrate import simpson
# import numpy as np

def calculate_holding_features(seg4, minima_pair):
    """
    基于 minima_pair 划分顶锻和保压阶段，并计算保压阶段的特征。
    # 保压阶段压力和斜率（下降斜率可以反映泄压或泄漏）、保压阶段的时间
    :param seg4: 第四阶段的数据 (DataFrame)，包含 'TIME', 'DISPLACEMENT', 'PRESSURE' 列。
    :param minima_pair: 包含划分点的索引对列表。
    :return: 保压阶段的特征字典。
    """
    holding_features = {}

    # 获取保压阶段的开始索引
    holding_start_idx = minima_pair[0][0] + 1
    holding_time = seg4['TIME'].values[holding_start_idx:]
    holding_displacement = seg4['DISPLACEMENT'].values[holding_start_idx:]
    holding_pressure = seg4['PRESSURE'].values[holding_start_idx:]
    # 计算保压阶段的位移
    total_displacement = holding_displacement[-1] - holding_displacement[0]

    holding_features['holding_time'] = holding_time[-1] - holding_time[0]
    holding_features['holding_total_displacement'] = total_displacement
    holding_features['holding_velocity_mean'] = total_displacement/(holding_time[-1] - holding_time[0])

    for i, (start_idx, end_idx) in enumerate(minima_pair):
        # 保压阶段数据
        holding_time = seg4['TIME'].values[start_idx:end_idx + 1]
        holding_displacement = seg4['DISPLACEMENT'].values[start_idx:end_idx + 1]
        holding_pressure = seg4['PRESSURE'].values[start_idx:end_idx + 1]

        # 计算保压阶段的时间
        holding_duration = holding_time[-1] - holding_time[0]
        # 计算保压阶段的位移
        total_displacement = holding_displacement[-1] - holding_displacement[0]

        # 计算保压阶段的位移速度
        displacement_diff = np.diff(holding_displacement)
        time_diff = np.diff(holding_time)
        velocity = displacement_diff / time_diff
        velocity_mean = np.mean(velocity)

        # 计算保压阶段的做功（压力对位移的积分）
        work_done = simpson(holding_pressure, holding_displacement)

        # 计算保压阶段的压力斜率
        pressure_slope = np.gradient(holding_pressure, holding_time)

        # 保存特征
        holding_features[f'holding_time_{i + 1}'] = holding_duration
        holding_features[f'holding_displacement_{i + 1}'] = total_displacement
        holding_features[f'holding_velocity_mean_{i + 1}'] = velocity_mean
        holding_features[f'holding_work_done_{i + 1}'] = work_done
        holding_features[f'holding_pressure_mean_{i + 1}'] = np.mean(holding_pressure)
        holding_features[f'holding_pressure_slope_mean_{i + 1}'] = np.mean(pressure_slope)


    return holding_features



def calculate_cross_features(upset_feature_dict, stage3_feature_dict):
    """
    计算交叉特征：
    - 顶锻力峰值 / 平均烧化电流
    - 顶锻留量 / 总烧化留量

    :param upset_feature_dict: 顶锻阶段的特征字典，包含 'upset_pressure_peak' 和 'upset_total_displacement'。
    :param stage3_feature_dict: 烧化阶段的特征字典，包含 'stage3_displacement' 和 'stage3_current_variation_coefficient'。
    :return: 包含交叉特征的字典。
    """
    cross_features = {}

    # 顶锻力峰值 / 平均烧化电流
    pressure_peak = upset_feature_dict.get('upset_pressure_peak', 0)
    current_mean = stage3_feature_dict.get('stage3_current_mean', 0)
    if current_mean != 0:
        cross_features['pressure_peak_to_avg_current'] = pressure_peak / current_mean
    else:
        cross_features['pressure_peak_to_avg_current'] = None  # 避免除以 0

    # 顶锻留量 / 总烧化留量
    upset_displacement = upset_feature_dict.get('upset_total_displacement', 0)
    stage3_displacement = stage3_feature_dict.get('stage3_displacement', 0)
    if stage3_displacement != 0:
        cross_features['upset_displacement_to_total_displacement'] = upset_displacement / stage3_displacement
    else:
        cross_features['upset_displacement_to_total_displacement'] = None  # 避免除以 0

    return cross_features



def other_feature_gene():
    return



def feature_gene(segments):
    import numpy as np
    # 各阶段时间、位移量

    seg1 = segments['stage_1']
    stage1_feature_dict = calculate_stage1_features(seg1)

    seg2 = segments['stage_2']
    stage2_feature_dict = calculate_stage2_features(seg2)

    seg3 = segments['stage_3']
    stage3_feature_dict = calculate_stage3_features(seg3)

    seg4 = segments['stage_4']
    window_length = min(31, len(seg4['PRESSURE'].values) // 10 * 2 + 1)  # 确保是奇数
    if window_length > 2:
        pressure_smooth = savgol_filter(seg4['PRESSURE'].values, window_length, 2)
    else:
        pressure_smooth = seg4['PRESSURE'].values.copy()
    pressure_smooth = savgol_filter(seg4['PRESSURE'].values, window_length, 2)
    minima_pair = find_minima_pairs(pressure_smooth, seg4['TIME'].values)


    upset_feature_dict = calculate_upset_features(seg4, minima_pair)
    holding_feature_dict = calculate_holding_features(seg4, minima_pair)
    # stage4_feature_dict = upset_feature_dict + holding_feature_dict
    # stage4_feature_dict = {**upset_feature_dict, **holding_feature_dict}

    cross_feature_dict = calculate_cross_features(upset_feature_dict, stage3_feature_dict)

    feature_dict = {**stage1_feature_dict, **stage2_feature_dict, **stage3_feature_dict, **upset_feature_dict, **holding_feature_dict, **cross_feature_dict}

    return feature_dict


def shuffle_data(tensors, seed=42):
    import random
    """
    对数据进行打乱，同时保证实验的可重复性。

    :param tensors: 包含 (tensor, label) 的列表。
    :param seed: 随机种子，默认为 42。
    :return: 打乱顺序后的数据列表。
    """
    random.seed(seed)  # 设置随机种子
    random.shuffle(tensors)  # 打乱数据
    return tensors



def calculate_explosion_freq_psd_with_plot(time, current, lowcut=20, highcut=200):
    """
    带可视化的过梁爆破频率计算
    """
    import matplotlib.pyplot as plt
    from scipy import signal
    # 设置字体为黑体，解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['STHeiti'] # 替换为 macOS 自带的中文字体
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

    # 计算采样率
    sampling_rate = 1 / (time[1] - time[0])
    
    # 带通滤波
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    high = min(high, 0.999)  # 确保低频率不为零
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_current = signal.filtfilt(b, a, current)
    
    # 计算功率谱密度
    freqs, psd = signal.welch(filtered_current, sampling_rate, nperseg=1024)
    
    # 寻找主导频率
    freq_mask = (freqs >= lowcut) & (freqs <= highcut)
    interested_freqs = freqs[freq_mask]
    interested_psd = psd[freq_mask]
    
    if len(interested_psd) == 0:
        return None
        
    dominant_idx = np.argmax(interested_psd)
    dominant_freq = interested_freqs[dominant_idx]
    dominant_power = interested_psd[dominant_idx]
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 原始信号
    plt.subplot(3, 1, 1)
    plt.plot(time, current, 'b-', alpha=0.7)
    plt.title('原始电流信号')
    plt.ylabel('电流 (A)')
    plt.grid(True)
    
    # 滤波后信号
    plt.subplot(3, 1, 2)
    plt.plot(time, filtered_current, 'r-')
    plt.title('带通滤波后的电流信号 (20-200Hz)')
    plt.ylabel('电流 (A)')
    plt.grid(True)
    
    # 功率谱密度
    plt.subplot(3, 1, 3)
    plt.semilogy(freqs, psd, 'g-', label='全频谱')
    plt.semilogy(interested_freqs, interested_psd, 'b-', linewidth=2, label='感兴趣范围')
    plt.axvline(dominant_freq, color='red', linestyle='--', 
                label=f'主导频率: {dominant_freq:.1f} Hz')
    plt.title('功率谱密度')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率谱密度')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 250)
    
    plt.tight_layout()
    plt.show()
    
    print(f"过梁爆破主导频率: {dominant_freq:.1f} Hz")
    print(f"该频率的相对功率: {dominant_power:.2e}")
    
    return dominant_freq