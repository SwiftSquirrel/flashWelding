import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress



# 文件夹路径
# input_folder = "/Users/dawn/PYY/flashWelding/data_all/202606/good"  # 数据文件夹
# output_folder = "/Users/dawn/PYY/flashWelding/data_all/202606/202606_good_Plot"  # 保存图像的文件夹
input_folder = "/Users/dawn/PYY/flashWelding/U75VH/P"  # 数据文件夹
output_folder = "/Users/dawn/PYY/flashWelding/U75VH/P_Plot"  # 保存图像的文件夹
# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)



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
                slope, _, _, _, _ = linregress(time[start:end], displacement[start:end])
                slopes[i] = slope
            else:
                slopes[i] = 0
        return slopes

    # 对 time 进行线性化
    linearized_time = np.linspace(time[0], time[-1], len(time))
    time = linearized_time  # 替代原始 time

    window_size = 11  # 滑动窗口大小
    slope = calculate_slope_with_window(time, displacement, window_size)
    abs_slope = np.abs(slope)



    phases = {}

    # 规则 1: 在10-30s内，第一个斜率数值超过3的点
    range_10_30 = (time >= 10) & (time <= 30)
    point1_index = np.argmax((slope > 3) & range_10_30)
    phases['Phase 1'] = time[point1_index] if slope[point1_index] > 3 else None

    # 规则 2: 在100s之前，最后一个斜率绝对值超过3的点
    range_before_100 = time < 90
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



# # ==================== 特征提取函数（与训练时完全一致）====================
# def calculate_displacement_slope_and_detect_phases(time, displacement):
#     """计算位移斜率并按规则划分阶段."""
#     # 插值以增加样本点
#     interp_factor = 10  # 插值因子，表示每两个点之间插值的点数
#     interp_time = np.linspace(time[0], time[-1], len(time) * interp_factor)
#     interp_func = interp1d(time, displacement, kind='linear')  # 线性插值
#     interp_displacement = interp_func(interp_time)


#     # 计算插值后的斜率
#     slope = np.gradient(interp_displacement, interp_time)
#     abs_slope = np.abs(slope)

#     phases = {}

#     # 规则 1: 在10-30s内，第一个斜率数值超过3的点
#     range_10_30 = (interp_time >= 10) & (interp_time <= 30)
#     point1_index = np.argmax((slope > 3) & range_10_30)
#     phases['Phase 1'] = interp_time[point1_index] if slope[point1_index] > 3 else None

#     # 规则 2: 在100s之前，最后一个斜率绝对值超过5的点
#     range_before_100 = interp_time < 100
#     valid_indices_before_100 = np.where((abs_slope > 5) & range_before_100)[0]
#     point2_index = valid_indices_before_100[-1] if len(valid_indices_before_100) > 0 else None
#     phases['Phase 2'] = interp_time[point2_index] if point2_index is not None else None

#     # 规则 3: 在100s之后，第一个斜率绝对值超过40的点
#     range_after_100 = interp_time > 100
#     valid_indices_after_100 = np.where((abs_slope > 40) & range_after_100)[0]
#     point3_index = valid_indices_after_100[0] if len(valid_indices_after_100) > 0 else None
#     phases['Phase 3'] = interp_time[point3_index] if point3_index is not None else None


#     # slope = np.gradient(displacement, time)
#     # abs_slope = np.abs(slope)

#     # phases = {}

#     # # 规则 1: 在10-30s内，第一个斜率数值超过3的点
#     # range_10_30 = (time >= 10) & (time <= 30)
#     # point1_index = np.argmax((slope > 3) & range_10_30)
#     # phases['Phase 1'] = time[point1_index] if slope[point1_index] > 3 else None

#     # # 规则 2: 在100s之前，最后一个斜率绝对值超过5的点
#     # range_before_100 = time < 100
#     # valid_indices_before_100 = np.where((abs_slope > 5) & range_before_100)[0]
#     # point2_index = valid_indices_before_100[-1] if len(valid_indices_before_100) > 0 else None
#     # phases['Phase 2'] = time[point2_index] if point2_index is not None else None

#     # # 规则 3: 在100s之后，第一个斜率绝对值超过40的点
#     # range_after_100 = time > 100
#     # valid_indices_after_100 = np.where((abs_slope > 40) & range_after_100)[0]
#     # point3_index = valid_indices_after_100[0] if len(valid_indices_after_100) > 0 else None
#     # phases['Phase 3'] = time[point3_index] if point3_index is not None else None

#     return phases



def func(df):
    time = df.tail(1)['TIME'].values[0]

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

    return phases


i = 0
# 遍历文件夹中的所有文件
for file_name in os.listdir(input_folder):
    
    # target_list = [
    #                 '071T26061002',
    #                 '071T26061823',
    #                 '071T26060516',
    #                 '071T26060502',
    #                 '071T26061510',
    #                 '071T26060526',
    #                 '071T26060801',
    #                 '071T26061807',
    #                 '071T26061412'
    #             ]

    # flag = False
    # for target in target_list:
    #     if target in file_name:
    #         flag = True
    #         break
    # if not flag:
    #     continue
    
    # if '071T26061406' not in file_name:
    #     continue

    # if i > 10:
    #     continue
    i += 1
    file_path = os.path.join(input_folder, file_name)
    
    # 检查是否为文件（可以根据需要调整文件类型，如 .csv）
    if os.path.isfile(file_path) and (file_name.endswith('.CSV') or file_name.endswith('.csv')):
        try:
            # 使用 pandas 读取数据
            data = pd.read_csv(file_path)

            # 检查是否有至少四列
            if data.shape[1] < 4:
                print(f"文件 {file_name} 列数不足，跳过...")
                continue
            
            if 'data_all/202606' in file_path:
                data = data.rename(
                    columns={'时间(s)': 'TIME', '压力': 'PRESSURE', '电流': 'CURRENT', '位移(mm)': 'DISPLACEMENT'})
            phases = calculate_displacement_slope_and_detect_phases(data['TIME'].values, data['DISPLACEMENT'].values)

            # 提取列名
            time_col = data.columns[0]  # 第一列为时间
            col1, col2, col3 = data.columns[1:4]  # 后三列
            
            # 创建 3x1 子图
            fig, axes = plt.subplots(3, 1, figsize=(8, 12))
            
            # 绘制每列数据
            axes[0].plot(data[time_col], data[col1], label=col1, color='b')
            axes[0].set_title(f"{col1} vs {time_col}")
            axes[0].set_xlabel(time_col)
            axes[0].set_ylabel(col1)
            axes[0].legend()
            
            axes[1].plot(data[time_col], data[col2], label=col2, color='g')
            axes[1].set_title(f"{col2} vs {time_col}")
            axes[1].set_xlabel(time_col)
            axes[1].set_ylabel(col2)
            axes[1].legend()
            
            axes[2].plot(data[time_col], data[col3], label=col3, color='r')
            axes[2].set_title(f"{col3} vs {time_col}")
            axes[2].set_xlabel(time_col)
            axes[2].set_ylabel(col3)
            axes[2].legend()


            # 在每个子图中添加分界点的竖线
            for phase_name, phase_time in phases.items():
                if phase_time is not None:
                    for ax in axes:
                        ax.axvline(x=phase_time, color='k', linestyle='--', label=phase_name)


            # 调整布局
            plt.tight_layout()
            
            # # 保存图像
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_plot.png")
            plt.savefig(output_path)
            # plt.show()
            plt.close()
            
            print(f"文件 {file_name} 的图像已保存到 {output_path}")
        
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")


