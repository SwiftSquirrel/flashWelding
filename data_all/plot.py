import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 文件夹路径
input_folder = "/home/dawn/Documents/HJ/data_all/202606_Plot/good"  # 数据文件夹
output_folder = "/home/dawn/Documents/HJ/data_all/202606_Plot/good_Plot"  # 保存图像的文件夹

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)


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
    if i > 10:
        continue
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
            
            if 'HJ/data_all/202606' in file_path:
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
            
            # 保存图像
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_plot.png")
            plt.savefig(output_path)
            plt.close()
            
            print(f"文件 {file_name} 的图像已保存到 {output_path}")
        
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")