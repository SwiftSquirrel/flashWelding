# import numpy as np
# from scipy import signal
# import matplotlib.pyplot as plt

# def analyze_low_frequency_envelope(data, sampling_rate=125):
#     """
#     分析低频重复模式的包络
#     """
#     # 步骤1：提取包络线
#     analytic_signal = signal.hilbert(data)
#     amplitude_envelope = np.abs(analytic_signal)
    
#     # 步骤2：重采样到更低采样率（因为模式频率很低）
#     envelope_sampling_rate = 10  # 10Hz足够分析0.333Hz模式
#     num_samples = int(len(amplitude_envelope) * envelope_sampling_rate / sampling_rate)
#     time_original = np.arange(len(amplitude_envelope)) / sampling_rate
#     time_resampled = np.linspace(0, len(amplitude_envelope)/sampling_rate, num_samples)
    
#     envelope_resampled = np.interp(time_resampled, time_original, amplitude_envelope)
    
#     # 步骤3：寻找重复模式
#     autocorr = np.correlate(envelope_resampled, envelope_resampled, mode='full')
#     autocorr = autocorr[len(autocorr)//2:]
    
#     # 找到峰值（对应重复周期）
#     peaks, properties = signal.find_peaks(autocorr[:len(autocorr)//2], 
#                                         height=0.3*np.max(autocorr))
    
#     if len(peaks) > 0:
#         fundamental_period = peaks[0] / envelope_sampling_rate
#         print(f"检测到重复周期: {fundamental_period:.2f} 秒")
    
#     return {
#         "amplitude_envelope": amplitude_envelope,
#         "envelope_resampled": envelope_resampled,
#         "time_resampled": time_resampled
#     }


# # 构造测试数据
# sampling_rate = 125  # 原始采样率
# duration = 10  # 数据时长（秒）
# time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# # 模拟低频正弦波（0.333Hz）叠加高斯噪声
# low_freq_signal = np.sin(2 * np.pi * 0.333 * time)
# noise = np.random.normal(0, 0.1, len(time))
# data = low_freq_signal + noise

# # 调用函数
# result = analyze_low_frequency_envelope(data, sampling_rate)

# # 可视化结果
# plt.figure(figsize=(12, 6))

# # 原始信号
# plt.subplot(3, 1, 1)
# plt.plot(time, data, label="原始信号")
# plt.legend()

# # 包络线
# plt.subplot(3, 1, 2)
# plt.plot(time, result["amplitude_envelope"], label="包络线", color="orange")
# plt.legend()

# # 重采样后的包络线
# plt.subplot(3, 1, 3)
# plt.plot(result["time_resampled"], result["envelope_resampled"], label="重采样包络线", color="green")
# plt.legend()

# plt.tight_layout()
# plt.show()





# # # 使用示例
# # envelope, envelope_resampled, time_env = analyze_low_frequency_envelope(data)



import numpy as np
from scipy import signal
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def analyze_low_frequency_envelope(data, sampling_rate=125, envelope_sampling_rate=20):
    """
    分析低频重复模式的包络
    """
    # 步骤1：提取包络线
    analytic_signal = signal.hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    
    # 平滑包络线
    amplitude_envelope = savgol_filter(amplitude_envelope, window_length=51, polyorder=3)
    
    # 步骤2：重采样到更低采样率
    num_samples = int(len(amplitude_envelope) * envelope_sampling_rate / sampling_rate)
    time_original = np.arange(len(amplitude_envelope)) / sampling_rate
    time_resampled = np.linspace(0, len(amplitude_envelope)/sampling_rate, num_samples)
    
    envelope_resampled = np.interp(time_resampled, time_original, amplitude_envelope)
    
    # 步骤3：寻找重复模式
    autocorr = np.correlate(envelope_resampled, envelope_resampled, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # 找到峰值（对应重复周期）
    peaks, properties = signal.find_peaks(autocorr[:len(autocorr)//2], 
                                        height=0.3*np.max(autocorr))
    
    if len(peaks) > 0:
        fundamental_period = peaks[0] / envelope_sampling_rate
        print(f"检测到重复周期: {fundamental_period:.2f} 秒")
    
    return {
        "amplitude_envelope": amplitude_envelope,
        "envelope_resampled": envelope_resampled,
        "time_resampled": time_resampled
    }

# 构造测试数据
sampling_rate = 125  # 原始采样率
duration = 10  # 数据时长（秒）
time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# 模拟低频正弦波（0.333Hz）叠加高斯噪声和高频分量
low_freq_signal = np.sin(2 * np.pi * 0.333 * time)
high_freq_signal = 0.1 * np.sin(2 * np.pi * 5 * time)  # 添加高频分量
noise = np.random.normal(0, 0.1, len(time))
data = low_freq_signal + high_freq_signal + noise

# 调用函数
result = analyze_low_frequency_envelope(data, sampling_rate)

# 可视化结果
plt.figure(figsize=(12, 6))

# 原始信号
plt.subplot(3, 1, 1)
plt.plot(time, data, label="原始信号")
plt.legend()

# 包络线
plt.subplot(3, 1, 2)
plt.plot(time, result["amplitude_envelope"], label="包络线", color="orange")
plt.legend()

# 重采样后的包络线
plt.subplot(3, 1, 3)
plt.plot(result["time_resampled"], result["envelope_resampled"], label="重采样包络线", color="green")
plt.legend()

plt.tight_layout()
plt.show()