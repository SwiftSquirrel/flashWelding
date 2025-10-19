
# # import numpy as np
# # from scipy.interpolate import interp1d
# # from PIL import Image


# # def time_series_to_image(time, series_data, img_size=(256, 256)):
# #     """
# #     将时间序列数据转化为固定尺寸的图像。
    
# #     参数:
# #     - time: 时间向量。
# #     - series_data: 包含多个时间序列数据的字典，如{'current': [], 'pressure': [], 'displacement': []}。
# #     - img_size: 输出图像的目标尺寸，默认是(256, 256)。
    
# #     返回:
# #     - PIL Image对象。
# #     """
# #     # 目标长度，对应于图像的宽度
# #     target_length = img_size[0]
# #     channels = []

# #     for name, data in series_data.items():
# #         # 插值到目标长度
# #         f = interp1d(time, data, kind='linear')
# #         new_x = np.linspace(time.min(), time.max(), target_length)
# #         interpolated_data = f(new_x)

# #         # 归一化到0-255区间
# #         normalized_data = ((interpolated_data - np.min(interpolated_data)) /
# #                            (np.max(interpolated_data) - np.min(interpolated_data))) * 255

# #         channels.append(normalized_data)

# #     # 转换为图像
# #     img_array = np.array(channels).astype(np.uint8)
# #     img_array = np.transpose(img_array, (1, 0))  # 调整形状以匹配PIL Image期望的格式
# #     img = Image.fromarray(img_array, mode='RGB')  # 使用'RGB'模式创建图像

# #     # 调整大小
# #     img_resized = img.resize((img_size[1], img_size[0]))

# #     return img_resized


# # # 示例数据
# # time = np.linspace(0, 10, 1000)
# # series_data = {
# #     'current': np.sin(time),
# #     'pressure': np.cos(time),
# #     'displacement': np.tan(time)
# # }

# # # 转换并保存图像
# # img = time_series_to_image(time, series_data)
# # # img.save('output.png')


# import numpy as np
# from PIL import Image

# # 示例数据
# time = np.linspace(0, 10, 1000)
# series_data = {
#     'current': np.sin(time),
#     'pressure': np.cos(time),
#     'displacement': np.tan(time)
# }


# def time_series_to_image(time, series_data):
#     # 归一化数据到[0, 255]
#     data_normalized = {}
#     for key in series_data.keys():
#         data_normalized[key] = (series_data[key] - min(series_data[key])) / \
#             (max(series_data[key]) - min(series_data[key]))
#         data_normalized[key] *= 255
#         data_normalized[key] = data_normalized[key].astype(np.uint8)

#     # 创建一个空的RGB图像数组
#     height = len(time)  # 图像的高度等于时间序列的长度
#     width = 1  # 假设宽度为1，因为我们将每个时间点映射为一个像素
#     rgb_image_array = np.zeros((height, width, 3), dtype=np.uint8)

#     # 将归一化后的数据填充到图像数组中
#     for i in range(height):
#         rgb_image_array[i, :, 0] = data_normalized['pressure'][i]  # R通道对应于压力
#         rgb_image_array[i, :, 1] = data_normalized['current'][i]   # G通道对应于电流
#         # B通道对应于位移
#         rgb_image_array[i, :, 2] = data_normalized['displacement'][i]

#     # 如果需要，可以通过重复像素来放大图像以便查看
#     # scale_factor = 10  # 放大倍数
#     # rgb_image_array_large = np.repeat(rgb_image_array, scale_factor, axis=1)
#     # rgb_image_array_large = np.repeat(
#     #     rgb_image_array_large, scale_factor, axis=0)

#     # 创建RGB图像
#     img_rgb = Image.fromarray(rgb_image_array, mode='RGB')

#     return img_rgb


# # 转换并保存图像
# img = time_series_to_image(time, series_data)
# img.save('output.png')


import numpy as np
from PIL import Image

# 示例数据：(1000, 3) 的时间序列，每列代表一个变量
time = np.linspace(0, 10, 100)
series_data = {
    'current': np.sin(time),
    'pressure': np.cos(time),
    'displacement': np.tan(time)
}

# 构造 (1000, 3) 的数组
data = np.column_stack([
    series_data['current'],
    series_data['pressure'],
    series_data['displacement']
])

# 归一化到 [0, 255]，并转为 uint8（图像格式要求）
data_normalized = ((data - data.min()) / (data.max() -
                   data.min()) * 255).astype(np.uint8)

# 转换为灰度图像
# 注意：PIL 期望输入 shape 为 (H, W)，这里 (1000, 3) 就是 H=1000, W=3
img = Image.fromarray(data_normalized.T, mode='L')  # 'L' 表示 8-bit 灰度图

# 保存图像
img.save('output.png')

# 可选：查看图像（如果环境支持）
# img.show()

print("✅ 图像已保存为 output.png，尺寸为 1000×3 像素")
