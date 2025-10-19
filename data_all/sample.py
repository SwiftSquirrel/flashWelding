import os
import pandas as pd

# generate samples

# 文件夹路径
input_folder = "data_all/U75VH/valid_N"  # 数据文件夹
output_folder = "data_all/U75VH_sampled/valid_N"  # 保存采样后的文件夹

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)


# 遍历文件夹中的所有文件
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
    # 检查是否为文件（可以根据需要调整文件类型，如 .csv）
    if os.path.isfile(file_path) and (file_name.endswith('.CSV') or file_name.endswith('.csv')):
        # 使用 pandas 读取数据
        data = pd.read_csv(file_path)

        # 检查是否有至少四列
        if data.shape[1] < 4:
            print(f"文件 {file_name} 列数不足，跳过...")
            continue

        # 检查数据长度是否大于5000
        if len(data) > 5000:
            # 遍历多个采样间隔
            for interval in range(2, 6):  # 生成多个采样间隔，factor 控制间隔倍数
                # interval = (len(data) // 2600) * factor
                # if interval == 0:
                #     interval = 1  # 确保间隔至少为 1
                sampled_data = data.iloc[::interval]  # 基于采样间隔进行下采样

                # 如果采样后长度大于2600，保存文件
                if len(sampled_data) > 2600:
                    output_file_name = f"{os.path.splitext(file_name)[0]}_sample-{interval}.csv"
                    output_file_path = os.path.join(output_folder, output_file_name)
                    sampled_data.to_csv(output_file_path, index=False)
                    print(f"文件 {file_name} 已采样并保存为 {output_file_name}")
                else:
                    print(f"文件 {file_name} 使用间隔 {interval} 采样后长度不足 2600，跳过...")
        else:
            print(f"文件 {file_name} 长度不足 5000，无需采样，跳过...")