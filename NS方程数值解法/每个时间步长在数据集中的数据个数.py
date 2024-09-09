import scipy.io
import numpy as np

# 加载数据集
data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')

# 获取时间步数据
time_key = 'a'
time_data = data[time_key]

# 打印数据形状
print(f"数据 '{time_key}' 的形状: {time_data.shape}")

# 计算每个大时间步包含的小时间步数量
num_large_time_steps = 20
num_small_time_steps_per_large = time_data.shape[0] // num_large_time_steps

# 输出每个小时间步的数据个数
for large_step in range(num_large_time_steps):
    start_idx = large_step * num_small_time_steps_per_large
    end_idx = (large_step + 1) * num_small_time_steps_per_large
    
    print(f"大时间步 {large_step + 1}:")
    for small_step in range(start_idx, end_idx):
        data_points_count = np.prod(time_data[small_step].shape)
        print(f"  小时间步 {small_step + 1}: 数据个数 = {data_points_count}")

# 确认每个小时间步的数据个数是否一致
all_counts = [np.prod(time_data[i].shape) for i in range(time_data.shape[0])]
if len(set(all_counts)) == 1:
    print(f"所有小时间步的数据个数都是一致的，每个小时间步的数据个数 = {all_counts[0]}")
else:
    print("不同小时间步的数据个数不一致，请具体检查。")
