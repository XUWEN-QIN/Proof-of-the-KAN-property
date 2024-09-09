import numpy as np
import torch
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

# 加载数据集 (选择一小部分数据来测试)
file_path = 'KFvorticity_Re100_N50_T500.npy'
vorticity_data = np.load(file_path)

# 选取一小部分数据来测试，防止内存爆满
# 这里选择第一个实验，前10个时间步
vorticity_data = vorticity_data[0:1, 0:10, :, :]

print(vorticity_data.shape)  # 新数据形状应为 (1, 10, 64, 64)

vorticity_data = torch.tensor(vorticity_data, dtype=torch.float32, device=device)
