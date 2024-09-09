import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确定性设置
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
torch.use_deterministic_algorithms(True)

# 常量设置
initial_lr = 0.002
epochs = 1
batch_size = 832
l2_lambda = 1e-5

# 加载 .mat 文件
data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')
a = data['a']
u = data['u']
t = data['t']

# 打印数据形状以了解数据维度
print("a shape:", a.shape)
print("u shape:", u.shape)
print("t shape:", t.shape)

# 初始化条件和边界条件
initial_conditions = u[:, 0, :, :].astype(np.float32)
boundary_conditions = u[:, -1, :, :].astype(np.float32)
print("Initial conditions shape:", initial_conditions.shape)
print("Boundary conditions shape:", boundary_conditions.shape)

# 转换为张量
initial_conditions_tensor = torch.tensor(initial_conditions)
boundary_conditions_tensor = torch.tensor(boundary_conditions).to(initial_conditions_tensor.device)

# 展平初始条件和平滑系数
initial_conditions_flat = initial_conditions_tensor.reshape(initial_conditions_tensor.shape[0], -1)
boundary_conditions_flat = boundary_conditions_tensor.reshape(boundary_conditions_tensor.shape[0], -1)

# 对 `a` 进行展平
a_flat = a.reshape(a.shape[0], -1).astype(np.float32)
a_tensor = torch.tensor(a_flat)

# 对 `t` 进行展平以匹配批次时间信息
t_flat = np.repeat(t, u.shape[0], axis=0).reshape(u.shape[0], -1).astype(np.float32)
t_tensor = torch.tensor(t_flat)

# 将初始条件、时间特征和平滑系数拼接作为输入
input_features = torch.cat((initial_conditions_flat, a_tensor, t_tensor), dim=1)

# 绘图功能定义
def plot_time_slices(data, steps, height, width):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    for i, time_step in enumerate(steps):
        ax = axs[i // 5, i % 5]
        ax.imshow(data[time_step].reshape(height, width), cmap='jet')
        ax.set_title(f'Data at Time Step {time_step + 1}')
    plt.show()

def plot_2d_heatmap(data, steps, height, width):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    for i, time_step in enumerate(steps):
        ax = axs[i // 5, i % 5]
        im = ax.imshow(data[time_step].reshape(height, width), cmap='hot')
        ax.set_title(f'Heatmap at Time Step {time_step + 1}')
        plt.colorbar(im, ax=ax)
    plt.show()

def plot_vector_field(data, steps, height, width):
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    for i, time_step in enumerate(steps):
        ax = axs[i // 5, i % 5]
        ax.quiver(x, y, data[time_step].reshape(height, width), data[time_step].reshape(height, width))
        ax.set_title(f'Vector Field at Time Step {time_step + 1}')
    plt.show()

def plot_3d_vector_field(data, steps, height, width):
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    fig = plt.figure(figsize=(20, 8))
    for i, time_step in enumerate(steps):
        ax = fig.add_subplot(2, 5, i + 1, projection='3d')
        ax.quiver(x, y, np.zeros_like(x), data[time_step].reshape(height, width), data[time_step].reshape(height, width), np.zeros_like(x))
        ax.set_title(f'3D Vector Field at Time Step {time_step + 1}')
    plt.show()

def plot_3d_surface_and_contour(data, steps, height, width):
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    fig, axs = plt.subplots(2, 5, subplot_kw={'projection': '3d'}, figsize=(20, 8))
    for i, time_step in enumerate(steps):
        ax = axs[i // 5, i % 5]
        ax.plot_surface(x, y, data[time_step].reshape(height, width), cmap='viridis')
        ax.set_title(f'Surface at Time Step {time_step + 1}')
    plt.show()

    fig, axs = plt.subplots(2, 5, subplot_kw={'projection': '3d'}, figsize=(20, 8))
    for i, time_step in enumerate(steps):
        ax = axs[i // 5, i % 5]
        ax.contour3D(x, y, data[time_step].reshape(height, width), 50, cmap='viridis')
        ax.set_title(f'Contour at Time Step {time_step + 1}')
    plt.show()

# 设置要绘制的时间步（前10个时间步的数据）
steps = list(range(10))

# 假设原始数据为 `u` 的前10个时间步
original_data = u[0, :, :, :]

# 调试时检查形状
print(f"Original data shape: {original_data.shape}")

# 使用原始数据进行绘图
plot_time_slices(original_data, steps, u.shape[2], u.shape[3])
plot_2d_heatmap(original_data, steps, u.shape[2], u.shape[3])
plot_vector_field(original_data, steps, u.shape[2], u.shape[3])
plot_3d_vector_field(original_data, steps, u.shape[2], u.shape[3])
plot_3d_surface_and_contour(original_data, steps, u.shape[2], u.shape[3])
