import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 绘图功能定义
def plot_time_slices(data, steps, height, width):
    num_steps = len(steps)
    fig, axs = plt.subplots(1, num_steps, figsize=(6 * num_steps, 6))
    for i, time_step in enumerate(steps):
        ax = axs[i]
        ax.imshow(data[time_step].reshape(height, width), cmap='jet')
        ax.set_title(f'Data at Time Step {time_step + 1}')
    plt.show()

def plot_2d_heatmap(data, steps, height, width):
    num_steps = len(steps)
    fig, axs = plt.subplots(1, num_steps, figsize=(6 * num_steps, 6))
    for i, time_step in enumerate(steps):
        ax = axs[i]
        im = ax.imshow(data[time_step].reshape(height, width), cmap='hot')
        ax.set_title(f'Heatmap at Time Step {time_step + 1}')
        plt.colorbar(im, ax=ax)
    plt.show()

def plot_vector_field(data, steps, height, width):
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    num_steps = len(steps)
    fig, axs = plt.subplots(1, num_steps, figsize=(6 * num_steps, 6))
    for i, time_step in enumerate(steps):
        ax = axs[i]
        ax.quiver(x, y, data[time_step].reshape(height, width), data[time_step].reshape(height, width))
        ax.set_title(f'Vector Field at Time Step {time_step + 1}')
    plt.show()

def plot_3d_vector_field(data, steps, height, width):
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    num_steps = len(steps)
    fig = plt.figure(figsize=(6 * num_steps, 6))
    for i, time_step in enumerate(steps):
        ax = fig.add_subplot(1, num_steps, i + 1, projection='3d')
        ax.quiver(x, y, np.zeros_like(x), data[time_step].reshape(height, width), data[time_step].reshape(height, width), np.zeros_like(x))
        ax.set_title(f'3D Vector Field at Time Step {time_step + 1}')
    plt.show()

def plot_3d_surface_and_contour(data, steps, height, width):
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    num_steps = len(steps)
    
    # 3D Surface Plot
    fig, axs = plt.subplots(1, num_steps, subplot_kw={'projection': '3d'}, figsize=(6 * num_steps, 6))
    for i, time_step in enumerate(steps):
        ax = axs[i]
        ax.plot_surface(x, y, data[time_step].reshape(height, width), cmap='viridis')
        ax.set_title(f'Surface at Time Step {time_step + 1}')
    plt.show()

    # 3D Contour Plot
    fig, axs = plt.subplots(1, num_steps, subplot_kw={'projection': '3d'}, figsize=(6 * num_steps, 6))
    for i, time_step in enumerate(steps):
        ax = axs[i]
        ax.contour3D(x, y, data[time_step].reshape(height, width), 50, cmap='viridis')
        ax.set_title(f'Contour at Time Step {time_step + 1}')
    plt.show()

# 假设原始数据为 `predictions` 的指定时间步
specified_steps = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 39]

# 调试时检查形状
print(f"Predictions shape: {predictions.shape}")

# 使用预测数据进行绘图
plot_time_slices(predictions, specified_steps, 64, 64)
plot_2d_heatmap(predictions, specified_steps, 64, 64)
plot_vector_field(predictions, specified_steps, 64, 64)
plot_3d_vector_field(predictions, specified_steps, 64, 64)
plot_3d_surface_and_contour(predictions, specified_steps, 64, 64)
