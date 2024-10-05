import torch
import os
import numpy as np
import scipy.io
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置设备，优先使用GPU，如果没有就使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置PyTorch使用确定性算法以确保结果的可重复性
torch.use_deterministic_algorithms(True)

# 配置CUDA的工作空间以避免非确定性行为
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

print("Using device:", device)

# 加载数据集，数据存储在.mat文件中
data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')

# 将数据转换为PyTorch张量并移动到指定设备上
a = torch.tensor(data['a'], dtype=torch.float32).to(device)  # 初始条件
u = torch.tensor(data['u'], dtype=torch.float32).to(device)  # 速度场
t = torch.tensor(data['t'], dtype=torch.float32).to(device)  # 时间步长

# 打印数据的形状以检查是否加载正确
print("Shape of a:", a.shape)
print("Shape of u:", u.shape)
print("Shape of t:", t.shape)

# 使用reshape方法将数据展平，便于输入到神经网络中
X = a.reshape(1200, -1)  # 将 a 展平为 (1200, 4096)
U = u[:, :, :, 0].reshape(1200, -1)  # 将 u 的第一个时间步展平为 (1200, 4096)

# 将数据集拆分为80%的训练集和20%的测试集
split_index = int(0.8 * X.size(0))
X_train, X_test = X[:split_index], X[split_index:]
U_train, U_test = U[:split_index], U[split_index:]

# 定义KAN模型，继承自torch.nn.Module
class KAN(torch.nn.Module):
    def __init__(self, width, grid, k, grid_eps, noise_scale):
        super(KAN, self).__init__()
        # 使用Sequential定义一系列线性层和激活函数
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(width[0], width[1]),
            torch.nn.Tanh(),
            torch.nn.Linear(width[1], width[2]),
            torch.nn.Tanh(),
            torch.nn.Linear(width[2], width[3])
        )

    def forward(self, x):
        # 前向传播函数
        return self.layers(x)

# 创建KAN模型实例，并将其移动到指定设备上
model = KAN(width=[4096, 1024, 512, 4096], grid=10, k=3, grid_eps=1.0, noise_scale=0.25).to(device)

# 定义均方误差损失函数
def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

# 训练过程
def train():
    # 使用LBFGS优化器
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1,
                                  history_size=10, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-32, tolerance_change=1e-32)

    steps = 100  # 训练步数
    pbar = tqdm(range(steps), desc='Training Progress')  # 进度条

    for step in pbar:
        # 定义闭包函数，用于计算损失和反向传播
        def closure():
            optimizer.zero_grad()  # 清零梯度
            y_pred = model(X_train)  # 预测
            loss = mse_loss(y_pred, U_train)  # 计算损失
            loss.backward()  # 反向传播
            return loss

        optimizer.step(closure)  # 执行优化步骤
        current_loss = closure().item()  # 获取当前损失值
        if step % 10 == 0:
            pbar.set_description("Step: %d | Loss: %.6f" % (step, current_loss))  # 更新进度条描述

        writer.add_scalar('Loss/train', current_loss, step)  # 记录损失到TensorBoard

# 绘制时间切片图
def plot_time_slices(actual_list, predicted_list, indices, output_dir):
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i, idx in enumerate(indices):
        actual = actual_list[i].cpu().numpy().flatten()  # 获取实际值并展平
        predicted = predicted_list[i].cpu().numpy().flatten()  # 获取预测值并展平

        axes[0, i].plot(actual, label='Actual', linestyle='--')  # 绘制实际值
        axes[0, i].set_title(f'Actual {idx}')
        axes[0, i].axis('off')

        axes[1, i].plot(predicted, label='Predicted', linestyle='-')  # 绘制预测值
        axes[1, i].set_title(f'Predicted {idx}')
        axes[1, i].axis('off')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_slices.png'))  # 保存图像
    plt.close()

# 绘制2D热图
def plot_2d_heatmap(actual_list, predicted_list, indices, output_dir):
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i, idx in enumerate(indices):
        actual = actual_list[i].cpu().numpy().reshape(64, 64)  # 重新形状为64x64
        predicted = predicted_list[i].cpu().numpy().reshape(64, 64)

        axes[0, i].imshow(actual, cmap='viridis', aspect='auto')  # 显示实际值热图
        axes[0, i].set_title(f'Actual {idx}')
        axes[0, i].axis('off')

        axes[1, i].imshow(predicted, cmap='viridis', aspect='auto')  # 显示预测值热图
        axes[1, i].set_title(f'Predicted {idx}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2d_heatmap.png'))  # 保存图像
    plt.close()

# 绘制矢量场
def plot_vector_field(actual_list, predicted_list, indices, output_dir):
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i, idx in enumerate(indices):
        actual = actual_list[i].cpu().numpy().reshape(64, 64)
        predicted = predicted_list[i].cpu().numpy().reshape(64, 64)

        x, y = np.meshgrid(np.arange(64), np.arange(64))

        # 绘制实际值的矢量场
        axes[0, i].quiver(x, y, actual, actual)
        axes[0, i].set_title(f'Actual {idx}')
        axes[0, i].axis('off')

        # 绘制预测值的矢量场
        axes[1, i].quiver(x, y, predicted, predicted)
        axes[1, i].set_title(f'Predicted {idx}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vector_field.png'))
    plt.close()

# 绘制3D矢量场
def plot_3d_vector_field(actual_list, predicted_list, indices, output_dir):
    fig = plt.figure(figsize=(20, 8))
    for i, idx in enumerate(indices):
        actual = actual_list[i].cpu().numpy().reshape(64, 64)
        predicted = predicted_list[i].cpu().numpy().reshape(64, 64)

        x, y = np.meshgrid(np.arange(64), np.arange(64))

        # 绘制实际值的3D矢量场
        ax = fig.add_subplot(2, 10, i + 1, projection='3d')
        ax.quiver(x, y, actual, actual, actual, actual)
        ax.set_title(f'Actual {idx}')
        ax.axis('off')

        # 绘制预测值的3D矢量场
        ax = fig.add_subplot(2, 10, i + 11, projection='3d')
        ax.quiver(x, y, predicted, predicted, predicted, predicted)
        ax.set_title(f'Predicted {idx}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_vector_field.png'))
    plt.close()

# 绘制3D表面和轮廓图
def plot_3d_surface_and_contour(actual_list, predicted_list, indices, output_dir):
    fig = plt.figure(figsize=(20, 8))
    for i, idx in enumerate(indices):
        actual = actual_list[i].cpu().numpy().reshape(64, 64)
        predicted = predicted_list[i].cpu().numpy().reshape(64, 64)

        x, y = np.meshgrid(np.arange(64), np.arange(64))

        # 绘制实际值的3D表面和轮廓图
        ax = fig.add_subplot(2, 10, i + 1, projection='3d')
        ax.plot_surface(x, y, actual, cmap='viridis', edgecolor='none')
        ax.contour(x, y, actual, zdir='z', offset=-2, cmap='viridis')
        ax.set_title(f'Actual {idx}')
        ax.axis('off')

        # 绘制预测值的3D表面和轮廓图
        ax = fig.add_subplot(2, 10, i + 11, projection='3d')
        ax.plot_surface(x, y, predicted, cmap='viridis', edgecolor='none')
        ax.contour(x, y, predicted, zdir='z', offset=-2, cmap='viridis')
        ax.set_title(f'Predicted {idx}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_surface_and_contour.png'))
    plt.close()

# 预测过程
def predict():
    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    with torch.no_grad():  # 禁用梯度计算，提高预测效率
        y_pred = model(X_test)  # 对测试集进行预测

        # 选择中间10个时间刻进行绘图
        num_samples = X_test.size(0)
        indices = np.linspace(0, num_samples - 1, num=10, dtype=int)

        actual_list = [U_test[idx] for idx in indices]
        predicted_list = [y_pred[idx] for idx in indices]

        plot_time_slices(actual_list, predicted_list, indices, output_dir)  # 绘制时间切片图
        plot_2d_heatmap(actual_list, predicted_list, indices, output_dir)  # 绘制2D热图
        plot_vector_field(actual_list, predicted_list, indices, output_dir)  # 绘制矢量场
        plot_3d_vector_field(actual_list, predicted_list, indices, output_dir)  # 绘制3D矢量场
        plot_3d_surface_and_contour(actual_list, predicted_list, indices, output_dir)  # 绘制3D表面和轮廓图

# 记录和保存结果
writer = SummaryWriter()  # 创建TensorBoard记录器
start_time = time.time()  # 记录开始时间
train()  # 开始训练
predict()  # 进行预测
end_time = time.time()  # 记录结束时间
training_duration = end_time - start_time  # 计算训练时长

# 将训练时长记录到TensorBoard
writer.add_text('[KAN_NS]Training/Duration', f'Total training time: {training_duration:.2f} seconds')

writer.close()  # 关闭TensorBoard记录器
