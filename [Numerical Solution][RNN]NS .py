import torch
import os
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

print("Using device:", device)

# 加载数据集
data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')
a = torch.tensor(data['a'], dtype=torch.float32).to(device)  # 初始条件
u = torch.tensor(data['u'], dtype=torch.float32).to(device)  # 速度场
t = torch.tensor(data['t'], dtype=torch.float32).to(device)  # 时间步长

# 检查形状
print("Shape of a:", a.shape)
print("Shape of u:", u.shape)
print("Shape of t:", t.shape)

# 使用 reshape 而不是 view
X = a.reshape(1200, -1)  # 将 a 展平为 (1200, 4096)
U = u[:, :, :, 0].reshape(1200, -1)  # 将 u 的第一个时间步展平为 (1200, 4096)

# 数据集拆分
train_size = int(0.8 * X.size(0))  # 前80%用于训练
X_train = X[:train_size]
U_train = U[:train_size]
X_test = X[train_size:]  # 后20%用于预测
U_test = U[train_size:]

# 定义RNN模型
class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 假设输入是展平后的初始条件
input_size = X_train.shape[1]
hidden_size = 512
output_size = U_train.shape[1]
model = RNNModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)

# 定义损失函数
def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

# 训练过程
def train_with_data():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    steps = 100
    pbar = tqdm(range(steps), desc='Training Progress')

    for step in pbar:
        optimizer.zero_grad()
        y_pred = model(X_train.unsqueeze(1))  # 将 X_train 加一个时间维度
        loss = mse_loss(y_pred, U_train)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if step % 10 == 0:
            pbar.set_description("Step: %d | Loss: %.6f" % (step, current_loss))

        writer.add_scalar('Loss/train', current_loss, step)

# 预测过程
def predict():
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.unsqueeze(1))
    return y_pred

# 绘制时间切片图
def plot_time_slices(y_true, y_pred, time_indices, output_dir):
    fig, axes = plt.subplots(2, len(time_indices), figsize=(20, 4))
    for i, idx in enumerate(time_indices):
        actual = y_true[idx].cpu().numpy().flatten()
        predicted = y_pred[idx].cpu().numpy().flatten()

        axes[0, i].plot(actual, label='Actual', linestyle='--')
        axes[0, i].set_title(f'Actual {idx}')
        axes[0, i].axis('off')

        axes[1, i].plot(predicted, label='Predicted', linestyle='-')
        axes[1, i].set_title(f'Predicted {idx}')
        axes[1, i].axis('off')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_slices.png'))
    plt.close()


# 绘制2D热图
def plot_2d_heatmap(y_true, y_pred, time_indices, output_dir):
    fig, axes = plt.subplots(2, len(time_indices), figsize=(20, 4))
    for i, idx in enumerate(time_indices):
        actual = y_true[idx].cpu().numpy().reshape(64, 64)
        predicted = y_pred[idx].cpu().numpy().reshape(64, 64)

        axes[0, i].imshow(actual, cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Actual {idx}')
        axes[0, i].axis('off')

        axes[1, i].imshow(predicted, cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'Predicted {idx}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2d_heatmap.png'))
    plt.close()


# 绘制矢量场
def plot_vector_field(y_true, y_pred, time_indices, output_dir):
    fig, axes = plt.subplots(2, len(time_indices), figsize=(20, 4))
    X, Y = np.meshgrid(np.arange(64), np.arange(64))
    for i, idx in enumerate(time_indices):
        actual = y_true[idx].cpu().numpy().reshape(64, 64)
        predicted = y_pred[idx].cpu().numpy().reshape(64, 64)

        axes[0, i].quiver(X, Y, actual, actual)
        axes[0, i].set_title(f'Actual {idx}')
        axes[0, i].axis('off')

        axes[1, i].quiver(X, Y, predicted, predicted)
        axes[1, i].set_title(f'Predicted {idx}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vector_field.png'))
    plt.close()


# 绘制3D矢量场
def plot_3d_vector_field(y_true, y_pred, time_indices, output_dir):
    fig = plt.figure(figsize=(20, 8))
    X, Y = np.meshgrid(np.arange(64), np.arange(64))
    for i, idx in enumerate(time_indices):
        actual = y_true[idx].cpu().numpy().reshape(64, 64)
        predicted = y_pred[idx].cpu().numpy().reshape(64, 64)

        ax = fig.add_subplot(2, len(time_indices), i + 1, projection='3d')
        ax.quiver(X, Y, actual, actual, actual, actual)
        ax.set_title(f'Actual {idx}')
        ax.axis('off')

        ax = fig.add_subplot(2, len(time_indices), len(time_indices) + i + 1, projection='3d')
        ax.quiver(X, Y, predicted, predicted, predicted, predicted)
        ax.set_title(f'Predicted {idx}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_vector_field.png'))
    plt.close()


# 绘制3D表面和等高线
def plot_3d_surface_and_contour(y_true, y_pred, time_indices, output_dir):
    fig = plt.figure(figsize=(20, 8))
    X, Y = np.meshgrid(np.arange(64), np.arange(64))
    for i, idx in enumerate(time_indices):
        actual = y_true[idx].cpu().numpy().reshape(64, 64)
        predicted = y_pred[idx].cpu().numpy().reshape(64, 64)

        ax = fig.add_subplot(2, len(time_indices), i + 1, projection='3d')
        ax.plot_surface(X, Y, actual, cmap='viridis', edgecolor='none')
        ax.contour(X, Y, actual, zdir='z', offset=np.min(actual), cmap='viridis')
        ax.set_title(f'Actual {idx}')
        ax.axis('off')

        ax = fig.add_subplot(2, len(time_indices), len(time_indices) + i + 1, projection='3d')
        ax.plot_surface(X, Y, predicted, cmap='viridis', edgecolor='none')
        ax.contour(X, Y, predicted, zdir='z', offset=np.min(predicted), cmap='viridis')
        ax.set_title(f'Predicted {idx}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_surface_and_contour.png'))
    plt.close()


# 记录和保存结果
writer = SummaryWriter()
start_time = time.time()
train_with_data()
end_time = time.time()
training_duration = end_time - start_time

writer.add_text('[RNN_NS]Training/Duration', f'Total training time: {training_duration:.2f} seconds')

# 进行预测
y_pred_test = predict()

# 创建输出目录
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# 绘制中间10个时间刻度的图
mid_indices = np.linspace(0, X_test.size(0) - 1, 10, dtype=int)
plot_time_slices(U_test, y_pred_test, mid_indices, output_dir)
plot_2d_heatmap(U_test, y_pred_test, mid_indices, output_dir)
plot_vector_field(U_test, y_pred_test, mid_indices, output_dir)
plot_3d_vector_field(U_test, y_pred_test, mid_indices, output_dir)
plot_3d_surface_and_contour(U_test, y_pred_test, mid_indices, output_dir)

writer.close()
