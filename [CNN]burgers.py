import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
import time
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.use_deterministic_algorithms(True)

print("Using device:", device)

# 定义常量
rho = torch.tensor(1.2e-3, requires_grad=False)
nu = torch.tensor(1e-5, requires_grad=False)

# 参数调整
width, height, num_points_x, num_points_y, num_points_t = 10.0, 2.0, 100, 20, 20
time_duration = 10.0

# 确保坐标张量形状一致
x = torch.linspace(0, width, num_points_x, requires_grad=False)
y = torch.linspace(0, height, num_points_y, requires_grad=False)
t = torch.linspace(0, time_duration, num_points_t, requires_grad=False)
X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
coordinates = torch.stack([X, Y, T], dim=0).unsqueeze(0)
coordinates.requires_grad_()
print("Coordinates shape:", coordinates.shape)

# 定义CNN模型
class MyUpdatedModel(nn.Module):
    def __init__(self):
        super(MyUpdatedModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3)  # 改为3
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        
        # 更新全连接层的输入大小
        self.fc1_input_size = 128 * 98 * 18 * 18
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.fc2 = nn.Linear(512, 3 * 20 * 20)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print("After conv1:", x.shape)

        x = torch.relu(self.conv2(x))
        print("After conv2:", x.shape)

        x = torch.relu(self.conv3(x))
        print("After conv3:", x.shape)

        # Flatten the output to feed into fully connected layers
        x_flattened = x.view(x.size(0), -1)
        print("Before fc1:", x_flattened.shape)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(x_flattened))
        x = self.fc2(x).view(-1, 3, 20, 20)
        return x

dummy_input = torch.randn(1, 3, 100, 20, 20)
model = MyUpdatedModel().to(device)
output = model(dummy_input)
print("Output shape:", output.shape)

def burgers_residuals(coords, model, device):
    coords.requires_grad_(True)
    coords = coords.to(device).float()
    
    y_pred = model(coords)

    # 假设 y_pred 的形状为 [batch_size, 3, num_points_x, num_points_y]
    batch_size = y_pred.size(0)
    num_points_x = y_pred.size(2)
    num_points_y = y_pred.size(3)

    # 提取 u 和 v
    u = y_pred[:, 0, :, :].contiguous().view(batch_size, num_points_x, num_points_y)
    v = y_pred[:, 1, :, :].contiguous().view(batch_size, num_points_x, num_points_y)

    # 计算梯度
    u_grad = torch.autograd.grad(outputs=u, inputs=coords, grad_outputs=torch.ones_like(u), retain_graph=True, allow_unused=True)[0]
    v_grad = torch.autograd.grad(outputs=v, inputs=coords, grad_outputs=torch.ones_like(v), retain_graph=True, allow_unused=True)[0]
    
    if u_grad is None or v_grad is None:
        raise RuntimeError("Gradient computation failed. Ensure that the model output depends on coords.")

    # 计算导数
    u_x = u_grad[:, 0, :, :].squeeze()
    v_y = v_grad[:, 1, :, :].squeeze()

    eps = torch.tensor(1.2e-3, device=coords.device, requires_grad=False)

    continuity = u_x + v_y + eps

    u_y = u_grad[:, 1, :, :].squeeze()
    u_t = u_grad[:, 2, :, :].squeeze()
    v_x = v_grad[:, 0, :, :].squeeze()
    v_t = v_grad[:, 2, :, :].squeeze()

    momentum_u = (u * u_x + v * u_y - eps * (u_x + u_t))
    momentum_v = (u * v_x + v * v_y - eps * (v_x + v_t))

    # 计算损失
    continuity_loss = torch.mean(continuity ** 2)
    momentum_u_loss = torch.mean(momentum_u ** 2)
    momentum_v_loss = torch.mean(momentum_v ** 2)
    
    total_loss = continuity_loss + momentum_u_loss + momentum_v_loss

    return total_loss




# 函数：计算解析解
def analytical_solution(x, y, t):
    u_true = -torch.sin(torch.pi * x) * torch.cos(torch.pi * y) * torch.exp(-torch.pi**2 * t)
    v_true = torch.cos(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(-torch.pi**2 * t)
    p_true = torch.zeros_like(x)
    return u_true, v_true, p_true

X_flatten = X.flatten()
Y_flatten = Y.flatten()
T_flatten = T.flatten()
u_true, v_true, p_true = analytical_solution(X_flatten, Y_flatten, T_flatten)

writer = SummaryWriter()
losses = []
accuracies = []
time_logs = []
num_epochs = 100
learning_rate = 0.001
epoch=20

def compute_accuracy(output, target):
    return torch.mean((output - target) ** 2).item()

start_time = time.time()

# 训练函数
def train():
    model.to(device)  # Move model to CUDA
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    coordinates = torch.rand([1, 3, 100, 20, 20], device=device, requires_grad=True)

    def closure():
        optimizer.zero_grad()  # 清零梯度
        loss = burgers_residuals(coordinates, model, device)
        loss.backward()  # 不使用 retain_graph=True
        return loss

    for step in range(10):
        optimizer.step(closure)  # 确保这里调用的是定义好的 closure
        current_loss = closure().item()
        if step % 5 == 0:
            print(f'Step: {step} | Loss: {current_loss:.3f}')
        losses.append(current_loss)
        writer.add_scalar('Loss/train', current_loss, step)

        with torch.no_grad():
            y_preds = []
            for t in range(num_points_t):
                coords_t = coordinates[:, :, :, :, t]
                coords_t_flat = coords_t.view(-1, 3, 20, 20)
                y_pred = model(coords_t_flat).view(-1, 3, 20, 20)
                y_preds.append(y_pred)
            y_preds = torch.cat(y_preds, dim=0)

            u_pred = model(coordinates.to(device)).view(-1, 3)[:, 0].view(num_points_t, num_points_y, num_points_x)
            v_pred = model(coordinates.to(device)).view(-1, 3)[:, 1].view(num_points_t, num_points_y, num_points_x)
            p_pred = model(coordinates.to(device)).view(-1, 3)[:, 2].view(num_points_t, num_points_y, num_points_x)

            u_acc = compute_accuracy(u_pred, u_true.reshape(num_points_t, num_points_y, num_points_x).to(device))
            v_acc = compute_accuracy(v_pred, v_true.reshape(num_points_t, num_points_y, num_points_x).to(device))
            p_acc = compute_accuracy(p_pred, p_true.reshape(num_points_t, num_points_y, num_points_x).to(device))

            accuracy = (u_acc + v_acc + p_acc) / 3
            accuracies.append(accuracy)
            writer.add_scalar('Accuracy/train', accuracy, step)

        elapsed_time = time.time() - start_time
        time_logs.append(elapsed_time)
        print(f'Epoch {epoch + 1}/{num_epochs}, Step {step + 1}/10, Loss {current_loss:.3f}, Accuracy {accuracy:.3f}, Elapsed Time {elapsed_time:.2f}s')

train()


training_duration = time.time() - start_time
writer.add_text('Training/Duration', f'Total training time: {training_duration:.2f} seconds')

# 保存训练时间日志
with open("[CNN_burgers]Training.txt", "w") as f:
    for elapsed_time in time_logs:
        f.write(f"{elapsed_time:.6f} seconds\n")

# 保存准确性日志
with open("[CNN_burgers]Accuracy.txt", "w") as f:
    for accuracy in accuracies:
        f.write(f"{accuracy:.6f}\n")

writer.close()

# 预测和可视化
u_pred = model(coordinates, coordinates).view(-1, 3)[:, 0].view(num_points_t, num_points_y, num_points_x)
v_pred = model(coordinates, coordinates).view(-1, 3)[:, 1].view(num_points_t, num_points_y, num_points_x)
p_pred = model(coordinates, coordinates).view(-1, 3)[:, 2].view(num_points_t, num_points_y, num_points_x)

time_step = 10
magnitude = torch.sqrt(u_pred[time_step] ** 2 + v_pred[time_step] ** 2).cpu().numpy()

# 定义可视化函数
def save_plot(data, title, filename, cmap='viridis'):
    plt.figure(figsize=(10, 5))
    plt.imshow(data, extent=(0, width, 0, height), origin='lower', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

save_plot(magnitude, f'[CNN_burgers]Velocity Magnitude Contour at t={t[time_step]:.2f}', '[CNN]Velocity Magnitude Contour.png')
save_plot(u_pred[time_step].cpu().numpy(), f'[CNN_burgers]u-Component Velocity Field at t={t[time_step]:.2f}', '[CNN]u-Component Velocity Field.png', cmap='coolwarm')
save_plot(v_pred[time_step].cpu().numpy(), f'[CNN_burgers]v-Component Velocity Field at t={t[time_step]:.2f}', '[CNN]v-Component Velocity Field.png', cmap='coolwarm')
save_plot(p_pred[time_step].cpu().numpy(), f'[CNN_burgers]Pressure Field Distribution at t={t[time_step]:.2f}', '[CNN]Pressure Field Distribution.png', cmap='coolwarm')

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('[CNN_burgers]Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('[CNN_burgers]Training Loss.png', dpi=300)
plt.close()

if __name__ == "__main__":
    train()
