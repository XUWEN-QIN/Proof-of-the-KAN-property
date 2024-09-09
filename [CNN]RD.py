import torch
import os
import matplotlib.pyplot as plt
from torch import autograd, nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 环境配置
torch.cuda.empty_cache()

#print("Using device:", device)

# 定义常量
mu = torch.tensor(0.1, device=device, requires_grad=False)
epsilon = 1e-8

width, height = 10.0, 2.0
batch_size = 32
num_time_points = 10
num_points_x = 100
num_points_y = 20 
x = torch.linspace(0, width, num_points_x, device=device, requires_grad=False)
y = torch.linspace(0, height, num_points_y, device=device, requires_grad=False)
X, Y = torch.meshgrid(x, y, indexing='ij')
coordinates = torch.stack([X.flatten(), Y.flatten()], dim=1)
coordinates.requires_grad = True

time_points = torch.linspace(0, 1, num_time_points, device=device)

# 定义真值解析函数
def analytical_solution(x, y, t):
    u_true = torch.exp(-(x + y + t))
    v_true = torch.exp(-2 * (x + y + t))
    return u_true, v_true

# 重新调整真值解的形状以匹配模型的输出
u_true, v_true = analytical_solution(X.flatten().repeat(num_time_points), 
                                     Y.flatten().repeat(num_time_points), 
                                     time_points.repeat_interleave(X.flatten().shape[0]))

u_true = u_true.view(num_time_points, num_points_x, num_points_y)
v_true = v_true.view(num_time_points, num_points_x, num_points_y)

# CNN 模型定义
class SimpleCNN(nn.Module):
    def __init__(self, num_points_x, num_points_y):
        super(SimpleCNN, self).__init__()
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_final = nn.Conv2d(128, 3, kernel_size=3, padding=1)  # 将输出个数改为3

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv_final(x)
        return x

# 定义RandomDataset类
class RandomDataset(Dataset):
    def __init__(self, num_samples, num_points_x, num_points_y):
        super(RandomDataset, self).__init__()
        self.num_samples = num_samples
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs = torch.randn(3, self.num_points_x, self.num_points_y)
        # 模拟生成的真值解决方案
        u_true = torch.sin(torch.linspace(0, 1, self.num_points_x)).repeat(self.num_points_y, 1).T
        v_true = torch.cos(torch.linspace(0, 1, self.num_points_y)).repeat(self.num_points_x, 1)
        p_true = torch.zeros(self.num_points_x, self.num_points_y)  # 假设 p_true 是零张量
        return inputs, (u_true, v_true, p_true)

# 生成真值解
def generate_true_solutions(num_samples, input_width, input_height, device):
    u_true = torch.sin(torch.linspace(0, 1, num_samples).view(-1, 1, 1)).repeat(1, input_width, input_height).to(device)
    v_true = torch.cos(torch.linspace(0, 1, num_samples).view(-1, 1, 1)).repeat(1, input_width, input_height).to(device)
    return u_true, v_true

u_true, v_true = generate_true_solutions(batch_size, num_points_x, num_points_y, device)

# 损失函数定义
def reaction_diffusion_residuals(predictions, true_solutions):
    u_pred, v_pred, p_pred = predictions.split(1, dim=1)
    u_pred = u_pred.squeeze(1)
    v_pred = v_pred.squeeze(1)
    p_pred = p_pred.squeeze(1)

    u_true, v_true, p_true = true_solutions

    assert u_pred.shape == u_true.shape, f"Shape mismatch: u_pred {u_pred.shape} vs u_true {u_true.shape}"
    assert v_pred.shape == v_true.shape, f"Shape mismatch: v_pred {v_pred.shape} vs v_true {v_true.shape}"
    assert p_pred.shape == p_true.shape, f"Shape mismatch: p_pred {p_pred.shape} vs p_true {p_true.shape}"

    loss_u = torch.nn.functional.mse_loss(u_pred, u_true)
    loss_v = torch.nn.functional.mse_loss(v_pred, v_true)
    loss_p = torch.nn.functional.mse_loss(p_pred, p_true)

    return loss_u + loss_v + loss_p

def train(model, dataloader, num_epochs=20, accumulation_steps=4, device='cuda'):
    scaler = torch.amp.GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    accuracies = []
    time_logs = []
    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0
        for i, data in tqdm(enumerate(dataloader, 0)):
            inputs, (u_true, v_true, p_true) = data
            inputs = inputs.to(device)
            u_true = u_true.to(device)
            v_true = v_true.to(device)
            p_true = p_true.to(device)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = reaction_diffusion_residuals(outputs, (u_true, v_true, p_true))

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            losses.append(loss.item())

        avg_loss = total_loss / len(dataloader)
        accuracies.append(avg_loss) 

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training completed in: {training_duration // 3600}h {training_duration % 3600 // 60}m {training_duration % 60}s")

    return losses, accuracies, time_logs
    # 保存训练时间到文档
    with open("[CNN_RD]Training.txt", "w") as f:
        for elapsed_time in time_logs:
            f.write(f"{elapsed_time:.6f} seconds\n")

    # 保存准确性到文档
    with open("[CNN_RD]Accuracy.txt", "w") as f:
        for accuracy in accuracies:
            f.write(f"{accuracy:.6f}\n")

    return losses, accuracies, time_logs

def save_plots_and_images(model, coordinates, time_points, width, height, num_points_x, num_points_y, losses):
    time_idx = -1
    # 获取 num_points
    num_points = coordinates.size(0)

    # 确保 num_points 是 num_points_x 和 num_points_y 的乘积
    assert num_points == num_points_x * num_points_y, "num_points x num_points_y 必须等于 num_points"

    # 合并坐标和时间
    input_tensor = torch.cat([
        coordinates, 
        time_points[time_idx].expand_as(coordinates[:, :1])
    ], dim=1)

    # 调整形状确保能符合模型输入要求
    input_tensor = input_tensor.view(1, num_points_y, num_points_x, 3)  # 改变形状和维度顺序
    input_tensor = input_tensor.permute(0, 3, 1, 2)  # 将维度重新排列为 [1, 3, num_points_y, num_points_x]

    # 计算预测量
    output_tensor = model(input_tensor).detach().squeeze()
    u_pred = output_tensor[0].view(num_points_y, num_points_x).T
    v_pred = output_tensor[1].view(num_points_y, num_points_x).T
    p_pred = output_tensor[2].view(num_points_y, num_points_x).T

    # 计算速度矢量大小
    magnitude = torch.sqrt(u_pred ** 2 + v_pred ** 2).cpu().numpy()

    # 保存各类图像
    save_image(u_pred.cpu().numpy(), '[CNN_RD]u_Component.png', '[CNN_RD]u-Component Velocity Field', width, height)
    save_image(v_pred.cpu().numpy(), '[CNN_RD]v_Component.png', '[CNN_RD]v-Component Velocity Field', width, height)
    save_image(magnitude, '[CNN_RD]Velocity_Magnitude_Contour.png', '[CNN_RD]Velocity Magnitude Contour', width, height)
    save_loss_plot(losses, '[CNN_RD]Training_Loss.png', '[CNN_RD]Training Loss over Time')

def save_image(data, filename, title, width, height):
    plt.figure(figsize=(10, 5))
    plt.imshow(data, extent=(0, width, 0, height), origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def save_loss_plot(losses, filename, title):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_points_x = 100
    num_points_y = 20


    dataset = RandomDataset(num_samples=1000, num_points_x=num_points_x, num_points_y=num_points_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    model = SimpleCNN(num_points_x, num_points_y).to(device)
    losses, accuracies, time_logs = train(model, dataloader, num_epochs=20, accumulation_steps=4, device=device)

    save_plots_and_images(model, coordinates, time_points, width, height, num_points_x, num_points_y, losses)
