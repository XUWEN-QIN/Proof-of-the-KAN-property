import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time

class SimpleCNN(nn.Module):
    def __init__(self, num_points_x, num_points_y):
        super(SimpleCNN, self).__init__()
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y

        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(num_points_x * num_points_y * 64, 256)
        self.fc2 = nn.Linear(256, num_points_x * num_points_y * 3)

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)  # 使用reshape替换view来处理非连续内存
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(-1, 3, self.num_points_x, self.num_points_y)  # 使用reshape替换view来处理非连续内存
        return x

def get_coordinates(num_points_x, num_points_y):
    x = torch.linspace(0, 1, num_points_x)
    y = torch.linspace(0, 1, num_points_y)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    coordinates = torch.stack([grid_x, grid_y], dim=-1)
    return coordinates

def check_nan_inf(tensor):
    if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
        raise ValueError("Tensor contains NaN or Inf values.")
    
def navier_stokes_residuals(model, coordinates):
    coordinates.requires_grad_(True)
    predictions = model(coordinates.contiguous())  # 添加contiguous
    u, v, p = predictions[:, 0], predictions[:, 1], predictions[:, 2]

    # 计算一阶和二阶导数
    u_x = torch.autograd.grad(u, coordinates, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,:,0]
    u_y = torch.autograd.grad(u, coordinates, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,:,1]
    v_x = torch.autograd.grad(v, coordinates, grad_outputs=torch.ones_like(v), create_graph=True)[0][:,:,0]
    v_y = torch.autograd.grad(v, coordinates, grad_outputs=torch.ones_like(v), create_graph=True)[0][:,:,1]
    p_x = torch.autograd.grad(p, coordinates, grad_outputs=torch.ones_like(p), create_graph=True)[0][:,:,0]
    p_y = torch.autograd.grad(p, coordinates, grad_outputs=torch.ones_like(p), create_graph=True)[0][:,:,1]

    u_xx = torch.autograd.grad(u_x, coordinates, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,:,0]
    u_yy = torch.autograd.grad(u_y, coordinates, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:,:,1]
    v_xx = torch.autograd.grad(v_x, coordinates, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:,:,0]
    v_yy = torch.autograd.grad(v_y, coordinates, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:,:,1]

    # 残差计算
    residual_u = u_x + v_y  # 取一简化形式
    residual_v = v_x - u_y  # 也取一简化形式
    residual_mass = u_xx + u_yy + v_xx + v_yy  # 普通散度形式

    residuals = torch.cat((residual_u, residual_v, residual_mass), dim=0)
    loss = torch.mean(residuals ** 2)
    return loss

def train():
    num_epochs = 50
    num_points_x = 300
    num_points_y = 300

    model = SimpleCNN(num_points_x, num_points_y).to(device)
    optimizer = optim.Adamax(model.parameters(), lr=0.001)

    coordinates = get_coordinates(num_points_x, num_points_y).reshape(
        1, num_points_x, num_points_y, 2).permute(0, 3, 1, 2).to(device)
    
    check_nan_inf(coordinates)
    start_time = time.time()

    loss_history = []
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = navier_stokes_residuals(model, coordinates)
        check_nan_inf(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 新增梯度裁剪这一行

        check_nan_inf(next(model.parameters()).grad)
        optimizer.step()

        loss_number = loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_number}")
        loss_history.append(loss_number)
        writer.add_scalar('Training Loss', loss_number, epoch + 1)
    
    end_time = time.time()
    training_duration = end_time - start_time

    writer.add_text('[CNN_NS]Training/Duration', f'Total training time: {training_duration:.2f} seconds')
    writer.close()

    with open("[CNN_NS]Training.txt", "w") as f:
        f.write(f'Total training time: {training_duration:.2f} seconds\n')

    with open("[CNN_NS]Loss_History.txt", "w") as f:
        for loss in loss_history:
            f.write(f"{loss:.6f}\n")

    np.save('[CNN_NS]loss_history.npy', np.array(loss_history))
    np.save('[CNN_NS]training_time.npy', np.array([training_duration]))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_history)
    plt.title('[CNN_NS]Training Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('[CNN_NS]Training_Loss.png', dpi=300)
    plt.close()

    model.eval()
    with torch.no_grad():
        u_pred = model(coordinates)[:, 0].detach().reshape(num_points_x, num_points_y).cpu().numpy()
        v_pred = model(coordinates)[:, 1].detach().reshape(num_points_x, num_points_y).cpu().numpy()
        p_pred = model(coordinates)[:, 2].detach().reshape(num_points_x, num_points_y).cpu().numpy()


    magnitude = np.sqrt(u_pred ** 2 + v_pred ** 2)

    plt.figure(figsize=(10, 5))
    plt.imshow(magnitude, extent=(0, num_points_x, 0, num_points_y), origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('[CNN_NS]Velocity Magnitude Contour')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('[CNN_NS]Velocity_Magnitude_Contour.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.imshow(u_pred, extent=(0, num_points_x, 0, num_points_y), origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.title('[CNN_NS]u-Component Velocity Field')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('[CNN_NS]u-Component Velocity Field.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.imshow(v_pred, extent=(0, num_points_x, 0, num_points_y), origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.title('[CNN_NS]v-Component Velocity Field')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('[CNN_NS]v-Component Velocity Field.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.imshow(p_pred, extent=(0, num_points_x, 0, num_points_y), origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.title('[CNN_NS]Pressure Field Distribution')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('[CNN_NS]Pressure Field Distribution.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train()
