import torch
import os
import matplotlib.pyplot as plt
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from kan import KAN, LBFGS
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

print("Using device:", device)

mu = torch.tensor(0.1, device=device, requires_grad=False)
epsilon = 1e-8

width, height = 10.0, 2.0
num_points_x, num_points_y = 100, 20

x = torch.linspace(0, width, num_points_x, device=device, requires_grad=False)
y = torch.linspace(0, height, num_points_y, device=device, requires_grad=False)
X, Y = torch.meshgrid(x, y, indexing='ij')
coordinates = torch.stack([X.flatten(), Y.flatten()], dim=1)
coordinates.requires_grad = True

num_time_points = 10
time_points = torch.linspace(0, 1, num_time_points, device=device)

model = KAN(width=[2, 6, 6, 3], grid=10, k=3, grid_eps=1.0, noise_scale=0.10).to(device)

def batch_jacobian(func, x, create_graph=False):
    def _func_sum(x):
        return func(x).sum(dim=0)
    return autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 0, 2)

def batch_hessian(func, x):
    jacobian = batch_jacobian(func, x, create_graph=True)
    hessians = []
    for i in range(jacobian.size(1)):
        grad = autograd.grad(jacobian[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
        hessians.append(grad.unsqueeze(1))
    return torch.cat(hessians, dim=1)

def reaction_diffusion_residuals(coords, time_coords):
    coords = coords.clone().detach().requires_grad_(True)
    time_coords = time_coords.clone().detach().requires_grad_(True)
    
    uvt_coords = torch.cat([coords, time_coords], dim=1)
    
    y_pred = model(uvt_coords)
    grads = batch_jacobian(model, uvt_coords, create_graph=True)
    hessians = batch_hessian(model, uvt_coords)

    u, v = y_pred[:, 0], y_pred[:, 1]
    u_t = grads[:, 0, 2]
    v_t = grads[:, 1, 2]
    u_xx = hessians[:, 0, 0]
    u_yy = hessians[:, 0, 1]
    v_xx = hessians[:, 1, 0]
    v_yy = hessians[:, 1, 1]

    u_rhs = mu * (u_xx + u_yy) + u - u**3 - v + 0.01
    v_rhs = mu * (v_xx + v_yy) + 0.25 * (u - v)

    residuals = torch.mean((u_t - u_rhs)**2) + torch.mean((v_t - v_rhs)**2)
    return residuals

def analytical_solution(x, y, t):
    u_true = torch.exp(-x - y - t)
    v_true = torch.exp(-2 * x - 2 * y - 2 * t)
    return u_true, v_true

# 从解析解生成基准真值数据
X_flatten = X.flatten()
Y_flatten = Y.flatten()
time_points_broadcasted = time_points.unsqueeze(1).repeat(1, X_flatten.shape[0]).flatten()

X_flatten_repeated = X_flatten.repeat(num_time_points)
Y_flatten_repeated = Y_flatten.repeat(num_time_points)

u_true, v_true = analytical_solution(X_flatten_repeated, Y_flatten_repeated, time_points_broadcasted)

# note: u_true and v_true 应重新调整形状以匹配预测张量形状
u_true = u_true.view(num_time_points, num_points_x, num_points_y)
v_true = v_true.view(num_time_points, num_points_x, num_points_y)

def compute_accuracy(output, target):
    return torch.mean((output - target) ** 2).item()

coordinates_time = coordinates.repeat(num_time_points, 1)
time_grid = time_points.repeat_interleave(coordinates.size(0)).unsqueeze(1)

writer = SummaryWriter()
losses = []
accuracies = []
time_logs = []

start_time = time.time()

def train():
    optimizer = LBFGS(model.parameters(), lr=0.1,
                      history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

    steps = 100
    pbar = tqdm(range(steps), desc='Training Progress')

    for step in pbar:
        def closure():
            optimizer.zero_grad()
            loss = reaction_diffusion_residuals(coordinates_time, time_grid)
            loss.backward()
            return loss

        optimizer.step(closure)
        current_loss = closure().item()
        losses.append(current_loss)
        pbar.set_description(f"Step: {step} | Loss: {current_loss:.3f}")
        writer.add_scalar('Loss/train', current_loss, step)

        with torch.no_grad():
            predictions = model(torch.cat([coordinates_time, time_grid], dim=1))
            u_pred = predictions[:, 0].reshape(num_time_points, num_points_x, num_points_y)
            v_pred = predictions[:, 1].reshape(num_time_points, num_points_x, num_points_y)

            u_acc = compute_accuracy(u_pred, u_true)
            v_acc = compute_accuracy(v_pred, v_true)

            accuracy = (u_acc + v_acc) / 2
            accuracies.append(accuracy)
            writer.add_scalar('Accuracy/train', accuracy, step)

        current_time = time.time()
        elapsed_time = current_time - start_time
        time_logs.append(elapsed_time)

train()

end_time = time.time()
training_duration = end_time - start_time

writer.add_text('[KAN_RD]Training/Duration', f'Total training time: {training_duration:.2f} seconds')

with open("[KAN_RD]Training.txt", "w") as f:
    for elapsed_time in time_logs:
        f.write(f"{elapsed_time:.6f} seconds\n")

with open("[KAN_RD]Accuracy.txt", "w") as f:
    for accuracy in accuracies:
        f.write(f"{accuracy:.6f}\n")

writer.close()

time_idx = -1
u_pred = model(torch.cat([coordinates, time_points[time_idx].expand_as(coordinates[:, :1])], dim=1))[:, 0].detach().reshape(num_points_x, num_points_y).T
v_pred = model(torch.cat([coordinates, time_points[time_idx].expand_as(coordinates[:, :1])], dim=1))[:, 1].detach().reshape(num_points_x, num_points_y).T

magnitude = torch.sqrt(u_pred ** 2 + v_pred ** 2).cpu().numpy()

# 保存 u 分量图像
plt.figure(figsize=(10, 5))
plt.imshow(u_pred.cpu().numpy(), extent=(0, width, 0, height), origin='lower', cmap='coolwarm')
plt.colorbar()
plt.title('[KAN_RD]u-Component Velocity Field')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.tight_layout()
plt.savefig('[KAN_RD]u_Component.png', dpi=300)
plt.close()

# 保存 v 分量图像
plt.figure(figsize=(10, 5))
plt.imshow(v_pred.cpu().numpy(), extent=(0, width, 0, height), origin='lower', cmap='coolwarm')
plt.colorbar()
plt.title('[KAN_RD]v-Component Velocity Field')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.tight_layout()
plt.savefig('[KAN_RD]v_Component.png', dpi=300)
plt.close()

# 保存速度矢量大小图像
plt.figure(figsize=(10, 5))
plt.imshow(magnitude, extent=(0, width, 0, height), origin='lower', cmap='viridis')
plt.colorbar()
plt.title('[KAN_RD]Velocity Magnitude Contour')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.tight_layout()
plt.savefig('[KAN_RD]Velocity_Magnitude_Contour.png', dpi=300)
plt.close()

# 保存损失曲线图像
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('[KAN_RD]Training Loss over Time')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('[KAN_RD]Training_Loss.png', dpi=300)
plt.close()
