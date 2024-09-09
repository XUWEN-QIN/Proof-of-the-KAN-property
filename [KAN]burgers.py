#%%
import torch
import os
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from kan import KAN, LBFGS
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

print("Using device:", device)

rho = torch.tensor(1.2e-3, device=device, requires_grad=False)
nu = torch.tensor(1e-5, device=device, requires_grad=False)
eps = torch.tensor(1e-8, device=device, requires_grad=False)

width, height, num_points_x, num_points_y, num_points_t = 10.0, 2.0, 100, 20, 20
time_duration = 10.0
x = torch.linspace(0, width, num_points_x, device=device, requires_grad=False)
y = torch.linspace(0, height, num_points_y, device=device, requires_grad=False)
t = torch.linspace(0, time_duration, num_points_t, device=device, requires_grad=False)
X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
coordinates = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
coordinates.requires_grad = True  # Ensure coordinates require grad

model = KAN(width=[3, 6, 6, 3], k=3, grid_eps=1.0).to(device)

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

def pde_residuals(coords):
    coords = coords.clone().detach().requires_grad_(True)  # Ensure coords require grad
    y_pred = model(coords)
    grads = batch_jacobian(model, coords, create_graph=True)
    hessians = batch_hessian(model, coords)

    u, v, p = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    u_x, u_y, u_t = grads[:, 0, 0], grads[:, 0, 1], grads[:, 0, 2]
    v_x, v_y, v_t = grads[:, 1, 0], grads[:, 1, 1], grads[:, 1, 2]

    u_xx, u_yy = hessians[:, 0, 0], hessians[:, 0, 1]
    v_xx, v_yy = hessians[:, 1, 0], hessians[:, 1, 1]

    continuity = u_x + v_y + eps * p
    u_eq = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy) - (1 - u**2 - v**2) * u - (u**2 + v**2) * v
    v_eq = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy) + (u**2 + v**2) * u - (1 - u**2 - v**2) * v

    no_slip_mask = (coords[:, 1] == 0) | (coords[:, 1] == height)
    inlet_mask = (coords[:, 0] == 0)
    outlet_mask = (coords[:, 0] == width)

    no_slip_loss = torch.mean(u[no_slip_mask] ** 2 + v[no_slip_mask] ** 2)
    inlet_loss = torch.mean((u[inlet_mask] - 1) ** 2)
    outlet_pressure_loss = torch.mean(p[outlet_mask] ** 2)

    bc_loss = no_slip_loss + inlet_loss + outlet_pressure_loss
    total_loss = torch.mean(continuity ** 2 + u_eq ** 2 + v_eq ** 2) + bc_loss
    return total_loss

# 分析解函数
def analytical_solution(x, y, t):
    u_true = torch.sin(torch.pi * x) * torch.cos(torch.pi * y) * torch.exp(-t)
    v_true = -torch.cos(torch.pi * x) * torch.sin(torch.pi * y) * torch.exp(-t)
    p_true = torch.ones_like(x)  # 假设压力场为常数
    return u_true, v_true, p_true

# 从解析解生成基准真值数据
X_flatten = X.flatten()
Y_flatten = Y.flatten()
T_flatten = T.flatten()
u_true, v_true, p_true = analytical_solution(X_flatten, Y_flatten, T_flatten)

writer = SummaryWriter()
losses = []
accuracies = []
time_logs = []

def compute_accuracy(output, target):
    return torch.mean((output - target) ** 2).item()

start_time = time.time()

def train():
    optimizer = LBFGS(model.parameters(), lr=0.1,
                      history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
    
    steps = 100  # Total number of training steps
    pbar = tqdm(range(steps), desc='Training Progress')

    for step in pbar:
        def closure():
            optimizer.zero_grad()
            loss = pde_residuals(coordinates)
            loss.backward()
            return loss

        optimizer.step(closure)
        current_loss = closure().item()
        if step % 5 == 0:
            pbar.set_description("Step: %d | Loss: %.3f" % (step, current_loss))
        losses.append(current_loss)
        writer.add_scalar('Loss/train', current_loss, step)

        # 计算当前步的精度
        with torch.no_grad():
            predictions = model(coordinates)
            u_pred = predictions[:, 0].reshape(num_points_x, num_points_y, num_points_t)
            v_pred = predictions[:, 1].reshape(num_points_x, num_points_y, num_points_t)
            p_pred = predictions[:, 2].reshape(num_points_x, num_points_y, num_points_t)

            u_acc = compute_accuracy(u_pred, u_true.reshape(num_points_x, num_points_y, num_points_t))
            v_acc = compute_accuracy(v_pred, v_true.reshape(num_points_x, num_points_y, num_points_t))
            p_acc = compute_accuracy(p_pred, p_true.reshape(num_points_x, num_points_y, num_points_t))

            accuracy = (u_acc + v_acc + p_acc) / 3
            accuracies.append(accuracy)
            writer.add_scalar('[KAN_burgers]Accuracy/train', accuracy, step)

        # 记录时间日志
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_logs.append(elapsed_time)

train()

end_time = time.time()
training_duration = end_time - start_time

writer.add_text('[KAN_burgers]Training/Duration', f'Total training time: {training_duration:.2f} seconds')

# 保存训练时间日志
with open("[KAN_burgers]Training.txt", "w") as f:
    for elapsed_time in time_logs:
        f.write(f"{elapsed_time:.6f} seconds\n")

# 保存精度日志
with open("[KAN_burgers]Accuracy.txt", "w") as f:
    for accuracy in accuracies:
        f.write(f"{accuracy:.6f}\n")

writer.close()

num_coords = num_points_x * num_points_y * num_points_t
u_pred = model(coordinates)[:, 0].detach().reshape(num_points_x, num_points_y, num_points_t).permute(2, 0, 1)
v_pred = model(coordinates)[:, 1].detach().reshape(num_points_x, num_points_y, num_points_t).permute(2, 0, 1)
p_pred = model(coordinates)[:, 2].detach().reshape(num_points_x, num_points_y, num_points_t).permute(2, 0, 1)

time_step = 10
magnitude = torch.sqrt(u_pred[time_step] ** 2 + v_pred[time_step] ** 2).cpu().numpy()

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

save_plot(magnitude, f'[KAN_burgers]Velocity Magnitude Contour at t={t[time_step]:.2f}', '[KAN_burgers]Velocity Magnitude Contour.png')
save_plot(u_pred[time_step].cpu().numpy(), f'[KAN_burgers]u-Component Velocity Field at t={t[time_step]:.2f}', '[KAN_burgers]u-Component Velocity Field.png', cmap='coolwarm')
save_plot(v_pred[time_step].cpu().numpy(), f'[KAN_burgers]v-Component Velocity Field at t={t[time_step]:.2f}', '[KAN_burgers]v-Component Velocity Field.png', cmap='coolwarm')
save_plot(p_pred[time_step].cpu().numpy(), f'[KAN_burgers]Pressure Field Distribution at t={t[time_step]:.2f}', '[KAN_burgers]Pressure Field Distribution.png', cmap='coolwarm')

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('[KAN_burgers]Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('[KAN_burgers]Training Loss.png', dpi=300)
plt.close()
