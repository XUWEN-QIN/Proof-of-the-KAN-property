import torch
import os
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from MLP.MLP import MLP_for_NS
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

print("Using device:", device)

rho = torch.tensor(1.2e-3, device=device, requires_grad=False)
nu = torch.tensor(1e-5, device=device, requires_grad=False)
eps = torch.tensor(1e-8, device=device, requires_grad=False)

width, height = 10.0, 2.0
num_points_x, num_points_y = 100, 20  # Changed grid size to 10

x = torch.linspace(0, width, num_points_x, device=device, requires_grad=False)
y = torch.linspace(0, height, num_points_y, device=device, requires_grad=False)
X, Y = torch.meshgrid(x, y, indexing='ij')
coordinates = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
coordinates.requires_grad = True  # Ensure coordinates require grad

# 创建 MLP 模型
model = MLP_for_NS().to(device)

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

def navier_stokes_residuals(coords):
    coords = coords.clone().detach().requires_grad_(True)  # Ensure coords require grad
    y_pred = model(coords)
    grads = batch_jacobian(model, coords, create_graph=True)
    hessians = batch_hessian(model, coords)

    u, v, p = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    u_x, u_y = grads[:, 0, 0], grads[:, 0, 1]
    v_x, v_y = grads[:, 1, 0], grads[:, 1, 1]
    p_x, p_y = grads[:, 2, 0], grads[:, 2, 1]

    u_xx, u_yy = hessians[:, 0, 0], hessians[:, 0, 1]
    v_xx, v_yy = hessians[:, 1, 0], hessians[:, 1, 1]

    continuity = u_x + v_y + eps * p
    x_momentum = u * u_x + v * u_y + (1 / rho) * p_x - nu * (u_xx + u_yy)
    y_momentum = u * v_x + v * v_y + (1 / rho) * p_y - nu * (v_xx + v_yy)

    no_slip_mask = (coords[:, 1] == 0) | (coords[:, 1] == height)
    inlet_mask = (coords[:, 0] == 0)
    outlet_mask = (coords[:, 0] == width)

    no_slip_loss = torch.mean(u[no_slip_mask] ** 2 + v[no_slip_mask] ** 2)
    inlet_loss = torch.mean((u[inlet_mask] - 1) ** 2)
    outlet_pressure_loss = torch.mean(p[outlet_mask] ** 2)

    bc_loss = no_slip_loss + inlet_loss + outlet_pressure_loss
    total_loss = torch.mean(continuity ** 2 + x_momentum ** 2 + y_momentum ** 2) + bc_loss
    return total_loss

def calculate_accuracy(u_pred, v_pred, u_target, v_target):
    u_acc = torch.mean(((u_pred - u_target).abs() < 0.1).float()).item()
    v_acc = torch.mean(((v_pred - v_target).abs() < 0.1).float()).item()
    return (u_acc + v_acc) / 2

writer = SummaryWriter()
losses = []
accuracies = []
time_logs = []
start_time = time.time()

def train():
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1,
                                  history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-10)

    steps = 100  # 20 steps are enough
    pbar = tqdm(range(steps), desc='Training Progress')

    for step in pbar:
        def closure():
            optimizer.zero_grad()
            loss = navier_stokes_residuals(coordinates)
            loss.backward()
            return loss

        optimizer.step(closure)
        current_loss = closure().item()
        if step % 5 == 0:
            pbar.set_description("Step: %d | Loss: %.3f" % (step, current_loss))
        losses.append(current_loss)

        # Dummy target values for accuracy calculation
        u_target = torch.zeros_like(coordinates[:, 0])
        v_target = torch.zeros_like(coordinates[:, 1])

        u_pred = model(coordinates)[:, 0].detach()
        v_pred = model(coordinates)[:, 1].detach()

        accuracy = calculate_accuracy(u_pred, v_pred, u_target, v_target)
        accuracies.append(accuracy)

        writer.add_scalar('Loss/train', current_loss, step)
        writer.add_scalar('Accuracy/train', accuracy, step)

        elapsed_time = time.time() - start_time
        time_logs.append(elapsed_time)

train()

end_time = time.time()
training_duration = end_time - start_time

writer.add_text('[MLP_NS]Training/Duration', f'Total training time: {training_duration:.2f} seconds')

with open("[MLP_NS]Training.txt", "w") as f:
    for elapsed_time in time_logs:
        f.write(f"{elapsed_time:.6f} seconds\n")

with open("[MLP_NS]Accuracy.txt", "w") as f:
    for accuracy in accuracies:
        f.write(f"{accuracy:.6f}\n")

writer.close()

# 提取并绘制预测结果
u_pred = model(coordinates)[:, 0].detach().reshape(num_points_x, num_points_y).T
v_pred = model(coordinates)[:, 1].detach().reshape(num_points_x, num_points_y).T
p_pred = model(coordinates)[:, 2].detach().reshape(num_points_x, num_points_y).T

magnitude = torch.sqrt(u_pred ** 2 + v_pred ** 2).cpu().numpy()

# 绘制速度幅值
plt.figure(figsize=(10, 5))
plt.imshow(magnitude, extent=(0, width, 0, height), origin='lower', cmap='viridis')
plt.colorbar()
plt.title('[MLP_NS]Velocity Magnitude Contour')
plt.xlabel('X Position')
plt.ylabel('Height')
plt.axis('equal')
plt.tight_layout()
plt.savefig('[MLP_NS]Velocity_Magnitude_Contour.png', dpi=300)
plt.close()

# 绘制 u 分量
plt.figure(figsize=(10, 5))
plt.imshow(u_pred.cpu().numpy(), extent=(0, width, 0, height), origin='lower', cmap='coolwarm')
plt.colorbar()
plt.title('[MLP_NS]u-Component Velocity Field')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.tight_layout()
plt.savefig('[MLP_NS]u-Component Velocity Field.png', dpi=300)
plt.close()

# 绘制 v 分量
plt.figure(figsize=(10, 5))
plt.imshow(v_pred.cpu().numpy(), extent=(0, width, 0, height), origin='lower', cmap='coolwarm')
plt.colorbar()
plt.title('[MLP_NS]v-Component Velocity Field')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.tight_layout()
plt.savefig('[MLP_NS]v-Component Velocity Field.png', dpi=300)
plt.close()

# 绘制压力场
plt.figure(figsize=(10, 5))
plt.imshow(p_pred.cpu().numpy(), extent=(0, width, 0, height), origin='lower', cmap='coolwarm')
plt.colorbar()
plt.title('[MLP_NS]Pressure Field Distribution')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.tight_layout()
plt.savefig('[MLP_NS]Pressure Field Distribution.png', dpi=300)
plt.close()

# 绘制训练损失
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('[MLP_NS]Training Loss over Time')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('[MLP_NS]Training_Loss.png', dpi=300)
plt.close()
