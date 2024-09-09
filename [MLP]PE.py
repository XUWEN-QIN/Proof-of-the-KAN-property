import torch
import time
import os
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from MLP.MLP import MLP_for_PE
from torch.optim import LBFGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

print("Using device:", device)

# Problem setup
width, height = 10.0, 2.0
num_points_x, num_points_y = 100, 20

x = torch.linspace(0, width, num_points_x, device=device, requires_grad=False)
y = torch.linspace(0, height, num_points_y, device=device, requires_grad=False)
X, Y = torch.meshgrid(x, y, indexing='ij')
coordinates = torch.stack([X.flatten(), Y.flatten()], dim=1)
coordinates.requires_grad = True  # Ensure coordinates require grad

# Model setup
model = MLP_for_PE().to(device)

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

def poisson_residuals(coords):
    coords = coords.clone().detach().requires_grad_(True)  # Ensure coords require grad
    phi_pred = model(coords)
    hessians = batch_hessian(model, coords)

    phi_xx, phi_yy = hessians[:, 0, 0], hessians[:, 0, 1]

    # \nabla^2 \phi = \partial^2 \phi / \partial x^2 + \partial^2 \phi / \partial y^2
    laplacian = phi_xx + phi_yy

    # Assume right-hand side function f(x) is constant, e.g., f(x) = 1 here
    source_term = torch.ones_like(phi_pred)
    residual = laplacian - source_term

    # Boundary conditions
    bc_loss = 0
    bc_loss += torch.mean((phi_pred[0:num_points_x] - 0) ** 2)  # Left boundary x=0
    bc_loss += torch.mean((phi_pred[(num_points_y-1)*num_points_x:] - 1) ** 2)  # Right boundary x=width
    bc_loss += torch.mean((phi_pred[::num_points_x] - 0) ** 2)  # Bottom boundary y=0
    bc_loss += torch.mean((phi_pred[num_points_x-1::num_points_x] - 1) ** 2)  # Top boundary y=height
    
    total_loss = torch.mean(residual ** 2) + bc_loss
    return total_loss

writer = SummaryWriter()
losses = []
time_logs = []
start_time = time.time()
def train():
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1,
                                  history_size=10, line_search_fn="strong_wolfe", 
                                  tolerance_grad=1e-32, tolerance_change=1e-32)
    
    steps = 100
    pbar = tqdm(range(steps), desc='Training Progress')
    start_time = time.time()

    for step in pbar:
        def closure():
            optimizer.zero_grad()
            loss = poisson_residuals(coordinates)
            loss.backward()
            return loss

        optimizer.step(closure)
        current_loss = closure().item()
        if step % 5 == 0:
            pbar.set_description("Step: %d | Loss: %.3f" % (step, current_loss))
        losses.append(current_loss)
        writer.add_scalar('Loss/train', current_loss, step)
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_logs.append(time.time() - start_time)

train()

# Post-training operations
end_time = time.time()
training_duration = end_time - start_time

writer.add_text('[MLP_PE]Training/Duration', f'Total training time: {training_duration:.2f} seconds')

with open("[MLP_PE]Training.txt", "w") as f:
    for elapsed_time in time_logs:
        f.write(f"{elapsed_time:.6f} seconds\n")

with open("[MLP_PE]Accuracy.txt", "w") as f:
    for loss in losses:
        f.write(f"{loss:.6f}\n")

writer.close()

phi_pred = model(coordinates)[:, 0].detach().reshape(num_points_x, num_points_y).T

# Plot potential field
plt.figure(figsize=(10, 5))
plt.imshow(phi_pred.cpu().numpy(), extent=(0, width, 0, height), origin='lower', cmap='viridis')
plt.colorbar()
plt.title('[MLP_PE]Potential Field Distribution')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.tight_layout()
plt.savefig('[MLP_PE]Potential_Field_Distribution.png', dpi=300)
plt.close()

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('[MLP_PE]Training Loss over Time')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('[MLP_PE]Training_Loss.png', dpi=300)
plt.close()

# Show 3D plot of the potential field
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
X_flatten = X.flatten()
Y_flatten = Y.flatten()
phi_flatten = phi_pred.flatten()
ax.plot_trisurf(X_flatten.cpu().numpy(), Y_flatten.cpu().numpy(), phi_flatten.cpu().numpy(), cmap='viridis', edgecolor='none')
ax.set_title('[MLP-PE]3D Potential Field Distribution')

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Potential')
plt.tight_layout()
plt.savefig('[MLP-PE]3D_Potential_Field_Distribution.png', dpi=300)
plt.close()
