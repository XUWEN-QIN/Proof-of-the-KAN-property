import torch
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch import autograd
from tqdm import tqdm
from kan import KAN, LBFGS
import matplotlib.pyplot as plt
import matplotlib

# 设置用于呈现中文字符的字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 或 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# Ensure that Torch uses deterministic algorithms for reproducibility
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TORCH_USE_CUDA_DSA' ]= 'enable'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("Using device:", device)

# Load Data
file_path = 'KFvorticity_Re100_N50_T500.npy'
vorticity_data = np.load(file_path)

# Select a subset of the data (e.g., the first experiment, first 10 time steps)
vorticity_data = vorticity_data[0:1, 0:10, :, :]
print(vorticity_data.shape)  # Should print (1, 10, 64, 64)

vorticity_data = torch.tensor(vorticity_data, dtype=torch.float32, device=device)

# Space and time parameters
width, height = 10.0, 2.0
num_points_x, num_points_y = vorticity_data.shape[2], vorticity_data.shape[3]
num_time_points = vorticity_data.shape[1]

x = torch.linspace(0, width, num_points_x, device=device, requires_grad=False)
y = torch.linspace(0, height, num_points_y, device=device, requires_grad=False)
X, Y = torch.meshgrid(x, y, indexing='ij')
coordinates = torch.stack([X.flatten(), Y.flatten()], dim=1)
coordinates.requires_grad = True

time_points = torch.linspace(0, 1, num_time_points, device=device)

# Initialize model
model = KAN(width=[3,9,9,3], grid=10, k=3, grid_eps=1.0, noise_scale=0.25).to(device)

# Function to compute the batch Jacobian
def batch_jacobian(func, x, create_graph=False):
    def _func_sum(x):
        return func(x).sum(dim=0)
    return autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1, 0, 2)

# Function to compute the batch Hessian
def batch_hessian(func, x):
    jacobian = batch_jacobian(func, x, create_graph=True)
    hessians = []
    for i in range(jacobian.size(1)):
        grad = autograd.grad(jacobian[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
        hessians.append(grad.unsqueeze(1))
    return torch.cat(hessians, dim=1)

# Function to compute vorticity residuals
def vorticity_residuals(coordinates, time_coords):
    coordinates = coordinates.clone().detach().requires_grad_(True)
    time_coords = time_coords.clone().detach().requires_grad_(True)
    
    uvt_coordinates = torch.cat([coordinates, time_coords], dim=1)
    
    output = model(uvt_coordinates)
    w_pred = output[:, 0]
    # f_pred corresponds to some other predicted quantity related to the velocity field
    f_pred = output[:, 1]
    
    grads = batch_jacobian(model, uvt_coordinates, create_graph=True)
    hessians = batch_hessian(model, uvt_coordinates)

    w_t = grads[:, 0, 2]
    w_x = grads[:, 0, 0]
    w_y = grads[:, 0, 1]
    w_xx = hessians[:, 0, 0]
    w_yy = hessians[:, 0, 1]

    # The convective term is derived from f_pred which represents a function of the velocity field
    convective_term = f_pred * (w_x + w_y)
    
    diffusion_term = 1.0 / Re * (w_xx + w_yy)

    w_rhs = -convective_term + diffusion_term
    residuals = torch.mean((w_t - w_rhs) ** 2)
    return residuals

# Function to compute model accuracy
def compute_accuracy(output, target):
    return torch.mean((output - target) ** 2).item()

# Preparing the coordinates with time included
coordinates_time = coordinates.repeat(num_time_points, 1)
time_grid = time_points.repeat_interleave(coordinates.size(0)).unsqueeze(1)

writer = SummaryWriter()
losses = []
accuracies = []
time_logs = []

start_time = time.time()

# Set Reynolds number
Re = 100

# Training function
def train():
    optimizer = LBFGS(model.parameters(), lr=0.1,
                      history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

    steps = 200
    pbar = tqdm(range(steps), desc='Training Progress')

    for step in pbar:
        def closure():
            optimizer.zero_grad()
            loss = vorticity_residuals(coordinates_time, time_grid)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        current_loss = closure().item()
        losses.append(current_loss)
#        pbar.set_description(f"Step: {step} | Loss: {current_loss:.3f}")
        pbar.set_description("Step: %d | Loss: %.3f" %(step, current_loss))
        writer.add_scalar('Loss/train', current_loss, step)

        with torch.no_grad():
            time_idx = torch.randint(num_time_points, (1,)).item()
            current_coordinates = torch.cat([coordinates, time_points[time_idx].expand_as(coordinates[:, :1])], dim=1)
            predictions = model(current_coordinates)[:, 0]
            true_values = vorticity_data[0, time_idx].flatten()

            accuracy = compute_accuracy(predictions, true_values)
            accuracies.append(accuracy)
            writer.add_scalar('Accuracy/train', accuracy, step)

        current_time = time.time()
        elapsed_time = current_time - start_time
        time_logs.append(elapsed_time)

train()

end_time = time.time()
training_duration = end_time - start_time

writer.add_text('[KAN]Training/Duration', f'Total training time: {training_duration:.2f} seconds')

with open("[KAN-Re100]Training.txt", "w") as f:
    for elapsed_time in time_logs:
        f.write(f"{elapsed_time:.6f} seconds\n")

with open("[KAN-Re100]Accuracy.txt", "w") as f:
    for accuracy in accuracies:
        f.write(f"{accuracy:.6f}\n")

writer.close()

# Plot predictions at the final time step
time_idx = -1
final_coordinates = torch.cat([coordinates, time_points[time_idx].expand_as(coordinates[:, :1])], dim=1)
w_pred = model(final_coordinates)[:, 0].detach().reshape(num_points_x, num_points_y).T

# Save vorticity field plot
plt.figure(figsize=(10, 5))
plt.imshow(w_pred.cpu().numpy(), extent=(0, width, 0, height), origin='lower', cmap='coolwarm')
plt.colorbar()
plt.title('[KAN]Vorticity Field')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.tight_layout()
plt.savefig('[KAN]Re100_Vorticity_Field.png', dpi=300)
plt.close()

# Save true vorticity field plot
w_true = vorticity_data[0, time_idx].reshape(num_points_x, num_points_y).cpu().numpy()
plt.figure(figsize=(10, 5))
plt.imshow(w_true, extent=(0, width, 0, height), origin='lower', cmap='coolwarm')
plt.colorbar()
plt.title('[KAN]True Vorticity Field')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.tight_layout()
plt.savefig('[KAN]Re100_True_Vorticity_Field.png', dpi=300)
plt.close()

# Save prediction error plot
error = (w_pred.cpu().numpy() - w_true) ** 2
plt.figure(figsize=(10, 5))
plt.imshow(error, extent=(0, width, 0, height), origin='lower', cmap='hot')
plt.colorbar()
plt.title('[KAN]Vorticity Prediction Error')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.tight_layout()
plt.savefig('[KAN]Re100_Vorticity_Prediction_Error.png', dpi=300)
plt.close()

# Save loss plot
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('[KAN]Training Loss over Time')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('[KAN]Re100_Training_Loss.png', dpi=300)
plt.close()
