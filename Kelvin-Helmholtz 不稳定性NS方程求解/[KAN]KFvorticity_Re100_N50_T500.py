import torch
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from kan import KAN
import matplotlib

# 设置用于呈现中文字符的字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 或 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# Ensure that Torch uses deterministic algorithms for reproducibility
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = 'enable'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("Using device:", device)

# Load Data
file_path = 'KFvorticity_Re100_N50_T500.npy'
vorticity_data = np.load(file_path)

# Determine how many time steps we have and split the data
num_time_points = vorticity_data.shape[1]
split_idx = int(0.8 * num_time_points)  # Index to split 80% train and 20% test

# Split the data
train_data = vorticity_data[:, :split_idx, :, :]  # First 80% time steps for training
test_data = vorticity_data[:, split_idx:, :, :]   # Last 20% time steps for testing

print("train_data shape:", train_data.shape)
print("test_data shape:", test_data.shape)

# Space and time parameters
width, height = 10.0, 2.0
num_points_x, num_points_y = train_data.shape[2], train_data.shape[3]

# Generate space coordinates
x = torch.linspace(0, width, num_points_x, dtype=torch.float64, device=device, requires_grad=False)
y = torch.linspace(0, height, num_points_y, dtype=torch.float64, device=device, requires_grad=False)
X, Y = torch.meshgrid(x, y, indexing='ij')
coordinates = torch.stack([X.flatten(), Y.flatten()], dim=1)
coordinates.requires_grad = True

# Create time points as a sequence
train_time_points_count = train_data.shape[1]
test_time_points_count = test_data.shape[1]

train_time_points = torch.arange(train_time_points_count, dtype=torch.float64, device=device).long()
test_time_points = torch.arange(train_time_points_count, num_time_points, dtype=torch.float64, device=device).long()

# Preparing the coordinates with time included for training
train_coordinates = coordinates.repeat(train_time_points_count, 1)
train_time_grid = train_time_points.repeat_interleave(coordinates.size(0)).unsqueeze(1)
train_coordinates_time = torch.cat([train_coordinates, train_time_grid], dim=1).to(device)

# Preparing the coordinates with time included for testing
test_coordinates = coordinates.repeat(test_time_points_count, 1)
test_time_grid = test_time_points.repeat_interleave(coordinates.size(0)).unsqueeze(1)
test_coordinates_time = torch.cat([test_coordinates, test_time_grid], dim=1).to(device)

# Normalize the combined coordinates and time
scaler = StandardScaler()

# Detach from computation graph before converting to numpy
train_coordinates_time_np = train_coordinates_time.detach().cpu().numpy()
train_coordinates_time_np = scaler.fit_transform(train_coordinates_time_np)
train_coordinates_time = torch.tensor(train_coordinates_time_np, device=device, dtype=torch.float64)

test_coordinates_time_np = test_coordinates_time.detach().cpu().numpy()
test_coordinates_time_np = scaler.transform(test_coordinates_time_np)
test_coordinates_time = torch.tensor(test_coordinates_time_np, device=device, dtype=torch.float64)

print("train_coordinates_time example:", train_coordinates_time[:5])
print("test_coordinates_time example:", test_coordinates_time[:5])

# Flatten train_data and test_data
train_data_flat = train_data.reshape(train_data.shape[0], -1).T  # Transpose to match sklearn's (samples, features) format
test_data_flat = test_data.reshape(test_data.shape[0], -1).T

# Normalize flat data
train_data_flat = scaler.fit_transform(train_data_flat)
train_data = torch.tensor(train_data_flat.T.reshape(train_data.shape), dtype=torch.float64, device=device)

test_data_flat = scaler.transform(test_data_flat)
test_data = torch.tensor(test_data_flat.T.reshape(test_data.shape), dtype=torch.float64, device=device)

print("Normalized train_data example:", train_data[:, 0, :, :])
print("Normalized test_data example:", test_data[:, 0, :, :])

model = KAN(width=[2, 2, 2,1], grid=500, k=3, grid_eps=1.0, noise_scale=0.1).to(device).double()

# Function to compute model accuracy
def compute_mse(output, target):
    return torch.mean((output - target) ** 2).item()

# Validate function to compute the validation loss
def validate():
    model.eval()
    with torch.no_grad():
        perm = torch.randperm(test_coordinates_time.size(0), device=device)
        total_loss = 0

        for i in range(0, test_coordinates_time.size(0), batch_size):
            indices = perm[i:i + batch_size]
            valid_coordinates = test_coordinates_time[indices]
            valid_times = test_time_grid[indices]

            time_indices = indices // (num_points_x * num_points_y)
            spatial_indices = indices % (num_points_x * num_points_y)

            valid_targets = test_data[0, time_indices, spatial_indices // num_points_y, spatial_indices % num_points_y]
            valid_targets = valid_targets.squeeze()

            uvt_coordinates = valid_coordinates  # We assume the model internally processes the coordinates with time
            output = model(uvt_coordinates)[:, 0]

            loss = torch.nn.functional.mse_loss(output, valid_targets)
            total_loss += loss.item()

        avg_loss = total_loss / (test_coordinates_time.size(0) // batch_size)
        return avg_loss

writer = SummaryWriter()
losses = []
accuracies = []
time_logs = []
batch_size = 102400
steps = 10
start_time = time.time()

def train():
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    model.training_flag = True
    pbar = tqdm(range(steps), desc='训练进度')

    for step in pbar:
        perm = torch.randperm(train_coordinates_time.size(0), device=device)
        total_loss = 0

        for i in range(0, train_coordinates_time.size(0), batch_size):
            indices = perm[i:i + batch_size]
            batch_coordinates = train_coordinates_time[indices]
            batch_times = train_time_grid[indices]

            time_indices = indices // (num_points_x * num_points_y)
            spatial_indices = indices % (num_points_x * num_points_y)

            batch_targets = train_data[0, time_indices, spatial_indices // num_points_y, spatial_indices % num_points_y]
            batch_targets = batch_targets.squeeze()

            uvt_coordinates = batch_coordinates

            optimizer.zero_grad()
            output = model(uvt_coordinates)[:, 0]
            loss = torch.nn.functional.mse_loss(output, batch_targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = validate()
        losses.append(avg_loss)
        pbar.set_description(f"Step：{step} | Loss: {avg_loss:.3f}")
        writer.add_scalar('Loss/train', avg_loss, step)

        scheduler.step(avg_loss)

        with torch.no_grad():
            time_idx = torch.randint(train_time_points.shape[0], (1,), device=device).item()
            current_coordinates = torch.cat([coordinates, train_time_points[time_idx].expand_as(coordinates[:, :1])], dim=1)
            predictions = model(current_coordinates)[:, 0]
            true_values = train_data[0, time_idx].flatten()

            accuracy = compute_mse(predictions, true_values)
            accuracies.append(accuracy)
            writer.add_scalar('Accuracy/train', accuracy, step)

            print(f"Step: {step}, Train Loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f}")

        current_time = time.time()
        elapsed_time = current_time - start_time
        time_logs.append(elapsed_time)

    # Save the final model
    torch.save(model.state_dict(), 'trained_kan_model.pth')

train()
model.training_flag = False  # 切换到测试模式

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

# 直接使用训练好的模型进行预测
model.eval()
predicts = []
with torch.no_grad():
    for i in range(test_time_points.shape[0]):
        test_coords = torch.cat([coordinates, test_time_points[i].expand_as(coordinates[:, :1])], dim=1)
        predictions = model(test_coords)[:, 0].detach().cpu().numpy().reshape(num_points_x, num_points_y)
        predicts.append(predictions)

def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('[KAN]Re100_Training_Loss.png', dpi=300)
    plt.show()

plot_losses(losses)

# Plot and compare predictions and true values for test time points
for i in range(test_time_points.shape[0]):
    w_pred = predicts[i]
    w_true = test_data[0, i].cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(w_pred, extent=(0, width, 0, height), origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.title('预测的涡量场')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.imshow(w_true, extent=(0, width, 0, height), origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.title('真实的涡量场')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')

    plt.suptitle(f'时间片 {i + split_idx + 1}')
    plt.tight_layout()

    plt.savefig(f'[KAN]Re100_Test_Vorticity_Comparison_Timestep_{i + split_idx + 1}.png', dpi=300)
    plt.close()
