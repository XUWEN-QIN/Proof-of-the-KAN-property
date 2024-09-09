import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from train_KAN import KAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "NS_KAN_checkpoint.pth"
log_file = "training_log.txt"
npy_file = "NS_KAN_predictions.npy"

mode = 'train'  # 'train' or 'predict'
slice_indices = [9, 10]  # Training slices
epochs = 1000
batch_size = 64
learning_rate = 0.001

data = scipy.io.loadmat('./data/NavierStokes_V1e-5_N1200_T20.mat')
a = data['a']  # shape = (1200, 64, 64)
u = data['u']  # shape = (1200, 64, 64, 20)
t = data['t']  # shape = (1, 20)

print(f"a shape: {a.shape}")
print(f"u shape: {u.shape}")
print(f"t shape: {t.shape}")


train_a_base = torch.tensor(a).to(device)  # shape = (1200, 64, 64)
train_u_base = torch.tensor(u).to(device)  # shape = (1200, 64, 64, 20)

train_a = train_a_base[:, :, :]  # shape = (1200, 64, 64)
train_u = train_u_base[:, :, :, slice_indices].permute(0, 3, 1, 2).contiguous()  # shape = (1200, len(train_slices), 64, 64)

print(f"train_a shape: {train_a.shape}")
print(f"train_u shape: {train_u.shape}")


input_dim = 64 * 64  # Each image is 64x64
output_dim = 64 * 64 * len(slice_indices)
hidden_dims = [128, 64, 32]

print("初始化模型...")
model = KAN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)


print("开始训练...")
train_a = train_a.reshape(-1, input_dim)  # Reshape to (1200, 64*64)
train_u = train_u.reshape(-1, output_dim)
model.fit(train_a, train_u, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, checkpoint=checkpoint)
    
def read_log_file(filename):
    epochs = []
    losses = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("Epoch"):
                try:
                    parts = line.strip().split(" ")
                    epoch_part = parts[1].strip('[],')
                    epoch = int(epoch_part.split('/')[0])
                    loss = float(parts[-1])
                    epochs.append(epoch)
                    losses.append(loss)
                except ValueError as e:
                    print(f"无法解释行: {line.strip()}，错误: {e}")
    return epochs, losses

epochs_list, losses = read_log_file(log_file)

def plot_loss_curve(epochs, losses, filename='training_loss_curve.png'):
    plt.figure()
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig(filename)
    plt.show()

plot_loss_curve(epochs_list, losses)
