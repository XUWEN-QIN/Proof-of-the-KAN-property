import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from NS_KAN import KAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "NS_KAN_checkpoint.pth"
log_file = "training_log.txt"
npy_file = "NS_KAN_predictions.npy"

mode = 'train'  # 'train' or 'predict'
train_slices = [9, 10]  # Training slices
predict_slices = [11, 12]  # Prediction slices
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

slice_indices = train_slices if mode == 'train' else predict_slices

train_a_base = torch.tensor(a).to(device)  # shape = (1200, 64, 64)
train_u_base = torch.tensor(u).to(device)  # shape = (1200, 64, 64, 20)

train_a = train_a_base[:, :, :]  # shape = (1200, 64, 64)
train_u = train_u_base[:, :, :, train_slices].permute(0, 3, 1, 2).contiguous()  # shape = (1200, len(train_slices), 64, 64)
test_a = train_a_base[:, :, :]  # shape = (1200, 64, 64)
test_u = train_u_base[:, :, :, predict_slices].permute(0, 3, 1, 2).contiguous()  # shape = (1200, len(predict_slices), 64, 64)

print(f"train_a shape: {train_a.shape}")
print(f"train_u shape: {train_u.shape}")
print(f"test_a shape: {test_a.shape}")
print(f"test_u shape: {test_u.shape}")

input_dim = 64 * 64  # Each image is 64x64
output_dim = 64 * 64 * len(slice_indices)
hidden_dims = [128, 64, 32]

print("初始化模型...")
model = KAN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)

if mode == 'train':
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

def reshape_for_saving(pred, batch_size, spatial_dim, time_len):
    total_elements = batch_size * np.prod(spatial_dim) * time_len
    if pred.numel() != total_elements:
        raise ValueError(f"Pred shape mismatch: expected {total_elements} but got {pred.numel()}")
    pred = pred.view(batch_size, time_len, *spatial_dim)
    return pred

if mode == 'predict':
    print(f"Predicting with test_a shape: {test_a.shape}")
    print(f"Predicting with test_u shape: {test_u.shape}")
    print("开始预测...")
    test_a = test_a.reshape(-1, input_dim)  # Reshape to (1200, 64*64)
    test_u = test_u.reshape(1200, -1)  # Reshape to (1200, 64*64*len(predict_slices))
    pred_a, pred_u = model.predict(test_a, test_u, checkpoint=checkpoint, slice_indices=predict_slices)
    
    time_len = len(predict_slices)
    print(f"预测输出 pred_a 形状: {pred_a.shape}")
    print(f"预测输出 pred_u 形状: {pred_u.shape}")
    
    pred_a_reshaped = reshape_for_saving(pred_a, 1200, (64, 64), time_len)
    pred_u_reshaped = reshape_for_saving(pred_u, 1200, (64, 64), time_len)
    
    predictions = {
        'a': pred_a_reshaped.cpu().numpy(),
        'u': pred_u_reshaped.cpu().numpy(),
        't': t.flatten()
    }
    model.save_predictions(pred_a, pred_u, test_a, test_u)
    np.savez(npy_file, **predictions)
