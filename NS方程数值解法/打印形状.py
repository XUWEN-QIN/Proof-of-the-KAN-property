import numpy as np
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确定性设置
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
torch.use_deterministic_algorithms(True)



# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(86016, 128)
        self.fc2 = torch.nn.Linear(128, 64 * 64 * 20)  # 适配回到 (64, 64, 20)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
        
        
proportion = 1 / 10
# 模拟输入特征数组
input_features = np.random.randn(1200, 86016)  # 示例数据，实际数据应来自您的数据加载部分
num_total_samples = input_features.shape[0]
num_last_samples = int(num_total_samples * proportion)
input_features_flat = torch.tensor(input_features, dtype=torch.float32).to(device)

# 初始化模型并将其移动到设备上（CPU或GPU）
model = SimpleModel().to(device)

# 关闭梯度计算（推理模式）
model.eval()
with torch.no_grad():
    # 对所有 1200 个样本进行预测
    predicted_u_flat = model(input_features_flat)
    
    predicted_u = predicted_u_flat.type(torch.LongTensor)
    
    # 提取最后 120 个样本

    predicted_u_last = predicted_u[-num_last_samples:].cpu().numpy()

    print(f"Model output flat shape: {predicted_u_flat.shape}")
    print(f"Model output reshaped shape: {predicted_u.shape}")
    print(f"Selected last 120 predictions shape: {predicted_u_last.shape}")

# 图形绘制

# 创建一个包含若干个子图的图形
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
channels = 20
height = 64
width = 64
# 重塑predicted_u_last为(120, 20, 64, 64)
predicted_u_last_reshaped = predicted_u_last.reshape(-1, channels, height, width)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for i, ax in enumerate(axes.flatten()):
    # 选择一个通道进行绘制，这里选择第一个通道
    channel_index = 0
    sample = predicted_u_last_reshaped[i, channel_index]  # 获取特定通道的图像数据
    ax.imshow(sample, cmap='viridis')
    ax.set_title(f'Sample {i+1}, Channel {channel_index}')
    ax.axis('off')

plt.tight_layout()
plt.show()