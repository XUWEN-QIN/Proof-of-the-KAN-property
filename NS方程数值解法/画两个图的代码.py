import matplotlib.pyplot as plt
import numpy as np

# 示例数据，真实数据和预测数据
# 假设 true_data 和 predicted_data 是相同尺寸的二维数组
true_data = np.random.rand(50, 50)  # 真实数据（50x50随机数矩阵作为示例）
predicted_data = true_data + np.random.normal(0, 0.1, true_data.shape)  # 模拟预测数据（添加噪声）

# 设置画布和子图布局
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 绘制真实数据的2D热力图
im1 = axes[0].imshow(true_data, cmap='viridis', origin='lower')
axes[0].set_title('真实数据')
axes[0].set_xlabel('X 轴')
axes[0].set_ylabel('Y 轴')
fig.colorbar(im1, ax=axes[0])

# 绘制模型预测数据的2D热力图
im2 = axes[1].imshow(predicted_data, cmap='viridis', origin='lower', vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])
axes[1].set_title('模型预测数据')
axes[1].set_xlabel('X 轴')
axes[1].set_ylabel('Y 轴')
fig.colorbar(im2, ax=axes[1])

# 调整子图布局
plt.tight_layout()
plt.show()
