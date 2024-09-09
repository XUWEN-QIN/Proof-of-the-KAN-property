import scipy.io
import matplotlib.pyplot as plt

# 读取MAT文件
data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')

# 提取变量
a = data['a']
u = data['u']
t = data['t'].flatten()

# 绘制不同时间步 a 的数据
for i in [0, 9, 19]:  # 绘制t=1, t=10, t=20
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(a[:,:,i], cmap='viridis')
    plt.colorbar()
    plt.title(f'a distribution at time step {t[i]:.1f}')
    
    plt.subplot(1, 2, 2)
    # 假设 u 的第一个分量是某方向的速度分量
    plt.imshow(u[:,:,i,0], cmap='viridis')
    plt.colorbar()
    plt.title(f'u (component 0) distribution at time step {t[i]:.1f}')
    
    plt.show()
