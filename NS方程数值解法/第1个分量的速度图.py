import scipy.io
import matplotlib.pyplot as plt
data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')
# 提取第二个速度分量
u_second_component = data['u'][:,:,:,1] # 这里假设第二个分量在第四个维度
a = data['a']
u = data['u']
t = data['t'].flatten()
# 绘制不同时间步 u 第二个分量的数据
for i in [0, 9, 19]:  # 绘制t=1, t=10, t=20
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(a[:,:,i], cmap='viridis')
    plt.colorbar()
    plt.title(f'a distribution at time step {t[i]:.1f}')
    
    plt.subplot(1, 2, 2)
    plt.imshow(u_second_component[:,:,i], cmap='viridis')
    plt.colorbar()
    plt.title(f'u (component 1) distribution at time step {t[i]:.1f}')
    
    plt.show()
