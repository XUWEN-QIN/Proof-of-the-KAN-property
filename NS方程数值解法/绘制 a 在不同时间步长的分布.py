import scipy.io
import matplotlib.pyplot as plt

# 读取MAT文件
data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')

# 提取变量
a = data['a']
t = data['t'].flatten()

# 绘制不同时间步数据
for i in range(len(t)):
    plt.imshow(a[:,:,i], cmap='viridis')
    plt.colorbar()
    plt.title(f'a distribution at time step {t[i]:.1f}')
    plt.show()
