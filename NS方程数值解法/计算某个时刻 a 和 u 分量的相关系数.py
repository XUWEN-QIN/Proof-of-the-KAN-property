import numpy as np
import scipy.io
import matplotlib.pyplot as plt
data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')
# 提取第二个速度分量
u_second_component = data['u'][:,:,:,1] # 这里假设第二个分量在第四个维度
a = data['a']
u = data['u']
t = data['t'].flatten()
# 在某些时刻计算a与u第二个分量的相关系数:
for i in [0, 9, 19]:
    a_flatten = a[:,:,i].flatten()
    u_component_flatten = u[:,:,:,1][:,:,i].flatten()
    correlation = np.corrcoef(a_flatten, u_component_flatten)[0, 1]
    print(f'Correlation at time step {t[i]:.1f}: {correlation:.2f}')
