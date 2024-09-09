import scipy.io

# 加载 .mat 文件
data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')

# 查看文件中的变量
print(data.keys())

# 查看具体变量的信息，例如 velocity 和 pressure
if 'velocity' in data:
    print(data['velocity'].shape)
if 'pressure' in data:
    print(data['pressure'].shape)
a = data['a']  # 当前 shape = (1200, 64, 64)
u = data['u']  # 当前 shape = (1200, 64, 64, 20)
t = data['t']  # 当前 shape = (20,)
print("a shape:", a.shape)
print("u shape:", u.shape)
print("t shape:", t.shape)
