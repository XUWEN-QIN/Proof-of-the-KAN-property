import scipy.io

# 加载 .mat 文件
#data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')

# 查看文件中的变量
#print(data.keys())

# 查看具体变量的信息，例如 velocity 和 pressure
#if 'velocity' in data:
#    print(data['velocity'].shape)
#if 'pressure' in data:
#    print(data['pressure'].shape)
#a = data['a']  # 当前 shape = (1200, 64, 64)
#u = data['u']  # 当前 shape = (1200, 64, 64, 20)
#t = data['t']  # 当前 shape = (20,)
file_path='NavierStokes_V1e-5_N1200_T20.mat'
#print("a shape:", a.shape)
#print("u shape:", u.shape)
#print("t shape:", t.shape)
# 文件路径
#file_path = 'NavierStokes_V1e-5_N1200_T20.mat'

# 定义要读取的字节数（1K = 1024字节）
bytes_to_read = 1024

# 使用二进制模式打开文件，并读取前1K字节
with open(file_path, 'rb') as file:  # 注意这里使用了 'rb' 而不是 'r'
    # 读取前1K字节的数据
    binary_data = file.read(bytes_to_read)

# 打印二进制数据的十六进制表示
print(binary_data.hex())