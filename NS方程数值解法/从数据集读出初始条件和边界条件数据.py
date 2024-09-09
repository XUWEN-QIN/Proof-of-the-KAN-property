import scipy.io

# 加载 .mat 文件
data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')

# 查看文件中的变量
print(data.keys())  # 列出所有变量

# 假设初始条件和边界条件的变量名为 'initial_conditions' 和 'boundary_conditions'
if 'initial_conditions' in data:
    initial_conditions = data['initial_conditions']
    print("Initial Conditions Shape:", initial_conditions.shape)
    print(initial_conditions)  # 打印初始条件数据

if 'boundary_conditions' in data:
    boundary_conditions = data['boundary_conditions']
    print("Boundary Conditions Shape:", boundary_conditions.shape)
    print(boundary_conditions)  # 打印边界条件数据


# 获取初始条件
initial_conditions = data['u']
print("Initial Conditions Shape:", initial_conditions.shape)
print(initial_conditions)

# 如果有边界条件，可以类似地提取
# 假设 `a` 是边界条件（需根据实际情况判断）
boundary_conditions = data['a']
print("Boundary Conditions Shape:", boundary_conditions.shape)
print(boundary_conditions)
