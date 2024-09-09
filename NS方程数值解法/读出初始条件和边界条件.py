import scipy.io

# 加载 .mat 文件
data = scipy.io.loadmat('NavierStokes_V1e-5_N1200_T20.mat')

# 检查变量内容
for key in data.keys():
    if not key.startswith('__'):  # 忽略以 '__' 开头的键
        print(f"{key}: {data[key]}")  # 打印变量及其内容
