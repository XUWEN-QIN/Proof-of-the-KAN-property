import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置用于呈现中文字符的字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 或 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 加载数据文件
file_path = 'KFvorticity_Re100_N50_T500.npy'
vorticity_data = np.load(file_path)

# 打印数据的基本信息
print("数据类型:", type(vorticity_data))
print("数据形状:", vorticity_data.shape)
print("数据的维度:", vorticity_data.ndim)
print("数据类型 (内部元素):", vorticity_data.dtype)

# 数据的基本统计信息
print("数据的最小值:", np.min(vorticity_data))
print("数据的最大值:", np.max(vorticity_data))
print("数据的平均值:", np.mean(vorticity_data))
print("数据的标准差:", np.std(vorticity_data))

# 如果数据包含多个字段/分量，打印部分数据示例
if vorticity_data.ndim > 1:
    print("数据的前5个项:", vorticity_data[:5])
else:
    print("数据的前5个点:", vorticity_data[:5])

# 可视化数据的部分切片（如适用）
import matplotlib.pyplot as plt

# 假设数据的第一个维度表示时间或样本，第二个和第三个维度表示空间坐标
# 这里取第一个时间点的数据进行可视化
if vorticity_data.ndim == 3:
    plt.imshow(vorticity_data[0], cmap='viridis', extent=(0, 1, 0, 1))
    plt.colorbar()
    plt.title("第一个时间点的涡量场")
    plt.show()
elif vorticity_data.ndim == 4:
    # 假设数据的第三个维度是物理量，第四个维度是空间
    # 仅展示第一个时间点，第一个物理量
    plt.imshow(vorticity_data[0, 0], cmap='viridis', extent=(0, 1, 0, 1))
    plt.colorbar()
    plt.title("第一个时间点的第一个物理量涡量场")
    plt.show()
