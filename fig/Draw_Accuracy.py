import matplotlib.pyplot as plt
import numpy as np

# 定义一个函数来读取数据文件
def read_data(filename):
    with open(filename, 'r') as file:
        # 跳过文件头
        next(file)
        # 读取数据，转换为浮点数
        data = [float(line.strip()) for line in file if line.strip()]
    return data

# 定义一个字典来存储数据和标签
datasets = {}
file_names = [
    '[KAN_NS]Accuracy.txt',
    '[KAN_burgers]Accuracy.txt',
    '[KAN_PE]Accuracy.txt',
    '[KAN_RD]Accuracy.txt',
    '[MLP_NS]Accuracy.txt',
    '[MLP_burgers]Accuracy.txt',
    '[MLP_PE]Accuracy.txt',
    '[MLP_RD]Accuracy.txt',
    # ... 添加其他文件名
]
# 读取数据并存储到字典中
for filename in file_names:
    label = filename.split('Accuracy.txt')[0].strip('[]')
    data = read_data(filename)
    datasets[label] = data

# 准备绘制图表
plt.figure(figsize=(10, 6))

# 为不同的数据集定义不同的颜色和标记
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 's', '^', 'v', '>', '<', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

# 绘制每个数据集
for i, (label, data) in enumerate(datasets.items()):
    plt.plot(range(len(data)), data, label=label, color=colors[i % len(colors)], marker=markers[i % len(markers)])

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Accuracy Over Time')
plt.xlabel('Time Step')
plt.ylabel('Accuracy')
plt.savefig('Accuracy.png', dpi=300, bbox_inches='tight')
# 显示图表
plt.show()
