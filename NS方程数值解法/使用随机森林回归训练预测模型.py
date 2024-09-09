import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 假设a代表的是某个速度分量，你可以使用u中的其他速度分量预测a
X = np.stack([u[:,:,:,0], u[:,:,:,1]], axis=-1).reshape(-1, 2)  # 其他两个速度分量作为特征
y = a.flatten()  # a作为目标变量

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机森林回归训练预测模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 评估模型
score = model.score(X_test, y_test)
print(f'Model R^2 score: {score:.2f}')
