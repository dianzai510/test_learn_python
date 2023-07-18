# 代码来至 https://blog.csdn.net/helloword111222/article/details/121279599

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# 生成训练数据
X = 0.3*rng.randn(100,2)#从一个正太分布中随机采样
X_train = np.r_[X+2,X-2]#行方向堆叠两个矩阵（垂直方向拼接）

# 
X = 0.3*rng.randn(20, 2)
X_test = np.r_[X+2,X-2]

#
X_outliers = rng.uniform(low=-4, high=4, size=(20,2))#从一个均匀分布[low,high)中随机采样(左闭右开)
print(X_outliers)

#训练模型
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)


# xx,yy = np.meshgrid(np.linspace(-5,5,50), np.linspace(-5,5,50))
# Z = clf.decision_function(np.c_[xx.ravel(),yy.reval()])
# Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
b1 = plt.scatter(X_train[:,0], X_train[:,1], c='white', s=20, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green', s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')

plt.axis('tight')
plt.xlim((-5,5))
plt.ylim((-5,5))
plt.legend([b1, b2, c],['a','b','c'])
plt.show()