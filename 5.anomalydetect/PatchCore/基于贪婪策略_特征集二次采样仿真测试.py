"""
目标是将一个大型的特征集合进行缩小，缩小后的特征集合尽量尽量分布均匀才能代码原始特征集。
而随机采样效果不是很好。
"""
import numpy as np
import matplotlib.pyplot as plt

#region 1、随机采样
xx = np.array([range(200)])
yy = np.array([range(200)])
xx,yy = np.meshgrid(xx,yy)
xx,yy = xx.reshape(-1),yy.reshape(-1)
pts = np.stack([xx,yy],axis=0).T
idx = np.random.choice(len(pts), 1000)
subpts = pts[idx]

plt.plot(subpts[:,0],subpts[:,1],".r")
plt.axis([0,200,0,200])#
plt.show()
#endregion

#region 2、贪婪策略
"""
1、随机采集10个点，计算其与集合的平均距离
2、选择其中的最大值
3、计算选择点与集合的
"""
idx = np.random.choice(range(len(pts)),1)

for i in range(len(pts)):
    
    pass

#endregion
pass

 