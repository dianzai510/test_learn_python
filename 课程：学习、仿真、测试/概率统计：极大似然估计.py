import numpy as np
import matplotlib.pyplot as plt

mean = 10
sigma = 5
x = np.random.randn(100)
x = x*sigma+mean

plt.hist(x,bins=50)#直方图，绘制每个数值出现的次数，可以直接表达概率分布情况
plt.show()



