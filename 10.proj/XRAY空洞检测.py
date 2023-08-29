import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.interpolate import interp1d
from scipy import interpolate

'''
https://www.blobmaker.app/
目标：生成与上述链接一样效果的随机图像
思路：
1、在极坐标系下随机生成theta和r
2、将坐标转换到笛卡尔坐标系下
3、采用滑动方式依次进行样条插值

https://www.codenong.com/33962717/
'''

# for i in range(100):
#     num = 3

#     theta = np.linspace(0, 2*pi - 2*pi/num, num)
#     r = np.random.rand(num)*30+0

#     x = r*np.cos(theta)
#     y = r*np.sin(theta)

    
#     tck, u = interpolate.splprep([x,y], s=0, per=True)

#     plt.xlim(-50,50)
#     plt.ylim(-50,50)
#     plt.plot(x,y, '.')
#     plt.show()
    
#     pass



import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

for i in range(100):
    num = 3
    theta = np.linspace(0, 2*pi - 2*pi/num, num)
    r = np.random.rand(num)*30+0
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    # append the starting x,y coordinates
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    tck, u = interpolate.splprep([x, y], s=0, per=True)

    # evaluate the spline fits for 1000 evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

    # plot the result
    fig, ax = plt.subplots(1, 1)

    plt.xlim(-50,50)
    plt.ylim(-50,50)
    ax.plot(x, y, 'or')
    ax.plot(xi, yi, '-b')
    plt.show()
    pass