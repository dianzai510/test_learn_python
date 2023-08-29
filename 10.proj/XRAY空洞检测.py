import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.interpolate import interp1d
from scipy import interpolate
import cv2

'''
https://www.blobmaker.app/
目标：生成与上述链接一样效果的随机图像
思路：
1、在极坐标系下随机生成theta和r
2、将坐标转换到笛卡尔坐标系下
3、采用滑动方式依次进行样条插值

https://www.codenong.com/33962717/
'''


for i in range(700,900):
    num = 9
    theta = np.linspace(0, 2*pi - 2*pi/num, num)
    r = np.random.rand(num)*60+30
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    x = np.append(x, x[0])
    y = np.append(y, y[0])

    #插值
    tck, u = interpolate.splprep([x, y], s=0, per=True)
    xi, yi = interpolate.splev(np.linspace(0, 1, 100), tck)
    
    #opencv 显示
    xi = xi - np.min(xi)
    yi = yi - np.min(yi)
    pts = np.stack([xi,yi],axis=0)
    w, h = int(np.max(xi))+1, int(np.max(yi))+1
    dis = np.zeros([h,w], dtype=np.uint8)
    pts = np.int32(pts).T
    dis = cv2.fillPoly(dis, [pts], 255)
    cv2.imshow("dis", dis)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()

    cv2.imwrite(f'D:/desktop/mask/{i}.png', dis)
    pass