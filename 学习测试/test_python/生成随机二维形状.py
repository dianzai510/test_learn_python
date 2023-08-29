import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from math import *
import cv2

for i in range(100):
    #https://www.codenong.com/33962717/
    num = 7
    theta = np.linspace(0, 2*pi - 2*pi/num, num)
    r = np.random.rand(num)*60+30
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    x = np.append(x, x[0])
    y = np.append(y, y[0])

    #插值
    tck, u = interpolate.splprep([x, y], s=0, per=True)
    xi, yi = interpolate.splev(np.linspace(0, 1, 100), tck)

    # plt.xlim(-50,50)
    # plt.ylim(-50,50)
    plt.plot(x, y, '.r')
    plt.plot(xi, yi, '-b')
    plt.show()
    
    #opencv 显示
    xi = xi - np.min(xi)
    yi = yi - np.min(yi)
    pts = np.stack([xi,yi],axis=0)
    w, h = int(np.max(xi))+1, int(np.max(yi))+1
    dis = np.zeros([h,w,3], dtype=np.uint8)
    pts = np.int32(pts).T
    dis = cv2.fillPoly(dis, [pts], (0,0,255))
    cv2.imshow("dis", dis)
    cv2.waitKey()
    cv2.destroyAllWindows()
    pass