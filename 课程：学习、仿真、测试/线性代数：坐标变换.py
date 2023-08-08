import math
import numpy as np

pt = np.array([[5],[4],[1]])  # 二维点的齐次坐标
H = np.array(
    [
        [1,0,5],
        [0,1,4],
        [0,0,1]
    ])
print(H)
print(np.linalg.inv(H))

p = np.linalg.inv(H)@pt
print(p)