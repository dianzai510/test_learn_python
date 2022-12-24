"""
BA优化思路：思路寻找一个相机姿态，使得对目标3D点的投影与已知的投影的误差最小。
测试思路：
1、给定3D点，给定相机姿态对3D点进行投影，并对得到的图像坐标添加高斯噪声
2、分别使用线性法和非线性法

实施步骤：
1、看懂别人的BA优化代码

思考：
1、优化问题的关键是找到导数，BA优化中的导数怎么得到？
2、在清洗机定位问题中，由于3D点已知且数量有限，而其对应的图像特征点可能有多
个，考虑在给定初值的情况下，对相机姿态和图像特征点一起进行优化。
"""

import sys

from cv2 import *

print("测试BA")

sys.exit()
