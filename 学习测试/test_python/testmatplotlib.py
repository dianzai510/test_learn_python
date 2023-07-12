from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# X=np.linspace(-np.pi,np.pi,256,endpoint=True)#-π to+π的256个值
# C,S=np.cos(X),np.sin(X)
# plt.plot(X,C)
# plt.plot(X,S)
# #在ipython的交互环境中需要这句话才能显示出来
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Make data
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = 10 * np.outer(np.cos(u), np.sin(v))
# y = 10 * np.outer(np.sin(u), np.sin(v))
# z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
 
# # Plot the surface
# ax.plot_surface(x, y, z, color='b')
 
# plt.show()

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.axes3d
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10
 
fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve')
ax.legend()
 
plt.show()