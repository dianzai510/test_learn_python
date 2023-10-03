"""
https://blog.csdn.net/qq_42856191/article/details/123776656
"""

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

#path = "D:/work/files/deeplearn_datasets/其它数据集/切割道检测/芯片尺寸/6380B24L1805/10-111.jpg"
path = "D:/desktop/7.jpg"
# img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# result = 20*np.log(np.abs(fshift))
# plt.subplot(121)
# plt.imshow(img, cmap='gray')
# plt.title('original')
# plt.axis('off')

# plt.subplot(122)
# plt.imshow(result, cmap='gray')
# plt.title('result')
# plt.axis('off')

# plt.show()


#opencv 傅里叶变换及逆变换
import cv2
import matplotlib.pyplot as plt
import numpy as np

original = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

dft = cv2.dft(np.float32(original), flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)     #  将图像中的低频部分移动到图像的中心
result = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))   # 将实部和虚部转换为实部，乘以20是为了使得结果更大

plt.subplot(121), plt.imshow(original, cmap='gray')
plt.title('original')
plt.axis('off')

plt.subplot(122), plt.imshow(result, cmap='gray')
plt.title('fft')
plt.axis('off')
plt.show()


original = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

dft = cv2.dft(np.float32(original), flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)     # 将图像中的低频部分移动到图像的中心
result = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))   # 将实部和虚部转换为实部，乘以20是为了使得结果更大

filt = np.zeros_like(dftShift)
x0,y0 = filt.shape[1]//2, filt.shape[0]//2
filt = cv2.circle(filt,(x0,y0),100,(1,1),-1)
dftShift = filt*dftShift


ishift = np.fft.ifftshift(dftShift)     # 低频部分从图像中心移开
iImg = cv2.idft(ishift)                 # 傅里叶反变换
iImg = cv2.magnitude(iImg[:, :, 0], iImg[:, :, 1])      # 转化为空间域

plt.subplot(131), plt.imshow(original, cmap='gray')
plt.title('original')
plt.axis('off')

plt.subplot(132), plt.imshow(result, cmap='gray')
plt.title('fft')
plt.axis('off')

plt.subplot(133), plt.imshow(iImg, cmap='gray')
plt.title('ifft')
plt.axis('off')
plt.show()
pass