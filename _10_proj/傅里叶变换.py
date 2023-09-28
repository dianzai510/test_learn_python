import cv2
import numpy as np
from matplotlib import pyplot as plt

path = "D:/work/files/deeplearn_datasets/其它数据集/切割道检测/芯片尺寸/6380B24L1805/10-111.jpg"
img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
result = 20*np.log(np.abs(fshift))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('original')
plt.axis('off')

plt.subplot(122)
plt.imshow(result, cmap='gray')
plt.title('result')
plt.axis('off')

plt.show()

pass