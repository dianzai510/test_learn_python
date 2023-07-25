import cv2
import numpy as np
import halcon
from our1314.myutils.myutils import rad

class templateMatch():
    def __init__(self, path_temp):
        
        temp = halcon.read_image(path_temp)
        region = halcon.gen_rectangle1(row_1=0, column_1=0, row_2=1, column_2=1)
        halcon.reduce_domain(temp, region=region)
        halcon.create_scaled_shape_model(temp, 5, -rad(10), rad(10), rad(0.1), -0.8, 0.8, 0.01, ('none', 'nopregeneration'), 'auto', [10,30,30], 10)
        pass


if __name__ == '__main__':
    from our1314.cv.mouseselect import mouseSelect

    path = 'd:/desktop/1.jpg'
    src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)#type:np.ndarray
    a = mouseSelect(src)

    