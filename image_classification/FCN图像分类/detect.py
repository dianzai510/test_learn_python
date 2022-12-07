import argparse
import os

from PIL.Image import Image

from image_classification.FCN图像分类.data import data_xray


def detect(opt):
    list_files = os.listdir(opt.img_path)
    for f in list_files:
        img = Image.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='run/train/exp/weights/best.pth')
    parser.add_argument('--input_size', default=data_xray.input_size, type=dict)
    parser.add_argument('--img_path', default='', type=str)
    parser.add_argument('--out_path', default='run/detect/exp', type=str)

    opt = parser.parse_args()
    detect(opt)
