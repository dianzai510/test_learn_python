import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class data_oqa(Dataset):
    def __init__(self, data_path, transform=None):
        super(data_oqa, self).__init__()
        self.transform = transform
        self.image_files = os.listdir(data_path + "\images")
        self.label_files = os.listdir(data_path + "\labels")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index])
        label = Image.open(self.label_files[index])

        return image, label


if __name__ == '__main__':
    a = data_oqa()
