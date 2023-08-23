import torch
import os
import PIL
from enum import Enum
from torchvision.transforms import transforms
import cv2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class CJJDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        imagesize=256,
        split=DatasetSplit.TRAIN,
    ):
        super().__init__()
        self.source = source
        self.split = split

        self.imgpaths = self.get_image_data()

        self.default_transform = transforms.Compose([
            transforms.Resize(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        self.transform_ae = transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2)
        ])

        if split == DatasetSplit.TRAIN:
            self.transform = transforms.Compose(self.transform_img)
        else:
            self.transform = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)


    def __getitem__(self, idx):
        image_path = self.imgpaths[idx]
        image = PIL.Image.open(image_path).convert("RGB")

        if self.split == DatasetSplit.TRAIN:
            return self.default_transform(image), self.default_transform(self.transform_ae(image))
        else:
            return self.default_transform(image)

    def __len__(self):
        return len(self.imgpaths)

    def get_image_data(self):
        data_dir = os.path.join(self.source, self.split.value)
        imgpaths = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
        return imgpaths
    