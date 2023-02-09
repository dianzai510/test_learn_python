import os
import torch
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

input_size = (200, 200)
class_num = 2


class SquarePad:
    def __call__(self, image):
        # image = torchvision.transforms.Pad()
        w, h = image.size
        max_wh = 195  # max(w, h)

        left = int((max_wh - w) / 2)
        right = max_wh - w - left
        top = int((max_wh - h) / 2)
        bottom = max_wh - h - top

        padding = [left, top, right, bottom]
        image = torchvision.transforms.Pad(padding=padding, fill=0)(image)  # left, top, right and bottom
        return image


class Dataset_agl(Dataset):
    def __init__(self, data_path, transform=None):
        self.list_files = []
        self.transform = transform
        # 将字符分类文件夹下的文件保存至list_dir
        list_dir = os.listdir(data_path)
        for Dir in list_dir:
            if Dir != '。':
                path = os.path.join(data_path, Dir)
                files = os.listdir(path)
                for f in files:
                    self.list_files.append(os.path.join(path, f))

    def __getitem__(self, item):
        index = item % len(self.list_files)
        image_path = self.list_files[index]
        img = Image.open(image_path).convert('RGB')  # 图像有3通、4通道，道如果不加这句图像的通道数不一致会报错。

        if item >= len(self.list_files):
            img = img.transpose(Image.ROTATE_90)

        if self.transform is not None:
            img = self.transform(img)

        label = 0 if item < len(self.list_files) else 1
        return img, label

    def __len__(self):
        length = 2 * len(self.list_files)
        return length


trans_train = torchvision.transforms.Compose([
    torchvision.transforms.GaussianBlur(kernel_size=(3, 15), sigma=(0.1, 15.0)),  # 随机高斯模糊
    SquarePad(),
    torchvision.transforms.ColorJitter(brightness=(0.3, 1.5), contrast=(0.5, 1.5), saturation=0.9),  # 亮度、对比度、饱和度
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.RandomAffine(degrees=20, scale=[0.5, 1.3]),
    torchvision.transforms.ToTensor(),
])

trans_val = torchvision.transforms.Compose([
    SquarePad(),
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.ToTensor(),
])

datasets_train = Dataset_agl('d:/work/files/deeplearn_datasets/OQA/cls', transform=trans_train)
datasets_val = Dataset_agl('d:/work/files/deeplearn_datasets/OQA/cls', transform=trans_val)
dataloader_train = DataLoader(datasets_train, 30, shuffle=True)
dataloader_val = DataLoader(datasets_val, 30, shuffle=True)

if __name__ == '__main__':
    data = Dataset_agl('C:/work/files/deeplearn_datasets/OQA/cls', transform=trans_train)
    img = data[37][0]  # type:torch.Tensor
    image1 = torchvision.transforms.ToPILImage()(img)  # type:import torch
    image1.show()

    img = data[1214 + 37][0]
    image2 = torchvision.transforms.ToPILImage()(img)  # type:import torch
    image2.show()

    # datasets = ImageFolder('C:/Users/pc/Desktop/ocr', torchvision.transforms.ToTensor())
    # dataloader_test = DataLoader(datasets, batch_size=1)
    # dataloader_test = DataLoader(datasets_train, batch_size=1)

    dataloader_train = DataLoader(data, batch_size=20, shuffle=True)
    max_width = 0
    max_height = 0
    for imgs, labels in dataloader_train:
        img1 = imgs[0, :, :, :]
        img1 = torchvision.transforms.ToPILImage()(img1)
        img1.show()
        if img1.size[0] > max_height:
            max_height = img1.size[0]
        if img1.size[1] > max_width:
            max_width = img1.size[1]
        pass
    print(max_width, max_height)
