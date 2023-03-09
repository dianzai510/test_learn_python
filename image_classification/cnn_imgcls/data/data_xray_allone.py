from torch import nn
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from myutils.myutils import Resize3, PadSquare

input_size = (360, 360)
class_num = 2

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    Resize3(round(input_size[0]*0.8)),
    PadSquare(),
    torchvision.transforms.Pad(100),

    torchvision.transforms.RandomVerticalFlip(0.5),
    torchvision.transforms.RandomHorizontalFlip(0.5),
    # torchvision.transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
    torchvision.transforms.RandomRotation(90, expand=False, interpolation=InterpolationMode.BILINEAR),
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
    torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 亮度、对比度、饱和度
    torchvision.transforms.CenterCrop(input_size),
])

transform_val = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    Resize3(round(input_size[0] * 0.8)),
    PadSquare(),
    torchvision.transforms.Pad(100),
    torchvision.transforms.CenterCrop(input_size),
])

datasets_train = ImageFolder("D:/work/files/deeplearn_datasets/x-ray/cls-dataset/sot23/train", transform=transform_train)
datasets_val = ImageFolder("D:/work/files/deeplearn_datasets/x-ray/cls-dataset/sot23/val", transform=transform_val)

if __name__ == '__main__':
    dataloader_train = DataLoader(datasets_train, 10, shuffle=True)
    dataloader_val = DataLoader(datasets_val, 4, shuffle=True)
    for imgs, labels in dataloader_train:
        img1 = imgs[0, :, :, :]
        img1 = torchvision.transforms.ToPILImage()(img1)  # type:PIL.Image.Image
        img1.show()
        img1.close()
        print(img1.size)
        pass
