import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

input_size = (110, 310)

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.RandomVerticalFlip(0.5),
    torchvision.transforms.RandomHorizontalFlip(0.5),
    #torchvision.transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
    torchvision.transforms.RandomRotation(2, expand=False),
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    torchvision.transforms.ColorJitter(brightness=(0.8, 1.1)),#亮度、对比度、饱和度
    torchvision.transforms.ToTensor()
])

transform_val = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.ToTensor()
])

path = "D:/work/files/data/DeepLearningDataSets/x-ray/xray-cls"
datasets_train = ImageFolder(path, transform=transform_train)
datasets_val = ImageFolder(path, transform=transform_val)

# dataloader_train = DataLoader(datasets_train, 10, shuffle=True)
# dataloader_val = DataLoader(datasets_val, 4, shuffle=True)

if __name__ == '__main__':

    dataloader_train = DataLoader(datasets_train, 10, shuffle=True)
    dataloader_val = DataLoader(datasets_val, 4, shuffle=True)
    for imgs, labels in dataloader_train:
        img1 = imgs[0,:,:,:]
        img1 = torchvision.transforms.ToPILImage()(img1)  # type:PIL.Image.Image
        img1.show()
        img1.close()
        pass
