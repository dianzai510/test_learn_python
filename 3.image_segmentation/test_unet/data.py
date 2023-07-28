from torch.utils.data import Dataset, DataLoader
import torch
import os
import torchvision
from PIL import Image
import cv2
from torchvision.transforms import InterpolationMode
from our1314.myutils.ext_transform import Resize1, PadSquare

input_size = (448, 448)

transform_basic = [
    Resize1(input_size),# 按比例缩放
    PadSquare(),# 填充为正方形
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomVerticalFlip(0.5),
    torchvision.transforms.RandomHorizontalFlip(0.5),

    torchvision.transforms.RandomRotation(90, interpolation=InterpolationMode.NEAREST)
    # torchvision.transforms.RandomRotation(90, expand=False, interpolation=InterpolationMode.BILINEAR),
    # torchvision.transforms.CenterCrop(input_size),
]
torchvision.transforms.ToPILImage
transform_advan = [
    # torchvision.transforms.Pad(300, padding_mode='symmetric'),
    torchvision.transforms.GaussianBlur(kernel_size=(3, 7)),  # 随机高斯模糊
    torchvision.transforms.ColorJitter(brightness=(0.5, 1.5))
    # , contrast=(0.8, 1.2), saturation=(0.8, 1.2)),  # 亮度、对比度、饱和度
    # torchvision.transforms.ToTensor()
]

trans_train_mask = torchvision.transforms.Compose(transform_basic)
trans_train_image = torchvision.transforms.Compose(transform_basic + transform_advan)


transform_val = torchvision.transforms.Compose([
    Resize1(300),  # 按比例缩放
    PadSquare(),  # 四周补零
    torchvision.transforms.ToTensor()])


class data_seg(Dataset):
    def __init__(self, data_path, transform_image=None, transform_mask=None):
        self.transform_image = transform_image
        self.transform_mask = transform_mask

        self.Images = [data_path + '/' + f for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.Labels = [data_path + '/' + f for f in os.listdir(data_path) if f.endswith('.png')]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        '''
        由于输出的图片的尺寸不同，我们需要转换为相同大小的图片。首先转换为正方形图片，然后缩放的同样尺度(256*256)。
        否则dataloader会报错。
        '''

        # 取出图片路径
        image_path = self.Images[item]
        label_path = self.Labels[item]

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # type:cv2.Mat
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)  # type:cv2.Mat

        h, w, c = image.shape
        scale = input_size[0] / max(h, w)
    
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        h, w, c = image.shape
        #_, label = cv2.threshold(label, 0, 255, type=cv2.THRESH_BINARY)
    
        left = (input_size[0] - w)//2
        right = input_size[0] - w - left
        top = (input_size[1] - h)//2
        bottom = input_size[1] - h - top
    
        image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
        label = cv2.copyMakeBorder(label, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    
        image = torchvision.transforms.ToTensor()(image)
        label = torchvision.transforms.ToTensor()(label)

        print(torch.max(label))
        print(torch.min(label))
        return image, label


class SEGData(Dataset):
    def __init__(self, data_path):
        '''
        根据标注文件去取图片
        '''
        IMG_PATH = data_path + '/images'
        SEGLABE_PATH = data_path + '/masks'
        self.img_path = IMG_PATH
        self.label_path = SEGLABE_PATH
        # print(self.label_path)
        # print(self.img_path)
        # self.img_data = os.listdir(self.img_path)
        self.label_data = os.listdir(self.label_path)
        self.totensor = torchvision.transforms.ToTensor()
        self.resizer = torchvision.transforms.Resize((512, 512))

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, item):
        '''
        由于输出的图片的尺寸不同，我们需要转换为相同大小的图片。首先转换为正方形图片，然后缩放的同样尺度(256*256)。
        否则dataloader会报错。
        '''
        # 取出图片路径

        img_name = os.path.join(self.img_path, self.label_data[item].split('.p')[0])

        img_name = os.path.split(img_name)

        img_name = img_name[1] + '.png'
        # print(img_name+ "GGGGGGGGGGGG")
        img_data = os.path.join(self.img_path, img_name)
        label_data = os.path.join(self.label_path, self.label_data[item])
        # 将图片和标签都转为正方形
        # print("ggggggg" + label_data)
        img = Image.open(img_data)

        label = Image.open(label_data)
        w, h = img.size
        # 以最长边为基准，生成全0正方形矩阵
        slide = max(h, w)
        black_img = torchvision.transforms.ToPILImage()(torch.zeros(3, slide, slide))
        black_label = torchvision.transforms.ToPILImage()(torch.zeros(3, slide, slide))
        black_img.paste(img, (0, 0, int(w), int(h)))  # patse在图中央和在左上角是一样的
        black_label.paste(label, (0, 0, int(w), int(h)))

        # img.show()
        # label.show()
        # 变为tensor,转换为统一大小256*256
        img = self.resizer(black_img)
        label = self.resizer(black_label)

        # img.show()
        # label.show()
        # label1 = np.array(label)
        # img1 = np.array(img)
        # label = cv2.cvtColor(label1,cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        # label = label.reshape(1,label.shape[0],label.shape[1])
        # img = img.reshape(1,img.shape[0],img.shape[1])
        # i

        # cv2.imwrite("D:\\unet\\testData\\1.jpg",img1)
        # cv2.imwrite("D:\\unet\\testData\\2.jpg",label1)
        img = self.totensor(img)
        label = self.totensor(label)

        m1 = torch.max(img)
        m2 = torch.min(img)
        m3 = torch.max(label)
        m4 = torch.min(label)

        return img, label


if __name__ == '__main__':
    data = data_seg('D:/desktop/choujianji/roi/mask', transform_image=trans_train_image, transform_mask=trans_train_mask)

    data_loader = DataLoader(data, batch_size=1, shuffle=True)
    for img, label in data_loader:
        torchvision.transforms.ToPILImage()(img[0] * label[0]).show()

        
