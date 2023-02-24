from torch.utils.data import Dataset
import torch
import os
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
from torchvision.transforms import InterpolationMode

input_size = (512, 512)

trans_train = torchvision.transforms.Compose([
    torchvision.transforms.Pad(300, padding_mode='symmetric'),
    torchvision.transforms.GaussianBlur(kernel_size=(3, 15), sigma=(0.1, 15.0)),  # 随机高斯模糊
    torchvision.transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=0.9),  # 亮度、对比度、饱和度
    torchvision.transforms.RandomRotation(90, expand=False, interpolation=InterpolationMode.BILINEAR),
    torchvision.transforms.CenterCrop(input_size),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.ToTensor(),
])

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.Pad(100, padding_mode='symmetric'),
    torchvision.transforms.RandomVerticalFlip(0.5),
    torchvision.transforms.RandomHorizontalFlip(0.5),
    # torchvision.transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
    torchvision.transforms.RandomRotation(10, expand=False, interpolation=InterpolationMode.BILINEAR),
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.02, 0.01)),
    torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 亮度、对比度、饱和度
    # torchvision.transforms.ToTensor(),
    torchvision.transforms.CenterCrop(input_size),
])


class data_seg(Dataset):
    def __init__(self, data_path):
        ext = ['.jpg', '.png', '.bmp']

        images_path = os.path.join(data_path, "images")
        labels_path = os.path.join(data_path, "masks")

        self.Images = [images_path + '/' + f for f in os.listdir(images_path) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.Labels = [labels_path + '/' + f for f in os.listdir(labels_path) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]

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

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # type:cv2.Mat
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)  # type:cv2.Mat

        h, w, c = image.shape
        scale = input_size[0] / max(h, w)
        th, tw = round(scale * h), round(scale * w)

        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        _, label = cv2.threshold(label, 0, 255, type=cv2.THRESH_BINARY)

        pad_bottom = input_size[0] - th
        pad_right = input_size[1] - tw

        image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT)
        label = cv2.copyMakeBorder(label, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT)

        # cv2.imshow("dis1", image)
        # cv2.waitKey(1)
        #
        # cv2.imshow("dis2", label)
        # cv2.waitKey()

        image = torchvision.transforms.ToTensor()(image)
        label = torchvision.transforms.ToTensor()(label)

        # image = trans_train(image)
        # img = torchvision.transforms.ToPILImage()(image)  # type:PIL.Image.Image
        # img.show()

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
    d = data_seg('D:/work/files/deeplearn_datasets/test_datasets/xray_real')
    a = d[0]
    pass
