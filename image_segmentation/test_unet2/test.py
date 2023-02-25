import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
import myutils.myutils
from image_segmentation.test_unet2.data import data_seg, transform_val
from image_segmentation.test_unet2.model import UNet


def detect(opt):
    datasets_test = data_seg('D:/work/files/deeplearn_datasets/test_datasets/xray_real', transform_val,
                             transform_val)
    dataloader_test = DataLoader(datasets_test, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    net = UNet()
    checkpoint = torch.load(opt.weights)
    net.load_state_dict(checkpoint['net'], strict=False)  # 加载checkpoint的网络权重

    index = 0
    for img, label in dataloader_test:
        out = net(img)
        aa = out[0] + img[0]

        aa = (aa - torch.min(aa)) / (torch.max(aa) - torch.min(aa))
        img1 = torchvision.transforms.ToPILImage()(aa)
        img1.show()
        img1.save(f"D:\desktop\{index}.png")
        index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./run/best.pth')
    parser.add_argument('--img_path',
                        default='D:/work/files/deeplearn_datasets/test_datasets/gen_xray/out/super_test/images',
                        type=str)
    parser.add_argument('--out_path', default='./run/detect/exp', type=str)

    opt = parser.parse_args()
    detect(opt)
