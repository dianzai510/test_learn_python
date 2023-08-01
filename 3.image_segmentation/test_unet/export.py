import argparse
import onnx
import torch.onnx.utils
from data import input_size
from model import UNet
from our1314.myutils import exportsd, importsd


def export(opt):
    path = opt.weights
    f = path.replace('.pth', '.onnx')

    size = (1, 3) + input_size
    x = torch.randn(size)
    checkpoint = torch.load(path)

    net = UNet()  # classify_net1()
    net.load_state_dict(checkpoint['net'])
    net.eval()
    torch.onnx.export(net,
                      x,
                      f,
                      opset_version=11,
                      # do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      verbose='True')

    # Checks 参考 yolov7
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    print('export onnx success!')

    # 导出pt,用于torchSharp
    net_ = torch.jit.trace(net, x)
    f = path.replace('.pth', '.pt')
    net_.save(f)
    print('export pt success!')

    # Saving a TorchSharp format model in Python
    save_path = path.replace('.pth', '.dat')
    f = open(save_path, "wb")
    exportsd.save_state_dict(net.to("cpu").state_dict(), f)
    f.close()
    # Loading a TorchSharp format model in Python
    f = open(save_path, "rb")
    net.load_state_dict(importsd.load_state_dict(f))
    f.close()
    print('export TorchSharp model success!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./run/train/best.pth')  # 修改

    opt = parser.parse_args()
    export(opt)
