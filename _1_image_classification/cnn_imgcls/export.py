import argparse
import onnx
import torch.onnx.utils
from data import data_xray_sot23, data_xray_sc88, data_xray_sc70, data_xray_sc89, \
    data_xray_sod123, data_xray_sod323, data_xray_sot23_juanpan, data_xray_sod523, data_xray_sod723, data_xray_sot25, \
    data_xray_sot26, data_xray_sot23e, data_oqa_chr, data_oqa_agl, data_cleaner
from models import net_xray
from our1314.myutils import exportsd, importsd


def export(opt):
    path = opt.weights
    f = path.replace('.pth', '.onnx')

    input_size = (1, 3) + opt.data.input_size
    x = torch.randn(input_size)
    checkpoint = torch.load(path)

    net = net_xray(False, opt.data.class_num)  # classify_net1()
    net.load_state_dict(checkpoint['net'])
    net.eval()
    torch.onnx.export(net,
                      x,
                      f,
                      opset_version=10,
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
    parser.add_argument('--weights', default='./run/train/exp_xray_sot23_juanpan/weights/best.pth')  # 修改
    parser.add_argument('--data', default=data_xray_sot23_juanpan, type=dict)  # 修改

    opt = parser.parse_args()
    export(opt)
