import os
import torch
from model import UNet


if __name__ == "__main__":
    path_best = f"./run/train/best.pth"
    checkpoint = torch.load(path_best)

    net = UNet()
    net.load_state_dict(checkpoint['net'])
    