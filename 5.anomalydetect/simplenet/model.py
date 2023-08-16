import torch
from torchvision.models.resnet import resnet50, ResNet50_Weights,wide_resnet50_2,Wide_ResNet50_2_Weights
class SimpleNet(torch.nn.Module):
    def __init__(self, device):
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.layers_to_extract_from = None
        
    def forward(self, x):
        pass