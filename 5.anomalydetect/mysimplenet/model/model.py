import torch
from torch import nn
from torchvision import models
from patchmaker import PatchMaker
import common
from discriminator import Discriminator


_BACKBONES = {
    "cait_s24_224" : "cait.cait_S24_224(True)",
    "cait_xs24": "cait.cait_XS24(True)",
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet18": "models.resnet18(pretrained=True)",
    "resnet50": "models.resnet50(pretrained=True)",
    "mc3_resnet50": "load_mc3_rn50()", 
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "ref_wideresnet50": "load_ref_wrn50()",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
}

class simplenet(nn.Module):
    def __init__(self):
        super(simplenet, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #0、参数
        self.input_shape = (3,288,288)
        self.pre_proj = 1

        #1、主干网
        self.backbone = eval(_BACKBONES['wideresnet50'])
        
        self.patch_maker = PatchMaker(patchsize=3, stride=1)

        #2、特征提取和聚合模块(结合了主干网)
        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(self.backbone, self.layers_to_extract_from, self.device, train_backbone=False)
        feature_dimensions = feature_aggregator.feature_dimensions(self.input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator#特征预适应聚合器

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension=1536)
        self.forward_modules["preprocessing"] = preprocessing#预处理器

        self.target_embed_dimension = 1536
        preadapt_aggregator = common.Aggregator(target_dim=self.target_embed_dimension)
        preadapt_aggregator.to(device=self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator#预适应聚合器

        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=self.input_shape[-2:])

        #3、判别器模块
        self.th = 0.5#判别器阈值
        self.lr = 0.0002
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=2, hidden=1024)
        self.discriminator.to(self.device)

    def forward(self, x, train=True):
        if train == False:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](x, eval=False)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](x)

        
        pass

if __name__ == "__main__":
    print(_BACKBONES['wideresnet50'])
    net = eval(_BACKBONES['wideresnet50'])
    print(net)
    pass