import torch
from torch import nn


class loss_fn(nn.Module):
    def __init__(self):
        super(loss_fn, self).__init__()

    def iou(self, bbox1, bbox2):
        pass

    def forward(self, pred, label):
        S, B, C = 7, 2, 2
        N = 12

        coord_mask = label[:, :, :, 4] > 0  # 获取同样尺寸的mask
        coord_mask = coord_mask.unsqueeze(-1)  # 在末尾插入维度
        coord_mask = coord_mask.expand_as(pred)  # 复制末端维度

        noobj_mask = label[:, :, :, 4] == 0
        noobj_mask = noobj_mask.unsqueeze(-1)  # 在末尾插入维度
        noobj_mask = noobj_mask.expand_as(label)  # 复制末端维度

        coord_pred = pred[coord_mask].view(-1, N)
        bbox_pred = coord_pred[:, :5 * B].contiguous().view(-1, 5)
        cls_pred = coord_pred[:, 5 * B:]

        noobj_pred = pred[noobj_mask].view(-1, N)
        noobj_label = label[noobj_mask].view(-1, N)

        noobj_conf_mask = 0
        noobj_pred_conf = noobj_label[noobj_conf_mask]

        # 计算损失
        # 有无目标损失、坐标损失、分类损失
        pass


if __name__ == '__main__':
    pred = torch.randint(0, 3, (3, 7, 7, 12))
    label = torch.randint(0, 3, (3, 7, 7, 12))
    loss = loss_fn()

    loss(pred, label)
