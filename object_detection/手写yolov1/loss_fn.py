import torch
from torch import nn
import torch.nn.functional as F

class loss_fn(nn.Module):
    def __init__(self, S, B, C):
        super(loss_fn, self).__init__()
        self.S = S
        self.B = B
        self.C = C

    def xywh2xyxy(self, xywh):
        xy, wh = xywh[:2], xywh[2:]
        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2

        return x1y1

    def compute_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = self.xywh2xyxy(bbox1)
        a1, b1, a2, b2 = self.xywh2xyxy(bbox2)
        ax = max(x1, a1)  # 相交区域左上角横坐标
        ay = max(y1, b1)  # 相交区域左上角纵坐标
        bx = min(x2, a2)  # 相交区域右下角横坐标
        by = min(y2, b2)  # 相交区域右下角纵坐标

        area_bbox1 = (x2 - x1) * (y2 - y1)  # bbox1的面积
        area_bbox2 = (a2 - a1) * (b2 - b1)  # bbox2的面积

        w = max(0, bx - ax)
        h = max(0, by - ay)
        area_X = w * h  # 交集
        result = area_X / (area_bbox1 + area_bbox2 - area_X)
        return result

    def forward(self, pred, label):
        S, B, C = self.S, self.B, self.C
        N = 12

        # 1、获取含有目标的索引
        coobj_mask = label[:, :, :, 4] > 0  # 获取同样尺寸的mask,shape=[batch, 7, 7]
        coobj_mask = coobj_mask.unsqueeze(-1)  # 在末尾插入维度,shape=[batch, 7, 7, 1]
        coobj_mask = coobj_mask.expand_as(pred)  # 复制末端维度,shape=[batch, 7, 7, 12]

        # 2、获取不含有目标的索引
        noobj_mask = label[:, :, :, 4] == 0
        noobj_mask = noobj_mask.unsqueeze(-1)  # 在末尾插入维度
        noobj_mask = noobj_mask.expand_as(label)  # 复制末端维度

        # 3、从预测值中提取应该含有目标的数据，并分为：bbox数据和cls数据
        coobj_pred = pred[coobj_mask].view(-1, N)  # 从预测值中提取含有目标的数据，并reshape成若干行,12列
        bbox_pred = coobj_pred[:, :5 * B].contiguous().view(-1, 5)  # 从含目标的数据中提取box信息，并reshape成(n,5)
        cls_pred = coobj_pred[:, 5 * B:]  # 从含目标的数据中提取分类信息

        # 4、从label中提取有目标的数据,并分为：bbox数据和cls数据
        coobj_label = label[coobj_mask].view(-1, N)
        bbox_label = coobj_label[:, :5 * B].contiguous().view(-1, 5)
        cls_label = coobj_label[:, 5 * B:]

        # 5、计算不含目标的损失
        noobj_pred = pred[noobj_mask].view(-1, N)#从预测值中提取应该不含目标的数据
        noobj_lable = label[noobj_mask].view(-1, N)#从标签中提取不含目标的数据
        noobj_pred_mask = torch.ByteTensor(noobj_pred.size())
        noobj_pred_mask.zero_()
        noobj_pred_mask[:, 4] = 1
        noobj_pred_mask[:, 9] = 1
        noobj_pred_c = noobj_pred[noobj_pred_mask]# 预测数据中不含目标的置信度
        noobj_label_c = noobj_lable[noobj_pred_mask]#标签中不含目标的置信度
        noobj_loss = F.mse_loss(noobj_pred_c, noobj_label_c, reduce='sum')  # 计算无目标的损失

        # 6、计算含有目标的损失
        coobj_response_mask = torch.ByteTensor(bbox_label.size())
        coobj_response_mask.zero_()
        coobj_not_response_mask = torch.ByteTensor(bbox_label.size())
        coobj_not_response_mask.zero_()
        bbox_label_iou = torch.zeros(bbox_label.size())
        for i in range(0, bbox_label.size()[0], 2):
            # 选择最佳IOU box
            box_pred = bbox_pred[i:i+self.B]
            box_pred_xyxy = torch.FloatTensor(box_pred.size())
            # (x,y,w,h)
            box_pred_xyxy[:,  :2] = box_pred[:,:2] / self.S - 0.5 * box_pred[:,2:4]
            box_pred_xyxy[:, 2:4] = box_pred[:,:2] / self.S + 0.5 * box_pred[:,2:4]

            box_label = bbox_label[i].view(-1, 5)
            box_label_xyxy = torch.FloatTensor(box_label.size())
            box_label_xyxy[:, :2] = box_label[:,:2] / self.S - 0.5 * box_label[:,2:4]
            box_label_xyxy[:,2:4] = box_label[:,:2] / self.S - 0.5 * box_label[:,2:4]

            iou = self.compute_iou(box_pred_xyxy[:,:4], box_label_xyxy[:,:4])

            #label匹配到的box,在self.B个预测box中获取与label box iou值最大的那个box的索引
            max_iou, max_index = iou.max(0)
            coobj_response_mask[i+max_index] = 1
            coobj_not_response_mask[i+1-max_index] = 1

            bbox_label_iou[i+max_index, 4] = max_iou
            pass

        bbox_label_iou = Variable(bbox_label_iou)

        # noobj_pred = pred[noobj_mask].view(-1, N)
        # noobj_label = label[noobj_mask].view(-1, N)
        #
        # noobj_conf_mask = 0
        # noobj_pred_conf = noobj_label[noobj_conf_mask]

        # 计算损失
        # 有无目标损失、坐标损失、分类损失
        pass


if __name__ == '__main__':
    pred = torch.randint(0, 3, (1, 7, 7, 12))
    label = torch.randint(0, 3, (1, 7, 7, 12))
    loss = loss_fn()

    loss(pred, label)
