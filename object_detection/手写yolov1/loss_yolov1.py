import torch


class loss_yolov1:
    def __init__(self):
        pass

    '''
    通过预测值与标签值计算损失
    '''

    def __call__(self, preds, labels):
        batch_size = preds.size(0)
        coord_mask = labels[:, :, :, 4] > 0
        noobj_mask = labels[:, :, :, 4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(labels)

        pass
