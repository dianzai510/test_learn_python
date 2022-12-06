import torch
from torch import nn


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(CBL, self).__init__()
        self._cbl = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self._cbl(x)


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """

    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C, H, W]
        Output:
            y: (Tensor) -> [B, 4C, H, W]
        """
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        y = torch.cat([x, x_1, x_2, x_3], dim=1)

        return y


def encode123(boxes, labels, grid_size, num_bbox, num_cls):
    """ Encode box coordinates and class labels as one target tensor.
    Args:
        boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...], normalized from 0.0 to 1.0 w.r.t. image width/height.
        labels: (tensor) [c_obj1, c_obj2, ...]
    Returns:
        An encoded tensor sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
    """

    S, B, C = grid_size, num_bbox, num_cls  # self.S, self.B, self.C
    N = 5 * B + C

    target = torch.zeros(S, S, N)
    cell_size = 1.0 / float(S)
    boxes_wh = boxes[:, 2:] - boxes[:, :2]  # width and height for each box, [n, 2]
    boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0  # center x & y for each box, [n, 2]
    for b in range(boxes.size(0)):
        xy, wh, label = boxes_xy[b], boxes_wh[b], int(labels[b])

        ij = (xy / cell_size).ceil() - 1.0
        i, j = int(ij[0]), int(ij[1])  # y & x index which represents its location on the grid.
        x0y0 = ij * cell_size  # x & y of the cell left-top corner.
        xy_normalized = (xy - x0y0) / cell_size  # x & y of the box on the cell, normalized from 0.0 to 1.0.

        # TBM, remove redundant dimensions from target tensor.
        # To remove these, loss implementation also has to be modified.
        for k in range(B):
            s = 5 * k
            target[j, i, s:s + 2] = xy_normalized
            target[j, i, s + 2:s + 4] = wh
            target[j, i, s + 4] = 1.0
        target[j, i, 5 * B + label] = 1.0

    return target


def encode(labels, grid_size, num_bbox, num_cls):
    """ Encode box coordinates and class labels as one target tensor.
    Args:
        boxes: (tensor) [[cx, cy, w, h]_obj1, ...], normalized from 0.0 to 1.0 w.r.t. image width/height.
        labels: (tensor) [c_obj1, c_obj2, ...]
    Returns:
        An encoded tensor sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        其中x,y为相对于当前网格的偏移量
    """

    S, B, C = grid_size, num_bbox, num_cls  # self.S, self.B, self.C
    N = 5 * B + C

    target = torch.zeros(S, S, N)
    label = labels[:, :1]
    boxes_xy = labels[:, 1:3]  # box中心坐标(归一化)
    boxes_wh = labels[:, 3:5]  # box宽高(归一化)
    for index in range(labels.size(0)):
        cls, xy, wh = int(labels[index]), boxes_xy[index], boxes_wh[index]
        ij = (xy * S).floor()  # 归一化到0~S,取整,得到目标在哪个网格
        delta_xy = (xy - ij / S) * S  # 归一化至0~1,目标位置相对于当前网格的偏移量,因为(xy - ij/S)∈[0,1/S]

        i, j = ij[0], ij[1]
        for k in range(B):
            s = 5 * k
            target[j, i, s:s + 2] = delta_xy  # xy_normalized
            target[j, i, s + 2:s + 4] = wh
            target[j, i, s + 4] = 1.0
        target[j, i, 5 * B + label] = 1.0

    return target
