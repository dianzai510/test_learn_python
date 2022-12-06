import os
import torch
import torchvision
from PIL.Image import Image
from torch.utils.data import Dataset


# 将label转换为7x7x30的张量.
class data(Dataset):
    def __init__(self, data_path, image_size=448, grid_size=7, num_bboxes=2, num_classes=20):
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.Images = []
        self.Labels = []

        images_path = os.path.join(data_path, "Images")
        labels_path = os.path.join(data_path, "Labels")

        self.Images = [f for f in os.listdir(images_path) if f.endswith('.jpg') or f.endswith('.png')]
        self.Labels = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]
        label_path = self.Labels[item]

        img = Image.open(image_path)
        img = torchvision.transforms.ToTensor()(img)

        label = torch.zeros(self.grid_size, self.grid_size, (self.num_bboxes * (4 + 1) + self.num_classes))
        with open(label_path, 'r') as f:
            lines = f.readlines()
            lines = [l.split(' ') for l in lines]

            for data_list in lines:
                data_list = [float(f.strip()) for f in data_list]
                cls = data_list[0]
                x = int(data_list[1] * 7)  # 向下取整
                y = int(data_list[2] * 7)  # 向下取整
                w = data_list[3]
                h = data_list[4]

                # label = torch.tensor([x, y, w, h])
                label[x, y] = torch.tensor([w, h, 1, cls])
        return img, label

    def encode(self, boxes, labels, grid_size, num_bbox, num_cls):
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


if __name__ == '__main__':
    Images = [f for f in os.listdir('D:/下载/Rotated-RetinaNet-master') if f.endswith('.py')]

    a = torch.rand(3, 4)
    print(a)
    mask = torch.randint(2, (3, 4))
    print(mask)
    mask = mask == 1
    print(mask)

    print(a[mask])

    # expand(),expand_as()函数只能将size = 1的维度扩展到更大的尺寸，如果扩展其他size（）的维度会报错。
