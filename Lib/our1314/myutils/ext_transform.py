import torchvision

# 按比例将长边缩放至目标尺寸
class Resize1:
    def __init__(self, width):
        self.width = width

    def __call__(self, x):
        _, h, w = x.shape
        scale = self.width / max(w, h)
        W, H = round(w * scale), round(h * scale)
        x = torchvision.transforms.Resize((H, W))(x)
        return x


# class Resize2():
#     def __init__(self, width):
#         self.width = width
#
#     def __call__(self, img):
#         h, w, c = img.shape
#         scale = self.width / max(w, h)
#         W, H = round(scale * w), round(scale * h)
#         dst = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
#         return dst


class PadSquare:
    def __call__(self, x):
        _, h, w = x.shape
        width = max(w, h)
        pad_left = round((width - w) / 2.0)
        pad_right = width - w - pad_left
        pad_up = round((width - h) / 2.0)
        pad_down = width - h - pad_up
        x = torchvision.transforms.Pad((pad_left, pad_up, pad_right, pad_down))(x)
        return x