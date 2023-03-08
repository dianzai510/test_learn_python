import pathlib
from torchvision.models import resnet18

#
# result_path = pathlib.Path('results.txt')
# # result_path.write_text('aaa')  # write results.txt
# # result_path.write_text('ddd',newline=Tru
# with result_path.open('w') as fp:
#     fp.write('aaaaa\n')
import torch
from torch import nn

from myutils import exportsd, myutils


class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.Conv2d(64, 3, 3))
        # self.conv3 = nn.Conv2d(64, 32, 3)
        # self.conv4 = nn.Conv2d(32, 3, 3)
        self.flatten = nn.Flatten()

        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = self.flatten(x)
        return x


if __name__ == '__main__':
    x = torch.rand((1, 3, 64, 64))
    net = mynet()
    print(net)
    y = net(x)
    print(y.shape)

    # Saving a TorchSharp format model in Python
    path = "D:/desktop/a.pth"
    save_path = path.replace('.pth', '.dat')
    f = open(save_path, "wb")
    exportsd.save_state_dict(net.to("cpu").state_dict(), f)
    f.close()
    pass
