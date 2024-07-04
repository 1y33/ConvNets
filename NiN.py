import torch
import torch.nn as nn
import torch.nn.functional as F


class NiN_Block(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride,padding):
        super().__init__()

        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride,padding=padding)
        self.conv1 = nn.Conv2d(out_ch,out_ch,kernel_size=1)
        self.conv2 = nn.Conv2d(out_ch,out_ch,kernel_size=1)

    def forward(self,x):
        x = F.relu(self.conv(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x

class NiN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()

        self.net = nn.Sequential(
            NiN_Block(3,96,11,4,0),
            nn.MaxPool2d(3,stride=2),
            NiN_Block(96,256,5,1,2),
            nn.MaxPool2d(3,stride=2),
            NiN_Block(256,384,3,1,1),
            nn.MaxPool2d(3,stride=2),
            nn.Dropout(0.5),
            NiN_Block(384,num_classes,3,1,1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

    def forward(self,x):
        return self.net(x)

def main():
    model = NiN(10)
    x = torch.randn(size=(3,224,224))
    print(model)
    print(model(x))


if __name__ == '__main__':
    main()