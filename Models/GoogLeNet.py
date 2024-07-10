import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self,c1,c2,c3,c4,**kwargs):
        super().__init__()
        self.b1_1 = nn.LazyConv2d(c1,kernel_size=1)

        self.b2_1 = nn.LazyConv2d(c2[0],kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1],kernel_size=3,padding=1)

        self.b3_1 = nn.LazyConv2d(c3[0],kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1],kernel_size=5,padding=2)

        self.b4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.b4_2 = nn.LazyConv2d(c4,kernel_size=1)

    def forward(self,x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))

        return torch.cat((b1,b2,b3,b4),dim=1)


class GoogLeNet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.LazyConv2d(64,kernel_size=7,stride=2,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.b2 = nn.Sequential(
            nn.LazyConv2d(64,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(192,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.b3 = nn.Sequential(
            Inception(64,(96,128),(16,32),32),
            Inception(128,(128,192),(32,64),64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )

        self.b4 = nn.Sequential(
            Inception(192,(96,208),(16,48),64),
            Inception(160,(112,224),(24,64),64),
            Inception(128,(128,256),(24,64),64),
            Inception(112,(144,288),(32,64),64),
            Inception(256,(160,320),(32,128),128),
            nn.MaxPool2d(kernel_size=3,stride= 2,padding=1)
        )

        self.b5 = nn.Sequential(
            Inception(256,(160,320),(32,128),128),
            Inception(384,(192,384),(48,128),128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

        self.fc = nn.LazyLinear(num_classes)

    def forward(self,x):
        x = self.b5(self.b4(self.b3(self.b2(self.b1(x)))))
        x = self.fc(x)

        return x

