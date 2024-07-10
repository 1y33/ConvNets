import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=stride,padding=1)
        self.conv2 = nn.Conv2d(out_ch,out_ch,kernel_size=3,stride=stride,padding=1)

        if(in_ch!=out_ch):
            self.conv3 = nn.Conv2d(in_ch,out_ch,kernel_size=1)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self,x):
        res = F.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(x))
        if self.conv3:
            x = self.conv3(x)
        res += x
        return res

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = self.create_block(3,64,2,True)
        self.block2 = self.create_block(64,128,4)
        # self....



    def create_block(self,in_ch,out_ch,nr_blocks,first_block=False):
        blocks = []
        for i in range(nr_blocks):
            if i==0 and not first_block :
                blocks.append(ResidualBlock(in_ch,out_ch,nr_blocks,stride=2))
            else :
                blocks.append(ResidualBlock(3,out_ch,nr_blocks,stride=1))

        return nn.Sequential(*blocks)
    def forward(self):
        x = self.block1(x)
        x = self.block2(x)

        #. ....

        return x
