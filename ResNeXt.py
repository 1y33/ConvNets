import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNeXt_Block(nn.Module):
    def __init__(self,num_chans,groups,bot_mul,use1x1_conv=False,strides=1):
        super().__init__()

        bot_channels = int(round(num_chans*bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels,kernel_size=1,stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels,kernel_size=3,stride=strides,
                                   groups = bot_channels/groups)
        self.conv3 = nn.LazyConv2d(num_chans,kernel_size=1,stride=1)

        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()

        if use1x1_conv:
            self.conv4 = nn.LazyConv2d(num_chans,kernel_size=1,stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self,x):
        res = F.relu(self.bn1(self.conv1(x)))
        res = F.relu(self.bn2(self.conv2(res)))
        res = self.bn3(self.conv3(res))
        if self.conv4:
            res = self.bn4(self.conv4)

        return F.relu(x+res)

