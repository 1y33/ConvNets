import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,6,kernel_size=5,padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv2 = nn.Conv2d(6,16,kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.fc1 = nn.Linear(400,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = F.sigmoid(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1,400)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return x


def main():
    model = LeNet()
    x = torch.randn(size=(1,28,28))
    print(model)
    print(model(x))


if __name__ == '__main__':
    main()