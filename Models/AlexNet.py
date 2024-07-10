import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,96,kernel_size=11,stride=4,padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96,256,kernel_size=5,padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256,384,kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384,384,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(384,256,kernel_size=3,padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)



        self.fc1 = nn.Linear(6400,4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096,4096)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(4096,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)

        x = x.view(-1,6400)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


def main():
    model = AlexNet()
    x = torch.randn(size=(3,224,224))
    print(model)
    print(model(x))


if __name__ == '__main__':
    main()