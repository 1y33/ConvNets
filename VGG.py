import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG_Block(nn.Module):
    def __init__(self,nr_convs,inch,outch):
        super().__init__()

        self.layers = []
        self.layers.append(nn.Conv2d(inch,outch,3,padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        for _ in range(nr_convs-1):
            self.layers.append(nn.Conv2d(outch,outch,3,padding=1))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(2,2))

        self.layers = nn.Sequential(*self.layers)

    def forward(self,x):
        return self.layers(x)

class VGG(nn.Module):
    def __init__(self,arch,num_cls):
        super().__init__()

        self.blocks = []
        for (nr_convs,in_ch,out_ch) in arch:
            self.blocks.append(VGG_Block(nr_convs,in_ch,out_ch))

        self.model = nn.Sequential(
            *self.blocks,
            nn.Flatten(),
            nn.Linear(25088,4096),nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )

    def forward(self,x):
        return self.model(x)

def main():
    arch =((1,3,64),
           (1,64,128),
           (2,128,256),
           (2,256,512),
           (2,512,512))
    model = VGG(arch,10)
    x = torch.randn(size=(1,3,224,224))
    print(model)
    print(model(x))


if __name__ == '__main__':
    main()

