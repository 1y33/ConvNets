 import torch
 import torch.nn as nn

def conv_block(input):
     return nn.Sequential(
         nn.LazyBatchNorm2d(),
         nn.ReLU(inplace=True),
         nn.LazyConv2d(input,kernel_size=3,padding=1),
     )


class DenseBlock(nn.Module):
    def __init__(self,input,num_convs):
        super().__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(input))
        self.net = nn.Sequential(*layer)

    def forward(self,x):
        for layer in self.net:
            y = layer(x)
            x = torch.cat((x,y),dim=1)

        return x

def transition_block(input):
    return nn.Sequential(
        nn.LazyBatchNorm2d(),
        nn.ReLU(inplace=True),
        nn.LazyConv2d(input,kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )

# Class denset you play with the building blocks like in the paper .