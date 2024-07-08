import torch
import torch.nn as nn
import torch.functional as F

class ConvMixerLayer(nn.Module):
    def __init__(self,dim,kernel_size):
        super().__init__()

        self.depth_wise_conv = nn.Conv2d(dim,dim,
                                         kernel_size=kernel_size,
                                         groups = dim,
                                         padding= (kernel_size-1)//2)
        self.act = nn.GELU()
        self.norm1 = nn.BatchNorm2d(dim)
        self.point_wise_conv = nn.Conv2d(dim,dim,kernel_size=1)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self,x:torch.Tensor):
        residual = x

        x = self.depth_wise_conv(x)
        x = self.act(x)
        x = self.norm1(x)

        x += residual

        x = self.point_wise_conv(x)
        x = self.act(x)
        x = self.norm2(x)

        return x


class PatchEmbeddings(nn.Module):
    def __init__(self,dim,patch_size,in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,dim,kernel_size=patch_size,stride=patch_size)
        self.att = nn.GELU()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self,x:torch.Tenosr):
        return self.norm(self.att(self.conv(x)))


class Classification(nn.Module):
    def __init__(self,dim,num_cls):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(dim,num_cls)

    def forward(self,x):
        x = self.pool(x)
        x = x[:,:,0,0]
        x = self.linear(x)

        return x

class ConvMixer(nn.Module):
    def __init__(self,
                 conv_mixer_layer: ConvMixerLayer,
                 n_layers = 10,
                 path_emb = PatchEmbeddings,
                 classification = Classification):
        super().__init__()

        self.patch_emb = path_emb
        self.classification = classification
        self.conv_mixer_layer = nn.ModuleList([ConvMixerLayer for _ in range(n_layers)])

    def forward(self,x):
        x = self.patch_emb(x)
        for layer in self.conv_mixer_layer:
            x = layer(x)
        x = self.classification(x)

        return x

