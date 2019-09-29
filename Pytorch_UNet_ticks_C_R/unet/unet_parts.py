# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, additive = True, skip_connection = True, additive_dim = 4):
        super(up, self).__init__()

        self.additive = additive
        self.skip_connection = skip_connection
        self.additive_dim = additive_dim
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        # Processing the in channel dimension
        if skip_connection == True:
            in_ch = in_ch
            if additive == True:
                add_ch = in_ch//2
                if add_ch%self.additive_dim == 0:
                    in_ch = in_ch//2 + add_ch//self.additive_dim
                else:
                    in_ch = in_ch//2 + add_ch//self.additive_dim + add_ch%self.additive_dim
        
        else:
            in_ch = in_ch//2
            if additive == True:
                if add_ch%self.additive_dim == 0:
                    in_ch = in_ch//self.additive_dim
                else:
                    in_ch = in_ch//self.additive_dim + in_ch%self.additive_dim

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        b,c,h,w = x1.shape

        if self.additive == True:
            print(x1[:,:(c-c%self.additive_dim),:,:].shape)
            x1 = torch.cat([
                x1[:,:(c-c%self.additive_dim),:,:].reshape(b,c//self.additive_dim,self.additive_dim,h,w).sum(2), 
                x1[:,-(c%self.additive_dim):,:,:]
            ], dim = 1)
        
        if self.skip_connection == True:
        # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                            diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1 

        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
