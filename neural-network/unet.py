import torch
import torch.nn as nn
import torch.functional as F

from attention import Attention
from misc import ResNet

class Unet(nn.Module):
    def __init__(self, device="cpu"):
        '''
        x --> conv1 --> res1 --> downsample --> attention --> res2 --> upsample --> res3 --> conv2 --> y
        input size: bs, 1, 28, 28
        _,1,28,28 --> _,4,26,26 --> _,4,12,12 --> _,4,12,12 --> _,4,24,24 --> _,1,26,26

        x _,1,28,28
        Down:
            - conv1 --> _,4,26,26
            - res1
            - downsample1 --> _,4,12,12
            - conv2 --> -,8,10,10
            - downsample1 --> _,8,5,5
        Middle:
            - attention1 
            - res3
            - attention2
        Up:
            - upsample --> -, 8, 10, 10
            - conv3 + res --> -, 4, 13, 13
            - upsample --> -, 4, 26, 26
            - conv4 + res --> -, 1, 28, 28
        '''
        super().__init__()
        self.device = device

        # Down
        self.conv1 = nn.Conv2d(1,4,5,padding=1,stride=1)
        self.res1 = ResNet(4, 4, 26**2, device=self.device)
        self.downSample1 = nn.Conv2d(4,4,3,stride=2)
        self.conv2 = nn.Conv2d(4,8,5,padding=1,stride=1)
        self.norm1 = nn.LayerNorm([8,10,10])
        self.res2 = ResNet(8, 8, 10**2, device=self.device)
        self.downSample2 = nn.Conv2d(8,8,4,padding=1,stride=2)

        # Middle
        self.attention1 = Attention(5**2, 8, device=self.device)
        self.res3 = ResNet(8, 8, 5**2, device=self.device)
        self.attention2 = Attention(5**2, 8, device=self.device)

        # Up 
        self.upSample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3 = nn.Conv2d(8, 4, 2,stride=1,padding=2)
        self.res4 = ResNet(4, 4, 13**2, device=self.device)
        self.conv4 = nn.Conv2d(4, 1, 5, stride=1, padding=3)
        self.res5 = ResNet(1, 1, 28**2, device=self.device)
        self.activation = Swish()
        self.norm = nn.LayerNorm([1,28,28])
    
    def forward(self, x, t):
        x = self.conv1(x)
        x, x_res1 = self.res1(x, t), self.res1(x, t)
        x = self.downSample1(x)
        x = self.conv2(x)
        x = self.norm1(x)
        x, x_res2 = self.res2(x, t), self.res2(x, t)
        x, x_res3 = self.downSample2(x), self.downSample2(x)
        
        x = self.attention1(x)
        x = self.res3(x, t)
        x = self.attention2(x)

        x = self.upSample(x + x_res3)
        x = self.conv3(x)
        x = self.res4(x, t)
        x = self.upSample(x)
        x = self.conv4(x + x_res1)
        x = self.res5(x, t)
        x = self.activation(x)
        x = self.norm(x)
        
        return x