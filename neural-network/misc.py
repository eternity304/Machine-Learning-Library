import torch
import torch.nn as nn
import torch.functional as F

class WeightStandardizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same"):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
    
    def forward(self, x):
        mu = self.weight.mean(dim=[1,2,3], keepdim=True)
        sig = self.weight.var(dim=[1,2,3], keepdim=True)
        weight = (self.weight - mu) / torch.sqrt(sig)
        
        return F.conv2d(
            x,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Sigmoid()
    def forward(self, x):
        return x * self.activation(x)

class Positional_Embedding(nn.Module):
    def __init__(self, t, dim, device="cpu"):
        super().__init__()
        self.t = float(t)
        self.activation = Swish()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            self.linear1,
            self.activation,
            self.linear2
        )
        self.device = device

    def forward(self, x):
        b, c, w, h = x.shape
        dim = w * h                   
        denom_even = 10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        denom_odd = 10000 ** (torch.arange(1, dim, 2, dtype=torch.float32) / dim)
        pe = torch.zeros(dim, dtype=torch.float32).to(self.device)
        pe[0::2] = torch.sin(self.t * (denom_even ** -1))
        pe[1::2] = torch.cos(self.t * (denom_odd -1))
        pe = self.mlp(pe)
        pe = pe.reshape((w, h)).unsqueeze(0).unsqueeze(0).expand(b, c, w, h)
        return pe + x

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, dim, t=1.0, kernel_size=3, stride=1, group=4, padding="same", device="cpu"):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=out_channels // 4 if out_channels != 1 else 1, num_channels=in_channels)
        self.conv1 = WeightStandardizedConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.activation = Swish()
        self.timeEmbedding = Positional_Embedding(t, dim, device=device)
        self.norm2 = nn.GroupNorm(num_groups=out_channels // 4 if out_channels != 1 else 1, num_channels=out_channels)
        self.conv2 = WeightStandardizedConv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        self.resNetBlock = nn.Sequential(
            self.norm1,
            self.activation,
            self.conv1,
            self.timeEmbedding,
            self.norm2,
            self.activation,
            self.conv2
        )

    def forward(self, x):
        return self.activation(self.resNetBlock(x) + x)