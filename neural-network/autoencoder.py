import torch
import torch.nn as nn
import torch.functional as F

from misc import ResNet
from attention import Attention

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import math
import matplotlib.pyplot as plt
import tqdm as tqdm

class AutoEncoder(nn.Module):
    def __init__(self, device="cpu"):
        '''
        x --> conv1 --> res1 --> downsample --> attention --> res2 --> upsample --> res3 --> conv2 --> y
        input size: bs, 1, 28, 28
        _,1,28,28 --> _,4,26,26 --> _,4,12,12 --> _,4,12,12 --> _,4,24,24 --> _,1,26,26
        '''
        super().__init__()
        self.device = device

        # Down
        self.conv1 = nn.Conv2d(1,4,5,padding=1)
        self.res1 = ResNet(4, 4, 26**2, device=self.device)
        self.downSample = nn.Conv2d(4,4,3,stride=2)
        self.res2 = ResNet(4, 4, 12**2, device=self.device)
        
        # Middle
        self.attention = Attention(12**2, 4, device=self.device)

        # Up 
        self.upSample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv2d(4, 1, 5,stride=1,padding=3)
        self.res3 = ResNet(1, 1, 26**2, device=self.device)
        self.conv3 = nn.Conv2d(1, 1, 5, stride=1, padding=3)

        self.ae = nn.Sequential(
            self.conv1,
            self.res1,
            self.downSample,
            self.attention,
            self.res2,
            self.upSample,
            self.conv2,
            self.res3,
            self.conv3
        )
    
    def forward(self, x):
        return self.ae(x)

    def train(model, dataloader, epochs, lr=0.001, device="cpu"):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        all_loss = []
        device = device
        model = model.to(device)

        for epoch in tqdm(range(epochs)):
            model.train()
            epoch_loss = 0

            for x, _ in tqdm(dataloader, leave=False, desc=f"Epoch {epoch}"):
                x = x.to(device)
                y = model(x)
                loss = criterion(y, x)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item() / x.shape[0]

            scheduler.step()
            all_loss.append(epoch_loss / len(dataloader))

            print(f"Epoch {epoch}, loss : {epoch_loss}")

            if epoch != 0 and epoch+1 % 5 == 0:
                plt.plot([i for i in range(len(all_loss))], all_loss)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.show()
