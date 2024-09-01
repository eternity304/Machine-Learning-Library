import torch
import torch.nn

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import DenoisingMatchingLoss

import math
import matplotlib.pyplot as plt
import tqdm as tqdm

class Diffusion(nn.Module):
    def __init__(self, model: nn.Module, dim: tuple, step: int=1000, beta_1: float=10e-4, beta_t: float=0.02, device: str="cpu"):
        super().__init__()
        self.model = model.to(device).double()
        self.device=device
        self.dim = dim
        self.betas = torch.tensor(
            [0] + [beta_1 + t * ((beta_t-beta_1) / step) for t in range(step)],
            dtype=torch.float64  # Use float64 for higher precision
        )
        
    def alpha_t(self, timestamp: int) -> float:
        '''
        alpha_t = 1 - beta_t
        '''
        return 1 - self.betas[timestamp]
    
    def alpha_bar_t(self, timestamp):
        '''
        alpha_bar_t = prod_i^t alpha_t
        '''
        out = 1.0
        if timestamp != 1:
            for t in range(1, timestamp+1):
                out *= self.alpha_t(t)
            return out
        else:
            return self.alpha_t(1)
    
    def sigma_sqr_t(self, timestamp: int) -> float:
        # return (1 - self.alpha_bar_t(timestamp-1)) / (1 - self.alpha_bar_t(timestamp)) * self.betas[timestamp]
        return self.betas[timestamp]

    def denoise(self, x: torch.Tensor, timestamp: int) -> torch.Tensor:
        epsilon = torch.randn_like(x).to(self.device)
        alpha_t = self.alpha_t(timestamp)
        alpha_bar_t = self.alpha_bar_t(timestamp)
        return (1 / math.sqrt(alpha_t)) * (x - ((1 - alpha_t) / math.sqrt(1 - alpha_bar_t)) * self.model(x, timestamp)) + math.sqrt(self.sigma_sqr_t(timestamp)) * epsilon

    def encode(self, x0: torch.Tensor, timestamp: int) -> torch.Tensor:
        '''
        x0, has shape bs, c, w, h
        epsilon, sample noise of length c * w * h, expanded to the bs's dimension, epsilon is sampled from standard normal
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon 
        '''
        epsilon = torch.randn_like(x0)
        return  math.sqrt(self.alpha_bar_t(timestamp))* x0 + math.sqrt(1 - self.alpha_bar_t(timestamp)) * epsilon

    def decode(self, start_timestamp: int=1000, final_timestamp: int=1, batch_size: int=1, x: torch.Tensor=None) -> torch.Tensor:
        '''
        1. decode from the t=1000 up to and including t=1

        3 cases:
        1000 -> 2 -> 1, no input x
        t init -> 1,  
        t init -> t final, yes input x, t final != 1
    
        '''
        h, w = self.dim
       
        z_t = torch.randn_like(torch.randn((batch_size, 1, h, w))).to(self.device).double()

        for t in tqdm(range(start_timestamp, final_timestamp-1, -1), leave=False, desc="Denosing Data"):
            z_t = self.denoise(z_t, t)
        
        return z_t
            
    def train(self, dataloader, epochs: int, lr: float=5.0e-5, save_every: int=10, save_dir: str=None, save_fig: int= 5) -> None:
        criterion = DenoisingMatchingLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        all_loss = []
        self.model.to(self.device).double()

        for epoch in tqdm(range(epochs)):
            self.model.train()
            epoch_loss = 0
            counter = 1

            for x, _ in tqdm(dataloader, leave=False, desc=f"Epcoh {epoch}"):
                x = x.to(self.device).double()
                t = torch.randint(1, 1000, (1,)).item()
                epsilon = torch.randn_like(x).to(self.device)
                y = self.model(self.encode(x, t), t)
                
                loss = criterion(y, epsilon, t) 
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

                counter += 1
                
            scheduler.step()
            all_loss.append(epoch_loss / len(dataloader))
            print(f"Epoch {epoch+1}, Batch Loss : {epoch_loss / len(dataloader)}")

            if epoch != 0 and (epoch+1) % save_fig == 0:
                plt.plot([i for i in range(len(all_loss))], all_loss)
                plt.title(f"Per Epoch Average Batch Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(f"{save_dir}epoch{epoch + 1}_loss.jpeg")
            
            if (epoch + 1) % save_every == 0 and save_dir != None:
                torch.save(self.model.state_dict(), f"{save_dir}epoch{epoch+1}_model.pth")
