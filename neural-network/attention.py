import torch
import torch.nn as nn
import torch.functional as F

class Attention(nn.Module):
    def __init__(self, sequence_length, in_channels, num_head=4, dim_head=None, bias=False, device="cpu"):
        super().__init__()
        self.in_channels=in_channels
        self.seq_len = sequence_length
        self.num_head=num_head
        self.dim_head=dim_head if dim_head != None else self.in_channels
        self.dim_Q = self.dim_head
        self.dim_K = self.dim_Q
        self.dim_V = self.dim_head
        self.bias = bias

        self.multi_head = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.in_channels, self.dim_Q, bias=self.bias),  # W_ki
                nn.Linear(self.in_channels, self.dim_K, bias=self.bias),  # W_qi
                nn.Linear(self.in_channels, self.dim_V, bias=self.bias)   # W_vi
            ]) for _ in range(num_head)
        ])
        
        self.W_o = nn.Linear(self.num_head * self.in_channels, self.in_channels)
        self.device = device

    def self_attention(self, W, x):
        '''
        x has dim b, l, c where l = w * h
        K has dim b, l, dimK
        Q has dim b, l, dimQ
        V has dim b, l, dimV
        '''
        # Compute KQV 
        K = W[0](x)                                                    
        Q = W[1](x)                                                    
        V = W[2](x)                                                 
        
        # Compute Softmax
        QK_t = torch.einsum("blc, bcL -> blL", Q, K.transpose(-1, -2) )
                                          
        return torch.einsum("blL,bLc->blc", QK_t, V)     
        
    def forward(self, x):
        b, c, w, h = x.shape
        
        if self.seq_len !=  w * h:
            raise valueError(f"Expect a sequence lenght of {self.num_pixel}, but got {w*h}")  

        # Reshape image from batch, channel, width, height --> batch, pixel, channel
        x = x.permute(0, 2, 3, 1).reshape(b, self.seq_len, c)

        # Compute multi-head
        heads = [self.self_attention(W_KQV, x) for W_KQV in self.multi_head]
        
        # Compute final-embbeding
        z = self.W_o(torch.concat(heads, dim=2))
        return z.permute(0,2,1).reshape(b,c,w,h)