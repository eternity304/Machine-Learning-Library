import torch

class DenoisingMatchingLoss(nn.Module):
    def __init__(self, step=1000, beta_1=10e-4, beta_t=0.02):
        super(DenoisingMatchingLoss, self).__init__()
        self.betas = torch.tensor(
            [0] + [beta_1 + t * ((beta_t-beta_1) / step) for t in range(step)],
            dtype=torch.float64  # Use float64 for higher precision
        )
    
    def alpha_t(self, timestamp):
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

    def sigma_sqr_t(self, t: int) -> float:
        # return (1 - self.alpha_t(t)) * (1 - self.alpha_bar_t(t-1)) / (1 - self.alpha_bar_t(t))
        return self.betas[t]

    def forward(self, prediction, label, timestamp):
        alpha_t = self.alpha_t(timestamp)
        alpha_bar_t = self.alpha_bar_t(timestamp)
        alpha_bar_t_prev = self.alpha_bar_t(timestamp - 1)
        sigma_sqr = self.sigma_sqr_t(timestamp)
        coef = ((1 - alpha_t) ** 2) / (2 * sigma_sqr * (1 - alpha_bar_t) * alpha_t)
        return coef * torch.norm(prediction - label, p=2)