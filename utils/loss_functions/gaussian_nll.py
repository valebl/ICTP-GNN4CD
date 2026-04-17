import torch
import torch.nn as nn


class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, sigma, target):
        return torch.mean(0.5 * ((target - mu)**2 / (sigma**2) + 2 * torch.log(sigma)))
