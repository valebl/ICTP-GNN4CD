import torch
import torch.nn as nn

#-----------------------------------------------------
#--------------- CUSTOM LOSS FUNCTIONS ---------------
#-----------------------------------------------------

class weighted_mse_loss():
    def __call__(input_batch, target_batch, weights):
        e = (input_batch - target_batch) ** 2
        return torch.sum(weights * e) / torch.sum(weights)


class weighted_mae_loss():
    def __call__(input_batch, target_batch, weights):
        e = torch.abs(input_batch - target_batch)
        return torch.sum(weights * e) / torch.sum(weights)


class quantized_loss():
    '''
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, alpha=0.025):
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        print(f"alpha: {self.alpha}")

    def __call__(self, prediction_batch, target_batch, bins):
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        loss_quantized = 0
        bins = bins.int()
        for b in torch.unique(bins):
            mask_b = (bins == b)
            loss_quantized += self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])
        return loss_mse + self.alpha * loss_quantized, loss_mse, loss_quantized
    

class quantized_loss_scaled():
    '''
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, gamma=0.5, scale=0.001):
        self.gamma = gamma
        self.scale = scale
        print(f"gamma: {self.gamma}, scale: {self.scale}")

    def __call__(self, prediction_batch, target_batch, bins):
        loss_quantized = 0
        bins = bins.int()
        for b in torch.unique(bins):
            mask_b = (bins == b)
            loss_quantized += torch.sum((prediction_batch[mask_b] - target_batch[mask_b])**2) * (1/torch.sum(mask_b))**self.gamma
        return self.scale * loss_quantized
    

class quantized_loss_mod():
    '''
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, alpha=1):
        self.mse_loss = nn.MSELoss(reduction="none")
        self.alpha = alpha
        print(f"alpha: {self.alpha}")

    def __call__(self, prediction_batch, target_batch, bins):
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        loss_quantized = []
        bins = bins.int()
        alpha_vector = []
        for b in torch.unique(bins):
            mask_b = (bins == b)
            loss_quantized += alpha_vector * self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])
        loss_quantized = torch.sum(torch.stack(loss_quantized))
        return loss_mse + self.alpha * loss_quantized, loss_mse, loss_quantized


class quantized_loss_bins():
    '''
    Used in inference to derive the QMSE term for the individual bins
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, alpha=0.025):
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        print(f"alpha: {self.alpha}")

    def __call__(self, prediction_batch, target_batch, bins, accelerator, nbins=12):
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        loss_quantized = 0
        bins = bins.int()
        losses = torch.ones((nbins)).to(accelerator.device) * torch.nan
        for b in torch.unique(bins):
            mask_b = (bins == b)
            losses[b] = self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])
        return losses, None, None

