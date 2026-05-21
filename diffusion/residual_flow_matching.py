import torch


class ResidualFlowMatching:
    """
    Conditional flow matching for stochastic high-frequency residuals.

    The model learns a vector field from Gaussian noise to the high-pass
    residual between log1p(target precipitation) and log1p(GNN prediction).
    The final precipitation field is reconstructed outside this class as:

        log_pr = log1p(gnn_pr) + sampled_high_frequency_residual
    """

    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min

    def get_training_sample(self, residual):
        """
        residual: (B, N) high-frequency residual in log1p space.
        """
        x0 = torch.randn_like(residual)
        t = torch.rand(residual.shape[0], device=residual.device)
        t_bc = t.view(-1, 1)
        x_t = (1.0 - t_bc) * x0 + t_bc * residual
        x_t = x_t + self.sigma_min * torch.randn_like(x_t)
        target_v = residual - x0
        return x_t, target_v, t

    @torch.no_grad()
    def sample(self, model, cond, static_feats, low2high_edge_index, lr_feats,
               n_steps=50, noise_scale=1.0):
        """
        cond:     (B, N) log1p GNN precipitation.
        lr_feats: (B, N_low, n_lr)
        """
        x = noise_scale * torch.randn_like(cond)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((x.shape[0],), t_val, device=x.device)
            v = model(x, cond, static_feats, low2high_edge_index, lr_feats, t)
            x = x + dt * v

        return x
