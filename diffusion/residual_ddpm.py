import torch


class ResidualDDPM:
    """
    DDPM-style noise prediction for full residuals.

    Forward process:
        x_t = sqrt(alpha_bar_t) * residual
              + sqrt(1 - alpha_bar_t) * epsilon

    Training target:
        epsilon_theta(x_t, condition, t) ~= epsilon

    Reverse process:
        x_{t-1} = 1/sqrt(alpha_t) *
                  (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta)
                  + sqrt(beta_t) * z

    This follows the residual diffusion branch in the reference code, but keeps
    the deterministic condition as the existing GNN output.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device=None):
        self.timesteps = int(timesteps)
        beta = torch.linspace(beta_start, beta_end, self.timesteps, device=device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def to(self, device):
        for name in (
            "beta",
            "alpha",
            "alpha_bar",
            "sqrt_alpha_bar",
            "sqrt_one_minus_alpha_bar",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def _gather(self, values, t, ndim):
        out = values[t].float()
        return out.view(-1, *([1] * (ndim - 1)))

    def get_training_sample(self, residual):
        """
        residual: (B, N) full residual.
        """
        bsz = residual.shape[0]
        t = torch.randint(0, self.timesteps, (bsz,), device=residual.device)
        noise = torch.randn_like(residual)
        sqrt_ab = self._gather(self.sqrt_alpha_bar, t, residual.ndim)
        sqrt_omab = self._gather(self.sqrt_one_minus_alpha_bar, t, residual.ndim)
        x_t = sqrt_ab * residual + sqrt_omab * noise
        return x_t, noise, t

    def predict_x0_from_eps(self, x_t, eps_pred, t):
        sqrt_ab = self._gather(self.sqrt_alpha_bar, t, x_t.ndim)
        sqrt_omab = self._gather(self.sqrt_one_minus_alpha_bar, t, x_t.ndim)
        return (x_t - sqrt_omab * eps_pred) / (sqrt_ab + 1e-8)

    @torch.no_grad()
    def sample(
        self,
        model,
        cond,
        static_feats,
        low2high_edge_index,
        lr_feats,
        noise_scale=1.0,
        reverse_noise_scale=1.0,
        snapshot_interval=0,
    ):
        """
        cond:     (B, N) GNN conditional mean in model space.
        lr_feats: (B, N_low, n_lr)

        If snapshot_interval > 0, return intermediate residual fields every
        snapshot_interval completed reverse steps.
        """
        x = noise_scale * torch.randn_like(cond)
        snapshots = {}

        for completed, step in enumerate(reversed(range(self.timesteps)), start=1):
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            eps_theta = model(x, cond, static_feats, low2high_edge_index, lr_feats, t.float())

            beta_t = self._gather(self.beta, t, x.ndim)
            alpha_t = self._gather(self.alpha, t, x.ndim)
            alpha_bar_t = self._gather(self.alpha_bar, t, x.ndim)

            x_mean = torch.rsqrt(alpha_t) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t + 1e-8)) * eps_theta
            )

            if step > 0:
                x = x_mean + reverse_noise_scale * torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = x_mean

            if snapshot_interval > 0 and completed % snapshot_interval == 0:
                snapshots[completed] = x.detach().clone()

        if snapshot_interval > 0 and self.timesteps not in snapshots:
            snapshots[self.timesteps] = x.detach().clone()
            return x, snapshots

        if snapshot_interval > 0:
            return x, snapshots

        return x
