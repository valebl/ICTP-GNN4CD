import argparse
import torch
import torch.nn as nn
from .registry import register_loss


@register_loss("CRPS_Loss")
class CRPS_Loss(nn.Module):
    """       
    Fair Continuous Ranked Probability Score (CRPS) for ensemble predictions,
    with an optional spectral (FFT-domain) component.

    Computes CRPS on ensemble predictions in the spatial domain, and
    optionally adds a spectral CRPS term computed in the FFT domain to
    penalize errors in the spatial power spectrum (e.g. blurry/oversmoothed
    ensembles that match pointwise CRPS but miss fine-scale structure).

    Parameters:
        y_dim (int): Height of spatial domain. Required if use_spectral=True.
        x_dim (int): Width of spatial domain. Required if use_spectral=True.
        beta (int): Power parameter for CRPS.
        alpha (float): Almost-fair CRPS parameter in (0, 1], following
            Lang et al. (2024b). alpha=1.0 gives the fully fair CRPS.
            Smaller alpha shrinks the pairwise-spread correction toward the
            conventional (biased) CRPS, which avoids a degeneracy in the fully fair
            score at small ensemble sizes, particularly relevant to keep
            M_train low. Default 0.95, matching Nordhagen et al. (2025) / Lang et al. (2024b).
        use_spectral (bool): Whether to compute and add the spectral CRPS
            term. If False, this behaves as a plain fair CRPS loss and
            y_dim/x_dim/spatial_resolution are not required.
        lambda_spectral (float): Weight for spectral CRPS. Ignored if use_spectral=False.
        spatial_resolution (float): Spatial resolution for low-pass filtering
            in the spectral term. Ignored if use_spectral=False.
        ignore_nans (bool): Ignore NaNs in target domain.
    """

    output_dim = 1

    @staticmethod
    def add_loss_specific_args(parser):
        parser.add_argument('--y_dim', type=int)
        parser.add_argument('--x_dim', type=int)
        parser.add_argument('--beta', type=float, default=1)
        parser.add_argument('--alpha', type=float, default=0.95)
        parser.add_argument('--use_spectral', action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--lambda_spectral', type=float, default=0.1)
        parser.add_argument('--spatial_resolution', type=float, default=None)
        parser.add_argument('--ignore_nans', action=argparse.BooleanOptionalAction, default=True)
        return parser

    def __init__(
        self,
        ignore_nans: bool = True,
        use_spectral: bool = False,
        y_dim: int = None,
        x_dim: int = None,
        beta: int = 1,
        alpha: float = 0.95,
        lambda_spectral: float = 0.1,
        spatial_resolution: float = None,
    ) -> None:
        super(CRPS_Loss, self).__init__()
        self.ignore_nans = ignore_nans
        self.use_spectral = use_spectral

        if self.use_spectral and (y_dim is None or x_dim is None):
            raise ValueError(
                "y_dim and x_dim must be provided when use_spectral=True."
            )

        self.y_dim = y_dim
        self.x_dim = x_dim
        self.beta = beta
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
        self.alpha = alpha
        self.lambda_spectral = lambda_spectral
        if spatial_resolution is not None and spatial_resolution <= 0:
            raise ValueError("spatial_resolution must be > 0 when provided.")
        self.spatial_resolution = spatial_resolution
        self.filter_nans = (
            False  # Control whether to filter out nans in _CRPS_pointwise
        )

    def _as_member_list(self, output) -> list:
        """
        Normalizes ensemble predictions to a list of tensors, one per member.
        Accepts either a list of tensors (as in the reference implementation)
        or a single stacked tensor of shape (M, ...), as produced e.g. by
        model.generate_ensemble().
        """
        if isinstance(output, torch.Tensor):
            return [output[i] for i in range(output.shape[0])]
        return output

    @staticmethod
    def _align_shape(a: torch.Tensor, b: torch.Tensor):
        """
        Squeezes trailing singleton dims from whichever of a/b has more
        dimensions, until their shapes match exactly. Raises a clear error
        if they can't be reconciled this way, e.g. if one still carries
        an output_dim=1 axis the other doesn't, this avoids of silent
        broadcasting into an unintended (and potentially enormous) tensor.
        """
        while a.dim() > b.dim() and a.shape[-1] == 1:
            a = a.squeeze(-1)
        while b.dim() > a.dim() and b.shape[-1] == 1:
            b = b.squeeze(-1)
        if a.shape != b.shape:
            raise ValueError(
                f"CRPSLoss: target/output shapes cannot be reconciled by "
                f"squeezing trailing singleton dims: {tuple(a.shape)} vs "
                f"{tuple(b.shape)}. Check that target and each ensemble "
                f"member represent the same quantity (e.g. one may still "
                f"carry an output_dim=1 axis the other doesn't)."
            )
        return a, b

    def _CRPS_pointwise(self, target: torch.Tensor, output) -> torch.Tensor:
        """
        Computes pointwise fair CRPS for ensemble predictions.
        Parameters:
            target (torch.Tensor): Target data.
            output (list): List of ensemble predictions.
        Returns:
            torch.Tensor: CRPS value.
        """

        # harmonize shapes
        target, output0 = self._align_shape(target, output[0])
        aligned_output = [output0]
        for out in output[1:]:
            _, out_aligned = self._align_shape(target, out)
            aligned_output.append(out_aligned)
        output = aligned_output

        if self.ignore_nans and self.filter_nans:
            nans_idx = torch.isnan(target)
            target = target[~nans_idx]
            output = [out[~nans_idx] for out in output]

        # Number of ensemble members
        M = len(output)

        # Error between target and each prediction
        first_term = 0.0
        for i in range(M):
            first_term += torch.abs(target - output[i]) ** self.beta
        first_term = first_term / M

        # Difference between all pairs of predictions (almost-fair CRPS
        # correction, Lang et al. 2024b):
        #   afCRPS_alpha = (1/M) sum_j|x_j-y| - [(1-eps)/(2M(M-1))] sum_jk|x_j-x_k|
        #   eps := (1-alpha)/M
        # alpha=1 (eps=0) recovers the fully fair CRPS.
        if M > 1:
            pairwise_term = 0.0
            for i in range(M):
                for j in range(M):
                    pairwise_term += torch.abs(output[i] - output[j]) ** self.beta
            eps = (1.0 - self.alpha) / M
            second_term = (1.0 - eps) * pairwise_term / (2 * M * (M - 1))
        else:
            second_term = 0.0

        loss = torch.mean(first_term - second_term)

        return loss

    def _reshape_to_grid(self, flat: torch.Tensor) -> torch.Tensor:
        """
        Reshapes a flat (N,) node tensor to (B, y_dim, x_dim), inferring B
        from the total length.
        """
        n = flat.shape[0]
        grid_size = self.y_dim * self.x_dim
        assert n % grid_size == 0, (
            f"CRPSLoss._FFT: length {n} is not a multiple of "
            f"y_dim*x_dim={grid_size}."
        )
        B = n // grid_size
        return flat.view(B, self.y_dim, self.x_dim)

    def _FFT(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes FFT for input data, applies low-pass filtering if needed.
        Only used when use_spectral=True.
        Parameters:
            data (torch.Tensor): Input data.
        Returns:
            torch.Tensor: FFT-transformed data.
        """

        # It does not make sense to filter out nans in the spectral domain
        self.filter_nans = False

        if isinstance(data, torch.Tensor):  # For the target
            data = [torch.nan_to_num(data, nan=0.0)]
        else:
            data = [torch.nan_to_num(d, nan=0.0) for d in data]

        data = [self._reshape_to_grid(member) for member in data]
        data = [torch.fft.rfft2(member) for member in data]

        if self.spatial_resolution is not None:
            k_nyquist = 2.0 * torch.pi / (2.0 * self.spatial_resolution)
            kx = (
                2.0
                * torch.pi
                * torch.fft.rfftfreq(self.x_dim, d=self.spatial_resolution)
            )
            ky = (
                2.0
                * torch.pi
                * torch.fft.fftfreq(self.y_dim, d=self.spatial_resolution)
            )
            k_radius = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
            low_pass_mask = k_radius <= k_nyquist
            low_pass_mask = low_pass_mask.to(device=data[0].device)
            data = [member * low_pass_mask for member in data]

        return data

    def forward(self, output, target: torch.Tensor) -> torch.Tensor:
        """
        Computes fair CRPS loss, optionally combined with spectral CRPS.
        Parameters (matches the loss_fn(y_out, y) call
        convention, output/prediction first, target second):
            output (list or torch.Tensor): Ensemble predictions, either a
                list of per-member tensors, or a single stacked tensor of
                shape (M, ...).
            target (torch.Tensor): Target data.
        Returns:
            torch.Tensor: Loss value.
        """

        output = self._as_member_list(output)

        self.filter_nans = True
        crps_field = self._CRPS_pointwise(target, output)

        if not self.use_spectral:
            return crps_field

        target_fft = self._FFT(target)[0]
        output_fft = self._FFT(output)
        crps_spectral = self._CRPS_pointwise(target_fft, output_fft)

        loss = crps_field + self.lambda_spectral * crps_spectral
        return loss