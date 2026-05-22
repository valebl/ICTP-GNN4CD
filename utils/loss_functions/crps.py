# SPDX-License-Identifier: GPL-3.0-or-later

"""
Continuous Ranked Probability Score (CRPS) loss functions.

Ported from deep4downscaling (deep/loss/crps.py).

Authors:
    Jose Gonzalez-Abad
    Carlota Garcia
"""

import torch
import torch.nn as nn


class CRPSLoss(nn.Module):
    """
    Fair Continuous Ranked Probability Score (CRPS). The second term is divided
    by 2*M*(M-1) instead of M**2.

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow the loss function to ignore NaNs in the target.
    """

    def __init__(self, ignore_nans: bool = True) -> None:
        super().__init__()
        self.ignore_nans = ignore_nans

    def forward(self, target: torch.Tensor, output, beta: int = 1) -> torch.Tensor:

        if isinstance(output, torch.Tensor):
            output = [output]

        if self.ignore_nans:
            nans_idx = torch.isnan(target)
            target = target[~nans_idx]
            output = [out[~nans_idx] for out in output]

        M = len(output)

        first_term = 0.0
        for i in range(M):
            first_term += torch.abs(target - output[i]) ** beta
        first_term = first_term / M

        if M > 1:
            second_term = 0.0
            for i in range(M):
                for j in range(M):
                    second_term += torch.abs(output[i] - output[j]) ** beta
            second_term = second_term / (2 * M * (M - 1))
        else:
            second_term = 0.0

        loss = torch.mean(first_term - second_term)
        return loss


class CRPSSpectralLoss(nn.Module):
    """
    Fair CRPS combined with a spectral CRPS computed on the 2D Fourier
    transform of the field (Nordhagen et al. 2025).

    Parameters
    ----------
    ignore_nans : bool
        Whether to allow ignoring NaNs in the pointwise term.
    H_shape, W_shape : int
        Height/width of the predictand grid.
    beta : int
        Power parameter.
    lambda_spectral : float
        Weight of the spectral term.
    spatial_resolution : float, optional
        If set, applies a low-pass filter at the Nyquist limit.
    """

    def __init__(self, ignore_nans: bool,
                 H_shape: int, W_shape: int,
                 beta: int = 1,
                 lambda_spectral: float = 0.1,
                 spatial_resolution: float = None) -> None:
        super().__init__()
        self.ignore_nans = ignore_nans
        self.H_shape = H_shape
        self.W_shape = W_shape
        self.beta = beta
        self.lambda_spectral = lambda_spectral
        if spatial_resolution is not None and spatial_resolution <= 0:
            raise ValueError("spatial_resolution must be > 0 when provided.")
        self.spatial_resolution = spatial_resolution
        self.filter_nans = False

    def _CRPS_pointwise(self, target: torch.Tensor, output) -> torch.Tensor:
        if self.ignore_nans and self.filter_nans:
            nans_idx = torch.isnan(target)
            target = target[~nans_idx]
            output = [out[~nans_idx] for out in output]

        M = len(output)

        first_term = 0.0
        for i in range(M):
            first_term += torch.abs(target - output[i]) ** self.beta
        first_term = first_term / M

        if M > 1:
            second_term = 0.0
            for i in range(M):
                for j in range(M):
                    second_term += torch.abs(output[i] - output[j]) ** self.beta
            second_term = second_term / (2 * M * (M - 1))
        else:
            second_term = 0.0

        return torch.mean(first_term - second_term)

    def _FFT(self, data: torch.Tensor):
        self.filter_nans = False

        if isinstance(data, torch.Tensor):
            data = [torch.nan_to_num(data, nan=0.0)]
        else:
            data = [torch.nan_to_num(d, nan=0.0) for d in data]

        B = data[0].shape[0]
        if data[0].ndim == 3:
            M = data[0].shape[1]

        if data[0].ndim == 2:
            data = [member.view(B, self.H_shape, self.W_shape) for member in data]
        elif data[0].ndim == 3:
            data = [member.view(B, M, self.H_shape, self.W_shape) for member in data]

        data = [torch.fft.rfft2(member) for member in data]

        if self.spatial_resolution is not None:
            k_nyquist = 2.0 * torch.pi / (2.0 * self.spatial_resolution)
            kx = 2.0 * torch.pi * torch.fft.rfftfreq(self.W_shape, d=self.spatial_resolution)
            ky = 2.0 * torch.pi * torch.fft.fftfreq(self.H_shape, d=self.spatial_resolution)
            k_radius = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
            low_pass_mask = (k_radius <= k_nyquist).to(device=data[0].device)
            data = [member * low_pass_mask for member in data]

        return data

    def forward(self, target: torch.Tensor, output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            output = [output]

        self.filter_nans = True
        crps_field = self._CRPS_pointwise(target, output)

        target_fft = self._FFT(target)[0]
        output_fft = self._FFT(output)
        crps_spectral = self._CRPS_pointwise(target_fft, output_fft)

        return crps_field + self.lambda_spectral * crps_spectral
