import torch
import torch.nn as nn
import numpy as np
from utils.helpers.tools import write_log
from .registry import register_loss


@register_loss("QMSE_Loss")
class QMSE_Loss(nn.Module):

    @staticmethod
    def add_loss_specific_args(parser):
        parser.add_argument("--alpha", type=float)
        parser.add_argument("--beta", type=float)
        parser.add_argument('--binmin', type=float)
        parser.add_argument('--binmax', type=float)
        parser.add_argument('--binwidth', type=float)
        parser.add_argument('--binscale', type=str)
        return parser

    def __init__(self, balance=None):
        super().__init__()
        self.balance = balance
        self.mse_loss = nn.MSELoss()

    def forward(self, prediction_batch, target_batch, bins):
        loss_quantized = 0
        bins = bins.int()
        bins_unique = torch.unique(bins)
        for b in bins_unique:
            mask_b = (bins == b)
            count_b = mask_b.sum()
            loss_b = self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])

            # Bin balancing
            if self.balance == "count":
                weight = 1.0 / count_b
            elif self.balance == "sqrt":
                weight = 1.0 / torch.sqrt(count_b.float())
            else:
                weight = 1.0
            loss_quantized += weight * loss_b

        return loss_quantized


def derive_qmse_bins(target, train_idxs, args, accelerator, binmin=np.log1p(0.1), binmax=np.log1p(200), binwidth=np.log1p(0.5), bins=None):

    # Precompute bins
    if bins is None:
        bins = np.arange(binmin, binmax, binwidth)

    # Histogram only on training subset (flatten for speed)
    train_vals = target[:, train_idxs].ravel()
    values_unif_log, edges_unif_log = np.histogram(train_vals, bins=bins)

    # searchsorted returns indices in [0, len(edges)-1]
    target_bins = np.searchsorted(edges_unif_log, target, side="left").astype(float, copy=False)

    # Handle NaNs in target
    nan_mask = np.isnan(target)
    if nan_mask.any():
        target_bins[nan_mask] = np.nan

    # Determine number of bins
    if nan_mask.any():
        max_bin = int(np.nanmax(target_bins))
    else:
        max_bin = int(target_bins.max())

    nbins = max_bin + 1

    # Fix case with an out-of-range bin
    if nbins > len(values_unif_log):
        write_log(
            f"\nBins min: {int(np.nanmin(target_bins))}, "
            f"bins max: {max_bin}, nbins: {nbins}, "
            f"len weights: {len(values_unif_log)}",
            args, accelerator, 'a'
        )

        # Clamp last bin
        target_bins[target_bins == nbins - 1] = nbins - 2
        nbins -= 1

        write_log("\nUpdating last bin...", args, accelerator, 'a')

    write_log(
        f"\nbins min: {int(np.nanmin(target_bins))}, "
        f"bins max: {int(np.nanmax(target_bins))}, nbins: {nbins}",
        args, accelerator, 'a'
    )

    write_log(f"\nBins: {bins}", args, accelerator, 'a')

    target_bins[target < 0.1] = np.nan

    return target_bins


