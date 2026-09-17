import argparse
import torch
import torch.nn as nn

from .registry import register_loss
from .crps import CRPS_Loss


@register_loss("MultiVariable_CRPS_Loss")
class MultiVariable_CRPS_Loss(nn.Module):
    """
    Fair CRPS loss for models predicting multiple target variables at once

    Computes a per-variable fair CRPS (optionally with the spectral term,
    see CRPS_Loss) and combines them into a single scalar via a weighted
    sum, using coefficients supplied through args.

    Conventions (must match graph_dataset.py / train.py / model):
      - target_variables: comma-separated variable names, e.g.
            TARGET_VARIABLES="tas,tasmax,tasmin,hurs,psl,uas,vas,pr"
        matching the order used when stacking target files in train.py, and
        the order of the model's predictor heads.
      - multivariable_loss_coeff: comma-separated floats, same length/order
        as target_variables, e.g.
            MULTIVARIABLE_LOSS_COEFF="1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0"
        Defaults to 1.0 for every variable if not provided.
      - target: (num_nodes, n_target_variables) [+ optional trailing
        singleton dim], as produced by Graph_Dataset for a stacked
        multivariable target.
      - output: ensemble predictions of the whole multivariable model
        output, i.e. either
            * a list of M tensors, each (num_nodes, n_target_variables, output_dim), or
            * a single stacked tensor of shape (M, num_nodes, n_target_variables, output_dim)
        where M is the number of ensemble members (e.g. produced by
        repeated stochastic forward passes at training time, matching the
        convention already used by CRPS_Loss) and output_dim is the width
        of a single predictor head's output (1 for the model as given).

    Parameters:
        target_variables (str): comma-separated variable names (required).
        multivariable_loss_coeff (str or None): comma-separated per-variable
            weights, same order as target_variables. Defaults to all 1.0.
        beta, alpha, use_spectral, lambda_spectral, spatial_resolution,
        y_dim, x_dim, ignore_nans: forwarded to every per-variable CRPS_Loss
            (shared hyperparameters across variables; extend this class if
            per-variable hyperparameters are ever needed).
    """

    output_dim = 1

    @staticmethod
    def add_loss_specific_args(parser):
        parser.add_argument("--multivariable_loss_coeff", type=str, default=None,
                             help='Comma-separated per-variable loss weights, same order as '
                                  '--target_variables, e.g. "1.0,1.0,2.0". Defaults to 1.0 each.')
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
        target_variables: str,
        multivariable_loss_coeff: str = None,
        ignore_nans: bool = True,
        use_spectral: bool = False,
        y_dim: int = None,
        x_dim: int = None,
        beta: int = 1,
        alpha: float = 0.95,
        lambda_spectral: float = 0.1,
        spatial_resolution: float = None,
    ) -> None:
        super(MultiVariable_CRPS_Loss, self).__init__()

        self.target_variables = target_variables.split(",")
        n_vars = len(self.target_variables)

        if multivariable_loss_coeff is None or multivariable_loss_coeff == "":
            coeffs = [1.0] * n_vars
        else:
            coeffs = [float(c) for c in multivariable_loss_coeff.split(",")]
            if len(coeffs) != n_vars:
                raise ValueError(
                    f"multivariable_loss_coeff has {len(coeffs)} entries but "
                    f"target_variables has {n_vars}: {self.target_variables}."
                )
        self.register_buffer("coeffs", torch.tensor(coeffs, dtype=torch.float))

        # One CRPS_Loss per variable, sharing hyperparameters. Using a
        # ModuleList (rather than a single shared instance)
        self.per_var_losses = nn.ModuleList([
            CRPS_Loss(
                ignore_nans=ignore_nans,
                use_spectral=use_spectral,
                y_dim=y_dim,
                x_dim=x_dim,
                beta=beta,
                alpha=alpha,
                lambda_spectral=lambda_spectral,
                spatial_resolution=spatial_resolution,
            )
            for _ in range(n_vars)
        ])

        self.components = self.target_variables

    def _as_member_list(self, output):
        """Normalizes the multivariable ensemble output to a list of M
        tensors, each (num_nodes, n_target_variables, output_dim)."""
        if isinstance(output, torch.Tensor):
            return [output[i] for i in range(output.shape[0])]
        return output

    def forward(self, output, target: torch.Tensor):
        members = self._as_member_list(output)  # list of M x (N, n_vars, output_dim)

        # target: (N, n_vars) or (N, n_vars, output_dim)
        if target.dim() == 2:
            target = target.unsqueeze(-1)  # (N, n_vars, 1)

        total = 0.0
        loss_components = []

        for i, crps_i in enumerate(self.per_var_losses):
            members_i = [member[:, i, :] for member in members]  # list of (N, output_dim)
            target_i = target[:, i, :]                            # (N, output_dim)

            loss_i = crps_i(members_i, target_i)
            loss_components.append(loss_i)

            total = total + self.coeffs[i] * loss_i

        return total, loss_components