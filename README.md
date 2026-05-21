## GNN4CD U-Net Flow Matching

This branch adds a U-Net Conditional Flow Matching head after the existing GNN4CD processor. 

### Main Components

- `models/gnn4cd_unet_model.py`
  - `GNN4CD_UNet_Model`: GNN4CD encoder/processor plus a U-Net velocity head.
  - Training returns `[v_pred, v_target]` in a single tensor.
  - Evaluation and prediction also return stochastic samples.

- `utils/losses/cfm_qmse_psd.py`
  - `CFM_QMSE_PSD_Loss = FM + alpha * QMSE + beta * PSD`.
  - The recommended precipitation setting in SA is:
    `ALPHA=0.005`, `BETA=0.008`, `LR=0.0008`, `BINWIDTH=0.25`.

- `predict/predict.py`
  - Saves raw sample ensembles and deterministic summaries.
  - Here we try:
    `pr_gnn4cd_best_member`.

- `scripts/run_predict_best_member.sh`
  - Runs prediction and plots only the best-member field.

### Best-Member Selection

For each predicted day, the sampler produces `N_SAMPLES` full spatial fields.
The best-member selector chooses one complete field, not one member per grid
point. It scores each candidate by closeness to an ensemble reference field,
with optional penalties for too-weak or too-strong tails.

Key parameters are exposed in the prediction config:

```bash
BEST_MEMBER_TRANSFORM="log1p"
BEST_MEMBER_REFERENCE_QUANTILE=0.50
BEST_MEMBER_TAIL_QUANTILE=0.98
BEST_MEMBER_TAIL_TARGET=0.75
BEST_MEMBER_TAIL_WEIGHT=0.50
BEST_MEMBER_TAIL_EXCESS_WEIGHT=0.00
```

Good reference quantiles to test are `0.30`, `0.40`, and `0.50`.
For closeness-only selection, set both tail weights to `0.00`.

### Templates

Training:

```bash
bash ${MAIN_PATH}/scripts/run_train.sh \
  ${MAIN_PATH}/config_templates/train/train_pr_unet_cfm_qmse_psd_template
```

Prediction:

```bash
bash ${MAIN_PATH}/scripts/run_predict_best_member.sh \
  ${MAIN_PATH}/config_templates/predict/predict_pr_unet_best_member_template
```

