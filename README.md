## GNN Output Residual Flow Matching

This branch contains a lightweight post-processing workflow for GNN4CD
downscaling outputs. The deterministic GNN prediction is used as the large-scale anchor, and a Conditional Flow Matching model learns the stochastic high-frequency residual between the GNN output and the high-resolution target.

For precipitation, training is done in `log1p(mm/day)` space:

```text
residual_raw = log1p(target_pr) - log1p(gnn_pr)
residual_hf  = residual_raw - smooth(residual_raw)
```

The flow-matching model learns only `residual_hf`. During prediction, the
sampled residual is added back to the GNN field and transformed back to
physical precipitation.

### Files

- `train_residual_fm_lr.py`: trains the residual flow-matching model.
- `predict_residual_fm_lr.py`: samples residuals and writes a GNN4CD-compatible
  `HeteroData` prediction file.
- `dataset_residual_fm.py`: aligns GNN predictions, target data, low-resolution
  predictors, and builds high-pass residual targets.
- `diffusion/residual_flow_matching.py`: linear conditional flow matching.
- `models/residual_denoiser_lr.py`: U-Net velocity model conditioned on GNN
  output, low-resolution predictors, static fields, and flow time.
- `losses.py`: FM endpoint reconstruction losses, PSD, quantile, coarse
  conservation, and wet-day losses.

### Templates

Training:

```bash
bash ${MAIN_PATH}/scripts/run_train_residual_fm.sh \
  ${MAIN_PATH}/config_templates/train/train_pr_residual_fm_SA_ESD_lr_template
```

Prediction:

```bash
bash ${MAIN_PATH}/scripts/run_predict_residual_fm.sh \
  ${MAIN_PATH}/config_templates/predict/predict_pr_residual_fm_SA_ESD_lr_template
```

The templates are filled with the SA ESD precipitation paths used in the
experiments. Update `MAIN_PATH`, input paths, output paths, and checkpoint paths
as needed for a different machine or experiment.

### Output

For precipitation, `predict_residual_fm_lr.py` writes:

- `pr_gnn4cd`: the residual-FM refined prediction used by the report scripts.
- `pr_gnn_baseline`: the original GNN prediction.
- `pr_residual_fm_mean`: the sample mean when `N_SAMPLES > 1`.

The prediction wrapper optionally calls the Valentina plotting report if the
output pickle exists.
