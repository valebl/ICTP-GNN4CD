# GNN + Residual DDPM With Reconstruction/Wet Loss

This experiment starts from a deterministic GNN downscaling output and learns a stochastic residual correction with DDPM.

The clean workflow is:

1. Predict the deterministic Attention baseline for the full train/validation period.
2. Train this residual DDPM on `target - GNN_prediction`, while splitting the full baseline file by year inside the DDPM dataset.
3. Select `best_model.pt` and `best_ema_model.pt` using a cheap validation loss at every epoch.
4. Run the separate prediction script for the final validation/report year.


## Model

The denoiser is `models/residual_denoiser_lr.py`.

Inputs:

- Noisy residual at diffusion step `t`.
- Deterministic baseline.
- Static high-resolution features: orography.
- Low-resolution predictors, normalized with `lr_norm.npz`.

Training target:

- DDPM noise prediction, i.e. the model learns the noise added to the true residual.

The trained EMA checkpoint is used by default for prediction:

```text
best_ema_model.pt
```

## Loss

The clean training loss is:

```text
L = L_noise + lambda_recon * L_recon + lambda_wet * L_wet
```

where:

- `L_noise`: MSE between predicted DDPM noise and true sampled noise.
- `L_recon`: MSE between reconstructed residual `x0_hat` and the true residual.
- `L_wet`: binary cross entropy for wet/dry precipitation after adding the reconstructed residual back to the GNN baseline.

The wet/dry loss is computed on reconstructed precipitation:

```text
pred_mm = GNN_baseline + residual_hat
target_mm = ground_truth
wet = pred_mm > WET_THRESHOLD
```

This helps the DDPM avoid erasing precipitation that the deterministic model already placed in a physically plausible region.

During training, checkpoint selection does not run the full reverse diffusion chain. Each epoch computes a cheap validation objective:

```text
val_loss = L_noise + lambda_recon * L_recon + lambda_wet * L_wet
```

This is much faster than generating a full yearly report every epoch and avoids using validation years in the training batches.

## Main Parameters

- `LAMBDA_RECON`: strength of reconstructed-residual MSE.
- `LAMBDA_WET`: strength of wet/dry auxiliary loss.
- `WET_THRESHOLD`: precipitation threshold in mm/day for wet/dry classification.
- `REVERSE_NOISE_SCALE`: stochasticity during reverse DDPM sampling.
- `APPLY_COARSE_CONSERVATION`: multiplicative coarse-scale correction for precipitation.
- `DDPM_TIMESTEPS`: diffusion chain length.

## Data Splits

For `ESD_pseudo_reality`, first generate one deterministic Attention prediction file covering 1961-1980. The DDPM training config uses 1961-1979 for training and 1980 for validation/checkpoint selection.

For `Emulator_hist_future`, first generate one deterministic Attention prediction file covering 1961-1980 and 2080-2099. The DDPM training config uses 1961-1980 and 2080-2097 for training, and 2098-2099 for validation/checkpoint selection. This clean DDPM package includes a prediction/report config for 2098 only.

## Commands

Generate the ESD Attention baseline file:

```bash
bash $HOME/benchmarks/Experiments_Attention/scripts/run_predict.sh \
  $HOME/benchmarks/Experiments_Attention/config/predict_pr_attention_SA_ESD_train_1961_1980
```

Train and predict ESD:

```bash
bash $HOME/benchmarks/Experiments_GNN_diffusion_ddpm_recon_wet_loss/run_residual_ddpm_lr.sh \
  $HOME/benchmarks/Experiments_GNN_diffusion_ddpm_recon_wet_loss/config/train_pr_recon_wet_SA_ESD

bash $HOME/benchmarks/Experiments_GNN_diffusion_ddpm_recon_wet_loss/run_predict_residual_ddpm_lr.sh \
  $HOME/benchmarks/Experiments_GNN_diffusion_ddpm_recon_wet_loss/config/predict_pr_recon_wet_SA_ESD_1980
```

Generate the Emulator_hist_future Attention baseline file:

```bash
bash $HOME/benchmarks/Experiments_Attention/scripts/run_predict.sh \
  $HOME/benchmarks/Experiments_Attention/config/predict_pr_attention_SA_Emulator_hist_future_train_1961_1980_2080_2099
```

Train and predict Emulator_hist_future:

```bash
bash $HOME/benchmarks/Experiments_GNN_diffusion_ddpm_recon_wet_loss/run_residual_ddpm_lr.sh \
  $HOME/benchmarks/Experiments_GNN_diffusion_ddpm_recon_wet_loss/config/train_pr_recon_wet_SA_Emulator_hist_future

bash $HOME/benchmarks/Experiments_GNN_diffusion_ddpm_recon_wet_loss/run_predict_residual_ddpm_lr.sh \
  $HOME/benchmarks/Experiments_GNN_diffusion_ddpm_recon_wet_loss/config/predict_pr_recon_wet_SA_Emulator_hist_future_2098
```
