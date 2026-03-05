# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GNN4CD-CORDEXML is a Graph Neural Network for statistical climate downscaling, used in the CORDEX-ML benchmark. It downscales low-resolution ERA5 reanalysis fields to high-resolution precipitation (`pr`) and maximum temperature (`tasmax`) using heterogeneous graphs connecting low- and high-resolution grid nodes.

The codebase runs on CINECA Leonardo HPC with SLURM, 4 GPUs, HuggingFace Accelerate (fp16 mixed precision), and WandB (offline mode).

## Running the Code

All runs are submitted to SLURM via wrapper shell scripts that source a config file and call `sbatch`:

```bash
# Training
bash run.sh config/training/<config_file>

# Inference - precipitation
bash run_inference_hist_pr.sh config/inference/<config_file>

# Inference - temperature
bash run_inference_hist_tasmax.sh config/inference/<config_file>

# Preprocessing
bash run_preprocessing.sh config/preprocessing_CORDEX
```

Config files under `config/` are shell scripts (`source`d by the run scripts) — not YAML or TOML. They define SLURM parameters, paths, hyperparameters, and flags as bash variables.

## Architecture

### Data flow

1. **Preprocessing** (`preprocessing/preprocessing.py`): Reads NetCDF ERA5 and CORDEX files, builds a PyTorch Geometric `HeteroData` heterogeneous graph saved as `low_high_graph.pkl`. Target tensors are saved as `.pkl` files.

2. **Graph structure** (`utils/graph.py`):
   - `low` nodes: low-resolution ERA5 grid, features shape `(num_nodes, time, n_vars, n_levels)`
   - `high` nodes: high-resolution CORDEX grid, static features (lat, lon, orography), shape `(num_nodes, 3)`
   - Edges: `('low','to','high')` bipartite, `('high','within','high')` spatial neighborhood

3. **Dataset** (`dataset.py`): `Dataset_Graph` wraps the graph. `Iterable_Graph` custom sampler selects time indices. The `__get_features` method extracts a sliding window of `seq_l+1` consecutive timesteps from low-res nodes, flattened to `(num_nodes, (seq_l+1) * n_vars * n_levels)`.

4. **Model** (see variants below): Takes low-res node features (temporal window) and high-res static features, outputs per-high-node scalar predictions.

5. **Training** (`train_Rall_new.py` + `utils/train_test.py`): Uses `Trainer.train_R_Rall`. Normalization stats (`means_low`, `stds_low`, `means_high`, `stds_high`) are computed on training indices only and saved to the checkpoint directory.

6. **Inference** (`inference_pr_Rall.py`, `inference_tasmax_Rall.py`): Loads saved normalization stats and checkpoint, runs forward pass over specified GCMs and periods.

### Model variants

| File | Key difference |
|------|---------------|
| `P_GNN4CD_model.py` | GRU temporal encoder (`nn.GRU`) |
| `P_GNN4CD_model_WOGRU.py` | No GRU — flat `nn.Linear` over concatenated `seq_l+1` timesteps |
| `P_GNN4CD_model_TOPO_Strengthed.py` | Enhanced topography signal |
| `T_GNN4CD_model.py` | Temperature-specific variant |

All models share the same three-stage structure:
- **Encoder**: temporal encoding of low-res node features
- **Downscaler**: `GraphConv` bipartite message passing from low to high nodes
- **Processor**: stack of `GATv2Conv` layers on high-res graph for spatial refinement → linear head

### Experiments

| `--experiment` | Training period | Validation year |
|---------------|----------------|-----------------|
| `ESD_pseudo_reality` | 1961–1980 | 1975 |
| `Emulator_hist_future` | 1961–1980 + 2080–2099 | 1975 |

### Target preprocessing

- **Precipitation**: `prepare_target_Rall` applies `log1p` transform with a 0.1 mm/day threshold. Loss: `quantized_loss` (per-bin MSE) or `mse_loss`.
- **Temperature**: `prepare_target_T_Rall` applies z-score normalization; stats saved to `target_stats.pkl`.

### Key hyperparameters

- `seq_l`: temporal context window (typically 2); model receives `seq_l+1` timesteps
- `high_in=3`: number of static high-res features (lat, lon, orography)
- `lr_scheduler`: `StepLR`, `CosineAnnealingLR` (preferred for pr), or `ReduceLROnPlateau`
- `model_name`: module name in `models/` — loaded dynamically via `importlib`

## Important Files

- `train_Rall_new.py` — main training entry point; handles both experiment types, index derivation, normalization
- `utils/train_test.py` — `Trainer` class with the training loop (`train_R_Rall`)
- `utils/train_test_TOPO.py` — trainer variant for TOPO-strengthened model
- `utils/tools.py` — `prepare_target_*`, `compute_input_statistics`, `standardize_input`, `derive_train_val_idxs_*`, `date_to_idxs_new`
- `utils/loss_functions.py` — `quantized_loss`, `weighted_mse_loss`
- `utils/graph.py` — graph construction utilities (`derive_edge_index_within`, `derive_edge_index_multiscale`)
- `format_predictions_for_CORDEX/format.py` — post-processing to CORDEX NetCDF format
- `evaluate_CORDEX_predictions/` — diagnostic metrics and indices

## Environment

```bash
source $HOME/Conda_init.txt
module load profile/deeplrn cuda/11.8 gcc/11.3.0
conda activate /leonardo/pub/userexternal/sdigioia/sdigioia/env/RLenv
accelerate config  # generate default accelerate config
```

WandB runs in offline mode (`WANDB_MODE=offline`). API key and project name are set in config files.
