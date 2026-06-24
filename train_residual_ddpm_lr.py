import argparse
import copy
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_residual_ddpm import ResidualDDPMDatasetLR
from diffusion import ResidualDDPM
from losses import (
    reconstruct_precip_mm,
    reconstruct_temperature,
    spatial_confidence_gate_from_precip,
    wet_day_loss,
)
from models import ResidualDenoiserLR


def write_log(msg, path, mode="a"):
    with open(path, mode) as f:
        f.write(msg)


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param, alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, buffer in model_buffers.items():
        if name in ema_buffers and ema_buffers[name].shape == buffer.shape:
            ema_buffers[name].copy_(buffer)


parser = argparse.ArgumentParser()
parser.add_argument("--gnn_pred_file", type=str, required=True)
parser.add_argument("--target_file", type=str, required=True)
parser.add_argument("--time_index_file", type=str, required=True)
parser.add_argument("--low_input_file", type=str, required=True)
parser.add_argument("--graph_file", type=str, required=True)
parser.add_argument("--static_file", type=str, default=None)
parser.add_argument("--train_years", type=str, default="")
parser.add_argument("--val_years", type=str, default="")
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--log_file", type=str, default="log.txt")
parser.add_argument("--target_type", type=str, default="precipitation")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1.5e-4)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--unet_base", type=int, default=32)
parser.add_argument("--lr_hidden", type=int, default=32)
parser.add_argument("--time_emb_dim", type=int, default=64)
parser.add_argument("--ddpm_timesteps", type=int, default=1000)
parser.add_argument("--beta_start", type=float, default=1e-4)
parser.add_argument("--beta_end", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--val_seed", type=int, default=12345)
parser.add_argument("--val_repeats", type=int, default=1)
parser.add_argument("--checkpoint_interval", type=int, default=10)
parser.add_argument("--grid_h", type=int, default=128)
parser.add_argument("--grid_w", type=int, default=128)
parser.add_argument("--use_spatial_confidence_gate", action="store_true")
parser.add_argument("--no-use_spatial_confidence_gate", dest="use_spatial_confidence_gate", action="store_false")
parser.set_defaults(use_spatial_confidence_gate=False)
parser.add_argument("--gate_smooth_kernel", type=int, default=9)
parser.add_argument("--gate_threshold", type=float, default=1.0)
parser.add_argument("--gate_tau", type=float, default=0.5)
parser.add_argument("--gate_min", type=float, default=0.0)
parser.add_argument("--lambda_recon", type=float, default=0.02)
parser.add_argument("--lambda_wet", type=float, default=0.01)
parser.add_argument("--wet_threshold", type=float, default=1.0)
parser.add_argument("--aux_residual_model_clip", type=float, default=8.0)
parser.add_argument("--aux_log_clip", type=float, default=8.0)
parser.add_argument("--grad_clip", type=float, default=5.0)
parser.add_argument("--normalize_residual", action="store_true")
parser.add_argument("--no-normalize_residual", dest="normalize_residual", action="store_false")
parser.set_defaults(normalize_residual=True)
parser.add_argument("--use_ema", action="store_true")
parser.add_argument("--no-use_ema", dest="use_ema", action="store_false")
parser.set_defaults(use_ema=True)
parser.add_argument("--ema_decay", type=float, default=0.999)


def compute_losses(
    model,
    ddpm,
    x_gnn,
    x_hr,
    residual,
    lr_feats,
    static_feats,
    low2high_edge_index,
    residual_mean_t,
    residual_std_t,
    args,
):
    residual_model = (residual - residual_mean_t) / residual_std_t
    x_t, target_noise, t_samp = ddpm.get_training_sample(residual_model)
    noise_pred = model(
        x_t, x_gnn, static_feats, low2high_edge_index, lr_feats, t_samp.float()
    )
    loss_noise = F.mse_loss(noise_pred, target_noise)

    loss_recon = torch.zeros((), device=residual.device)
    loss_wet = torch.zeros((), device=residual.device)
    if args.lambda_recon > 0 or args.lambda_wet > 0:
        residual_hat_model = ddpm.predict_x0_from_eps(x_t, noise_pred, t_samp)
        if args.aux_residual_model_clip > 0:
            clip = float(args.aux_residual_model_clip)
            residual_hat_model = torch.nan_to_num(
                residual_hat_model,
                nan=0.0,
                posinf=clip,
                neginf=-clip,
            ).clamp(min=-clip, max=clip)
        residual_hat = residual_hat_model * residual_std_t + residual_mean_t

        # NEW: optional spatial confidence gate. The original recon/wet
        # auxiliary path used the full residual directly:
        #     residual_for_aux = residual_hat
        # We keep that behavior when the flag is disabled. When enabled, only
        # the residual used to reconstruct the auxiliary precipitation field is
        # gated; the DDPM noise-prediction target above remains unchanged.
        residual_for_aux = residual_hat
        if (
            args.use_spatial_confidence_gate
            and args.target_type == "precipitation"
        ):
            gnn_mm = torch.expm1(torch.clamp(x_gnn, max=args.aux_log_clip)).clamp(min=0.0)
            gate = spatial_confidence_gate_from_precip(
                gnn_mm,
                grid_h=args.grid_h,
                grid_w=args.grid_w,
                smooth_kernel=args.gate_smooth_kernel,
                threshold=args.gate_threshold,
                tau=args.gate_tau,
                gate_min=args.gate_min,
            )
            residual_for_aux = residual_hat * gate

        if args.lambda_recon > 0:
            # Original ungated residual-space reconstruction:
            #     loss_recon = F.mse_loss(residual_hat, residual)
            loss_recon = F.mse_loss(residual_for_aux, residual)

        if args.lambda_wet > 0 and args.target_type == "precipitation":
            pred_mm, _ = reconstruct_precip_mm(
                x_gnn, residual_for_aux, log_clip=args.aux_log_clip
            )
            target_mm = torch.expm1(torch.clamp(x_hr, max=args.aux_log_clip)).clamp(min=0.0)
            loss_wet = wet_day_loss(pred_mm, target_mm, threshold=args.wet_threshold)
        elif args.lambda_wet > 0:
            pred_value, _ = reconstruct_temperature(x_gnn, residual_for_aux)
            loss_wet = F.mse_loss(pred_value, x_hr)

    loss = loss_noise + args.lambda_recon * loss_recon + args.lambda_wet * loss_wet
    return loss, {
        "total": loss,
        "noise": loss_noise,
        "recon": loss_recon,
        "wet": loss_wet,
    }


@torch.no_grad()
def evaluate_validation(
    model,
    dataloader,
    ddpm,
    static_feats,
    low2high_edge_index,
    residual_mean_t,
    residual_std_t,
    device,
    args,
):
    model.eval()
    meters = {"total": 0.0, "noise": 0.0, "recon": 0.0, "wet": 0.0}
    valid_batches = 0

    # Keep validation stochasticity fixed across epochs. The DDPM training
    # sample draws random t/noise; without a fixed validation stream, the
    # checkpoint selected as "best" can be dominated by Monte Carlo jitter.
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        for repeat in range(max(1, args.val_repeats)):
            seed = int(args.val_seed) + repeat
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            for x_gnn, x_hr, residual, lr_feats in dataloader:
                x_gnn = x_gnn.to(device)
                x_hr = x_hr.to(device)
                residual = residual.to(device)
                lr_feats = lr_feats.to(device)
                loss, losses = compute_losses(
                    model,
                    ddpm,
                    x_gnn,
                    x_hr,
                    residual,
                    lr_feats,
                    static_feats,
                    low2high_edge_index,
                    residual_mean_t,
                    residual_std_t,
                    args,
                )
                if not torch.isfinite(loss):
                    continue
                for key in meters:
                    meters[key] += losses[key].item()
                valid_batches += 1
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    if valid_batches == 0:
        raise RuntimeError("All validation batches produced non-finite loss.")
    for key in meters:
        meters[key] /= valid_batches
    return meters


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    log_path = os.path.join(args.output_path, args.log_file)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_log(
        "Attention-conditioned residual DDPM training with recon/wet losses\n"
        f"Device: {device}\n"
        f"DDPM timesteps: {args.ddpm_timesteps}\n",
        log_path,
        "w",
    )

    dataset = ResidualDDPMDatasetLR(
        gnn_pred_file=args.gnn_pred_file,
        target_file=args.target_file,
        time_index_file=args.time_index_file,
        low_input_file=args.low_input_file,
        target_type=args.target_type,
        train_years=args.train_years,
    )
    val_dataset = None
    if args.val_years:
        val_dataset = ResidualDDPMDatasetLR(
            gnn_pred_file=args.gnn_pred_file,
            target_file=args.target_file,
            time_index_file=args.time_index_file,
            low_input_file=args.low_input_file,
            target_type=args.target_type,
            train_years=args.val_years,
            lr_mean=dataset.lr_mean,
            lr_std=dataset.lr_std,
        )
    n_nodes = dataset.x_gnn.shape[1]
    n_lr = dataset.lr_feats.shape[2]
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
    write_log(
        f"Dataset: {len(dataset)} timesteps, {n_nodes} nodes, {n_lr} LR features\n"
        f"Training years: {args.train_years or 'all'}\n",
        log_path,
    )
    if val_dataset is not None:
        write_log(
            f"Validation dataset: {len(val_dataset)} timesteps\n"
            f"Validation years: {args.val_years}\n",
            log_path,
        )

    np.savez(os.path.join(args.output_path, "lr_norm.npz"), mean=dataset.lr_mean, std=dataset.lr_std)

    residual_mean = float(dataset.residual.mean())
    residual_std = float(dataset.residual.std() + 1e-8)
    if not args.normalize_residual:
        residual_mean = 0.0
        residual_std = 1.0
    np.savez(
        os.path.join(args.output_path, "residual_norm.npz"),
        mean=np.array(residual_mean, dtype=np.float32),
        std=np.array(residual_std, dtype=np.float32),
        enabled=np.array(args.normalize_residual, dtype=np.bool_),
    )
    residual_mean_t = torch.tensor(residual_mean, device=device, dtype=torch.float32)
    residual_std_t = torch.tensor(residual_std, device=device, dtype=torch.float32)
    write_log(
        f"Residual normalization: enabled={args.normalize_residual}, "
        f"mean={residual_mean:.6f}, std={residual_std:.6f}\n",
        log_path,
    )
    write_log(
        f"Validation RNG: seed={args.val_seed}, repeats={max(1, args.val_repeats)}\n"
        f"Checkpoint interval: {args.checkpoint_interval} epochs\n",
        log_path,
    )
    write_log(
        "Spatial confidence gate: "
        f"enabled={args.use_spatial_confidence_gate}, "
        f"kernel={args.gate_smooth_kernel}, "
        f"threshold={args.gate_threshold}, "
        f"tau={args.gate_tau}, "
        f"gate_min={args.gate_min}\n",
        log_path,
    )

    with open(args.graph_file, "rb") as f:
        graph = pickle.load(f)
    low2high_edge_index = graph["low", "to", "high"].edge_index.to(device)
    write_log(f"Low->high edges: {low2high_edge_index.shape[1]}\n", log_path)

    n_static = 0
    static_feats = torch.zeros(n_nodes, 0, device=device)
    if args.static_file and os.path.exists(args.static_file):
        orog = np.load(args.static_file).reshape(-1, 1).astype(np.float32)
        orog = (orog - orog.mean()) / (orog.std() + 1e-8)
        static_feats = torch.from_numpy(orog).to(device)
        n_static = 1
        write_log("Static features: orography\n", log_path)

    model = ResidualDenoiserLR(
        n_static=n_static,
        n_lr=n_lr,
        lr_hidden=args.lr_hidden,
        time_emb_dim=args.time_emb_dim,
        unet_base=args.unet_base,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
    ).to(device)
    write_log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n", log_path)

    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model).eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
        write_log(f"EMA enabled: decay={args.ema_decay:.6f}\n", log_path)

    ddpm = ResidualDDPM(
        timesteps=args.ddpm_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        meters = {"total": 0.0, "noise": 0.0, "recon": 0.0, "wet": 0.0}
        valid_batches = 0
        skipped_batches = 0
        t0 = time.time()

        for x_gnn, x_hr, residual, lr_feats in dataloader:
            x_gnn = x_gnn.to(device)
            x_hr = x_hr.to(device)
            residual = residual.to(device)
            lr_feats = lr_feats.to(device)
            loss, losses = compute_losses(
                model,
                ddpm,
                x_gnn,
                x_hr,
                residual,
                lr_feats,
                static_feats,
                low2high_edge_index,
                residual_mean_t,
                residual_std_t,
                args,
            )

            if not torch.isfinite(loss):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, args.ema_decay)

            meters["total"] += loss.item()
            meters["noise"] += losses["noise"].item()
            meters["recon"] += losses["recon"].item()
            meters["wet"] += losses["wet"].item()
            valid_batches += 1

        if valid_batches == 0:
            raise RuntimeError("All training batches produced non-finite loss.")
        for key in meters:
            meters[key] /= valid_batches

        elapsed = time.time() - t0
        val_meters = None
        if val_loader is not None:
            eval_model = ema_model if ema_model is not None else model
            val_meters = evaluate_validation(
                eval_model,
                val_loader,
                ddpm,
                static_feats,
                low2high_edge_index,
                residual_mean_t,
                residual_std_t,
                device,
                args,
            )

        msg = (
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_total={meters['total']:.6f} train_noise={meters['noise']:.6f} "
            f"train_recon={meters['recon']:.6f} train_wet={meters['wet']:.6f}"
        )
        if val_meters is not None:
            msg += (
                f" | val_total={val_meters['total']:.6f} "
                f"val_noise={val_meters['noise']:.6f} "
                f"val_recon={val_meters['recon']:.6f} "
                f"val_wet={val_meters['wet']:.6f}"
            )
        msg += f" | {elapsed:.1f}s\n"
        write_log(msg, log_path)
        if skipped_batches:
            write_log(f"Skipped non-finite batches: {skipped_batches}/{len(dataloader)}\n", log_path)

        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "n_static": n_static,
            "n_lr": n_lr,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "normalize_residual": args.normalize_residual,
        }
        ckpt_ema = None
        if ema_model is not None:
            ckpt_ema = dict(ckpt)
            ckpt_ema["model"] = ema_model.state_dict()
            ckpt_ema["ema_decay"] = args.ema_decay
            ckpt_ema["is_ema"] = True

        if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(ckpt, os.path.join(args.output_path, f"checkpoint_{epoch}.pt"))
            if ckpt_ema is not None:
                torch.save(ckpt_ema, os.path.join(args.output_path, f"checkpoint_ema_{epoch}.pt"))

        selection_loss = val_meters["total"] if val_meters is not None else meters["total"]
        if selection_loss < best_loss:
            best_loss = selection_loss
            torch.save(ckpt, os.path.join(args.output_path, "best_model.pt"))
            if ckpt_ema is not None:
                torch.save(ckpt_ema, os.path.join(args.output_path, "best_ema_model.pt"))

    metric_name = "validation loss" if val_loader is not None else "training loss"
    write_log(f"\nTraining complete. Best {metric_name}: {best_loss:.6f}\n", log_path)
