import argparse
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_residual_fm import ResidualFMDatasetLR
from diffusion.residual_flow_matching import ResidualFlowMatching
from losses import (
    coarse_conservation_loss,
    endpoint_from_velocity,
    psd_loss,
    quantile_loss,
    reconstruct_temperature,
    reconstruct_precip_mm,
    wet_day_loss,
)
from models.residual_denoiser_lr import ResidualDenoiserLR


def write_log(msg, path, mode="a"):
    with open(path, mode) as f:
        f.write(msg)


parser = argparse.ArgumentParser()
parser.add_argument("--gnn_pred_file", type=str, required=True)
parser.add_argument("--target_file", type=str, required=True)
parser.add_argument("--time_index_file", type=str, required=True)
parser.add_argument("--low_input_file", type=str, required=True)
parser.add_argument("--graph_file", type=str, required=True)
parser.add_argument("--static_file", type=str, default=None)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--log_file", type=str, default="log.txt")
parser.add_argument("--target_type", type=str, default="precipitation")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--unet_base", type=int, default=32)
parser.add_argument("--lr_hidden", type=int, default=16)
parser.add_argument("--time_emb_dim", type=int, default=64)
parser.add_argument("--sigma_min", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--grid_h", type=int, default=128)
parser.add_argument("--grid_w", type=int, default=128)
parser.add_argument("--highpass_kernel", type=int, default=9)
parser.add_argument("--coarse_block_size", type=int, default=8)
parser.add_argument("--lambda_psd", type=float, default=0.02)
parser.add_argument("--lambda_quantile", type=float, default=0.02)
parser.add_argument("--lambda_coarse", type=float, default=0.05)
parser.add_argument("--lambda_wet", type=float, default=0.01)
parser.add_argument("--wet_threshold", type=float, default=1.0)
parser.add_argument("--grad_clip", type=float, default=5.0)


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    log_path = os.path.join(args.output_path, args.log_file)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_log(f"Residual high-frequency FM training\nDevice: {device}\n", log_path, "w")

    dataset = ResidualFMDatasetLR(
        gnn_pred_file=args.gnn_pred_file,
        target_file=args.target_file,
        time_index_file=args.time_index_file,
        low_input_file=args.low_input_file,
        target_type=args.target_type,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        highpass_kernel=args.highpass_kernel,
    )
    n_nodes = dataset.x_gnn.shape[1]
    n_lr = dataset.lr_feats.shape[2]
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    write_log(
        f"Dataset: {len(dataset)} timesteps, {n_nodes} nodes, {n_lr} LR features\n",
        log_path,
    )
    np.savez(
        os.path.join(args.output_path, "lr_norm.npz"),
        mean=dataset.lr_mean,
        std=dataset.lr_std,
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
    n_params = sum(p.numel() for p in model.parameters())
    write_log(f"Model parameters: {n_params:,}\n", log_path)

    cfm = ResidualFlowMatching(sigma_min=args.sigma_min)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        meters = {
            "total": 0.0,
            "fm": 0.0,
            "psd": 0.0,
            "q": 0.0,
            "coarse": 0.0,
            "wet": 0.0,
        }
        t0 = time.time()

        for x_gnn, x_hr, residual, lr_feats in dataloader:
            x_gnn = x_gnn.to(device)
            x_hr = x_hr.to(device)
            residual = residual.to(device)
            lr_feats = lr_feats.to(device)

            x_t, target_v, t_samp = cfm.get_training_sample(residual)
            v_pred = model(
                x_t, x_gnn, static_feats, low2high_edge_index, lr_feats, t_samp
            )
            loss_fm = torch.mean((v_pred - target_v) ** 2)

            residual_hat = endpoint_from_velocity(x_t, v_pred, t_samp)
            if args.target_type == "precipitation":
                pred_phys, pred_field = reconstruct_precip_mm(x_gnn, residual_hat)
                target_phys = torch.expm1(x_hr).clamp(min=0.0)
                loss_wet = wet_day_loss(
                    pred_phys, target_phys, threshold=args.wet_threshold
                )
            else:
                pred_phys, pred_field = reconstruct_temperature(x_gnn, residual_hat)
                target_phys = x_hr
                loss_wet = torch.zeros((), device=device)

            loss_psd = psd_loss(pred_field, x_hr, grid_h=args.grid_h, grid_w=args.grid_w)
            loss_q = quantile_loss(pred_phys, target_phys)
            loss_coarse = coarse_conservation_loss(
                pred_phys,
                target_phys,
                grid_h=args.grid_h,
                grid_w=args.grid_w,
                block_size=args.coarse_block_size,
            )

            loss = (
                loss_fm
                + args.lambda_psd * loss_psd
                + args.lambda_quantile * loss_q
                + args.lambda_coarse * loss_coarse
                + args.lambda_wet * loss_wet
            )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            meters["total"] += loss.item()
            meters["fm"] += loss_fm.item()
            meters["psd"] += loss_psd.item()
            meters["q"] += loss_q.item()
            meters["coarse"] += loss_coarse.item()
            meters["wet"] += loss_wet.item()

        for key in meters:
            meters[key] /= len(dataloader)

        elapsed = time.time() - t0
        write_log(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"total={meters['total']:.6f} fm={meters['fm']:.6f} "
            f"psd={meters['psd']:.6f} q={meters['q']:.6f} "
            f"coarse={meters['coarse']:.6f} wet={meters['wet']:.6f} | "
            f"{elapsed:.1f}s\n",
            log_path,
        )

        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "n_static": n_static,
            "n_lr": n_lr,
        }
        if (epoch + 1) % 20 == 0:
            torch.save(ckpt, os.path.join(args.output_path, f"checkpoint_{epoch}.pt"))
        if meters["total"] < best_loss:
            best_loss = meters["total"]
            torch.save(ckpt, os.path.join(args.output_path, "best_model.pt"))

    write_log(f"\nTraining complete. Best loss: {best_loss:.6f}\n", log_path)
