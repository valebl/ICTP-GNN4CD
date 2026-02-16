"""
批量降水推理脚本
支持多个GCM、多个时间段、perfect/imperfect模式
"""
import os
import pickle
import numpy as np
import torch
import argparse
import time
import importlib
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from accelerate import Accelerator
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree

import dataset
from dataset import Dataset_Graph, Iterable_Graph

from utils.tools import write_log, standardize_input, set_seed_everything


# ==================== 参数定义 ====================
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 路径
parser.add_argument('--base_graph_path',  type=str, help='图文件根目录，如 .../SA_domain/ESD_pseudo_reality/test/')
parser.add_argument('--base_output_path', type=str, help='输出根目录，如 .../BENCHMARK/test/')
parser.add_argument('--train_path_reg',   type=str, help='训练模型路径（含统计量pkl）')
parser.add_argument('--checkpoint_reg',   type=str, default='checkpoint_44')
parser.add_argument('--log_path',         type=str, help='日志路径')
parser.add_argument('--log_file',         type=str, default='inference_log.txt')

# 模型配置
parser.add_argument('--model_name',    type=str, default='P_GNN4CD_model')
parser.add_argument('--dataset_name',  type=str, default='Dataset_Graph')
parser.add_argument('--seq_l',         type=int, default=2)
parser.add_argument('--batch_size',    type=int, default=32)
parser.add_argument('--seed',          type=int, default=42)

# 数据集起始日期（和训练时一致，用于seq_l偏移计算）
parser.add_argument('--pr_threshold',  type=float, default=0.1)

# 推理范围控制
parser.add_argument('--gcms',     type=str, nargs='+',
                    default=['ACCESS-CM2', 'NorESM2-MM'])
parser.add_argument('--periods',  type=str, nargs='+',
                    default=['historical', 'mid_century', 'end_century'])
parser.add_argument('--modes',    type=str, nargs='+',
                    default=['perfect', 'imperfect'])

# 加速器
parser.add_argument('--use_accelerate',    action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')
parser.set_defaults(use_accelerate=True)


# ==================== 辅助函数 ====================

def load_model(model_name, checkpoint_path, seq_l, accelerator):
    """加载模型和checkpoint"""
    model_file = importlib.import_module(f"models.{model_name}")
    Model = getattr(model_file, model_name)
    model = Model(seq_l=seq_l + 1)

    ckpt = torch.load(checkpoint_path, weights_only=True,
                      map_location=accelerator.device if accelerator else 'cpu')
    model.load_state_dict(ckpt)
    return model


def load_norm_stats(train_path):
    """加载标准化统计量"""
    with open(train_path + "means_low.pkl", 'rb') as f:
        means_low = pickle.load(f)
    with open(train_path + "stds_low.pkl", 'rb') as f:
        stds_low = pickle.load(f)
    with open(train_path + "means_high.pkl", 'rb') as f:
        means_high = pickle.load(f)
    with open(train_path + "stds_high.pkl", 'rb') as f:
        stds_high = pickle.load(f)
    return means_low, stds_low, means_high, stds_high


def run_inference(model, dataloader, args, accelerator, log_path, log_file):
    """运行推理，返回预测值和时间步"""
    model.eval()
    pr_list    = []
    times_list = []
    step = 0

    with torch.no_grad():
        for graph in dataloader:
            t = graph.t

            # 过滤accelerate补充的假样本（t<0）
            if (t < 0).any():
                step += 1
                continue

            # 空batch保护
            if graph.x_dict['low'].shape[1] == 0:
                step += 1
                continue

            y_pred = model(graph)
            pr_list.append(y_pred)
            times_list.append(t)

            if step % 50 == 0:
                if accelerator is None or accelerator.is_main_process:
                    with open(os.path.join(log_path, log_file), 'a') as f:
                        f.write(f"\n  Step {step}, t={t.cpu().numpy()}")
            step += 1

    pr    = torch.stack(pr_list)
    times = torch.stack(times_list)
    return pr, times


def postprocess_pr(pr_normalized, times, mask, accelerator, pr_threshold=0.1):
    """收集多卡结果并做反变换"""
    if accelerator is not None:
        accelerator.wait_for_everyone()
        times         = accelerator.gather(times).squeeze()
        pr_normalized = accelerator.gather(pr_normalized)

    times, sort_idx = torch.sort(times)
    times    = times.cpu().numpy()
    sort_idx = sort_idx.cpu().numpy()

    pr_norm = pr_normalized.squeeze().swapaxes(0, 1).cpu().numpy()[:, sort_idx]

    # expm1反变换
    pr_pred = np.where(np.isfinite(np.expm1(pr_norm)), np.expm1(pr_norm), np.nan)
    pr_pred[pr_pred < pr_threshold] = 0.0
    pr_pred = np.round(pr_pred, decimals=1)

    # 应用空间mask
    pr_pred = pr_pred[mask, :]

    return pr_pred, pr_norm, times


def make_seasonal_results(pr_pred, times, n_total_steps):
    """
    根据times（局部索引，0~N-1）计算四季切片
    由于推理时间段可能跨多年，这里按月份粗略分季节
    返回 [DJF, MAM, JJA, SON] 各自的数组
    """
    # times是局部索引（0-based），直接按比例划分四季
    # 更精确的做法需要知道具体日期，这里做简单的等分
    n = len(times)
    quarter = n // 4

    djf = pr_pred[:, :quarter]
    mam = pr_pred[:, quarter:2*quarter]
    jja = pr_pred[:, 2*quarter:3*quarter]
    son = pr_pred[:, 3*quarter:]

    return [djf, mam, jja, son]


def get_missing_mask(graph):
    """
    检测low节点中有缺失值的时间步
    返回有效时间步的布尔掩码 (shape: [n_times])
    """
    x_low = graph['low'].x  # (n_nodes, n_times, n_features) or after flatten (n_nodes, n_times, n_features_flat)
    # 检查每个时间步是否有任何节点/特征为NaN
    # x_low: (n_nodes, n_times, n_feats)
    nan_per_time = torch.isnan(x_low).any(dim=0).any(dim=-1)  # (n_times,)
    valid_mask = ~nan_per_time
    return valid_mask


# ==================== 主程序 ====================
if __name__ == '__main__':


    args = parser.parse_args()
    # ===== 添加这行，让write_log能找到路径 =====
    args.output_path = args.log_path
    
    set_seed_everything(args.seed)
    
    set_seed_everything(args.seed)

    os.makedirs(args.log_path, exist_ok=True)

    # ===== 不用accelerate，直接用单GPU =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accelerator = None

    write_log("="*80, args, accelerator, 'w')
    write_log("BATCH PRECIPITATION INFERENCE (Serial)", args, accelerator, 'a')
    write_log(f"Device: {device}", args, accelerator, 'a')
    write_log(f"GCMs    : {args.gcms}", args, accelerator, 'a')
    write_log(f"Periods : {args.periods}", args, accelerator, 'a')
    write_log(f"Modes   : {args.modes}", args, accelerator, 'a')

    # ===== 加载统计量 =====
    means_low, stds_low, means_high, stds_high = load_norm_stats(args.train_path_reg)

    try:
        with open(args.train_path_reg + "target_stats.pkl", 'rb') as f:
            target_stats = pickle.load(f)
        pr_threshold = target_stats.get('threshold', args.pr_threshold)
    except FileNotFoundError:
        pr_threshold = args.pr_threshold

    # ===== 加载模型（只加载一次，放到GPU上）=====
    checkpoint_path = os.path.join(args.train_path_reg, args.checkpoint_reg, "pytorch_model.bin")
    model_file = importlib.import_module(f"models.{args.model_name}")
    Model = getattr(model_file, args.model_name)
    model = Model(seq_l=args.seq_l + 1)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    write_log(f"Model loaded on {device}", args, accelerator, 'a')

    # ==================== 串行推理循环 ====================
    total_tasks = len(args.periods) * len(args.modes) * len(args.gcms)
    task_idx = 0
    failed_tasks = []

    for period_name in args.periods:
        for mode in args.modes:
            for gcm in args.gcms:
                task_idx += 1
                write_log(f"\n{'='*60}", args, accelerator, 'a')
                write_log(f"Task {task_idx}/{total_tasks}: {period_name} | {mode} | {gcm}", args, accelerator, 'a')
                write_log(f"{'='*60}", args, accelerator, 'a')

                # 路径
                graph_path = os.path.join(
                    args.base_graph_path,
                    period_name, mode,
                    f"{gcm}_low_graph.pkl"
                )
                out_dir = os.path.join(args.base_output_path, period_name, mode)
                os.makedirs(out_dir, exist_ok=True)
                out_file        = os.path.join(out_dir, f"{gcm}_pr_predictions.pkl")
                out_file_season = os.path.join(out_dir, f"{gcm}_pr_seasonal_predictions.pkl")

                # 跳过已完成
                if os.path.exists(out_file):
                    write_log(f"  Already exists, skipping.", args, accelerator, 'a')
                    continue

                if not os.path.exists(graph_path):
                    write_log(f"  WARNING: Graph not found: {graph_path}", args, accelerator, 'a')
                    failed_tasks.append(f"{period_name}/{mode}/{gcm} - graph not found")
                    continue

                try:
                    t_start = time.time()

                    # 加载图
                    with open(graph_path, 'rb') as f:
                        low_high_graph = pickle.load(f)
                    n_times = low_high_graph['low'].x.shape[1]
                    write_log(f"  Time steps: {n_times}", args, accelerator, 'a')

                    # 检测缺失值
                    x_low_raw = low_high_graph['low'].x
                    nan_per_time = torch.isnan(x_low_raw).any(dim=0).any(dim=-1).any(dim=-1)
                    valid_time_mask = ~nan_per_time
                    n_valid   = valid_time_mask.sum().item()
                    n_missing = (~valid_time_mask).sum().item()
                    write_log(f"  Valid/Missing: {n_valid}/{n_missing}", args, accelerator, 'a')

                    if n_valid == 0:
                        write_log("  ERROR: All NaN! Skipping.", args, accelerator, 'a')
                        failed_tasks.append(f"{period_name}/{mode}/{gcm} - all NaN")
                        continue

                    all_time_idxs = torch.where(valid_time_mask)[0]

                    # 标准化
                    x_low_original  = low_high_graph['low'].x.clone()
                    x_high_original = low_high_graph['high'].x.clone()

                    write_log(f"  Raw input mean: {torch.nanmean(x_low_original):.6f}", args, None, 'a')

                    low_high_graph['low'].x, low_high_graph['high'].x = standardize_input(
                        x_low_original.clone(),
                        x_high_original.clone(),
                        means_low, stds_low, means_high, stds_high,
                        args, accelerator
                    )

                    low_high_graph['low'].x = torch.flatten(low_high_graph['low'].x, start_dim=2, end_dim=-1)
                    # 安全索引
                    max_safe_idx = n_times - args.seq_l - 1
                    min_safe_idx = args.seq_l
                    valid_idxs = all_time_idxs[
                        (all_time_idxs >= min_safe_idx) &
                        (all_time_idxs <= max_safe_idx)
                    ]
                    write_log(f"  After trim: {len(valid_idxs)} steps", args, accelerator, 'a')

                    # DataLoader
                    Dataset_cls   = getattr(dataset, args.dataset_name)
                    dataset_graph = Dataset_cls(
                        targets=None, graph=low_high_graph,
                        model_name=args.model_name, seq_l=args.seq_l
                    )
                    custom_collate = getattr(dataset, 'custom_collate_fn_graph')
                    t_offset = valid_idxs.min().item()
                    sampler = Iterable_Graph(
                        dataset_graph=dataset_graph, shuffle=False,
                        idxs_vector=valid_idxs, t_offset=t_offset
                    )
                    dataloader = torch.utils.data.DataLoader(
                        dataset_graph, batch_size=args.batch_size,
                        num_workers=0, sampler=sampler,
                        collate_fn=custom_collate
                    )

                    # ===== 串行推理 =====
                    pr_list    = []
                    times_list = []
                    step = 0
                    n_nodes_high = low_high_graph["high"].num_nodes  # 16384

                    with torch.no_grad():
                        for graph in dataloader:
                            t = graph.t  # (batch_size,)

                            if (t < 0).any():
                                step += 1
                                continue
                            if graph.x_dict['low'].shape[1] == 0:
                                step += 1
                                continue

                            graph  = graph.to(device)
                            y_pred = model(graph)  # (n_nodes × batch_size, 1)

                            # 拆分batch: (n_nodes × batch_size, 1) → (batch_size, n_nodes, 1)
                            actual_batch = t.shape[0]
                            y_pred = y_pred.reshape(actual_batch, n_nodes_high, 1)  # (batch, n_nodes, 1)
                            y_pred = y_pred.squeeze(-1)                              # (batch, n_nodes)

                            # 每个时间步单独存
                            for i in range(actual_batch):
                                pr_list.append(y_pred[i].cpu())       # (n_nodes,)
                                times_list.append(t[i].cpu())         # scalar

                            if step % 50 == 0:
                                with open(os.path.join(args.log_path, args.log_file), 'a') as f:
                                    f.write(f"\n  [{period_name}/{mode}/{gcm}] Step {step}/{len(valid_idxs)}")
                            step += 1

                    # ===== 后处理 =====
                    # pr_list:    每个元素 (n_nodes,)，共n_valid_steps个
                    # times_list: 每个元素 scalar，共n_valid_steps个

                    times_cat = torch.stack(times_list)        # (n_valid_steps,)
                    times_sorted, sort_idx = torch.sort(times_cat)
                    times_np = times_sorted.numpy()

                    pr_stack = torch.stack(pr_list)            # (n_valid_steps, n_nodes)
                    pr_norm  = pr_stack[sort_idx].numpy().T    # (n_nodes, n_valid_steps)

                    write_log(f"  pr_norm shape: {pr_norm.shape}", args, accelerator, 'a')
                    # 期望: (16384, ~7296)
                    # 反变换 log1p → expm1
                    pr_pred_full = np.where(
                        np.isfinite(np.expm1(pr_norm)),
                        np.expm1(pr_norm),
                        np.nan
                    )
                    pr_pred_full[pr_pred_full < pr_threshold] = 0.0
                    pr_pred_full = np.round(pr_pred_full, decimals=1)

                    # 节点mask
                    degree_nodes = degree(
                        low_high_graph["high", "within", "high"].edge_index[1],
                        low_high_graph["high"].num_nodes
                    ).cpu().numpy()
                    node_mask = degree_nodes > 2

                    write_log(f"  pr_pred_full shape: {pr_pred_full.shape}", args, accelerator, 'a')
                    write_log(f"  node_mask sum     : {node_mask.sum()}", args, accelerator, 'a')

                    # 应用节点mask
                    pr_pred = pr_pred_full[node_mask, :]

                    t_end = time.time()
                    write_log(f"  Done in {t_end - t_start:.1f}s", args, accelerator, 'a')
                    write_log(f"  pr_pred shape : {pr_pred.shape}", args, accelerator, 'a')
                    write_log(f"  Mean: {np.nanmean(pr_pred):.3f} mm", args, accelerator, 'a')
                    write_log(f"  P99 : {np.nanpercentile(pr_pred, 99):.2f} mm", args, accelerator, 'a')
                    dry = (pr_pred == 0).sum() / (~np.isnan(pr_pred)).sum() * 100
                    write_log(f"  Dry day ratio: {dry:.1f}%", args, accelerator, 'a')

                    # 季节性
                    pr_seasons = make_seasonal_results(pr_pred, times_np, n_valid)

                    # 经纬度
                    lat_high = low_high_graph["high"].lat.cpu().numpy()[node_mask]
                    lon_high = low_high_graph["high"].lon.cpu().numpy()[node_mask]
                    lat_low  = low_high_graph["low"].lat.cpu().numpy()
                    lon_low  = low_high_graph["low"].lon.cpu().numpy()

                    # 输出结构
                    data = HeteroData()
                    data["low"].lat     = lat_low
                    data["low"].lon     = lon_low
                    data["high"].lat    = lat_high
                    data["high"].lon    = lon_high
                    data["high"].degree = degree_nodes[node_mask]
                    data.pr_normalized  = pr_norm
                    data.pr_pred        = pr_pred
                    data.pr_target      = None
                    data.times          = times_np
                    data.threshold      = pr_threshold
                    data.gcm            = gcm
                    data.period         = period_name
                    data.mode           = mode

                    results = {
                        "lon"               : lon_high,
                        "lat"               : lat_high,
                        "times"             : times_np,
                        "pr_gnn4cd"         : pr_pred,
                        "pr_gnn4cd_seasons" : pr_seasons,
                        "pr_target"         : None,
                        "pr_target_seasons" : None,
                        "threshold"         : pr_threshold,
                        "gcm"               : gcm,
                        "period"            : period_name,
                        "mode"              : mode,
                        "valid_time_steps"  : n_valid,
                        "missing_time_steps": n_missing,
                    }

                    # 保存
                    with open(out_file, 'wb') as f:
                        pickle.dump(data, f)
                    write_log(f"  Saved: {out_file}", args, accelerator, 'a')

                    with open(out_file_season, 'wb') as f:
                        pickle.dump(results, f)
                    write_log(f"  Saved: {out_file_season}", args, accelerator, 'a')

                    # 释放内存
                    del low_high_graph, dataset_graph, dataloader
                    del pr_list, times_list, pr_stack, pr_norm
                    torch.cuda.empty_cache()

                except Exception as e:
                    write_log(f"  ERROR: {e}", args, accelerator, 'a')
                    import traceback
                    write_log(traceback.format_exc(), args, accelerator, 'a')
                    failed_tasks.append(f"{period_name}/{mode}/{gcm} - {str(e)}")
                    continue

    # 总结
    write_log("\n" + "="*80, args, accelerator, 'a')
    write_log("BATCH INFERENCE COMPLETED", args, accelerator, 'a')
    write_log(f"Total  : {total_tasks}", args, accelerator, 'a')
    write_log(f"Failed : {len(failed_tasks)}", args, accelerator, 'a')
    for ft in failed_tasks:
        write_log(f"  - {ft}", args, accelerator, 'a')
    write_log("="*80, args, accelerator, 'a')