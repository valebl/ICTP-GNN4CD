import numpy as np
import pickle
import torch
import argparse
import time
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import importlib

from torch_geometric.data import HeteroData
from torch_geometric.utils import degree

import dataset
from dataset import Dataset_Graph, Iterable_Graph

from utils.tools import set_seed_everything, write_log, standardize_input


# ==================== 参数定义 ====================
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 路径
parser.add_argument('--base_graph_path',  type=str, help='图文件根目录')
parser.add_argument('--base_output_path', type=str, help='输出根目录')
parser.add_argument('--train_path_reg',   type=str, help='训练模型路径（含统计量pkl）')
parser.add_argument('--checkpoint_reg',   type=str, default='checkpoint_75')
parser.add_argument('--log_path',         type=str, help='日志路径')
parser.add_argument('--log_file',         type=str, default='inference_tasmax_log.txt')

# 模型配置
parser.add_argument('--model_name',    type=str, default='T_GNN4CD_model')
parser.add_argument('--dataset_name',  type=str, default='Dataset_Graph')
parser.add_argument('--seq_l',         type=int, default=2)
parser.add_argument('--batch_size',    type=int, default=32)
parser.add_argument('--seed',          type=int, default=42)
parser.add_argument('--stats_mode',    type=str, default='var')

# 推理范围
parser.add_argument('--gcms',    type=str, nargs='+',
                    default=['ACCESS-CM2', 'NorESM2-MM'])
parser.add_argument('--periods', type=str, nargs='+',
                    default=['historical', 'mid_century', 'end_century'])
parser.add_argument('--modes',   type=str, nargs='+',
                    default=['perfect', 'imperfect'])

# 加速器
parser.add_argument('--use_accelerate',    action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')
parser.set_defaults(use_accelerate=False)


# ==================== 反归一化函数 ====================
def denormalize_temperature(t_normalized, target_stats):
    if target_stats['procedure'] == 'z-score':
        return t_normalized * target_stats['std'] + target_stats['mean']
    else:  # minmax
        return t_normalized * (target_stats['max'] - target_stats['min']) + target_stats['min']


# ==================== 主程序 ====================
if __name__ == '__main__':

    args = parser.parse_args()

    # write_log需要args.output_path
    args.output_path = args.log_path

    set_seed_everything(args.seed)
    os.makedirs(args.log_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    write_log("="*80, args, None, 'w')
    write_log("BATCH TASMAX INFERENCE (Serial)", args, None, 'a')
    write_log("="*80, args, None, 'a')
    write_log(f"Device : {device}", args, None, 'a')
    write_log(f"GCMs   : {args.gcms}", args, None, 'a')
    write_log(f"Periods: {args.periods}", args, None, 'a')
    write_log(f"Modes  : {args.modes}", args, None, 'a')

    # ===== 加载标准化统计量（只加载一次）=====
    write_log("\n[1] Loading normalization statistics...", args, None, 'a')
    with open(args.train_path_reg + "means_low.pkl", 'rb') as f:
        means_low = pickle.load(f)
    with open(args.train_path_reg + "stds_low.pkl", 'rb') as f:
        stds_low = pickle.load(f)
    with open(args.train_path_reg + "means_high.pkl", 'rb') as f:
        means_high = pickle.load(f)
    with open(args.train_path_reg + "stds_high.pkl", 'rb') as f:
        stds_high = pickle.load(f)
    write_log(f"  means_low shape: {means_low.shape}", args, None, 'a')

    # ===== 加载目标标准化统计量（温度反归一化用）=====
    write_log("\n[2] Loading target statistics...", args, None, 'a')
    try:
        with open(args.train_path_reg + "target_stats.pkl", 'rb') as f:
            target_stats = pickle.load(f)
        write_log(f"  Procedure: {target_stats['procedure']}", args, None, 'a')
        if target_stats['procedure'] == 'z-score':
            write_log(f"  Mean: {target_stats['mean']:.4f} K", args, None, 'a')
            write_log(f"  Std : {target_stats['std']:.4f} K", args, None, 'a')
        else:
            write_log(f"  Min: {target_stats['min']:.4f} K", args, None, 'a')
            write_log(f"  Max: {target_stats['max']:.4f} K", args, None, 'a')
    except FileNotFoundError:
        write_log("  ERROR: target_stats.pkl not found! Required for temperature denormalization.", args, None, 'a')
        raise FileNotFoundError("target_stats.pkl is required for tasmax inference")

    # ===== 加载模型（只加载一次）=====
    write_log("\n[3] Loading model...", args, None, 'a')
    checkpoint_path = os.path.join(args.train_path_reg, args.checkpoint_reg, "pytorch_model.bin")
    model_file = importlib.import_module(f"models.{args.model_name}")
    Model = getattr(model_file, args.model_name)
    model = Model(seq_l=args.seq_l + 1)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(f"  Checkpoint  : {checkpoint_path}", args, None, 'a')
    write_log(f"  Parameters  : {total_params:,}", args, None, 'a')
    write_log(f"  Device      : {device}", args, None, 'a')

    # ==================== 批量推理循环 ====================
    write_log("\n[4] Starting batch inference...", args, None, 'a')

    total_tasks = len(args.periods) * len(args.modes) * len(args.gcms)
    task_idx    = 0
    failed_tasks = []

    for period_name in args.periods:
        for mode in args.modes:
            for gcm in args.gcms:
                task_idx += 1
                write_log(f"\n{'='*60}", args, None, 'a')
                write_log(f"Task {task_idx}/{total_tasks}: {period_name} | {mode} | {gcm}", args, None, 'a')
                write_log(f"{'='*60}", args, None, 'a')

                # ===== 路径 =====
                graph_path = os.path.join(
                    args.base_graph_path,
                    period_name, mode,
                    f"{gcm}_low_graph.pkl"
                )
                out_dir = os.path.join(args.base_output_path, period_name, mode)
                os.makedirs(out_dir, exist_ok=True)

                out_file        = os.path.join(out_dir, f"{gcm}_tasmax_predictions.pkl")
                out_file_season = os.path.join(out_dir, f"{gcm}_tasmax_seasonal_predictions.pkl")

                # 跳过已完成
                # if os.path.exists(out_file):
                #     write_log(f"  Already exists, skipping.", args, None, 'a')
                #     continue

                # 检查图文件
                if not os.path.exists(graph_path):
                    write_log(f"  WARNING: Graph not found: {graph_path}", args, None, 'a')
                    failed_tasks.append(f"{period_name}/{mode}/{gcm} - graph not found")
                    continue

                try:
                    t_start = time.time()

                    # ===== 加载图 =====
                    write_log(f"  Loading graph: {graph_path}", args, None, 'a')
                    with open(graph_path, 'rb') as f:
                        low_high_graph = pickle.load(f)

                    n_times = low_high_graph['low'].x.shape[1]
                    write_log(f"  Low  shape : {low_high_graph['low'].x.shape}", args, None, 'a')
                    write_log(f"  High shape : {low_high_graph['high'].x.shape}", args, None, 'a')

                    # ===== 检测缺失值 =====
                    x_low_raw    = low_high_graph['low'].x  # (n_nodes, n_times, n_vars, n_levels)
                    nan_per_time = torch.isnan(x_low_raw).any(dim=0).any(dim=-1).any(dim=-1)  # (n_times,)
                    valid_time_mask = ~nan_per_time
                    n_valid   = valid_time_mask.sum().item()
                    n_missing = (~valid_time_mask).sum().item()
                    write_log(f"  Valid/Missing time steps: {n_valid}/{n_missing}", args, None, 'a')

                    if n_valid == 0:
                        write_log("  ERROR: All NaN! Skipping.", args, None, 'a')
                        failed_tasks.append(f"{period_name}/{mode}/{gcm} - all NaN")
                        continue

                    all_time_idxs = torch.where(valid_time_mask)[0]

                    # ===== 保存原始数据（用clone避免后续标准化污染）=====
                    x_low_original  = low_high_graph['low'].x.clone()
                    x_high_original = low_high_graph['high'].x.clone()

                    # 打印原始数据统计（诊断不同任务输入是否不同）
                    write_log(f"  Raw input mean: {x_low_original.mean():.6f}", args, None, 'a')
                    write_log(f"  Raw input std : {x_low_original.std():.6f}", args, None, 'a')
                    write_log(f"  Raw input min : {x_low_original.min():.6f}", args, None, 'a')
                    write_log(f"  Raw input max : {x_low_original.max():.6f}", args, None, 'a')

                    # ===== 标准化（使用clone后的原始数据）=====
                    write_log("  Standardizing inputs...", args, None, 'a')
                    low_high_graph['low'].x, low_high_graph['high'].x = standardize_input(
                        x_low_original.clone(),
                        x_high_original.clone(),
                        means_low, stds_low, means_high, stds_high,
                        args, None
                    )

                    # 打印标准化统计（和验证代码一致）
                    vars_names = ['q', 't', 'u', 'v', 'z']
                    if args.stats_mode == "var":
                        for var in range(5):
                            write_log(
                                f"  Low var {vars_names[var]}: "
                                f"mean={low_high_graph['low'].x[:,:,var,:].mean():.6f}, "
                                f"std={low_high_graph['low'].x[:,:,var,:].std():.6f}",
                                args, None, 'a'
                            )

                    # Flatten vars×levels维度
                    low_high_graph['low'].x = torch.flatten(low_high_graph['low'].x, start_dim=2, end_dim=-1)
                    write_log(f"  After flatten: {low_high_graph['low'].x.shape}", args, None, 'a')

                    # ===== 安全索引范围 =====
                    max_safe_idx = n_times - args.seq_l - 1
                    min_safe_idx = args.seq_l
                    valid_idxs = all_time_idxs[
                        (all_time_idxs >= min_safe_idx) &
                        (all_time_idxs <= max_safe_idx)
                    ]
                    write_log(f"  After seq_l trim: {len(valid_idxs)} steps "
                              f"({valid_idxs[0].item()} to {valid_idxs[-1].item()})", args, None, 'a')

                    # ===== DataLoader =====
                    Dataset_cls   = getattr(dataset, args.dataset_name)
                    dataset_graph = Dataset_cls(
                        targets=None, graph=low_high_graph,
                        model_name=args.model_name, seq_l=args.seq_l
                    )
                    custom_collate = getattr(dataset, 'custom_collate_fn_graph')
                    t_offset = valid_idxs.min().item()
                    sampler  = Iterable_Graph(
                        dataset_graph=dataset_graph, shuffle=False,
                        idxs_vector=valid_idxs, t_offset=t_offset
                    )
                    dataloader = torch.utils.data.DataLoader(
                        dataset_graph, batch_size=args.batch_size,
                        num_workers=0, sampler=sampler,
                        collate_fn=custom_collate
                    )

                    # ===== 串行推理 =====
                    write_log("  Running inference...", args, None, 'a')
                    n_nodes_high = low_high_graph["high"].num_nodes
                    pr_list    = []
                    times_list = []
                    step = 0

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

                            # 打印第一个batch shape确认
                            if step == 0:
                                write_log(f"  y_pred shape: {y_pred.shape}", args, None, 'a')
                                write_log(f"  t shape     : {t.shape}", args, None, 'a')

                            # 拆分batch: (n_nodes × batch_size, 1) → (batch_size, n_nodes)
                            actual_batch = t.shape[0]
                            y_pred = y_pred.reshape(actual_batch, n_nodes_high, 1).squeeze(-1)  # (batch, n_nodes)

                            for i in range(actual_batch):
                                pr_list.append(y_pred[i].cpu())   # (n_nodes,)
                                times_list.append(t[i].cpu())      # scalar

                            if step % 50 == 0:
                                with open(os.path.join(args.log_path, args.log_file), 'a') as f:
                                    f.write(f"\n  [{period_name}/{mode}/{gcm}] Step {step}/{len(valid_idxs)}")
                            step += 1

                    # ===== 后处理 =====
                    times_cat = torch.stack(times_list)            # (n_steps,)
                    times_sorted, sort_idx = torch.sort(times_cat)
                    times_np = times_sorted.numpy()

                    tasmax_stack     = torch.stack(pr_list)        # (n_steps, n_nodes)
                    tasmax_norm      = tasmax_stack[sort_idx].numpy().T  # (n_nodes, n_steps)

                    write_log(f"  tasmax_norm shape: {tasmax_norm.shape}", args, None, 'a')
                    write_log(f"  Norm range: [{np.nanmin(tasmax_norm):.4f}, {np.nanmax(tasmax_norm):.4f}]",
                              args, None, 'a')

                    # 反归一化
                    tasmax_pred_full = denormalize_temperature(tasmax_norm, target_stats)

                    # 节点mask
                    degree_nodes = degree(
                        low_high_graph["high", "within", "high"].edge_index[1],
                        low_high_graph["high"].num_nodes
                    ).cpu().numpy()
                    node_mask = degree_nodes > 2

                    tasmax_pred = tasmax_pred_full[node_mask, :]

                    t_end = time.time()
                    write_log(f"  Done in {t_end - t_start:.1f}s", args, None, 'a')
                    write_log(f"  tasmax_pred shape: {tasmax_pred.shape}", args, None, 'a')
                    write_log(f"  Range: [{np.nanmin(tasmax_pred):.2f}, {np.nanmax(tasmax_pred):.2f}] K",
                              args, None, 'a')
                    write_log(f"  Mean : {np.nanmean(tasmax_pred):.2f} K", args, None, 'a')
                    write_log(f"  Std  : {np.nanstd(tasmax_pred):.2f} K", args, None, 'a')

                    # ===== 季节性（简单四等分）=====
                    n = tasmax_pred.shape[1]
                    q = n // 4
                    tasmax_seasons = [
                        tasmax_pred[:, :q],           # DJF
                        tasmax_pred[:, q:2*q],         # MAM
                        tasmax_pred[:, 2*q:3*q],       # JJA
                        tasmax_pred[:, 3*q:]           # SON
                    ]

                    # ===== 经纬度 =====
                    lat_high = low_high_graph["high"].lat.cpu().numpy()[node_mask]
                    lon_high = low_high_graph["high"].lon.cpu().numpy()[node_mask]
                    lat_low  = low_high_graph["low"].lat.cpu().numpy()
                    lon_low  = low_high_graph["low"].lon.cpu().numpy()

                    # ===== 偏差统计（无target，用pred自身统计）=====
                    write_log(f"\n  P5  : {np.nanpercentile(tasmax_pred, 5):.2f} K", args, None, 'a')
                    write_log(f"  P95 : {np.nanpercentile(tasmax_pred, 95):.2f} K", args, None, 'a')
                    write_log(f"  P99 : {np.nanpercentile(tasmax_pred, 99):.2f} K", args, None, 'a')

                    # ===== 构建输出结构（和验证代码格式一致）=====
                    data = HeteroData()
                    data["low"].lat          = lat_low
                    data["low"].lon          = lon_low
                    data["high"].lat         = lat_high
                    data["high"].lon         = lon_high
                    data["high"].degree      = degree_nodes[node_mask]
                    data.tasmax_normalized   = tasmax_norm
                    data.tasmax_pred         = tasmax_pred
                    data.tasmax_target       = None   # 推理无target
                    data.times               = times_np
                    data.gcm                 = gcm
                    data.period              = period_name
                    data.mode                = mode

                    results = {
                        "lon"                    : lon_high,
                        "lat"                    : lat_high,
                        "times"                  : times_np,
                        "tasmax_gnn4cd"          : tasmax_pred,
                        "tasmax_gnn4cd_seasons"  : tasmax_seasons,
                        "tasmax_target"          : None,
                        "tasmax_target_seasons"  : None,
                        "gcm"                    : gcm,
                        "period"                 : period_name,
                        "mode"                   : mode,
                        "valid_time_steps"       : n_valid,
                        "missing_time_steps"     : n_missing,
                    }

                    # ===== 保存 =====
                    with open(out_file, 'wb') as f:
                        pickle.dump(data, f)
                    write_log(f"  Saved: {out_file}", args, None, 'a')

                    with open(out_file_season, 'wb') as f:
                        pickle.dump(results, f)
                    write_log(f"  Saved: {out_file_season}", args, None, 'a')

                    # 释放内存
                    del low_high_graph, x_low_original, x_high_original
                    del dataset_graph, dataloader
                    del pr_list, times_list, tasmax_stack, tasmax_norm
                    torch.cuda.empty_cache()

                except Exception as e:
                    write_log(f"  ERROR: {e}", args, None, 'a')
                    import traceback
                    write_log(traceback.format_exc(), args, None, 'a')
                    failed_tasks.append(f"{period_name}/{mode}/{gcm} - {str(e)}")
                    continue

    # ==================== 总结 ====================
    write_log("\n" + "="*80, args, None, 'a')
    write_log("BATCH TASMAX INFERENCE COMPLETED", args, None, 'a')
    write_log(f"Total tasks  : {total_tasks}", args, None, 'a')
    write_log(f"Failed tasks : {len(failed_tasks)}", args, None, 'a')
    for ft in failed_tasks:
        write_log(f"  - {ft}", args, None, 'a')
    write_log("="*80, args, None, 'a')