import numpy as np
import pickle
import torch
import argparse
import time
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import importlib

from accelerate import Accelerator
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree

import dataset
from dataset import Dataset_Graph, Iterable_Graph

from utils.tools import date_to_idxs_new, set_seed_everything, derive_train_val_idxs_new
from utils.train_test import Tester
from utils.tools import date_to_idxs, write_log, standardize_input

        
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_path', type=str, help='path to logfile directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--train_path_reg', type=str)
parser.add_argument('--checkpoint_reg', type=str, default=None)
parser.add_argument('--output_file', type=str, default="full_validation_predictions.pkl")
parser.add_argument('--output_file_season', type=str, default="seasonal_predictions.pkl")

parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--target_file', type=str, default="pr_target.pkl") 
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--model_type', type=str, default=None)
parser.add_argument('--model', type=str, default=None) 
parser.add_argument('--mode', type=str, default="RC") 
parser.add_argument('--dataset_name', type=str, default=None) 
parser.add_argument('--test_idxs_file', type=str, default="")
parser.add_argument('--stats_mode', type=str, default="var") 
parser.add_argument('--target_type', type=str, default="precipitation")
parser.add_argument('--seq_l', type=int, default=24)

#-- start and end training dates
parser.add_argument('--validation_year', type=int)
parser.add_argument('--test_month_start', type=int)
parser.add_argument('--test_day_start', type=int)
parser.add_argument('--test_month_end', type=int)
parser.add_argument('--test_day_end', type=int)

parser.add_argument('--batch_size', type=int)
parser.add_argument('--seed', type=int)

parser.add_argument('--use_accelerate', action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')

parser.add_argument('--make_plots', action='store_true')
parser.add_argument('--no-make_plots', dest='make_plots', action='store_false')


if __name__ == '__main__':

    args = parser.parse_args()
    
    # Set all seeds
    set_seed_everything(seed=args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    if args.use_accelerate is True:
        accelerator = Accelerator()
    else:
        accelerator = None

    write_log("="*80, args, accelerator, 'w')
    write_log("Starting PRECIPITATION prediction...", args, accelerator, 'a')
    write_log("="*80, args, accelerator, 'a')
    write_log(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.", args, accelerator, 'a')

    validation_year = args.validation_year
    
    # ==================== 获取验证集索引 ====================
    if args.test_idxs_file == "":
        train_idxs, val_idxs = derive_train_val_idxs_new(
            1961, 1, 1, 1980, 11, 30, 1961, 
            args.model_name, idxs_not_all_nan=None, 
            validation_year=1975, args=None, accelerator=accelerator
        )
        
        write_log(f"\nValidation start idx: {val_idxs[0]}", args, accelerator, 'a')
        write_log(f"Validation end idx: {val_idxs[-1]}", args, accelerator, 'a')
        write_log(f"Total validation samples: {len(val_idxs)}", args, accelerator, 'a')
        
        test_val_idxs = torch.tensor([*range(val_idxs[0], val_idxs[-1])])
    else:
        with open(args.train_path_reg + args.test_idxs_file, 'rb') as f:
            test_idxs = pickle.load(f)
        write_log(f"Using the provided test idxs vector.", args, accelerator, 'a')

    # ==================== 加载降水目标数据 ====================
    with open(args.input_path + args.target_file, 'rb') as f:
        pr_target = pickle.load(f)
    
    write_log(f"\nLoaded precipitation target shape: {pr_target.shape}", args, accelerator, 'a')

    # ==================== 加载图数据 ====================
    with open(args.input_path + args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    # ==================== 加载训练时的标准化统计量 ====================
    with open(args.train_path_reg + "means_low.pkl", 'rb') as f:
        means_low = pickle.load(f)
    with open(args.train_path_reg + "stds_low.pkl", 'rb') as f:
        stds_low = pickle.load(f)
    with open(args.train_path_reg + "means_high.pkl", 'rb') as f:
        means_high = pickle.load(f)
    with open(args.train_path_reg + "stds_high.pkl", 'rb') as f:
        stds_high = pickle.load(f)
    
    write_log(f"\nLoaded input standardization statistics:", args, accelerator, 'a')
    write_log(f"  means_low shape: {means_low.shape}", args, accelerator, 'a')
    write_log(f"  stds_low shape: {stds_low.shape}", args, accelerator, 'a')
    
    # 尝试加载降水阈值（如果存在）
    try:
        with open(args.train_path_reg + "target_stats.pkl", 'rb') as f:
            target_stats = pickle.load(f)
        if 'threshold' in target_stats:
            pr_threshold = target_stats['threshold']
            write_log(f"\nLoaded precipitation threshold: {pr_threshold} mm", args, accelerator, 'a')
        else:
            pr_threshold = 0.1
            write_log(f"\nUsing default precipitation threshold: {pr_threshold} mm", args, accelerator, 'a')
    except:
        pr_threshold = 0.1
        write_log(f"\nNo target_stats.pkl found, using default threshold: {pr_threshold} mm", args, accelerator, 'a')

    # ==================== 标准化输入数据 ====================
    low_high_graph['low'].x, low_high_graph['high'].x = standardize_input(
        low_high_graph['low'].x, low_high_graph['high'].x, 
        means_low, stds_low, means_high, stds_high, args, accelerator
    )
    
    # 打印标准化后的统计信息
    vars_names = ['q', 't', 'u', 'v', 'z']
    levels = ['500', '750', '850']
    
    if args.stats_mode == "var":
        for var in range(5):
            write_log(f"\nLow var {vars_names[var]}: mean={low_high_graph['low'].x[:,:,var,:].mean():.6f}, std={low_high_graph['low'].x[:,:,var,:].std():.6f}",
                      args, accelerator, 'a')
    elif args.stats_mode == "field":
        for var in range(5):
            for lev in range(3):
                write_log(f"\nLow var {vars_names[var]} lev {levels[lev]}: mean={low_high_graph['low'].x[:,:,var,lev].mean():.6f}, std={low_high_graph['low'].x[:,:,var,lev].std():.6f}",
                          args, accelerator, 'a')
    
    write_log(f"\nHigh z: mean={low_high_graph['high'].x[:,0].mean():.6f}, std={low_high_graph['high'].x[:,0].std():.6f}",
              args, accelerator, 'a')

    # Flatten the vars + levels dimension
    low_high_graph['low'].x = torch.flatten(low_high_graph['low'].x, start_dim=2, end_dim=-1)

    # ==================== 准备数据加载器 ====================
    Dataset_Graph = getattr(dataset, args.dataset_name)
    dataset_graph = Dataset_Graph(targets=None, graph=low_high_graph, model_name=args.model_name, seq_l=args.seq_l)
    custom_collate_fn = getattr(dataset, 'custom_collate_fn_graph')
    sampler_graph = Iterable_Graph(dataset_graph=dataset_graph, shuffle=False, idxs_vector=val_idxs)
    dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph, collate_fn=custom_collate_fn)

    # ==================== 加载模型 ====================
    model_file = importlib.import_module(f"models.{args.model_name}")
    Model = getattr(model_file, args.model_name)
    model = Model(seq_l=args.seq_l+1)

    if accelerator is None:
        checkpoint_reg = torch.load(args.train_path_reg + args.checkpoint_reg + "/pytorch_model.bin", 
                                   map_location=torch.device('cpu'), weights_only=True)
        device = 'cpu'
    else:
        checkpoint_reg = torch.load(args.train_path_reg + args.checkpoint_reg + "/pytorch_model.bin", 
                                   weights_only=True)
        device = accelerator.device
    
    write_log("\nLoading model state dict...", args, accelerator, 'a')
    model.load_state_dict(checkpoint_reg)

    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    # ==================== 运行预测 ====================
    tester = Tester()
    start = time.time()
    
    write_log(f"\nStarting precipitation prediction...", args, accelerator, 'a')
    pr_normalized, times = tester.test(model, dataloader, args=args, accelerator=accelerator)
    
    end = time.time()
    write_log(f"\nPrediction completed in {end-start:.2f} seconds.", args, accelerator, 'a')

    # ==================== 后处理：降水特有处理 ====================
    write_log("\n" + "="*80, args, accelerator, 'a')
    write_log("POST-PROCESSING PRECIPITATION", args, accelerator, 'a')
    write_log("="*80, args, accelerator, 'a')
    
    # 处理目标值
    pr_target = pr_target.swapaxes(0, 1)
    
    # 降水目标值处理：应用阈值
    pr_target[pr_target < pr_threshold] = 0.0
    pr_target = np.round(pr_target, decimals=1)  # 降水精度到0.1mm
    
    write_log(f"\nProcessed precipitation target:", args, accelerator, 'a')
    write_log(f"  Range: [{np.nanmin(pr_target):.2f}, {np.nanmax(pr_target):.2f}] mm", 
             args, accelerator, 'a')
    write_log(f"  Mean: {np.nanmean(pr_target):.2f} mm", args, accelerator, 'a')
    
    pr_target = pr_target[:, val_idxs].numpy()
    mask_nan = np.isnan(pr_target)
    
    write_log(f"\nTarget shape after selection: {pr_target.shape}", args, accelerator, 'a')
    
    # 获取high resolution节点信息
    graph_high_nodes = low_high_graph["high"].num_nodes
    write_log(f"\nNumber of high-res nodes in graph: {graph_high_nodes}", args, accelerator, 'a')
    
    degree_nodes = degree(low_high_graph["high", "within", "high"].edge_index[1], 
                         low_high_graph["high"].num_nodes).cpu().numpy()
    write_log(f"Degree array shape: {degree_nodes.shape}", args, accelerator, 'a')
   
    # 创建mask：过滤掉度数太小或全是NaN的节点
    mask = degree_nodes > 2 * np.array([~np.isnan(pr_target[i,:]).all() for i in range(pr_target.shape[0])])
    mask_nan = mask_nan[mask, :]
    
    pr_target = pr_target[mask, :]
    pr_target[mask_nan] = np.nan
    degree_nodes = degree_nodes[mask]
    
    write_log(f"\nAfter filtering, kept {mask.sum()} nodes out of {graph_high_nodes}", args, accelerator, 'a')

    # 经纬度信息
    lat_low = low_high_graph["low"].lat.cpu().numpy()
    lon_low = low_high_graph["low"].lon.cpu().numpy()
    lat_high = low_high_graph["high"].lat.cpu().numpy()[mask]
    lon_high = low_high_graph["high"].lon.cpu().numpy()[mask]

    # ==================== Gather predictions from all processes ====================
    if accelerator is not None:
        accelerator.wait_for_everyone()
        times = accelerator.gather(times).squeeze()
        pr_normalized = accelerator.gather(pr_normalized)

    # 排序
    times, indices = torch.sort(times)
    times = times.cpu().numpy()
    indices = indices.cpu().numpy()

    write_log(f"\nFirst 10 time steps: {times[:10]}", args, accelerator, 'a')

    # 处理预测值（log变换后的）
    pr_normalized = pr_normalized.squeeze().swapaxes(0, 1).cpu().numpy()[:, indices]
    
    write_log(f"\nNormalized (log1p) predictions shape: {pr_normalized.shape}", args, accelerator, 'a')
    write_log(f"Normalized predictions range: [{np.nanmin(pr_normalized):.4f}, {np.nanmax(pr_normalized):.4f}]", 
             args, accelerator, 'a')
    
    # ==================== 【关键步骤】反变换：log1p -> expm1 ====================
    write_log("\nApplying expm1 transformation to predictions...", args, accelerator, 'a')
    
    # expm1是log1p的逆变换
    pr_pred_full = np.where(np.isfinite(np.expm1(pr_normalized)), 
                           np.expm1(pr_normalized), np.nan)
    
    # 应用降水阈值：<threshold的设为0
    pr_pred_full[pr_pred_full < pr_threshold] = 0.0
    
    # 四舍五入到0.1mm
    pr_pred_full = np.round(pr_pred_full, decimals=1)
    
    # 应用空间mask
    pr_pred = pr_pred_full[mask, :]
    
    write_log(f"\n" + "="*80, args, accelerator, 'a')
    write_log("PRECIPITATION PREDICTION STATISTICS", args, accelerator, 'a')
    write_log("="*80, args, accelerator, 'a')
    write_log(f"Predicted precipitation shape: {pr_pred.shape}", args, accelerator, 'a')
    write_log(f"Predicted precipitation range: [{np.nanmin(pr_pred):.2f}, {np.nanmax(pr_pred):.2f}] mm", 
             args, accelerator, 'a')
    write_log(f"Predicted precipitation mean: {np.nanmean(pr_pred):.3f} mm", args, accelerator, 'a')
    write_log(f"Predicted precipitation std: {np.nanstd(pr_pred):.3f} mm", args, accelerator, 'a')
    
    # 统计干湿天
    dry_ratio_pred = (pr_pred == 0).sum() / (~np.isnan(pr_pred)).sum() * 100
    dry_ratio_target = (pr_target == 0).sum() / (~np.isnan(pr_target)).sum() * 100
    wet_mean_pred = np.nanmean(pr_pred[pr_pred > pr_threshold])
    wet_mean_target = np.nanmean(pr_target[pr_target > pr_threshold])
    
    write_log(f"\n--- Dry/Wet Day Statistics ---", args, accelerator, 'a')
    write_log(f"Dry days (={pr_threshold}mm) - Target: {dry_ratio_target:.2f}%, Prediction: {dry_ratio_pred:.2f}%", 
             args, accelerator, 'a')
    write_log(f"Wet days mean - Target: {wet_mean_target:.2f} mm, Prediction: {wet_mean_pred:.2f} mm", 
             args, accelerator, 'a')
    
    # 极端降水统计
    p95_target = np.nanpercentile(pr_target, 95)
    p95_pred = np.nanpercentile(pr_pred, 95)
    p99_target = np.nanpercentile(pr_target, 99)
    p99_pred = np.nanpercentile(pr_pred, 99)
    
    write_log(f"\n--- Extreme Precipitation ---", args, accelerator, 'a')
    write_log(f"P95 - Target: {p95_target:.2f} mm, Prediction: {p95_pred:.2f} mm", args, accelerator, 'a')
    write_log(f"P99 - Target: {p99_target:.2f} mm, Prediction: {p99_pred:.2f} mm", args, accelerator, 'a')
    
    # 误差统计
    bias = np.nanmean(pr_pred - pr_target)
    rmse = np.sqrt(np.nanmean((pr_pred - pr_target)**2))
    mae = np.nanmean(np.abs(pr_pred - pr_target))
    
    write_log(f"\n--- Error Metrics ---", args, accelerator, 'a')
    write_log(f"Bias: {bias:.3f} mm", args, accelerator, 'a')
    write_log(f"RMSE: {rmse:.3f} mm", args, accelerator, 'a')
    write_log(f"MAE: {mae:.3f} mm", args, accelerator, 'a')
    write_log("="*80, args, accelerator, 'a')

    # ==================== 创建输出数据结构 ====================
    data = HeteroData()
    data["low"].lat = lat_low
    data["low"].lon = lon_low
    data["high"].lat = lat_high
    data["high"].lon = lon_high
    data["high"].degree = degree_nodes
    data.pr_normalized = pr_normalized  # 保存log变换后的版本（用于调试）
    data.pr_pred = pr_pred             # expm1反变换后的预测（最终结果）
    data.pr_target = pr_target         # 目标值
    data.times = times
    data.threshold = pr_threshold      # 保存阈值信息

    # ==================== 创建季节性结果字典 ====================
    results = {}
    results["lon"] = lon_high
    results["lat"] = lat_high
    results["times"] = times
    results["pr_target"] = pr_target
    results["pr_gnn4cd"] = pr_pred
    results["threshold"] = pr_threshold

    # 计算季节性统计
    pr_pred_seasons = []
    pr_target_seasons = []

    # 定义季节时间索引
    jf_start, jf_end = date_to_idxs_new(
        year_start=validation_year, month_start=1, day_start=1,
        year_end=validation_year, month_end=2, day_end=28,
        first_year=validation_year, first_month=1, first_day=1)
    
    mam_start, mam_end = date_to_idxs_new(
        year_start=validation_year, month_start=3, day_start=1,
        year_end=validation_year, month_end=5, day_end=31,
        first_year=validation_year, first_month=1, first_day=1)
    
    jja_start, jja_end = date_to_idxs_new(
        year_start=validation_year, month_start=6, day_start=1,
        year_end=validation_year, month_end=8, day_end=31,
        first_year=validation_year, first_month=1, first_day=1)
    
    son_start, son_end = date_to_idxs_new(
        year_start=validation_year, month_start=9, day_start=1,
        year_end=validation_year, month_end=11, day_end=30,
        first_year=validation_year, first_month=1, first_day=1)
    
    d_start, d_end = date_to_idxs_new(
        year_start=validation_year, month_start=12, day_start=1,
        year_end=validation_year, month_end=12, day_end=31,
        first_year=validation_year, first_month=1, first_day=1)
 
    # DJF (December-January-February)
    djf_idxs = np.arange(jf_start, jf_end).tolist()
    djf_idxs.extend(np.arange(d_start, d_end).tolist())

    pr_pred_seasons.append(pr_pred[:, djf_idxs])
    pr_pred_seasons.append(pr_pred[:, mam_start:mam_end])
    pr_pred_seasons.append(pr_pred[:, jja_start:jja_end])
    pr_pred_seasons.append(pr_pred[:, son_start:son_end])

    pr_target_seasons.append(pr_target[:, djf_idxs])
    pr_target_seasons.append(pr_target[:, mam_start:mam_end])
    pr_target_seasons.append(pr_target[:, jja_start:jja_end])
    pr_target_seasons.append(pr_target[:, son_start:son_end])

    results["pr_gnn4cd_seasons"] = pr_pred_seasons
    results["pr_target_seasons"] = pr_target_seasons

    # 计算偏差（降水用相对偏差更合理）
    pr_bias_avg = np.nanmean(pr_pred, axis=1) - np.nanmean(pr_target, axis=1)
    # 相对偏差（百分比）
    pr_bias_percentage_avg = pr_bias_avg / (np.nanmean(pr_target, axis=1) + 0.01) * 100
    results["pr_bias_avg"] = pr_bias_avg  # 绝对偏差 (mm)
    results["pr_bias_percentage_avg"] = pr_bias_percentage_avg  # 相对偏差 (%)

    write_log(f"\nBias statistics:", args, accelerator, 'a')
    write_log(f"  Mean absolute bias: {np.nanmean(pr_bias_avg):.4f} mm", args, accelerator, 'a')
    write_log(f"  Mean relative bias: {np.nanmean(pr_bias_percentage_avg):.2f} %", args, accelerator, 'a')
    write_log(f"  Std of bias: {np.nanstd(pr_bias_avg):.4f} mm", args, accelerator, 'a')

    write_log(f"\nCompleted. Writing output files...", args, accelerator, 'a')

    # ==================== 保存结果 ====================
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(data, f)
        write_log(f"Saved full predictions to {args.output_path + args.output_file}", args, accelerator, 'a')

        with open(args.output_path + args.output_file_season, 'wb') as f:
            pickle.dump(results, f)
        write_log(f"Saved seasonal results to {args.output_path + args.output_file_season}", args, accelerator, 'a')

    write_log(f"\n" + "="*80, args, accelerator, 'a')
    write_log("PRECIPITATION PREDICTION COMPLETED SUCCESSFULLY", args, accelerator, 'a')
    write_log("="*80, args, accelerator, 'a')