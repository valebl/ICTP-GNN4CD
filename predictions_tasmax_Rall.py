import numpy as np
import pickle
import torch
import argparse
import time
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import importlib

#import safetensors

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
parser.add_argument('--train_path_cl', type=str)
parser.add_argument('--checkpoint_reg', type=str, default=None)
parser.add_argument('--output_file', type=str, default="full_validation_predictions.pkl")
parser.add_argument('--output_file_season', type=str, default="seasonal_predictions.pkl")

parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--target_file', type=str, default="pr_target.pkl") 
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--model_type', type=str, default=None)
parser.add_argument('--model', type=str, default=None) 
parser.add_argument('--dataset_name', type=str, default=None) 
parser.add_argument('--mode', type=str, default="RC") 
parser.add_argument('--test_idxs_file', type=str, default="")
parser.add_argument('--stats_mode', type=str, default="var") 
parser.add_argument('--target_type', type=str, default="tasmax")
parser.add_argument('--seq_l', type=int, default=24)

#-- start and end training dates
parser.add_argument('--validation_year', type=int)
parser.add_argument('--test_month_start', type=int)
parser.add_argument('--test_day_start', type=int)
parser.add_argument('--test_month_end', type=int)
parser.add_argument('--test_day_end', type=int)

parser.add_argument('--batch_size', type=int)
parser.add_argument('--seed', type=int)

parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')

parser.add_argument('--make_plots',  action='store_true')
parser.add_argument('--no-make_plots', dest='make_plots', action='store_false')

# denormalization function for temperature predictions
#New added 0201 try
def denormalize_temperature(t_normalized, target_stats):
    """
    Denormalize temperature predictions
    
    Args:
        t_normalized: Normalized temperature predictions (numpy array)
        target_stats: Dictionary containing normalization statistics, formatted as:
                     {'mean': float, 'std': float, 'procedure': 'z-score'} or
                     {'min': float, 'max': float, 'procedure': 'minmax'}
    
    Returns:
        Denormalized temperature values (Kelvin)
    """
    if target_stats['procedure'] == 'z-score':
        # Z-score denormalization: T = T_normalized * std + mean
        t_denorm = t_normalized * target_stats['std'] + target_stats['mean']
    else:  # minmax
        # Min-Max denormalization: T = T_normalized * (max - min) + min
        t_denorm = t_normalized * (target_stats['max'] - target_stats['min']) + target_stats['min']
    
    return t_denorm
#0201 try end

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

    write_log("Starting the testing...", args, accelerator, 'w')
    write_log(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.", args, accelerator, 'a')

    validation_year = args.validation_year
    if args.test_idxs_file == "":
        train_idxs, val_idxs = derive_train_val_idxs_new(
            1961, 1, 1, 1980, 11, 30, 1961, 
            args.model_name, idxs_not_all_nan=None, 
            validation_year=1975, args=None, accelerator=accelerator)
        
        write_log(f"\nValidation start idx: {val_idxs[0]}", args, accelerator, 'a')
        write_log(f"Validation end idx: {val_idxs[-1]}", args, accelerator, 'a')
        write_log(f"Total validation samples: {len(val_idxs)}", args, accelerator, 'a')
        
        test_val_idxs = torch.tensor([*range(val_idxs[0], val_idxs[-1])])
    else:
        with open(args.train_path_reg + args.test_idxs_file, 'rb') as f:
            test_idxs = pickle.load(f)
        write_log(f"Using the provided test idxs vector.", args, accelerator, 'a')

    # Load the temperature target (tasmax)
    with open(args.input_path + args.target_file, 'rb') as f:
        tasmax_target = pickle.load(f)
    
    write_log(f"\nLoaded tasmax_target shape: {tasmax_target.shape}", args, accelerator, 'a')

    # Load the graph
    with open(args.input_path + args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    # Load input standardization statistics
    with open(args.train_path_reg + "means_low.pkl", 'rb') as f:
        means_low = pickle.load(f)
    with open(args.train_path_reg + "stds_low.pkl", 'rb') as f:
        stds_low = pickle.load(f)
    with open(args.train_path_reg + "means_high.pkl", 'rb') as f:
        means_high = pickle.load(f)
    with open(args.train_path_reg + "stds_high.pkl", 'rb') as f:
        stds_high = pickle.load(f)
    
    write_log(f"\nLoaded input statistics:", args, accelerator, 'a')
    write_log(f"  means_low shape: {means_low.shape}", args, accelerator, 'a')
    write_log(f"  stds_low shape: {stds_low.shape}", args, accelerator, 'a')
    
    ##0201 try
    # Load target (temperature) standardization statistics
    target_stats = None
    try:
        with open(args.train_path_reg + "target_stats.pkl", 'rb') as f:
            target_stats = pickle.load(f)
        write_log(f"\nLoaded target (tasmax) standardization statistics:", args, accelerator, 'a')
        write_log(f"  Procedure: {target_stats['procedure']}", args, accelerator, 'a')
        if target_stats['procedure'] == 'z-score':
            write_log(f"  Mean: {target_stats['mean']:.4f} K", args, accelerator, 'a')
            write_log(f"  Std: {target_stats['std']:.4f} K", args, accelerator, 'a')
        else:
            write_log(f"  Min: {target_stats['min']:.4f} K", args, accelerator, 'a')
            write_log(f"  Max: {target_stats['max']:.4f} K", args, accelerator, 'a')
    except FileNotFoundError:
        write_log(f"\n!!! ERROR: target_stats.pkl not found in {args.train_path_reg}", args, accelerator, 'a')
        write_log(f"!!! This file is required for denormalizing temperature predictions!", args, accelerator, 'a')
        write_log(f"!!! Please re-run training to generate this file.", args, accelerator, 'a')
        raise FileNotFoundError("target_stats.pkl is required for temperature prediction")
    #0201 try end

    # Standardize input data 
    low_high_graph['low'].x, low_high_graph['high'].x = standardize_input(
        low_high_graph['low'].x, low_high_graph['high'].x, 
        means_low, stds_low, means_high, stds_high, args, accelerator)
    
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

    #Prepare data loader for validation
    Dataset_Graph = getattr(dataset, args.dataset_name)
    dataset_graph = Dataset_Graph(targets=None, graph=low_high_graph, model_name=args.model_name, seq_l=args.seq_l)
    custom_collate_fn = getattr(dataset, 'custom_collate_fn_graph')
    sampler_graph = Iterable_Graph(dataset_graph=dataset_graph, shuffle=False, idxs_vector=val_idxs)
    dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph, collate_fn=custom_collate_fn)

    # Load model
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

    # Run prediction
    tester = Tester()
    start = time.time()
    
    write_log(f"\nStarting temperature (tasmax) prediction...", args, accelerator, 'a')
    tasmax_normalized, times = tester.test(model, dataloader, args=args, accelerator=accelerator)
    
    end = time.time()
    write_log(f"\nPrediction completed in {end-start:.2f} seconds.", args, accelerator, 'a')
    
    
    #0201 try
    # Post-processing: Denormalization 
    # Process target values
    tasmax_target = tasmax_target.swapaxes(0,1)
    tasmax_target = tasmax_target[:, val_idxs].numpy()
    mask_nan = np.isnan(tasmax_target)
    
    write_log(f"\nTarget tasmax shape: {tasmax_target.shape}", args, accelerator, 'a')
    write_log(f"Target tasmax range: [{np.nanmin(tasmax_target):.2f}, {np.nanmax(tasmax_target):.2f}] K", 
             args, accelerator, 'a')
    # 0201 try end

    # Get high resolution node information
    graph_high_nodes = low_high_graph["high"].num_nodes
    write_log(f"\nNumber of high-res nodes: {graph_high_nodes}", args, accelerator, 'a')
    
    degree = degree(low_high_graph["high", "within", "high"].edge_index[1], 
                   low_high_graph["high"].num_nodes).cpu().numpy()
    
    # Create mask: filter out nodes with too low degree or all NaN
    mask = degree > 2 * np.array([~np.isnan(tasmax_target[i,:]).all() for i in range(tasmax_target.shape[0])])
    mask_nan = mask_nan[mask, :]
    tasmax_target = tasmax_target[mask, :]
    tasmax_target[mask_nan] = np.nan
    degree = degree[mask]
    
    write_log(f"\nAfter filtering, kept {mask.sum()} nodes out of {graph_high_nodes}", args, accelerator, 'a')

    # Latitude and longitude information
    lat_low = low_high_graph["low"].lat.cpu().numpy()
    lon_low = low_high_graph["low"].lon.cpu().numpy()
    lat_high = low_high_graph["high"].lat.cpu().numpy()[mask]
    lon_high = low_high_graph["high"].lon.cpu().numpy()[mask]

    # Gather predictions from all processes
    if accelerator is not None:
        accelerator.wait_for_everyone()
        times = accelerator.gather(times).squeeze()
        tasmax_normalized = accelerator.gather(tasmax_normalized)

    times, indices = torch.sort(times)
    times = times.cpu().numpy()
    indices = indices.cpu().numpy()

    write_log(f"\nFirst 10 time steps: {times[0:10]}", args, accelerator, 'a')

    # 0201 try
    tasmax_normalized = tasmax_normalized.squeeze().swapaxes(0,1).cpu().numpy()[:, indices]
    
    write_log(f"\nNormalized predictions shape: {tasmax_normalized.shape}", args, accelerator, 'a')
    write_log(f"Normalized predictions range: [{np.nanmin(tasmax_normalized):.4f}, {np.nanmax(tasmax_normalized):.4f}]", 
             args, accelerator, 'a')
    
    # denormalize
    tasmax_pred_full = denormalize_temperature(tasmax_normalized, target_stats)
    tasmax_pred = tasmax_pred_full[mask, :]
    
    write_log(f"\n=== Denormalized Temperature Predictions ===", args, accelerator, 'a')
    write_log(f"Predicted tasmax shape: {tasmax_pred.shape}", args, accelerator, 'a')
    write_log(f"Predicted tasmax range: [{np.nanmin(tasmax_pred):.2f}, {np.nanmax(tasmax_pred):.2f}] K", 
             args, accelerator, 'a')
    write_log(f"Predicted tasmax mean: {np.nanmean(tasmax_pred):.2f} K", args, accelerator, 'a')
    write_log(f"Predicted tasmax std: {np.nanstd(tasmax_pred):.2f} K", args, accelerator, 'a')
    
    write_log(f"\n=== Target vs Prediction Comparison ===", args, accelerator, 'a')
    write_log(f"Target mean: {np.nanmean(tasmax_target):.2f} K", args, accelerator, 'a')
    write_log(f"Prediction mean: {np.nanmean(tasmax_pred):.2f} K", args, accelerator, 'a')
    write_log(f"Mean difference: {np.nanmean(tasmax_pred) - np.nanmean(tasmax_target):.2f} K", args, accelerator, 'a')
    #0201 try end

    
    # Create the HeteroData object
    data = HeteroData()
    data["low"].lat = lat_low
    data["low"].lon = lon_low
    data["high"].lat = lat_high
    data["high"].lon = lon_high
    data["high"].degree = degree
    data.tasmax_normalized = tasmax_normalized  # Save normalized version for debugging
    data.tasmax_pred = tasmax_pred  # Denormalized predictions
    data.tasmax_target = tasmax_target  # Target values
    data.times = times

    #Create seasonal results dictionary
    results = {}
    results["lon"] = lon_high
    results["lat"] = lat_high
    results["times"] = times
    results["tasmax_target"] = tasmax_target
    results["tasmax_gnn4cd"] = tasmax_pred

    # Calculate seasonal statistics
    tasmax_pred_seasons = []
    tasmax_target_seasons = []

    # Define seasonal time indices
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

    tasmax_pred_seasons.append(tasmax_pred[:, djf_idxs])
    tasmax_pred_seasons.append(tasmax_pred[:, mam_start:mam_end])
    tasmax_pred_seasons.append(tasmax_pred[:, jja_start:jja_end])
    tasmax_pred_seasons.append(tasmax_pred[:, son_start:son_end])

    tasmax_target_seasons.append(tasmax_target[:, djf_idxs])
    tasmax_target_seasons.append(tasmax_target[:, mam_start:mam_end])
    tasmax_target_seasons.append(tasmax_target[:, jja_start:jja_end])
    tasmax_target_seasons.append(tasmax_target[:, son_start:son_end])

    results["tasmax_gnn4cd_seasons"] = tasmax_pred_seasons
    results["tasmax_target_seasons"] = tasmax_target_seasons

    # Calculate bias statistics
    # AVERAGE BIAS IS ADDED
    tasmax_bias_avg = np.nanmean(tasmax_pred, axis=1) - np.nanmean(tasmax_target, axis=1)
    tasmax_bias_percentage_avg = tasmax_bias_avg / np.nanmean(tasmax_target, axis=1) * 100
    results["tasmax_bias_avg"] = tasmax_bias_avg  # Unit: K
    results["tasmax_bias_percentage_avg"] = tasmax_bias_percentage_avg  # Unit: %
    write_log(f"\nBias statistics:", args, accelerator, 'a')
    write_log(f"  Mean bias: {np.nanmean(tasmax_bias_avg):.4f} K", args, accelerator, 'a')
    write_log(f"  Std bias: {np.nanstd(tasmax_bias_avg):.4f} K", args, accelerator, 'a')
    write_log(f"  Mean percentage bias: {np.nanmean(tasmax_bias_percentage_avg):.4f} %", args, accelerator, 'a')
    write_log(f"\nCompleted. Writing output files...", args, accelerator, 'a')

    # Save full predictions and seasonal results
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(data, f)
        write_log(f"Saved full predictions to {args.output_path + args.output_file}", args, accelerator, 'a')

        with open(args.output_path + args.output_file_season, 'wb') as f:
            pickle.dump(results, f)
        write_log(f"Saved seasonal results to {args.output_path + args.output_file_season}", args, accelerator, 'a')

    write_log(f"\n=== Testing completed successfully ===", args, accelerator, 'a')