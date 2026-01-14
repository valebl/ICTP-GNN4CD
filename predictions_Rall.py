import numpy as np
import pickle
import torch
import argparse
import time
import os
import importlib

import safetensors

from accelerate import Accelerator

from torch_geometric.data import HeteroData
from torch_geometric.utils import degree

import dataset
from dataset import Dataset_Graph, Iterable_Graph

from utils.tools import date_to_idxs, set_seed_everything
from utils.train_test import Tester

from utils.tools import date_to_idxs, write_log, standardize_input
        

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--train_path_reg', type=str)
parser.add_argument('--train_path_cl', type=str)
parser.add_argument('--checkpoint_reg', type=str, default=None)
parser.add_argument('--checkpoint_cl', type=str, default=None)
parser.add_argument('--output_file', type=str, default="G_predictions.pkl")

parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--target_file', type=str, default="pr_target.pkl") 
parser.add_argument('--model_type', type=str, default=None)
parser.add_argument('--model', type=str, default=None) 
parser.add_argument('--dataset_name', type=str, default=None) 
parser.add_argument('--mode', type=str, default="RC") 
parser.add_argument('--test_idxs_file', type=str, default="")
parser.add_argument('--stats_mode', type=str, default="var") 
parser.add_argument('--target_type', type=str, default="precipitation")
parser.add_argument('--seq_l', type=int, default=24)

#-- start and end training dates
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

    if args.test_idxs_file == "":
        test_start_idx, test_end_idx = date_to_idxs_new(test_year_start, args.test_month_start,
            args.test_day_start, test_year_end, args.test_month_end,
            args.test_day_end, first_year)        
        if test_start_idx < 2:
            test_start_idx = 2
        test_idxs = torch.tensor([*range(test_start_idx, test_end_idx)])
        write_log(f"\nUsing the provided start and end test times to derive the test idxs.", args, accelerator, 'a')
    else:
        with open(args.train_path_reg+args.test_idxs_file, 'rb') as f:
            test_idxs = pickle.load(f)
        write_log(f"Using the provided test idxs vector.", args, accelerator, 'a')

    # Load the precipitation target
    with open(args.input_path+args.target_file, 'rb') as f:
        pr_target = pickle.load(f)

    # Load the graph
    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

   

    # Load the input data statistics used during training
    # (At the moment we assume that the same statistics has been used for
    # the regressor and classifier in the RC model case)
    with open(args.train_path_reg + "means_low.pkl", 'rb') as f:
        means_low = pickle.load(f)
    with open(args.train_path_reg + "stds_low.pkl", 'rb') as f:
        stds_low = pickle.load(f)
    with open(args.train_path_reg + "means_high.pkl", 'rb') as f:
        means_high = pickle.load(f)
    with open(args.train_path_reg + "stds_high.pkl", 'rb') as f:
        stds_high = pickle.load(f)

    # Standardizing the input data
    low_high_graph['low'].x, low_high_graph['high'].x = standardize_input(
        low_high_graph['low'].x, low_high_graph['high'].x, means_low, stds_low, means_high, stds_high, args, accelerator) # num_nodes, time, vars, levels
    
    vars_names = ['q', 't', 'u', 'v', 'z']
    levels = ['500', '700', '850']
    if args.stats_mode == "var":
        for var in range(5):
            write_log(f"\nLow var {vars_names[var]}: mean={low_high_graph['low'].x[:,:,var,:].mean()}, std={low_high_graph['low'].x[:,:,var,:].std()}",
                      args, accelerator, 'a')
    elif args.stats_mode == "field":
        for var in range(5):
            for lev in range(5):
                write_log(f"\nLow var {vars_names[var]} lev {levels[lev]}: mean={low_high_graph[:,:,var,lev].mean()}, std={low_high_graph[:,:,var,lev].std()}",
                          args, accelerator, 'a')
    
    write_log(f"\nHigh z: mean={low_high_graph['high'].x[:,0].mean()}, std={low_high_graph['high'].x[:,0].std()}",
              args, accelerator, 'a')
    write_log(f"\nHigh land_use: mean={low_high_graph['high'].x[:,1:].mean()}, std={low_high_graph['high'].x[:,1:].std()}",
              args, accelerator, 'a')
    
    if args.target_type == "temperature":
        low_high_graph['low'].x = torch.cat((low_high_graph['low'].x[:,:,:1,:], low_high_graph['low'].x[:,:,2:,:]), dim=2)

    low_high_graph['low'].x = torch.flatten(low_high_graph['low'].x, start_dim=2, end_dim=-1)   # num_nodes, time, vars*levels

    Dataset_Graph = getattr(dataset, args.dataset_name)
    
    dataset_graph = Dataset_Graph(targets=None, graph=low_high_graph, model_name=args.model_typel, seq_l=args.seq_l)

    custom_collate_fn = getattr(dataset, 'custom_collate_fn_graph')
        
    sampler_graph = Iterable_Graph(dataset_graph=dataset_graph, shuffle=False, idxs_vector=test_idxs)
        
    dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph, collate_fn=custom_collate_fn)

    model_file = importlib.import_module(f"models.{args.model_typel}")
    Model = getattr(model_file, args.model_typel)
    
    model = Model(seq_l=args.seq_l+1)

    if accelerator is None:
        
        checkpoint_reg = torch.load(args.train_path_reg+args.checkpoint_reg, map_location=torch.device('cpu'), weights_only=True)
        device = 'cpu'
    else:
        
        try:
                checkpoint_reg = torch.load(args.train_path_reg+args.checkpoint_reg+"/pytorch_model.bin", weights_only=True)
        except:
                checkpoint_reg = safetensors.torch.load_file(args.train_path_reg+args.checkpoint_reg+"/model.safetensors")
                torch.save(checkpoint_reg, args.train_path_reg+args.checkpoint_reg+"pytorch_model.bin")
        device = accelerator.device
    
    write_log("\nLoading state dict.", args, accelerator, 'a')
    
    model.load_state_dict(checkpoint_reg)

    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    # write_log(f"\nStarting the test, from idx {test_start_idx} to idx {test_end_idx}.", args, accelerator, 'a')

    tester = Tester()

    start = time.time()

    
    pr_Rall, times = tester.test(model, dataloader, args=args, accelerator=accelerator)
    

    end = time.time()

    ### POST-PROCESS
    pr_target = pr_target[:,test_idxs].numpy()

    threshold = 0.1
    mask_nan = np.isnan(pr_target)
    pr_target[pr_target < threshold] = 0
    pr_target = np.round(pr_target, decimals=1)

    mask =  degree > 2 * np.array([~np.isnan(pr_target[i,:]).all() for i in range(pr_target.shape[0])])
    mask_nan = mask_nan[mask,:]

    pr_target = pr_target[mask,:]
    pr_target[mask_nan] = np.nan
    degree = degree[mask]
  
    # LON LAT
    lat_low = low_high_graph["low"].lat.cpu().numpy()
    lon_low = low_high_graph["low"].lon.cpu().numpy()
    lat_high = low_high_graph["high"].lat.cpu().numpy()[mask]
    lon_high = low_high_graph["high"].lon.cpu().numpy()[mask]

    # Gather the values in *tensors* across all processes and concatenate them on the first dimension. Useful to
    # regroup the predictions from all processes when doing evaluation.
    if accelerator is not None:
        accelerator.wait_for_everyone()

        times = accelerator.gather(times).squeeze()
        pr_Rall = accelerator.gather(pr_R)
        


    times, indices = torch.sort(times)
    times = times.cpu().numpy()
    indices = indices.cpu().numpy()

    
    
    # not processed Rall output
    pr_Rall = pr_Rall.squeeze().swapaxes(0,1).cpu().numpy()[:,indices]
    
    # processed estimates, ready to use
    pr = np.where(np.isfinite(np.expm1(pr_Rall)), np.expm1(pr_Rall), np.nan)
    pr = pr[mask,:]; [pr<threshold] = 0; pr[mask_nan] = np.nan
    

    # Create the pyg object
    data = HeteroData()
    
    data.pr_target = pr_target
    data.times = times
    data["low"].lat = lat_low
    data["low"].lon = lon_low
    data["high"].lat = lat_high
    data["high"].lon = lon_high
    data["high"].degree = degree.cpu().numpy()

    
    data.pr_Rall_raw = pr_Rall
    data.pr_Rall = pr

    ## OPTIONAL: create a dictionary with some ready-to-use results
    #results dictionary stores seasonal target maps for the val year plus bias map
    results = {}
    results["lon"] = lon_high
    results["lat"] = lat_high
    results["times"] = times
    results["pr_original"] = pr_target
    results["pr_gnn4cd"] = pr

    pr_pred_seasons = []
    pr_target_seasons = []

    jf_start, jf_end = date_to_idxs_new(year_start=validation_year, month_start=1,day_start=1,year_end=validation_year,month_end=2,day_end=28,first_year=validation_year,first_month=1,first_day=1)
    mam_start, mam_end = date_to_idxs_new(year_start=validation_year, month_start=3,day_start=1,year_end=validation_year,month_end=5,day_end=31,first_year=validation_year,first_month=1,first_day=1)
    jja_start, jja_end = date_to_idxs_new(year_start=validation_year, month_start=6,day_start=1,year_end=validation_year,month_end=8,day_end=31,first_year=validation_year,first_month=1,first_day=1)
    son_start, son_end = date_to_idxs_new(year_start=validation_year, month_start=9,day_start=1,year_end=validation_year,month_end=11,day_end=31,first_year=validation_year,first_month=1,first_day=1)
    d_start, d_end =date_to_idxs_new(year_start=validation_year, month_start=12,day_start=1,year_end=validation_year,month_end=12,day_end=31,first_year=validation_year,first_month=1,first_day=1)
 
    djf_idxs = np.arange(jf_start, jf_end).tolist()
    djf_idxs.extend(np.arange(d_start, d_end).tolist())

    pr_pred_seasons.append(pr[:,djf_idxs])
    pr_pred_seasons.append(pr[:,mam_start: mam_end])
    pr_pred_seasons.append(pr[:,jja_start: jja_end])
    pr_pred_seasons.append(pr[:,son_start: son_end])

    pr_target_seasons.append(pr_target[:,djf_idxs])
    pr_target_seasons.append(pr_target[:,mam_start: mam_end])
    pr_target_seasons.append(pr_target[:,jja_start: jja_end])
    pr_target_seasons.append(pr_target[:,son_start: son_end])

    results["pr_gnn4cd_seasons"] = pr_pred_seasons
    results["pr_original_seasons"] = pr_target_seasons

    #Bias average
    pr_bias_avg = np.nanmean(pr, axis=1) - np.nanmean(pr_target, axis=1)
    pr_bias_percentage_avg = pr_bias_avg / np.nanmean(pr_target, axis=1) * 100

    results["pr_bias_percentage_avg"] = pr_bias_percentage_avg

    write_log(f"\nDone. Testing concluded in {end-start} seconds.\nWrite the files.", args, accelerator, 'a')

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(data, f)

        with open(args.output_path + args.output_file_season, 'wb') as f:
            pickle.dump(results, f)


    ## OPTIONAL: create a dictionary with some ready-to-use results
    # results = {}
    # results["lon"] = lon_high
    # results["lat"] = lat_high
    # results["times"] = times
    # results["pr_gripho"] = pr_target
    # results["pr_gnn4cd"] = pr

    # # sesasons

    # pr_pred_seasons = []
    # pr_target_seasons = []

    # jf_start, jf_end = date_to_idxs(year_start=2007, month_start=1,day_start=1,year_end=2007,month_end=2,day_end=28,first_year=2007,first_month=1,first_day=1)
    # mam_start, mam_end = date_to_idxs(year_start=2007, month_start=3,day_start=1,year_end=2007,month_end=5,day_end=31,first_year=2007,first_month=1,first_day=1)
    # jja_start, jja_end = date_to_idxs(year_start=2007, month_start=6,day_start=1,year_end=2007,month_end=8,day_end=31,first_year=2007,first_month=1,first_day=1)
    # son_start, son_end = date_to_idxs(year_start=2007, month_start=9,day_start=1,year_end=2007,month_end=11,day_end=30,first_year=2007,first_month=1,first_day=1)

    # d_start, d_end = date_to_idxs(year_start=2007, month_start=12,day_start=1,year_end=2007,month_end=12,day_end=31,first_year=2007,first_month=1,first_day=1)

    # djf_idxs = np.arange(jf_start, jf_end).tolist()
    # djf_idxs.extend(np.arange(d_start, d_end).tolist())

    # pr_pred_seasons.append(pr[:,djf_idxs])
    # pr_pred_seasons.append(pr[:,mam_start: mam_end])
    # pr_pred_seasons.append(pr[:,jja_start: jja_end])
    # pr_pred_seasons.append(pr[:,son_start: son_end])

    # pr_target_seasons.append(pr_target[:,djf_idxs])
    # pr_target_seasons.append(pr_target[:,mam_start: mam_end])
    # pr_target_seasons.append(pr_target[:,jja_start: jja_end])
    # pr_target_seasons.append(pr_target[:,son_start: son_end])

    # results["pr_gnn4cd_seasons"] = pr_pred_seasons
    # results["pr_gripho_seasons"] = pr_target_seasons

    # # Mean percentage bias

    # pr_bias_avg = np.nanmean(pr, axis=1) - np.nanmean(pr_target, axis=1)
    # pr_bias_percentage_avg = pr_bias_avg / np.nanmean(pr_target, axis=1) * 100

    # results["pr_bias_percentage_avg"] = pr_bias_percentage_avg

    # # Diurnal cycles

    # pr_pred_seasons_daily_cycle = np.zeros((4,24))
    # for s in range(4):
    #     pr_season = pr_pred_seasons[s]
    #     for i in range(0,24):
    #         pr_pred_seasons_daily_cycle[s,i] = np.nanmean(pr_season[:,i::24])

    # pr_gripho_seasons_daily_cycle = np.zeros((4,24))
    # for s in range(4):
    #     pr_season = pr_target_seasons[s]
    #     for i in range(0,24):
    #         pr_gripho_seasons_daily_cycle[s,i] = np.nanmean(pr_season[:,i::24])

    # t_gripho = 0.1
    # t = 0.1
    # pr_pred_seasons_daily_cycle_intensity = np.zeros((4,24))
    # pr_pred_seasons_daily_cycle_frequency = np.zeros((4,24))
    # for s in range(4):
    #     pr_season = pr_pred_seasons[s]
    #     for i in range(0,24):
    #         pr_pred_seasons_daily_cycle_intensity[s,i] = np.nanmean(pr_season[:,i::24][pr_season[:,i::24]>=t])
    #         pr_pred_seasons_daily_cycle_frequency[s,i] = (pr_season[:,i::24]>=t).sum() / pr_season[:,i::24].flatten().shape[0] * 100

    # pr_gripho_seasons_daily_cycle_intensity = np.zeros((4,24))
    # pr_gripho_seasons_daily_cycle_frequency = np.zeros((4,24))
    # for s in range(4):
    #     pr_season = pr_target_seasons[s]
    #     for i in range(0,24):
    #         pr_gripho_seasons_daily_cycle_intensity[s,i] = np.nanmean(pr_season[:,i::24][pr_season[:,i::24]>=t_gripho])
    #         pr_gripho_seasons_daily_cycle_frequency[s,i] = (pr_season[:,i::24]>=t_gripho).sum() / pr_season[:,i::24].flatten().shape[0] * 100

    # results["pr_gnn4cd_seasons_daily_cycle"] = pr_pred_seasons_daily_cycle
    # results["pr_gripho_seasons_daily_cycle"] = pr_gripho_seasons_daily_cycle
    # results["pr_gnn4cd_seasons_daily_cycle_intensity"] = pr_pred_seasons_daily_cycle_intensity
    # results["pr_gnn4cd_seasons_daily_cycle_frequency"] = pr_pred_seasons_daily_cycle_frequency
    # results["pr_gripho_seasons_daily_cycle_intensity"] = pr_gripho_seasons_daily_cycle_intensity
    # results["pr_gripho_seasons_daily_cycle_frequency"] = pr_gripho_seasons_daily_cycle_frequency

    # # PDF

    # hist_y, bin_edges_y = np.histogram(pr_target.flatten(), bins=np.arange(0,200,0.5).astype(np.float32), density=False)
    # hist_pr, bin_edges_pr = np.histogram(pr.flatten(), bins=np.arange(0,200,0.5).astype(np.float32), density=False)

    # Ntot_y = hist_y.sum()
    # Ntot_pr = hist_pr.sum()

    # bin_edges_y_centre = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
    # bin_edges_pr_centre = (bin_edges_pr[:-1] + bin_edges_pr[1:]) / 2

    # results["bin_edges_centre_gripho"] = bin_edges_y_centre
    # results["bin_edges_centre_gnn4cd"] = bin_edges_pr_centre
    # results["hist/Ntot_gripho"] = hist_y/Ntot_y
    # results["hist/Ntot_gnn4cd"] = hist_pr/Ntot_pr

    # # EXTREME PERCENTILES

    # p99_y = np.nanpercentile(pr_target, q=99, axis=1)
    # p99_pred = np.nanpercentile(pr, q=99, axis=1)
    # p99_bias = p99_pred - p99_y
    # p99_bias_percentile = p99_bias / p99_y * 100

    # p999_y = np.nanpercentile(pr_target, q=99.9, axis=1)
    # p999_pred = np.nanpercentile(pr, q=99.9, axis=1)
    # p999_bias = p999_pred - p999_y
    # p999_bias_percentile = p999_bias / p999_y * 100

    # results["pr_gripho_p99"] = p99_y
    # results["pr_gnn4cd_p99"] = p99_pred
    # results["p99_bias_percentile"] = p99_bias_percentile
    # results["pr_gripho_p999"] = p999_y
    # results["pr_gnn4cd_p999"] = p999_pred
    # results["p999_bias_percentile"] = p999_bias_percentile

    # if accelerator is None or accelerator.is_main_process:
    #     with open(args.output_path + "results_dict.pkl", 'wb') as f:
    #         pickle.dump(results, f)