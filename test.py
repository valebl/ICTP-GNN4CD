import numpy as np
import pickle
import torch
import json
import argparse
import time
import os
import importlib
from accelerate import Accelerator
from torch_geometric.data import HeteroData

from dataset import Dataset_Graph, custom_collate_fn_graph

from utils.helpers.tools import set_seed_everything
from data.loaders.complete_loader import load_dataset_CORDEXML
from utils.helpers.tools import write_log
from utils.testing.test_NLL import NLL_Tester
from utils.testing.test import Tester
from utils.helpers.tools import write_log, standardize_input, invert_normalization, date_to_idxs_from_timeindex

import xarray as xr

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--domain', type=str)
parser.add_argument('--experiment', type=str)
parser.add_argument('--period', type=str)
parser.add_argument('--train_path', type=str, help='path to logfile directory')
parser.add_argument('--input_path_P', type=str, help='path to test predictors directory')
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_path', type=str, help='path to logfile directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')
parser.add_argument('--predictors_filename', type=str, help='filename')
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--model_type', type=str, default="Rall")
parser.add_argument('--model', type=str, default="Rall")
parser.add_argument('--dataset_name', type=str, default=None)  
parser.add_argument('--seq_l', type=int, default=0)
parser.add_argument('--graph_file', type=str, default=None)
parser.add_argument('--output_file', type=str, default="test_predictions.pkl")
parser.add_argument('--output_file_season', type=str, default="test_seasonal_predictions.pkl")
parser.add_argument('--mode', type=str, default="RC") 
parser.add_argument('--stats_mode', type=str, default="var") 
parser.add_argument('--procedure', type=str, default='z-score')
parser.add_argument('--seed', type=int, default=80)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--checkpoint_R', type=str, default=None)
parser.add_argument('--orog_file', type=str, default=None)
parser.add_argument('--mask_sealand_file', type=str, default=None)
parser.add_argument('--coords_ij_file', type=str, default=None)
parser.add_argument('--metadata_file', type=str, default=None) 
parser.add_argument('--run_type', type=str)
parser.add_argument('--target_type', type=str, default="precipitation")
parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')

parser.add_argument('--make_plots',  action='store_true')
parser.add_argument('--no-make_plots', dest='make_plots', action='store_false')

def return_test_idxs_from_years_list(years_list, time_index, history_length):
    test_idxs_list = []
    test_idxs_valid_list = []
    test_idxs_list = []
    years = sorted(years_list)
    for year in years:
        test_start_idx, test_end_idx = date_to_idxs_from_timeindex(
            year_start=year, month_start=1, day_start=1,
            year_end=year, month_end=12, day_end=31,
            time_index=time_index
        )
        if test_start_idx - history_length < 0:
            test_start_idx = history_length
        # indices of output
        test_idxs_list.append(np.arange(test_start_idx - history_length, test_end_idx))
        # indices of input
        test_idxs_valid_list.append(np.arange(test_start_idx, test_end_idx))
    test_idxs = np.concatenate(test_idxs_list)
    test_idxs_valid = np.concatenate(test_idxs_valid_list)
    # indices of input but referred to test_idxs_valid
    test_idxs_valid_subset = np.where(np.isin(test_idxs, test_idxs_valid))[0]

    return test_idxs, test_idxs_valid_subset


THRESHOLD = 0.0

HISTORY_LENGTH_MAP = {
    "1h": 24,   # [t-24,...,t]
    "3h": 8,    # [t-24,t-21,...,t]
    "6h": 4,    # [t-24,t-18,t-12,t-6,t]
    "1d": 2,    # [t-2,t-1,t]
}   

HIGH_INDEPENDENT_VARS = True

if __name__ == '__main__':

    args = parser.parse_args()

    
    # Set all seeds
    set_seed_everything(seed=args.seed)

    if not os.path.exists(args.output_path):
       os.makedirs(args.output_path)
        
    if args.use_accelerate is True:
        accelerator = Accelerator()
        print("I am using accelerator")
    else:
        accelerator = None
        print("I am not using accelerator")

    write_log(f"Starting the testing for run: {args.train_path}", args, accelerator, 'w')
    write_log(f"\nCuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.", args, accelerator, 'a')

#--------------------------------------------------------
#--------------------- LOAD FILES -----------------------
#--------------------------------------------------------

    write_log(f"\nLoading files", args, accelerator, 'a')

    #-- 1. Graph
    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    #-- 2. Orography
    orog = np.load(args.input_path+args.orog_file)

    #-- 3. Mask sea-land
    if args.mask_sealand_file != "":
        mask_sealand = np.load(args.input_path+args.mask_sealand_file)
        use_mask_sealand = True
    else:
        use_mask_sealand = False
    
    #-- 4. Coords ij
    if args.coords_ij_file != "":
        coords_ij = np.load(args.input_path+args.coords_ij_file)
        use_coords_ij = True
    else:
        use_coords_ij = False

    #-- 5. Low input metadata
    with open(args.input_path + args.metadata_file, "r") as f:
        metadata = json.load(f)

    predictors_filename = args.input_path_P + args.predictors_filename
    # Load the input dataset
    params = ['q', 't', 'u', 'v', 'z']
    levels = ['850', '700', '500']
    load_dataset = load_dataset_CORDEXML

    # Load the input dataset
    input_ds, lat_low, lon_low, low_time_index, low_native_time_res, low_time_res = load_dataset(
        params=params, levels=levels, file_path=args.input_path_P, file=args.predictors_filename, args=args)

    if lat_low[0] > lat_low[-1]:
        write_log(f"\nFlipping the lat axes to have the origin in the bottom left corner", args, accelerator=None, mode='a')
        lat_low = np.flip(lat_low, axis=0)  # Flip the latitude array along the first axis
        input_ds = np.flip(input_ds, axis=3) # time, var, lev, lat, lon

    if lon_low[0] > lon_low[-1]:
        write_log(f"\nFlipping the lon axes to have the origin in the bottom left corner", args, accelerator=None, mode='a')
        lon_low = np.flip(lon_low, axis=0)  # Flip the latitude array along the first axis
        input_ds = np.flip(input_ds, axis=4) # time, var, lev, lat, lon

    lat_low, lon_low = np.meshgrid(lat_low, lon_low, indexing='ij')

    lat_low = lat_low.flatten()
    lon_low = lon_low.flatten()

    input_ds = np.transpose(input_ds, (3, 4, 0, 1, 2)) #torch.permute(input_ds, (3,4,0,1,2)) # lat, lon, time, vars, levels
    input_ds = input_ds.reshape(-1, *input_ds.shape[2:]) # num_nodes, time, vars, levels

    # conditional (depends on how the graph was preprocessed)
    src = low_high_graph["low", "to", "high"].edge_index[0,:]              # shape (2,num_edges)
    unique_src = np.unique(src)
    num_low = input_ds.shape[0]
    if unique_src.shape[0] != num_low:
        write_log(f"\nLoading unique_src.npy to update the predictors, keeping only the points corresponding to the Low nodes (from {num_low} to {unique_src.shape[0]})", args, accelerator=None, mode='a')
        unique_src = np.load(args.input_path + "unique_src.npy")
        input_ds = input_ds[unique_src]

    n_vars = input_ds.shape[2]
    n_levels = input_ds.shape[3]

    history_length = HISTORY_LENGTH_MAP.get(low_time_res) # lookup for the predictors
    if history_length is None:
        raise ValueError(f"Unknown time resolution: {low_time_res}")
    
    seq_length = history_length + 1 # total length of the sequence for the RNN model

    if args.period == 'historical':
        years_test = list(range(1981, 2001))
    elif args.period == 'mid_century':
        years_test = list(range(2041, 2061))
    elif args.period == 'end_century':
        years_test = list(range(2080, 2100))
                         
    test_idxs, test_idxs_valid_subset = return_test_idxs_from_years_list(years_test, low_time_index, history_length)

    # Statistics computed on training data
    means_low = np.load(args.train_path + "means_low.npy")
    stds_low = np.load(args.train_path + "stds_low.npy")
    means_high = np.load(args.train_path + "means_high.npy")
    stds_high = np.load(args.train_path + "stds_high.npy")
    
    write_log(f"\nLoaded input statistics:", args, accelerator, 'a')
    write_log(f"  means_low shape: {means_low.shape}", args, accelerator, 'a')
    write_log(f"  stds_low shape: {stds_low.shape}", args, accelerator, 'a')

    write_log(f"\ntest_idxs shape: {test_idxs.shape}", args, accelerator, 'a')
    write_log(f"\ntest_idxs: {test_idxs}", args, accelerator, 'a')
    
    #-- Slice time index and target
    time_index_test = low_time_index[test_idxs]
    low_input_test = input_ds[:, test_idxs, :, :] #

    write_log(f"\nTest start idx: {test_idxs_valid_subset[0]} - {time_index_test[test_idxs_valid_subset[0]]}", args, accelerator, 'a')
    write_log(f"\nTest end idx: {test_idxs_valid_subset[0]} - {time_index_test[test_idxs_valid_subset[-1]]}", args, accelerator, 'a')
    write_log(f"\nTotal test days: {len(test_idxs_valid_subset)}", args, accelerator, 'a')
    write_log(f"\n the first ten idx are {test_idxs[0:10]}", args, accelerator, 'a')
    write_log(f"\n val idx type {type(test_idxs)}", args, accelerator, 'a')

    write_log(f"\nStandardizing input data.", args, accelerator, 'a')
    low_input_test_std, high_input_std = standardize_input(
        x_low=low_input_test,
        x_high=orog,
        means_low=means_low,
        stds_low=stds_low,
        means_high=means_high,
        stds_high=stds_high,
        n_vars=n_vars,
        high_independent_vars=HIGH_INDEPENDENT_VARS,
    )
    
    #-- Add the other high-res features
    if use_mask_sealand:
        high_input_std = np.concatenate((high_input_std, mask_sealand), axis=-1)
        write_log(f"\nAdding mask sea-land node features", args, accelerator, 'a')

    if use_coords_ij:
        high_input_std = np.concatenate((high_input_std, coords_ij), axis=-1)
        write_log(f"\nAdding ij node features", args, accelerator, 'a')

    #-- Step 3 - torch tensors from numpy arrays
    test_idxs = torch.from_numpy(test_idxs).int()
    low_input_test_std = torch.from_numpy(low_input_test_std).float()
    high_input_std = torch.from_numpy(high_input_std).float()

    low_input_test_std = torch.flatten(low_input_test_std, start_dim=2, end_dim=-1)   # num_nodes, time, vars*levels

    #-----------------------------------------------------
    #------------------------ MODEL ----------------------
    #-----------------------------------------------------

    n_static_high = high_input_std.shape[1]
    write_log(f"\nn_vars: {n_vars}, n_levels: {n_levels}, n_static_high: {n_static_high}", args, accelerator, 'a')

    #-- Model
    models = importlib.import_module(f"models.{args.model_name}")
    Model = getattr(models, args.model_name)
    model_R = Model(h_in=n_vars*n_levels, h_hid=n_vars*n_levels, high_in=n_static_high, seq_length=seq_length)

    #-----------------------------------------------------
    #------------------ LOAD CHECKPOINT ------------------
    #-----------------------------------------------------

    if accelerator is None:
        checkpoint_R = torch.load(args.train_path + args.checkpoint_R, map_location=torch.device('cpu'), weights_only=True)
        device = 'cpu'
    else:
        checkpoint_R = torch.load(args.train_path + args.checkpoint_R + "/pytorch_model.bin", weights_only=True)
        device = accelerator.device
    
    write_log("\nLoading state dict.", args, accelerator, 'a')
    
    model_R.load_state_dict(checkpoint_R)

    #-----------------------------------------------------
    #-------------- DATASET AND DATALOADER ---------------
    #-----------------------------------------------------
    
    write_log(f"\nDefining dataset and dataloader.", args, accelerator, 'a')
    dataset_graph_tmp = Dataset_Graph(
        low_input=low_input_test_std,
        high_input=high_input_std,
        graph=low_high_graph,
        target=None,
        history_length=history_length,
    )

    dataset_graph = torch.utils.data.Subset(dataset_graph_tmp, test_idxs_valid_subset) # it's just a view of the original dataset

    dataloader = torch.utils.data.DataLoader(
        dataset_graph,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn_graph,
        num_workers=0
    )

    #-----------------------------------------------------
    #---------------- ACCELERATOR PREPARE ----------------
    #-----------------------------------------------------

    if accelerator is not None:
        model_R, dataloader = accelerator.prepare(model_R, dataloader)

    #-----------------------------------------------------
    #----------------------- TEST ------------------------
    #-----------------------------------------------------

    write_log(f"\nStarting the test, from " +
              f"{time_index_test[test_idxs_valid_subset.min()]} to idx {time_index_test[test_idxs_valid_subset.max()]}.", args, accelerator, 'a')

    start = time.time()

    if args.model_name == "GNN4CD_model_mod1_GaussianNLL":
        y_pred, sigma_pred, idxs = NLL_Tester().test_GaussianNLL(
            model_R, dataloader, args=args, accelerator=accelerator
        )
    elif args.model_name == "GNN4CD_model_mod1_BernoulliGammaNLL":
        y_pred, idxs = NLL_Tester().test_BernoulliGammaNLL(
            model_R, dataloader, args=args, accelerator=accelerator
        )
    else:
        y_pred, idxs = Tester().test(
            model_R, dataloader, args=args, accelerator=accelerator
        )
    
    end = time.time()

    write_log(f"\nTest Done! \nNow post-processing results.", args, accelerator, 'a')

    # POST PROCESS PREDICTIONS

    if accelerator is not None:
        accelerator.wait_for_everyone()

        # Gather the values in *tensor* across all processes and concatenate them on the first dimension. Useful to
        # regroup the predictions from all processes when doing evaluation.

        idxs = accelerator.gather(idxs)[: len(dataset_graph)]
        idxs, indices = torch.sort(idxs)
        idxs = idxs.cpu().numpy()
        indices = indices.cpu().numpy()

        y_pred = accelerator.gather(y_pred)[: len(dataset_graph),:] # (times, nodes)
        y_pred = y_pred.swapaxes(0,1).cpu().numpy()[:,indices] # (nodes, times)
        print(f"[Rank {accelerator.process_index}] y_pred.shape (after gather): {y_pred.shape}")
        
        if args.model_name == "GNN4CD_model_mod1_GaussianNLL":
            sigma_pred = accelerator.gather(sigma_pred)[: len(dataset_graph),:]
            sigma_pred = sigma_pred.swapaxes(0,1).cpu().numpy()[:,indices]

    if args.target_type == "precipitation":
        if args.model_name != "GNN4CD_model_mod1_BernoulliGammaNLL":
            y_pred = np.where(np.isfinite(np.expm1(y_pred)), np.expm1(y_pred), np.nan)
        y_pred[y_pred<THRESHOLD] = 0.0
    elif args.target_type == "temperature":
        y_pred = invert_normalization(y_pred, stats_path=args.train_path)

    # LON LAT
    lat_low = low_high_graph["low"].lat.cpu().numpy()
    lon_low = low_high_graph["low"].lon.cpu().numpy()
    lat_high = low_high_graph["high"].lat.cpu().numpy()
    lon_high = low_high_graph["high"].lon.cpu().numpy()

    # degree = degree(low_high_graph["high", "within", "high"].edge_index[0], low_high_graph["high"].num_nodes).cpu().numpy()

    # mask = degree > 2

    # np.save(args.output_path + "mask_degree.npy", mask)

    # if mask is not None:
    #     degree[~mask] = np.nan
    #     if y_pred is not None:
    #         y_pred[~mask,:] = np.nan

    # CREATE THE DATA OBJECT
    data = HeteroData()

    if args.target_type == "precipitation":
        data.pr_gnn4cd = y_pred
    elif args.target_type == "temperature":
        data.tasmax_gnn4cd = y_pred    

    data.times = time_index_test[idxs]
    data["low"].lat = lat_low
    data["low"].lon = lon_low
    data["high"].lat = lat_high
    data["high"].lon = lon_high

    if "NLL" in args.model_name:
        data.sigma_gnn4cd = sigma_pred

    write_log(f"\nDone. Testing concluded in {end-start} seconds.\nWrite the files.", args, accelerator, 'a')

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(data, f)

    