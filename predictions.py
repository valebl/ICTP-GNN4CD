import numpy as np
import pickle
import torch
import torch._dynamo
import argparse
import time
import os
import json
import importlib
import safetensors
from accelerate import Accelerator

torch._dynamo.config.disable = True  
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree
from dataset import Dataset_Graph, custom_collate_fn_graph

from utils.helpers.tools import date_to_idxs_from_timeindex, set_seed_everything
from utils.helpers.tools import write_log, standardize_input, invert_normalization
from utils.testing.test_NLL import NLL_Tester
from utils.testing.test import Tester

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--epoch', type=int)
parser.add_argument('--train_path_R', type=str)
parser.add_argument('--train_path_C', type=str)
parser.add_argument('--checkpoint_R', type=str, default=None)
parser.add_argument('--checkpoint_C', type=str, default=None)
parser.add_argument('--output_file', type=str, default="G_predictions.pkl")

parser.add_argument('--graph_file', type=str, default=None)
parser.add_argument('--low_input_file', type=str, default=None)
parser.add_argument('--target_file', type=str, default=None)
parser.add_argument('--orog_file', type=str, default=None)
parser.add_argument('--mask_sealand_file', type=str, default=None)
parser.add_argument('--coords_ij_file', type=str, default=None)
parser.add_argument('--model_type', type=str, default=None)
parser.add_argument('--model_name', type=str, default=None) 
parser.add_argument('--dataset_name', type=str, default=None) 
parser.add_argument('--test_idxs_file', type=str, default="")
parser.add_argument('--target_type', type=str, default="precipitation")
parser.add_argument('--run_type', type=str)

parser.add_argument('--metadata_file', type=str, help='metadata file')

#-- start and end training dates
parser.add_argument('--test_year_start', type=int)
parser.add_argument('--test_month_start', type=int)
parser.add_argument('--test_day_start', type=int)
parser.add_argument('--test_year_end', type=int)
parser.add_argument('--test_month_end', type=int)
parser.add_argument('--test_day_end', type=int)
parser.add_argument("--test_years", type=str, default="")

parser.add_argument('--batch_size', type=int)
parser.add_argument('--seed', type=int)

parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')

parser.add_argument('--make_plots',  action='store_true')
parser.add_argument('--no-make_plots', dest='make_plots', action='store_false')


### PARAMETERS THAT ARE NOW SET MANUALLY
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
    else:
        accelerator = None

    write_log(f"Starting the testing for epoch {args.epoch}...", args, accelerator, 'w')
    write_log(f"\nUsing {args.checkpoint_R}, training path: {args.train_path_R}", args, accelerator, 'w')
    write_log(f"\nCuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.", args, accelerator, 'a')

#--------------------------------------------------------
#--------------------- LOAD FILES -----------------------
#--------------------------------------------------------

    write_log(f"\nLoading files", args, accelerator, 'a')

    #-- 1. Graph
    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    #-- 2. Low res input
    low_input = np.load(args.input_path+args.low_input_file)

    #-- 3. Target
    target = np.load(args.input_path+args.target_file)

    #-- 4. Orography
    orog = np.load(args.input_path+args.orog_file)

    #-- 5. Mask sea-land
    if args.mask_sealand_file != "":
        mask_sealand = np.load(args.input_path+args.mask_sealand_file)
        use_mask_sealand = True
    else:
        use_mask_sealand = False
    
    #-- 6. Coords ij
    if args.coords_ij_file != "":
        coords_ij = np.load(args.input_path+args.coords_ij_file)
        use_coords_ij = True
    else:
        use_coords_ij = False

    #-- 7. Time index
    time_index = np.load(args.input_path+"time_index.npy", allow_pickle=True)

    #-- 8. Low input metadata
    with open(args.input_path + args.metadata_file, "r") as f:
        metadata = json.load(f)

#--------------------------------------------------------
#----------------- DERIVED QUANTITIES -------------------
#--------------------------------------------------------

    # High resolution lon and lat
    lon_high = low_high_graph["high"].lon
    lat_high = low_high_graph["high"].lat

    # Time resolution
    time_res = metadata.get("time_res", None)

    # Determine history length and seq_length based on time resolution
    history_length = HISTORY_LENGTH_MAP.get(time_res) # lookup for the predictors
    if history_length is None:
        raise ValueError(f"Unknown time resolution: {time_res}")
    
    seq_length = history_length + 1 # total length of the sequence for the RNN model
    
    n_vars = low_input.shape[2]
    n_levels = low_input.shape[3]

#-----------------------------------------------------
#---------------------- INDICES  ---------------------
#-----------------------------------------------------

    #-- Test indices
    if args.test_years == "":
        write_log(f"\nStart year-month-day and end year-month-day have been provided.", args, accelerator, 'a')
        # Input and predictions
        test_start_idx, test_end_idx = date_to_idxs_from_timeindex(
            year_start=args.test_year_start, month_start=args.test_month_start, day_start=args.test_day_start,
            year_end=args.test_year_end, month_end=args.test_month_end, day_end=args.test_day_end,
            time_index=time_index)
        
        if test_start_idx - history_length >= 0:
            test_idxs = np.arange(test_start_idx - history_length, test_end_idx)
            test_idxs_valid = np.arange(test_start_idx, test_end_idx)
            test_idxs_valid_subset = np.where(np.isin(test_idxs, test_idxs_valid))[0]
        else:
            raise ValueError(f"Invalid start date: {args.test_year_start}/{args.test_month_start}/{args.test_day_start}")
    else:
        write_log(f"\nA list of test years have been provided.", args, accelerator, 'a')
        test_idxs_list = []
        test_idxs_valid_list = []
        test_idxs_valid_subset_list = []
        test_idxs_list = []
        # Convert "year1_year2_year3_year4" → [year1, year2, year3, year4]
        years = sorted([int(y) for y in args.test_years.split("_")])
        for year in years:
            test_start_idx, test_end_idx = date_to_idxs_from_timeindex(
                year_start=year, month_start=1, day_start=1,
                year_end=year, month_end=12, day_end=31,
                time_index=time_index
            )
            if test_start_idx - history_length < 0:
                test_start_idx = history_length
                write_log(f"\nFor year {year} test starts from {time_index[test_start_idx]}", args, accelerator, 'a')
            # indices of output
            test_idxs_list.append(np.arange(test_start_idx - history_length, test_end_idx))
            # indices of input
            test_idxs_valid_list.append(np.arange(test_start_idx, test_end_idx))
        
        test_idxs = np.concatenate(test_idxs_list)
        test_idxs_valid = np.concatenate(test_idxs_valid_list)

        # indices of input but referred to test_idxs_valid
        test_idxs_valid_subset = np.where(np.isin(test_idxs, test_idxs_valid))[0]

    if accelerator.is_main_process:
        print(f"Output (start_idx, end_idx): {test_start_idx, test_end_idx}" +
        f" corresponding to {time_index[test_start_idx], time_index[test_end_idx-1]}")

    # Statistics computed on training data
    means_low = np.load(args.train_path_R + "means_low.npy")
    stds_low = np.load(args.train_path_R + "stds_low.npy")
    means_high = np.load(args.train_path_R + "means_high.npy")
    stds_high = np.load(args.train_path_R + "stds_high.npy")

    #-- Slice time index and target
    time_index_test = time_index[test_idxs]
    low_input_test = low_input[:, test_idxs, :, :] # num_nodes, time, vars, levels
    target_test = target[:, test_idxs][:, test_idxs_valid_subset]

#--------------------------------------------------------
#--------------------  PREPROCESSING --------------------
#--------------------------------------------------------

    #-- Stesp 1. Standardize input data
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
    
    #-- Step 2. Add the other high-res features
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
    target_test = torch.from_numpy(target_test).float()

    low_input_test_std = torch.flatten(low_input_test_std, start_dim=2, end_dim=-1)   # num_nodes, time, vars*levels

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
    #------------------------ MODEL ----------------------
    #-----------------------------------------------------

    n_static_high = high_input_std.shape[1]
    write_log(f"\nn_vars: {n_vars}, n_levels: {n_levels}, n_static_high: {n_static_high}", args, accelerator, 'a')

    # Model
    models = importlib.import_module(f"models.{args.model_name}")
    Model = getattr(models, args.model_name)
    model = Model(h_in=n_vars*n_levels, h_hid=n_vars*n_levels, high_in=n_static_high, seq_length=seq_length)

    
    #-----------------------------------------------------
    #------------------ LOAD CHECKPOINT ------------------
    #-----------------------------------------------------

    if accelerator is None:
        checkpoint = torch.load(args.train_path_R + args.checkpoint_R, map_location=torch.device('cpu'), weights_only=True)
        device = 'cpu'
    else:
        try:
            checkpoint = torch.load(args.train_path_R + args.checkpoint_R + "/pytorch_model.bin", weights_only=True)
        except:
            checkpoint = safetensors.torch.load_file(args.train_path_R + args.checkpoint_R + "/model.safetensors")
            torch.save(checkpoint, args.train_path_R + args.checkpoint_R + "pytorch_model.bin")
        device = accelerator.device
    
    write_log("\nLoading state dict.", args, accelerator, 'a')
    model.load_state_dict(checkpoint)

    #-----------------------------------------------------
    #---------------- ACCELERATOR PREPARE ----------------
    #-----------------------------------------------------

    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    #-----------------------------------------------------
    #----------------------- TEST ------------------------
    #-----------------------------------------------------

    write_log(f"\nStarting the test, from " +
              f"{time_index_test[test_idxs_valid_subset.min()]} to idx {time_index_test[test_idxs_valid_subset.max()]}.", args, accelerator, 'a')

    start = time.time()

    if args.model_name == "GNN4CD_model_mod1_GaussianNLL":
        y_pred, sigma_pred, idxs = NLL_Tester().test_GaussianNLL(
            model, dataloader, args=args, accelerator=accelerator
        )
    elif args.model_name == "GNN4CD_model_mod1_BernoulliGammaNLL":
        y_pred, idxs = NLL_Tester().test_BernoulliGammaNLL(
            model, dataloader, args=args, accelerator=accelerator
        )
    else:
        y_pred, idxs = Tester().test(
            model, dataloader, args=args, accelerator=accelerator
        )
    
    end = time.time()

    write_log(f"\nTest Done! \nNow post-processing results.", args, accelerator, 'a')

    #-----------------------------------------------------
    #------------------ POST-PROCESSING ------------------
    #-----------------------------------------------------

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
        y_pred = invert_normalization(y_pred, stats_path=args.train_path_R)

    # LON LAT
    lat_low = low_high_graph["low"].lat.cpu().numpy()
    lon_low = low_high_graph["low"].lon.cpu().numpy()
    lat_high = low_high_graph["high"].lat.cpu().numpy()
    lon_high = low_high_graph["high"].lon.cpu().numpy()

    degree = degree(low_high_graph["high", "within", "high"].edge_index[0], low_high_graph["high"].num_nodes).cpu().numpy()

    # If target and predictions have the same spatial shape, create the mask
    if target.shape[0] == y_pred.shape[0]:
        mask =  degree > 2 * np.array([~np.isnan(target[i,:]).all() for i in range(target.shape[0])])
    else:
        mask = degree > 2

    np.save(args.output_path + "mask_degree.npy", mask)

    if mask is not None:
        target_test[~mask,:] = np.nan # space, time
        degree[~mask] = np.nan
        if y_pred is not None:
            y_pred[~mask,:] = np.nan

    #-----------------------------------------------------
    #-------------------- SAVE RESULTS -------------------
    #-----------------------------------------------------

    data = HeteroData()

    if args.target_type == "precipitation":
        data.pr_gnn4cd = y_pred
    elif args.target_type == "temperature":
        data.tasmax_gnn4cd = y_pred
    
    data.target = target_test

    data.times = time_index_test[idxs]
    data.times_target = time_index_test
    data["low"].lat = lat_low
    data["low"].lon = lon_low
    data["high"].lat = lat_high
    data["high"].lon = lon_high

    if args.model_name == "GNN4CD_model_mod1_GaussianNLL":
        data.sigma_gnn4cd = sigma_pred

    write_log(f"\nDone. Testing concluded in {end-start} seconds.\nWrite the files.", args, accelerator, 'a')

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(data, f)
