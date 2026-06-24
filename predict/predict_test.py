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
import xarray as xr

import data.loaders  # noqa: F401 - register CORDEXML test loader
from args.predict.add_base_args_test import add_base_args_test
from args.predict.add_target_specific_args import add_target_specific_args
from args.shared.add_model_specific_args import add_model_specific_args
from models.build_model import build_model
from data.datasets.graph_dataset import Graph_Dataset, custom_collate_fn_graph
from data.loaders.registry import get_dataset_loader
from utils.predictions.predictor import Predictor
from utils.helpers.tools import set_seed_everything, write_log, date_to_idxs_from_timeindex
from utils.extractors.extract_prediction import extract_prediction
from utils.predictand_transforms.inverse_transform_predictand import inverse_transform_predictand
from utils.predictor_transforms.transform_predictors import transform_predictors
from utils.losses.registry import get_loss

def return_test_idxs(predictor: xr.Dataset, period: str):
    """Split test data by period using the NetCDF predictor time axis.

    This follows the original test workflow: the selected indices are taken
    directly from the predictor file, so they stay aligned with the raw test
    NetCDF calendar.
    """
    if period == 'historical':
        years_test = list(range(1981, 2001))
    elif period == 'mid_century':
        years_test = list(range(2041, 2061))
    else:
        years_test = list(range(2080, 2100))

    test_idxs = np.argwhere(np.isin(predictor['time'].dt.year, years_test))
    return test_idxs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_base_args_test(parser)

    args, unknown = parser.parse_known_args()

    # Update args with target-specific arguments
    parser = add_target_specific_args(parser, args.target_type)
    args = parser.parse_args()
    
    # Set all seeds
    set_seed_everything(seed=args.seed)

    if not os.path.exists(args.output_path):
       os.makedirs(args.output_path)
        
    if args.use_accelerate:
        accelerator = Accelerator()
        print("I am using accelerator")
    else:
        accelerator = None
        print("I am not using accelerator")

    write_log(f"Starting the testing for epoch {args.epoch}...", args, accelerator, 'w')
    write_log(f"\nUsing {args.checkpoint}, training path: {args.train_path}", args, accelerator, 'w')
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

    #--6 Original netcdf predictors
    predictors_filename = args.input_path_P + args.predictors_filename
    write_log(f"\nPredictor file: {predictors_filename}", args, accelerator, 'a')

    # Load the input dataset
    load_dataset = get_dataset_loader(args.dataset_name)
    params = metadata.get("variables", ['q', 't', 'u', 'v', 'z'])
    levels = metadata.get("levels", [850, 700, 500])
    levels = [str(level) for level in levels]
    write_log(f"\nUsing predictor channel order from metadata: params={params}, levels={levels}", args, accelerator, 'a')
    x_low, lat_low, lon_low, time_index, _, _ = load_dataset(
        params=params,
        levels=levels,
        file_path=args.input_path_P,
        file=args.predictors_filename,
        args=args
        )

    if lat_low[0] > lat_low[-1]:
        write_log(f"\nFlipping the lat axes to have the origin in the bottom left corner", args, accelerator=None, mode='a')
        x_low = np.flip(x_low, axis=3) # time, var, lev, lat, lon

    if lon_low[0] > lon_low[-1]:
        write_log(f"\nFlipping the lon axes to have the origin in the bottom left corner", args, accelerator=None, mode='a')
        x_low = np.flip(x_low, axis=4) # time, var, lev, lat, lon

    x_low = np.transpose(x_low, (3, 4, 0, 1, 2)) #torch.permute(x_low, (3,4,0,1,2)) # lat, lon, time, vars, levels
    x_low = x_low.reshape(-1, *x_low.shape[2:]) # num_nodes, time, vars, levels
    write_log(
        f"\nLoaded test predictors: shape={x_low.shape}, "
        f"min={np.nanmin(x_low):.6g}, max={np.nanmax(x_low):.6g}",
        args, accelerator, 'a'
    )

    # conditional (depends on how the graph was preprocessed)
    src = low_high_graph["low", "to", "high"].edge_index[0,:]              # shape (2,num_edges)
    unique_src = np.unique(src)
    num_low = x_low.shape[0]
    if unique_src.shape[0] != num_low:
        write_log(f"\nLoading unique_src.npy to update the predictors, keeping only the points corresponding to the Low nodes (from {num_low} to {unique_src.shape[0]})", args, accelerator=None, mode='a')
        unique_src = np.load(args.input_path + "unique_src.npy")
        x_low = x_low[unique_src]
        write_log(
            f"\nAfter unique_src filtering: shape={x_low.shape}, "
            f"min={np.nanmin(x_low):.6g}, max={np.nanmax(x_low):.6g}",
            args, accelerator=None, mode='a'
        )

    n_vars = x_low.shape[2]
    n_levels = x_low.shape[3]

    #-----------------------------------------------------
    #---------------------- INDICES  ---------------------
    #-----------------------------------------------------

    # Original test indexing: select the requested years from the raw predictor
    # NetCDF time axis, then drop the first history_length outputs.
    predictor = xr.open_dataset(predictors_filename, engine="netcdf4")
    test_idxs = return_test_idxs(predictor, args.period).squeeze()
    test_idxs_valid_subset = test_idxs[args.history_length:]
    predictor.close()

    #-- Slice time index and target
    time_index_test = time_index[test_idxs]
    x_low_test = x_low[:, test_idxs, :, :] # num_nodes, time, vars, levels

    write_log(f"\nTest start idx: {test_idxs_valid_subset[0]} - {time_index_test[test_idxs_valid_subset[0]]}", args, accelerator, 'a')
    write_log(f"\nTest end idx: {test_idxs_valid_subset[0]} - {time_index_test[test_idxs_valid_subset[-1]]}", args, accelerator, 'a')
    write_log(f"\nTotal test days: {len(test_idxs_valid_subset)}", args, accelerator, 'a')
    write_log(f"\n the first ten idx are {test_idxs[0:10]}", args, accelerator, 'a')
    write_log(f"\n val idx type {type(test_idxs)}", args, accelerator, 'a')

    write_log(f"\ntest_idxs shape: {test_idxs.shape}", args, accelerator, 'a')
    write_log(f"\ntest_idxs: {test_idxs}", args, accelerator, 'a')

    #-----------------------------------------
    #---------  TRANSFORM PREDICTORS ---------
    #-----------------------------------------

# 1. Transform predictors
    write_log(f"\nTransforming predictor data.", args, accelerator, 'a')

    predictors_stats = np.load(args.train_path + "predictors_stats.npz", allow_pickle=True)
    x_low_test_std, x_high_std = transform_predictors(
        x_low=x_low_test,
        x_high=orog,
        train_idxs=None,
        mode_low=None,     # inferred from predictors_stats
        mode_high=None,    # inferred from predictors_stats
        stats=predictors_stats,
        stats_save_path=None
    )

    n_vars = x_low_test_std.shape[2]
    n_levels = x_low_test_std.shape[3]

    # 1.2 Flatten x_low_test_std
    N, T = x_low_test_std.shape[:2] 
    x_low_test_std = x_low_test_std.reshape(N, T, -1) # num_nodes, time, vars*levels
    
    # 1.3 Add high_res predictors which are not transformed
    if use_mask_sealand:
        x_high_std = np.concatenate((x_high_std, mask_sealand), axis=-1)
        write_log(f"\nAdding mask sea-land node features", args, accelerator, 'a')

    if use_coords_ij:
        x_high_std = np.concatenate((x_high_std, coords_ij), axis=-1)
        write_log(f"\nAdding ij node features", args, accelerator, 'a')

    n_static_high = x_high_std.shape[1]

    write_log(f"\nn_vars: {n_vars}, n_levels: {n_levels}, n_static_high: {n_static_high}", args, accelerator, 'a')
    
    #-----------------------------------------------------
    #-------------- FROM NUMPY TO PYTORCH ----------------
    #-----------------------------------------------------

    # Predictors
    x_low_test_std = torch.from_numpy(x_low_test_std).float()
    x_high_std = torch.from_numpy(x_high_std).float()

    #--------------------------------------------
    #-------------- BUILD MODEL -----------------
    #--------------------------------------------

    # Update args with loss- and model-specific arguments
    parser = add_model_specific_args(parser, args.model_name)
    args = parser.parse_args()

    LossClass = get_loss(args.loss_name)
    output_dim = LossClass.output_dim
    
    model = build_model(
        x_low_var_dim=n_vars,
        x_low_lev_dim=n_levels,
        x_high_dim=n_static_high,
        output_dim=output_dim,
        args=args
    )

    #-----------------------------------------------------
    #------------------ LOAD CHECKPOINT ------------------
    #-----------------------------------------------------

    if accelerator is None:
        checkpoint = torch.load(args.train_path + args.checkpoint + "/pytorch_model.bin", map_location=torch.device('cpu'), weights_only=True)
        device = 'cpu'
    else:
        checkpoint = torch.load(args.train_path + args.checkpoint + "/pytorch_model.bin", weights_only=True)
        device = accelerator.device
    
    write_log("\nLoading state dict.", args, accelerator, 'a')
    model.load_state_dict(checkpoint)

    #-----------------------------------------------------
    #-------------- DATASET AND DATALOADER ---------------
    #-----------------------------------------------------
    
    write_log(f"\nDefining dataset and dataloader.", args, accelerator, 'a')
    graph_dataset_tmp = Graph_Dataset(
        graph=low_high_graph,
        low_input=x_low_test_std,
        high_input=x_high_std,
        target=None,
        history_length=args.history_length,
    )

    graph_dataset = torch.utils.data.Subset(graph_dataset_tmp, test_idxs_valid_subset) # it's just a view of the original dataset

    dataloader = torch.utils.data.DataLoader(
        graph_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn_graph,
        num_workers=0
    )

    #-----------------------------------------------------
    #---------------- ACCELERATOR PREPARE ----------------
    #-----------------------------------------------------

    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    #-----------------------------------------------------
    #------------------- PREDICTIONS ---------------------
    #-----------------------------------------------------

    write_log(f"\nStarting the test, from " +
              f"{time_index_test[test_idxs_valid_subset.min()]} to idx {time_index_test[test_idxs_valid_subset.max()]}.", args, accelerator, 'a')

    start = time.time()
    predictor = Predictor()
    y_pred_raw, idxs_sorted = predictor.predict(model, dataloader, pred_size=len(graph_dataset), args=args, accelerator=accelerator)
    end = time.time()

    write_log(f"\nTest Done! \nNow post-processing results.", args, accelerator, 'a')

    #-----------------------------------------------------
    #------------------ POST-PROCESSING ------------------
    #-----------------------------------------------------

    # from raw model prediction to actual pr/tasmax values
    predictand_stats = np.load(args.train_path + "predictand_stats.npz", allow_pickle=True)
    y_pred = inverse_transform_predictand(y_pred_raw, predictand_stats)

    if args.target_type == "precipitation":
        y_pred[y_pred < args.threshold] = 0.0
    write_log(
        f"\nPrediction diagnostics: raw_min={np.nanmin(y_pred_raw):.6g}, "
        f"raw_max={np.nanmax(y_pred_raw):.6g}, "
        f"phys_min={np.nanmin(y_pred):.6g}, phys_max={np.nanmax(y_pred):.6g}",
        args, accelerator, 'a'
    )

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

    #-----------------------------------------------------
    #-------------------- SAVE RESULTS -------------------
    #-----------------------------------------------------

    data = HeteroData()

    if args.target_type == "precipitation":
        data.pr_gnn4cd = y_pred
    elif args.target_type == "temperature":
        data.tasmax_gnn4cd = y_pred    

    # TEST FIX (2026-06): predictions are produced only for valid output
    # indices after the history window. Keep the same alignment used by
    # predict.py; otherwise daily reports can pair a prediction with the
    # wrong calendar date.
    data.times = time_index_test[test_idxs_valid_subset][idxs_sorted]
    data["low"].lat = lat_low
    data["low"].lon = lon_low
    data["high"].lat = lat_high
    data["high"].lon = lon_high

    write_log(f"\nDone. Testing concluded in {end-start} seconds.\nWrite the files.", args, accelerator, 'a')

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(data, f)
