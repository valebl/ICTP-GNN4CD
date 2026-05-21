import numpy as np
import pickle
import torch
import time
import os
import json
import importlib
import safetensors
from accelerate import Accelerator

from torch_geometric.data import HeteroData
from torch_geometric.utils import degree
from data.datasets.graph_dataset import Graph_Dataset, custom_collate_fn_graph

from args.predict.build_args import build_args

from models.build_model import build_model

from utils.helpers.tools import (
    set_seed_everything,
    write_log,
    date_to_idxs_from_timeindex,
    compute_predictions_idxs
)

from utils.predictions.predictor import Predictor
from utils.predictions.detect_predictions_idxs_config import detect_predictions_idxs_config
from utils.extractors.extract_prediction import extract_prediction
from utils.predictand_transforms.inverse_transform_predictand import inverse_transform_predictand
from utils.predictor_transforms.transform_predictors import transform_predictors
from utils.losses.registry import get_loss


def probability_matched_mean(samples):
    """Probability-matched mean over samples.

    samples: (nodes, time, samples)
    returns: (nodes, time)
    """
    mean_field = np.nanmean(samples, axis=-1)
    pmm = np.full_like(mean_field, np.nan)
    n_nodes, n_times = mean_field.shape

    for t in range(n_times):
        mean_t = mean_field[:, t]
        pool_t = samples[:, t, :].reshape(-1)
        valid_mean = np.isfinite(mean_t)
        pool_t = pool_t[np.isfinite(pool_t)]
        if valid_mean.sum() == 0 or pool_t.size == 0:
            continue

        order = np.argsort(mean_t[valid_mean])
        quantile_idx = np.linspace(0, pool_t.size - 1, int(valid_mean.sum()))
        matched_values = np.sort(pool_t)[np.rint(quantile_idx).astype(int)]

        valid_indices = np.where(valid_mean)[0]
        pmm[valid_indices[order], t] = matched_values

    return pmm


def select_best_member(samples, args):
    """Select one coherent ensemble member per time step.

    The score favors fields close to the ensemble median while softly
    penalizing members whose tail is too weak or, optionally, too strong.
    """
    if args.best_member_transform == "log1p":
        x = np.log1p(np.maximum(samples, 0.0)).astype(np.float32, copy=False)
    else:
        x = samples.astype(np.float32, copy=False)

    reference_field = np.nanquantile(x, args.best_member_reference_quantile, axis=-1)
    diff = x - reference_field[:, :, None]
    closeness = np.sqrt(np.nanmean(diff * diff, axis=0))  # (time, sample)

    tail = np.nanquantile(x, args.best_member_tail_quantile, axis=0)  # (time, sample)
    tail_target = np.nanquantile(tail, args.best_member_tail_target, axis=1)
    tail_deficit = np.maximum(tail_target[:, None] - tail, 0.0)
    tail_excess = np.maximum(tail - tail_target[:, None], 0.0)

    closeness_scale = np.nanmedian(closeness, axis=1, keepdims=True)
    closeness_scale = np.where(np.isfinite(closeness_scale) & (closeness_scale > 0), closeness_scale, 1.0)
    tail_scale = np.maximum(np.abs(tail_target[:, None]), 1e-6)

    score = (
        closeness / closeness_scale
        + args.best_member_tail_weight * tail_deficit / tail_scale
        + args.best_member_tail_excess_weight * tail_excess / tail_scale
    )
    score = np.where(np.isfinite(score), score, np.inf)
    member_idx = np.argmin(score, axis=1)

    pred = samples[:, np.arange(samples.shape[1]), member_idx]
    return pred.astype(np.float32), member_idx.astype(np.int64), score.astype(np.float32)


if __name__ == '__main__':

    args = build_args()
    
    # Set all seeds
    set_seed_everything(seed=args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    if args.use_accelerate is True:
        accelerator = Accelerator()
    else:
        accelerator = None

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

    #-- 2. Low res input
    x_low = np.load(args.input_path+args.low_input_file)

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
    time_index = np.load(args.input_path+"time_index.npy")

    #-- 8. Low input metadata
    with open(args.input_path + args.metadata_file, "r") as f:
        metadata = json.load(f)

#-----------------------------------------------------
#---------------------- INDICES  ---------------------
#-----------------------------------------------------

    cfg = detect_predictions_idxs_config(args)

    test_idxs, test_idxs_valid, test_idxs_valid_subset = compute_predictions_idxs(
        cfg,
        time_index,
        args.history_length,
        date_to_idxs_from_timeindex
    )

    if accelerator.is_main_process:
        print(f"Output (start_idx, end_idx): {test_idxs_valid[0], test_idxs_valid[-1]}" +
        f" corresponding to {time_index[test_idxs_valid[0]], time_index[test_idxs_valid[-1]]}")

    #-- Slice time index and target
    time_index_test = time_index[test_idxs]
    x_low_test = x_low[:, test_idxs, :, :] # num_nodes, time, vars, levels
    target_test = target[:, test_idxs][:, test_idxs_valid_subset]

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

    # Predictand
    target_test = torch.from_numpy(target_test).float()

    # test_idxs = torch.from_numpy(test_idxs).int()

    #--------------------------------------------
    #-------------- BUILD MODEL -----------------
    #--------------------------------------------

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
        checkpoint = torch.load(args.train_path + args.checkpoint, map_location=torch.device('cpu'), weights_only=True)
        device = 'cpu'
    else:
        try:
            checkpoint = torch.load(args.train_path + args.checkpoint + "/pytorch_model.bin", weights_only=True)
        except:
            checkpoint = safetensors.torch.load_file(args.train_path + args.checkpoint + "/model.safetensors")
            torch.save(checkpoint, args.train_path + args.checkpoint + "pytorch_model.bin")
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

    write_log(f"\nStarting the predictions, from " +
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
    y_pred_samples = None
    y_pred_sample0 = None
    y_pred_q90 = None
    y_pred_pmm = None
    y_pred_best_member = None
    best_member_idx = None
    best_member_score = None

    if y_pred.ndim == 3:
        write_log(f"\nComputing sample aggregations from shape {y_pred.shape}... ", args, accelerator, 'a')
        y_pred_samples = y_pred
        y_pred_sample0 = y_pred_samples[:, :, 0]
        y_pred_q90 = np.nanquantile(y_pred_samples, 0.9, axis=-1)
        y_pred_pmm = probability_matched_mean(y_pred_samples)
        y_pred_best_member, best_member_idx, best_member_score = select_best_member(y_pred_samples, args)
        y_pred = np.nanmean(y_pred_samples, axis=-1)
        write_log(f"to {y_pred.shape}.", args, accelerator, 'a')

    if args.target_type == "precipitation":
        y_pred[y_pred < args.threshold] = 0.0
        for arr in (y_pred_samples, y_pred_sample0, y_pred_q90, y_pred_pmm, y_pred_best_member):
            if arr is not None:
                arr[arr < args.threshold] = 0.0

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
        for arr in (y_pred_samples, y_pred_sample0, y_pred_q90, y_pred_pmm, y_pred_best_member):
            if arr is not None:
                arr[~mask, ...] = np.nan

    #-----------------------------------------------------
    #-------------------- SAVE RESULTS -------------------
    #-----------------------------------------------------

    data = HeteroData()

    if args.target_type == "precipitation":
        data.pr_gnn4cd = y_pred
        if y_pred_samples is not None:
            data.pr_gnn4cd_samples = y_pred_samples
            data.pr_gnn4cd_sample0 = y_pred_sample0
            data.pr_gnn4cd_q90 = y_pred_q90
            data.pr_gnn4cd_pmm = y_pred_pmm
            data.pr_gnn4cd_best_member = y_pred_best_member
    elif args.target_type == "temperature":
        data.tasmax_gnn4cd = y_pred
        if y_pred_samples is not None:
            data.tasmax_gnn4cd_samples = y_pred_samples
            data.tasmax_gnn4cd_sample0 = y_pred_sample0
            data.tasmax_gnn4cd_q90 = y_pred_q90
            data.tasmax_gnn4cd_pmm = y_pred_pmm
            data.tasmax_gnn4cd_best_member = y_pred_best_member
    
    data.target = target_test
    data.best_member_idx = best_member_idx
    data.best_member_score = best_member_score
    data.best_member_transform = args.best_member_transform
    data.best_member_reference_quantile = args.best_member_reference_quantile
    data.best_member_tail_quantile = args.best_member_tail_quantile
    data.best_member_tail_target = args.best_member_tail_target
    data.best_member_tail_weight = args.best_member_tail_weight
    data.best_member_tail_excess_weight = args.best_member_tail_excess_weight

    # Predictions are sorted by graph.idxs inside Predictor. These correspond to
    # test_idxs_valid_subset, not to positions starting from zero. Using
    # time_index_test[idxs_sorted] shifts daily plots by history_length days.
    data.times = time_index_test[test_idxs_valid_subset]
    data.times_target = time_index_test[test_idxs_valid_subset]
    data["low"].lat = lat_low
    data["low"].lon = lon_low
    data["high"].lat = lat_high
    data["high"].lon = lon_high

    write_log(f"\nDone. Testing concluded in {end-start} seconds.\nWrite the files.", args, accelerator, 'a')

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(data, f)
