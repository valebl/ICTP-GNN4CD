import torch
import numpy as np
import pickle
from dataset import Dataset_Graph, custom_collate_fn_graph
import time
import argparse
import os
import importlib
import json
import random

from utils.loss_functions.mse_qmse_psd import MSE_QMSE_PSD_Loss
from utils.loss_functions.gaussian_nll import GaussianNLLLoss
from utils.loss_functions.crps import CRPSLoss, CRPSSpectralLoss
from utils.helpers.tools import write_log, check_freezed_layers, set_seed_everything
from utils.helpers.tools import find_not_all_nan_times, derive_train_val_idxs, derive_train_val_idxs_years_list
from utils.helpers.tools import new_compute_input_statistics_and_standardize, derive_qmse_bins
from utils.helpers.tools import prepare_target_for_train
from utils.training.train_nll import NLL_Trainer
from utils.training.train_mse_qmse_psd import MSE_QMSE_PSD_Trainer
from utils.training.train_crps import CRPS_Trainer
from accelerate import Accelerator

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--low_input_file', type=str, default=None)
parser.add_argument('--orog_file', type=str, default=None)
parser.add_argument('--mask_sealand_file', type=str, default=None)
parser.add_argument('--target_file', type=str, default=None)
parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--coords_ij_file', type=str, default=None)

parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')

parser.add_argument('--metadata_file', type=str, help='metadata file')

#-- training hyperparameters
parser.add_argument('--experiment', type=str, default='ESD_pseudo_reality')
parser.add_argument('--domain', type=str, default='ALPS')
parser.add_argument('--epochs', type=int, default=15, help='number of total training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (global)')
parser.add_argument('--step_size', type=int, default=25, help='scheduler step size (global)')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (wd)')
parser.add_argument('--fine_tuning',  action='store_true')
parser.add_argument('--no-fine_tuning', dest='fine_tuning', action='store_false')
parser.add_argument('--lr_scheduler', type=str, default="StepLR")

parser.add_argument('--checkpoint_ctd', type=str, help='checkpoint to load to continue')
parser.add_argument('--ctd_training',  action='store_true')
parser.add_argument('--no-ctd_training', dest='ctd_training', action='store_false')
parser.add_argument('--make_val_plots', action='store_true')
parser.add_argument('--no-make_val_plots', dest='make_val_plots', action='store_false')

parser.add_argument('--loss_fn', type=str)
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--seed', type=int)
parser.add_argument('--n_gpu', type=int, default=4)

parser.add_argument('--model_type', type=str)
parser.add_argument('--model_name', type=str, default='HiResPrecipNet')
parser.add_argument('--dataset_name', type=str, default='Dataset_Graph')
parser.add_argument('--collate_name', type=str)

parser.add_argument('--stats_mode', type=str, default="var")
parser.add_argument('--target_type', type=str)
parser.add_argument('--run_type', type=str)

#-- start and end training dates
parser.add_argument('--train_year_start', type=int, default=None)
parser.add_argument('--train_month_start', type=int, default=None)
parser.add_argument('--train_day_start', type=int, default=None)
parser.add_argument('--train_year_end', type=int, default=None)
parser.add_argument('--train_month_end', type=int, default=None)
parser.add_argument('--train_day_end', type=int, default=None)
parser.add_argument('--validation_year', type=int, default=None)

parser.add_argument('--first_year', type=int, default=None)
parser.add_argument('--last_year', type=int, default=None)
parser.add_argument('--n_val_years', type=int, default=None)
parser.add_argument('--seq_l', type=int, default=2)

parser.add_argument('--wandb_api_key', type=str)
parser.add_argument('--wandb_project_name', type=str)
parser.add_argument('--wandb_username', type=str)
parser.add_argument('--wandb_team', type=str)

#-- CRPS / noisy-model parameters
parser.add_argument('--num_channels_noise', type=int, default=5,
                    help='number of noise channels injected at stage 1 of the noisy model')
parser.add_argument('--members_for_training', type=int, default=5,
                    help='number of noisy forward passes per training step (ensemble size for CRPS)')
parser.add_argument('--spatial_grad_weight', type=float, default=0.0,
                    help='weight of the optional spatial-gradient regulariser (CRPS trainer)')
parser.add_argument('--val_ensemble_size', type=int, default=1,
                    help='number of noisy passes averaged at validation time (CRPS trainer)')

#-- CRPSSpectralLoss parameters
parser.add_argument('--h_shape', type=int, default=0,
                    help='height of the predictand grid (required for CRPSSpectralLoss)')
parser.add_argument('--w_shape', type=int, default=0,
                    help='width of the predictand grid (required for CRPSSpectralLoss)')
parser.add_argument('--lambda_spectral', type=float, default=0.1,
                    help='weight of the spectral CRPS term (CRPSSpectralLoss)')
parser.add_argument('--spatial_resolution', type=float, default=0.0,
                    help='predictand spatial resolution in km for the optional Nyquist low-pass filter; '
                         '<=0 disables it (CRPSSpectralLoss)')

# parser.add_argument('--validation_year', type=lambda x : None if x == 'None' else int(x), default=None)

import argparse

### PARAMETERS THAT ARE NOW SET MANUALLY
THRESHOLD = 0.0
BINMIN = np.log1p(THRESHOLD)
BINMAX = np.log1p(400)
BINWIDTH = np.log1p(0.25) # 0.5*24 mm/day

HISTORY_LENGTH_MAP = {
    "1h": 24,   # [t-24,...,t]
    "3h": 8,    # [t-24,t-21,...,t]
    "6h": 4,    # [t-24,t-18,t-12,t-6,t]
    "1d": 2,    # [t-2,t-1,t]
}

HIGH_INDEPENDENT_VARS=True


if __name__ == '__main__':

    args = parser.parse_args()

    # Set all seeds
    set_seed_everything(seed=args.seed)
    #torch.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


#-----------------------------------------------------
#--------------- WANDB and ACCELERATE ----------------
#-----------------------------------------------------

    if args.use_accelerate is True:
        accelerator = Accelerator(log_with="wandb", step_scheduler_with_optimizer=False)
    else:
        accelerator = None
    
    #os.environ['WANDB_API_KEY'] = 'b3abf8b44e8d01ae09185d7f9adb518fc44730dd'
    #os.environ['WANDB_USERNAME'] = 'valebl'
    os.environ['WANDB_API_KEY'] = args.wandb_api_key
    #os.environ['WANDB_USERNAME'] = args.wandb_username
    os.environ['WANDB_ENTITY'] = args.wandb_team
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_CONFIG_DIR']='./wandb/'
    os.environ['WANDB_SERVICE_WAIT'] = '300'

    accelerator.init_trackers(
            project_name=args.wandb_project_name
        )

    write_log(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.", args, accelerator, 'w')

#--------------------------------------------------------
#--------------------- LOAD FILES -----------------------
#--------------------------------------------------------

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
    time_index = np.load(args.input_path+"time_index.npy")

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
#----------------------- LOSS ------------------------
#-----------------------------------------------------

    if args.loss_fn == "MSE_QMSE_PSD_Loss":
        loss_fn = MSE_QMSE_PSD_Loss(alpha=args.alpha, beta=args.beta)
    elif args.loss_fn == "GaussianNLLLoss":
        loss_fn = GaussianNLLLoss()
    elif args.loss_fn == "CRPSLoss":
        loss_fn = CRPSLoss(ignore_nans=True)
    elif args.loss_fn == "CRPSSpectralLoss":
        if args.h_shape <= 0 or args.w_shape <= 0:
            raise ValueError("CRPSSpectralLoss requires --h_shape and --w_shape > 0.")
        # Check that the high-res grid is regular and matches the requested
        # (H_shape, W_shape). CRPSSpectralLoss._FFT reshapes (B, N_high) to
        # (B, H_shape, W_shape) before computing the FFT, so the product must
        # match the number of high-res nodes per snapshot.
        n_high_nodes = low_high_graph["high"].num_nodes
        if args.h_shape * args.w_shape != n_high_nodes:
            raise ValueError(
                f"CRPSSpectralLoss: --h_shape * --w_shape "
                f"({args.h_shape} * {args.w_shape} = {args.h_shape*args.w_shape}) "
                f"does not match the number of high-res nodes "
                f"({n_high_nodes}). The spectral CRPS requires a regular grid."
            )
        write_log(
            f"\nCRPSSpectralLoss grid check OK: "
            f"H={args.h_shape}, W={args.w_shape}, N_high={n_high_nodes}.",
            args, accelerator, 'a'
        )
        loss_fn = CRPSSpectralLoss(
            ignore_nans=True,
            H_shape=args.h_shape,
            W_shape=args.w_shape,
            beta=1,
            lambda_spectral=args.lambda_spectral,
            spatial_resolution=(args.spatial_resolution
                                if args.spatial_resolution > 0 else None),
        )
    else:
        raise Exception(f"The provided loss: {args.loss_fn} is not implemented.")

#--------------------------------------------------------
#--------------------  PREPROCESSING --------------------
#--------------------------------------------------------

    #-- Step 1 - Prepare target
    target_prepared = prepare_target_for_train(target, args.target_type)

    #-- Step 2 - Find valid time indices
    idxs_not_all_nan = find_not_all_nan_times(target_prepared)

    write_log(f"\nAfter removing all nan time indexes, {len(idxs_not_all_nan)}" +
        f" time indexes are considered ({(len(idxs_not_all_nan) / target_prepared.shape[1] * 100):.1f} % of initial ones).",
        args, accelerator, 'a')

    #-- Step 3 - Compute train/val indices
    if args.first_year == 0 or args.last_year == 0:
        train_idxs, train_idxs_valid_subset, val_idxs, val_idxs_valid_subset = derive_train_val_idxs(
            args.train_year_start, args.train_month_start, args.train_day_start,
            args.train_year_end, args.train_month_end, args.train_day_end,
            history_length=history_length,
            time_index=time_index,
            idxs_not_all_nan=idxs_not_all_nan,
            validation_year=args.validation_year,
            args=args,
            accelerator=accelerator
        )
    else:
        # Build the full range
        all_years = list(range(args.first_year, args.last_year + 1))
        # Randomly sample validation years
        val_years = random.sample(all_years, args.n_val_years)
        # Training years are the complement
        train_years = [y for y in all_years if y not in val_years]
        train_idxs, train_idxs_valid_subset, val_idxs, val_idxs_valid_subset = derive_train_val_idxs_years_list(
            train_years,
            val_years,
            history_length=history_length,
            time_index=time_index,
            args=args,
            accelerator=accelerator
        )

    np.save(args.output_path + "train_idxs.npy", train_idxs)
    np.save(args.output_path + "train_idxs_valid_subset.npy", train_idxs_valid_subset)
    if args.validation_year is not None:
        np.save(args.output_path + "val_idxs.npy", val_idxs)
        np.save(args.output_path + "val_idxs_valid_subset.npy", val_idxs_valid_subset)


    #-- Step 4 - Compute QMSE bins
    if "QMSE" in args.loss_fn:
        target_bins = derive_qmse_bins(
            target_prepared,
            train_idxs_valid_subset,
            args,
            accelerator,
            binmin=BINMIN,
            binmax=BINMAX,
            binwidth=BINWIDTH
        )
        
    #-- Step 5 - Compute input statistics + standardize
    # At this point the high input coincides to orog, then mask and ij are added
    low_input_std = new_compute_input_statistics_and_standardize(
            x_low=low_input,
            train_idxs=train_idxs,
            n_vars=n_vars,
            n_levels=n_levels,
            apply_stats=True,
            args=args,
            accelerator=accelerator
        )
    
    high_input_std = orog/np.max(orog)
    
    #-- Step 6. Add the other high-res features
    if use_mask_sealand:
        high_input_std = np.concatenate((high_input_std, mask_sealand), axis=-1)
        write_log(f"\nAdding mask sea-land node features", args, accelerator, 'a')

    if use_coords_ij:
        high_input_std = np.concatenate((high_input_std, coords_ij), axis=-1)
        write_log(f"\nAdding ij node features", args, accelerator, 'a')

    #-- Step 7 - torch tensors from numpy arrays
    target_prepared = torch.from_numpy(target_prepared).float()
    train_idxs = torch.from_numpy(train_idxs).int()
    if args.validation_year is not None:
        val_idxs = torch.from_numpy(val_idxs).int()
    if "QMSE" in args.loss_fn:
        target_bins = torch.from_numpy(target_bins).int()
    low_input_std = torch.from_numpy(low_input_std).float()
    high_input_std = torch.from_numpy(high_input_std).float()
    
    low_input_std = torch.flatten(low_input_std, start_dim=2, end_dim=-1)   # num_nodes, time, vars*levels
    low_input_train = low_input_std[:, train_idxs, :]
    target_train = target_prepared[:, train_idxs]
    if "QMSE" in args.loss_fn:
        target_bins_train = target_bins[:, train_idxs]

    if args.validation_year is not None:
        low_input_val = low_input_std[:, val_idxs, :]
        target_val = target_prepared[:, val_idxs]
        if "QMSE" in args.loss_fn:
            target_bins_val = target_bins[:, val_idxs]

    if accelerator.is_main_process:
        print(f"low_input_train.shape: {low_input_train.shape}")
        print(f"target_train.shape: {target_train.shape}")
        if "QMSE" in args.loss_fn:
            print(f"target_bins_train.shape: {target_bins_train.shape}")
        if args.validation_year is not None:
            print(f"low_input_val.shape: {low_input_val.shape}")
            print(f"target_val.shape: {target_val.shape}")
            if "QMSE" in args.loss_fn:
                print(f"target_bins_val.shape: {target_bins_val.shape}")

    #-----------------------------------------------------
    #------------------------ MODEL ----------------------
    #-----------------------------------------------------

    n_static_high = high_input_std.shape[1]
    write_log(f"\nn_vars: {n_vars}, n_levels: {n_levels}, n_static_high: {n_static_high}", args, accelerator, 'a')

    # Model
    models = importlib.import_module(f"models.{args.model_name}")
    Model = getattr(models, args.model_name)
    model_kwargs = dict(
        h_in=n_vars*n_levels,
        h_hid=n_vars*n_levels,
        high_in=n_static_high,
        seq_length=seq_length,
    )
    if args.loss_fn in ("CRPSLoss", "CRPSSpectralLoss"):
        model_kwargs["num_channels_noise"] = args.num_channels_noise
        model_kwargs["members_for_training"] = args.members_for_training
    model = Model(**model_kwargs)
   
    #-----------------------------------------------------
    #-------------- DATASET AND DATALOADER ---------------
    #-----------------------------------------------------
  
    # Create two different datasets for efficiency
    dataset_graph_train_tmp = Dataset_Graph(
        low_high_graph,
        low_input_train,
        high_input_std,
        target_train,
        history_length,
    )

    if args.validation_year is not None or args.val_years is not None:
        dataset_graph_val_tmp = Dataset_Graph(
            low_high_graph,
            low_input_val,
            high_input_std,
            target_val,
            history_length,
        )

    if "QMSE" in args.loss_fn:
        dataset_graph_train_tmp.set_additional_features(w=target_bins_train)
        if args.validation_year is not None:
            dataset_graph_val_tmp.set_additional_features(w=target_bins_val)

    dataset_graph_train = torch.utils.data.Subset(dataset_graph_train_tmp, train_idxs_valid_subset) # it's just a view of the original dataset
    if args.validation_year is not None:
        dataset_graph_val = torch.utils.data.Subset(dataset_graph_val_tmp, val_idxs_valid_subset)
        
    # len(dataset_graph_train) will be the number of training exaples (inputs are bigger, considering the history length)
    if args.validation_year is not None:
        write_log(f'\nTrainset size = {len(dataset_graph_train)}, validationset size = {len(dataset_graph_val)}.', args, accelerator, 'a')
    else:
        write_log(f'\nTrainset size = {len(dataset_graph_train)}.', args, accelerator, 'a')

    # Define the dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_graph_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn_graph,
        num_workers=0
    )

    if args.validation_year is not None:
        dataloader_val = torch.utils.data.DataLoader(
            dataset_graph_val,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate_fn_graph,
            num_workers=0
        )
    else:
        dataloader_val = None

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        write_log(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %", args, accelerator, 'a')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    elif args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001, last_epoch=-1)
    else:
        lr_scheduler = None

#-----------------------------------------------------
#---------------- ACCELERATE PREPARE -----------------
#-----------------------------------------------------

    epoch_start=0
    
    if accelerator is not None:
        model, optimizer, dataloader_train, lr_scheduler, loss_fn = accelerator.prepare(
            model, optimizer, dataloader_train, lr_scheduler, loss_fn)
        if args.validation_year is not None:
            dataloader_val = accelerator.prepare(dataloader_val)
        write_log("\nUsing accelerator to prepare model, optimizer, dataloader and loss_fn...", args, accelerator, 'a')
    else:
        write_log("\nNot using accelerator to prepare model, optimizer, dataloader and loss_fn...", args, accelerator, 'a')
        model = model.cuda()

    if args.ctd_training:
        write_log("\nContinuing the training.")
        accelerator.load_state(args.checkpoint_ctd)
        epoch_start = torch.load(args.checkpoint_ctd+"epoch")["epoch"] + 1
        
    if not args.fine_tuning:
        net_names = ["rnn", "dense", "downscaler", "processor"]
        for net_name in net_names:
            [param.requires_grad_(False) for name, param in model.named_parameters() if net_name in name]
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        model, optimizer, dataloader_train, lr_scheduler, loss_fn = accelerator.prepare(
            model, optimizer, dataloader_train, lr_scheduler, loss_fn)
    
    check_freezed_layers(model, args.output_path, args.log_file, accelerator)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(f"\nTotal number of trainable parameters: {total_params}.", args, accelerator, 'a')

#-----------------------------------------------------
#----------------------- TRAIN -----------------------
#-----------------------------------------------------

    
    write_log(f"\nUsing lr={optimizer.param_groups[0]['lr']:.8f}, " +
                f"weight decay = {args.weight_decay} and epochs={args.epochs}." + 
                f"\nloss: {loss_fn}", args, accelerator, 'a') 
    if accelerator is None:
        write_log(f"\nModel = {args.model_name}, batch size = {args.batch_size}", args, accelerator, 'a') 
    else:
        write_log(f"\nModel = {args.model_name}, batch size = {args.batch_size*torch.cuda.device_count()}", args, accelerator, 'a')

    start = time.time()

    val_size = len(dataset_graph_val)

    if args.loss_fn == "GaussianNLLLoss":
        train_fn = NLL_Trainer().train_reg
    elif args.loss_fn == "MSE_QMSE_PSD_Loss":
        train_fn = MSE_QMSE_PSD_Trainer().train_reg
    elif args.loss_fn in ("CRPSLoss", "CRPSSpectralLoss"):
        train_fn = CRPS_Trainer(
            spatial_gradient_weight=args.spatial_grad_weight,
            val_ensemble_size=args.val_ensemble_size,
            use_spectral=(args.loss_fn == "CRPSSpectralLoss"),
        ).train_reg

    train_fn(
        model,
        dataloader_train,
        dataloader_val,
        optimizer,
        loss_fn,
        lr_scheduler,
        val_size,
        time_index[val_idxs][val_idxs_valid_subset],
        accelerator,
        args,
        epoch_start=epoch_start)

    end = time.time()

    write_log(f"\nCompleted in {end - start} seconds.\nDONE!", args, accelerator, 'a')

