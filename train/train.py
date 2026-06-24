import torch
import numpy as np
import pickle
import time
import os
import importlib
import json
import random
from accelerate import Accelerator

from args.train.build_args import build_args

# Models
from models.build_model import build_model

# Losses
from utils.losses.build_loss import build_loss
from utils.losses.qmse import derive_qmse_bins

# Helpers
from utils.helpers.tools import (
    write_log,
    inspect_model,
    set_seed_everything,
    find_not_all_nan_times,
    derive_train_val_idxs,
    derive_train_val_idxs_from_years_list,
)

# Training
from utils.training.detect_train_val_idxs_config import detect_train_val_idxs_config
from utils.training.trainer import Trainer

# Data
from data.datasets.graph_dataset import Graph_Dataset, custom_collate_fn_graph

# Transforms
from utils.predictand_transforms.transform_predictand import transform_predictand
from utils.predictor_transforms.transform_predictors import transform_predictors


if __name__ == '__main__':
    
    args = build_args()

    # Set all seeds
    set_seed_everything(seed=args.seed)

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
    
    if args.WANDB_API_KEY:
        os.environ['WANDB_API_KEY'] = args.WANDB_API_KEY
    if args.WANDB_USERNAME:
        os.environ['WANDB_USERNAME'] = args.WANDB_USERNAME
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
    x_low = np.load(args.input_path+args.low_input_file)

    #-- 3. Target
    target = np.load(args.input_path+args.target_file)

    #-- 4. Orography
    orog = np.load(args.input_path+args.orog_file)

    #-- 5. Mask sea-land
    if args.mask_sealand_file not in ("", "None", None):
        mask_sealand = np.load(args.input_path+args.mask_sealand_file)
        use_mask_sealand = True
    else:
        use_mask_sealand = False
    
    #-- 6. Coords ij
    if args.coords_ij_file not in ("", "None", None):
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
#-------------------  TRAIN/VAL IDXS --------------------
#--------------------------------------------------------

    idxs_not_all_nan = find_not_all_nan_times(
        data=target,
        L=args.history_length,
        args=args,
        accelerator=accelerator
        )

    cfg = detect_train_val_idxs_config(args)

    # Case 1 and 2: year-based logic
    if cfg["mode"] in ("years_list", "random_years"):
        train_idxs, train_idxs_valid_subset, val_idxs, val_idxs_valid_subset = \
            derive_train_val_idxs_from_years_list(
                cfg["train_years"],
                cfg["val_years"],
                history_length=args.history_length,
                time_index=time_index,
                args=args,
                accelerator=accelerator
            )

    # Case 3: date-based logic
    else:
        train_idxs, train_idxs_valid_subset, val_idxs, val_idxs_valid_subset = \
            derive_train_val_idxs(
                cfg["train_start"].year, cfg["train_start"].month, cfg["train_start"].day,
                cfg["train_end"].year,   cfg["train_end"].month,   cfg["train_end"].day,
                history_length=args.history_length,
                time_index=time_index,
                idxs_not_all_nan=idxs_not_all_nan,
                validation_year=cfg["validation_year"],
                args=args,
                accelerator=accelerator
            )


    np.save(args.output_path + "train_idxs.npy", train_idxs)
    np.save(args.output_path + "train_idxs_valid_subset.npy", train_idxs_valid_subset)
    np.save(args.output_path + "val_idxs.npy", val_idxs)
    np.save(args.output_path + "val_idxs_valid_subset.npy", val_idxs_valid_subset)

#--------------------------------------------------------
#---------  TRANSFORM PREDICTORS AND PREDICTAND ---------
#--------------------------------------------------------

    # 1. Transform predictors
    x_low_std, x_high_std = transform_predictors(
        x_low,
        x_high=orog,
        train_idxs=train_idxs,
        mode_low=args.predictor_low_transform_mode,      # e.g. "zscore_lowres_var"
        mode_high=args.predictor_high_transform_mode,    # e.g. "zscore_highres_grouped"
        stats=None,
        stats_save_path=args.output_path + "predictors_stats.npz"
    )

    n_vars = x_low_std.shape[2]
    n_levels = x_low_std.shape[3]

    # 1.2 Flatten x_low_std
    N, T = x_low_std.shape[:2] 
    x_low_std = x_low_std.reshape(N, T, -1) # num_nodes, time, vars*levels

    # 1.3 Add high_res predictors which are not transformed
    if use_mask_sealand:
        x_high_std = np.concatenate((x_high_std, mask_sealand), axis=-1)
        write_log(f"\nAdding mask sea-land node features", args, accelerator, 'a')

    if use_coords_ij:
        x_high_std = np.concatenate((x_high_std, coords_ij), axis=-1)
        write_log(f"\nAdding ij node features", args, accelerator, 'a')

    n_static_high = x_high_std.shape[1]

    write_log(f"\nn_vars: {n_vars}, n_levels: {n_levels}, n_static_high: {n_static_high}", args, accelerator, 'a')

    # 2. Transform predictand
    if args.loss_name == "Bernoulli_Gamma_NLL_Loss":
        target_trans = target
    else:
        target_trans = transform_predictand(
            target,
            mode=args.predictand_transform_mode,      # e.g. "log1p", "z_score", "minmax"
            train_idxs=train_idxs,
            stats_save_path=args.output_path + "predictand_stats.npz"
        )
    
    #-------------------------------------------
    #-------------- BUILD LOSS -----------------
    #-------------------------------------------

    loss_fn, output_dim = build_loss(args)

    # Eventually compute QMSE bins
    if getattr(loss_fn, "use_bins", False):
        if args.binscale == "log":
            binmin = np.log1p(args.binmin)
            binmax = np.log1p(args.binmax)
            binwidth = np.log1p(args.binwidth)
        else:
            binmin = args.binmin
            binmax = args.binmax
            binwidth = args.binwidth

        target_bins = derive_qmse_bins(
            target_trans,
            train_idxs[train_idxs_valid_subset],
            args,
            accelerator,
            binmin=binmin,
            binmax=binmax,
            binwidth=binwidth
        )
    
    #-----------------------------------------------------
    #---------------- SPLIT IN TRAIN/VAL -----------------
    #-----------------------------------------------------

    x_low_std_train = x_low_std[:, train_idxs, :]
    x_low_std_val = x_low_std[:, val_idxs, :]

    target_trans_train = target_trans[:, train_idxs]
    target_trans_val = target_trans[:, val_idxs]

    if getattr(loss_fn, "use_bins", False):
        target_bins_train = target_bins[:, train_idxs]
        target_bins_val = target_bins[:, val_idxs]

    #-----------------------------------------------------
    #-------------- FROM NUMPY TO PYTORCH ----------------
    #-----------------------------------------------------

    # Predictors
    x_low_std_train = torch.from_numpy(x_low_std_train).float()
    x_low_std_val = torch.from_numpy(x_low_std_val).float()
    x_high_std = torch.from_numpy(x_high_std).float()

    # Predictand and eventually QMSE bins
    target_trans_train = torch.from_numpy(target_trans_train).float()
    target_trans_val = torch.from_numpy(target_trans_val).float()

    if getattr(loss_fn, "use_bins", False):
        target_bins_train = torch.from_numpy(target_bins_train).int()
        target_bins_val = torch.from_numpy(target_bins_val).int()

    #--------------------------------------------
    #-------------- BUILD MODEL -----------------
    #--------------------------------------------

    model = build_model(
        x_low_var_dim=n_vars,
        x_low_lev_dim=n_levels,
        x_high_dim=n_static_high,
        output_dim=output_dim,
        args=args
    )

    #-----------------------------------------------------
    #-------------- DATASET AND DATALOADER ---------------
    #-----------------------------------------------------
  
    # Create two different datasets for efficiency
    graph_dataset_train_tmp = Graph_Dataset(
        low_high_graph,
        x_low_std_train,
        x_high_std,
        target_trans_train,
        args.history_length,
    )

    if args.validation_year is not None or args.val_years is not None:
        graph_dataset_val_tmp = Graph_Dataset(
            low_high_graph,
            x_low_std_val,
            x_high_std,
            target_trans_val,
            args.history_length,
        )

    if getattr(loss_fn, "use_bins", False):
        graph_dataset_train_tmp.set_additional_features(w=target_bins_train)
        graph_dataset_val_tmp.set_additional_features(w=target_bins_val)

    graph_dataset_train = torch.utils.data.Subset(graph_dataset_train_tmp, train_idxs_valid_subset) # it's just a view of the original dataset
    graph_dataset_val = torch.utils.data.Subset(graph_dataset_val_tmp, val_idxs_valid_subset)
        
    # len(graph_dataset_train) will be the number of training exaples (inputs are bigger, considering the history length)
    write_log(f'\nTrainset size = {len(graph_dataset_train)}, validationset size = {len(graph_dataset_val)}.', args, accelerator, 'a')

    # Define the dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        graph_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn_graph,
        num_workers=0
    )

    dataloader_val = torch.utils.data.DataLoader(
        graph_dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn_graph,
        num_workers=0
    )

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        write_log(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %", args, accelerator, 'a')

    #-----------------------------------------------------
    #------------ OPTIMIZER AND LR SCHEDULER -------------
    #-----------------------------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
    elif args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_eta_min,
            last_epoch=-1
        )
    elif args.lr_scheduler == "CosineAnnealingLR_with_warmup":
        # 2 epochs warmup
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,            # starts at 3e-5, ramps to 3e-4
            end_factor=1.0,
            total_iters=2,               # epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - 2,
            eta_min=args.lr_eta_min,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[2],
        )
    else:
        lr_scheduler = None

#-----------------------------------------------------
#---------------- ACCELERATE PREPARE -----------------
#-----------------------------------------------------

    epoch_start=0
    
    if accelerator is not None:
        model, optimizer, dataloader_train, lr_scheduler, loss_fn = accelerator.prepare(
            model, optimizer, dataloader_train, lr_scheduler, loss_fn)
        dataloader_val = accelerator.prepare(dataloader_val)
        write_log("\nUsing accelerator to prepare model, optimizer, dataloader and loss...", args, accelerator, 'a')
    else:
        write_log("\nNot using accelerator to prepare model, optimizer, dataloader and loss...", args, accelerator, 'a')
        model = model.cuda()

    if args.ctd_training:
        write_log("\nContinuing the training.")
        accelerator.load_state(args.checkpoint_ctd)
        epoch_start = torch.load(args.checkpoint_ctd+"epoch")["epoch"] + 1
    
    inspect_model(model, args, accelerator)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(f"\nTotal number of trainable parameters: {total_params}.", args, accelerator, 'a')

#-----------------------------------------------------
#----------------------- TRAIN -----------------------
#-----------------------------------------------------

    write_log(f"\nUsing lr={optimizer.param_groups[0]['lr']:.8f}, " +
                f"weight decay = {args.weight_decay} and epochs={args.epochs}." + 
                f"\nloss: {loss_fn}", args, accelerator, 'a') 
    
    effective_batch_size = args.batch_size if accelerator is None else args.batch_size*torch.cuda.device_count()
    write_log(f"\nModel = {args.model_name}, batch size = {effective_batch_size}", args, accelerator, 'a')

    val_size = len(graph_dataset_val)

    start = time.time()    

    trainer = Trainer()

    trainer.train(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        val_size=val_size,
        times=time_index[val_idxs][val_idxs_valid_subset],
        accelerator=accelerator,
        args=args,
        epoch_start=epoch_start
    )

    end = time.time()

    write_log(f"\nCompleted in {end - start} seconds.\nDONE!", args, accelerator, 'a')
    
