from torch_geometric.data import HeteroData
import torch

from torch import nn
import torchvision.ops
import matplotlib.pyplot as plt
import numpy as np
import pickle
from dataset import Iterable_Graph
import dataset
import time
import argparse
import sys
import os
import dataset
import importlib

import utils.loss_functions
from utils.tools import write_log, check_freezed_layers, set_seed_everything
from utils.tools import find_not_all_nan_times, derive_train_val_idxs, derive_train_val_test_idxs_random_months
from utils.tools import compute_input_statistics, standardize_input
from utils.train_test import Trainer
from accelerate import Accelerator

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--target_file', type=str, default=None)
parser.add_argument('--graph_file', type=str, default=None) 

parser.add_argument('--out_checkpoint_file', type=str, default="checkpoint.pth")
parser.add_argument('--out_loss_file', type=str, default="loss.csv")

parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')
parser.add_argument('--wandb_project_name', type=str)

#-- training hyperparameters
parser.add_argument('--pct_trainset', type=float, default=1.0, help='percentage of dataset in trainset')
parser.add_argument('--epochs', type=int, default=15, help='number of total training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (global)')
parser.add_argument('--step_size', type=int, default=10, help='scheduler step size (global)')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (wd)')
parser.add_argument('--fine_tuning',  action='store_true')
parser.add_argument('--no-fine_tuning', dest='fine_tuning', action='store_false')
parser.add_argument('--load_checkpoint',  action='store_true')
parser.add_argument('--no-load_checkpoint', dest='load_checkpoint', action='store_false')

parser.add_argument('--checkpoint_ctd', type=str, help='checkpoint to load to continue')
parser.add_argument('--ctd_training',  action='store_true')
parser.add_argument('--no-ctd_training', dest='ctd_training', action='store_false')

parser.add_argument('--loss_fn', type=str, default="mse_loss")
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--seed', type=int)
parser.add_argument('--n_gpu', type=int, default=4)

parser.add_argument('--model_type', type=str)
parser.add_argument('--model_name', type=str, default='HiResPrecipNet')
parser.add_argument('--dataset_name', type=str, default='Dataset_Graph')
parser.add_argument('--collate_name', type=str)

parser.add_argument('--stats_mode', type=str, default="var")
parser.add_argument('--target_type', type=str)

#-- start and end training dates
parser.add_argument('--train_year_start', type=int)
parser.add_argument('--train_month_start', type=int)
parser.add_argument('--train_day_start', type=int)
parser.add_argument('--train_year_end', type=int)
parser.add_argument('--train_month_end', type=int)
parser.add_argument('--train_day_end', type=int)
parser.add_argument('--first_year', type=int)
parser.add_argument('--validation_year', type=int, default=None)


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
    
    os.environ['WANDB_API_KEY'] = 'b3abf8b44e8d01ae09185d7f9adb518fc44730dd'
    os.environ['WANDB_USERNAME'] = 'valebl'
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_CONFIG_DIR']='./wandb/'
    os.environ['WANDB_SERVICE_WAIT'] = '300'

    accelerator.init_trackers(
            project_name=args.wandb_project_name
        )

    write_log(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.", args, accelerator, 'w')


#-----------------------------------------------------
#--------------- MODEL, LOSS, OPTIMIZER --------------
#-----------------------------------------------------

    models = importlib.import_module(f"models.{args.model_name}")
    Model = getattr(models, args.model_name)
    if args.target_type == "temperature":
        model = Model(h_in=4*5, h_hid=4*5, high_in=1)
    else:
        model = Model()

    # Loss
    if args.loss_fn == 'sigmoid_focal_loss':
        loss_fn = getattr(torchvision.ops, args.loss_fn)
    elif 'quantized_loss' in args.loss_fn:
        loss_fn = getattr(utils.loss_functions, args.loss_fn)(alpha=args.alpha)
    elif args.loss_fn == 'weighted_mse_loss':
        loss_fn = getattr(utils.loss_functions, args.loss_fn)()
    elif args.loss_fn == 'weighted_mae_loss':
        loss_fn = getattr(utils.loss_functions, args.loss_fn)()
    else:
        loss_fn = getattr(nn.functional, args.loss_fn) 
    

#-----------------------------------------------------
#--------------------- INDEXES -----------------------
#-----------------------------------------------------

    # Load the graph and target files
    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    with open(args.input_path+args.target_file, 'rb') as f:
        target_train = pickle.load(f)

    if args.target_type == "precipitation":
        # derive two masks:
        # - mask_not_nan, i.e. where the target is not nan
        # - mask_geq_threshold, i.e. where the target is larger than the preferred threshold (now 0.1mm)
        threshold = 0.1

        mask_nan = torch.isnan(target_train)
        mask_threshold = target_train < threshold #mm

        # set to 0.0 everything below sensitivity threshold
        target_train[mask_threshold] = 0.0
        # round to comply with instrument sensitivity
        target_train = torch.round(target_train, decimals=1)

        if args.model_type == "cl":
            #-- CLASSIFIER --#        
            target_train = torch.where(target_train >= threshold, 1, 0).float()
        elif args.model_type == "reg":
            #-- REGRESSOR ON pr >=threshold --#    
            target_train = torch.log1p(target_train)
            target_train[target_train < threshold] = torch.nan
        elif args.model_type =="all":
            #-- REGRESSOR ON ALL --#    
            target_train = torch.log1p(target_train)
    
        target_train[mask_nan] = torch.nan

    idxs_not_all_nan = find_not_all_nan_times(target_train)

    write_log(f"\nAfter removing all nan time indexes, {len(idxs_not_all_nan)}" +
            f" time indexes are considered ({(len(idxs_not_all_nan) / target_train.shape[1] * 100):.1f} % of initial ones).",
            args, accelerator, 'a')

    if args.validation_year is not None:
        train_idxs, val_idxs = derive_train_val_idxs(args.train_year_start, args.train_month_start, args.train_day_start, args.train_year_end,
                                                     args.train_month_end, args.train_day_end, args.first_year, idxs_not_all_nan, args.validation_year)
        write_log(f"\nTrain from {args.train_day_start}/{args.train_month_start}/{args.train_year_start} to " +
                  f"{args.train_day_end}/{args.train_month_end}/{args.train_year_end} with validation year {args.validation_year}",
                  args, accelerator, 'a')
    else:
        train_idxs, val_idxs = derive_train_val_test_idxs_random_months(args.train_year_start, args.train_month_start,
            args.train_day_start,args.train_year_end, args.train_month_end, args.train_day_end, args.first_year,
            idxs_not_all_nan, args, accelerator)

        
        write_log(f"\nTrain from {args.train_day_start}/{args.train_month_start}/{args.train_year_start} to " +
                f"{args.train_day_end}/{args.train_month_end}/{args.train_year_end} with validation and test years" +
                f"as 12 months chosen randomly within the {args.train_year_start}-{args.train_year_end} period..",
                args, accelerator, 'a')

    train_start_idx = train_idxs.min()
    train_end_idx = train_idxs.max()
    val_start_idx = val_idxs.min()
    val_end_idx = val_idxs.max()
    
    # Check that the size of the train and val idxs is multiple of n_gpu
    # to avoid issues with accelerator.gather_for_metrics; if not, simply
    # discard the last idxs to obtain a multiple of n_gpu
    tail_train_idxs = len(train_idxs) % args.n_gpu
    tail_val_idxs = len(val_idxs) % args.n_gpu
    if tail_train_idxs != 0:
        train_idxs = train_idxs[:-tail_train_idxs]
    if tail_val_idxs != 0:
        val_idxs = val_idxs[:-tail_val_idxs]

    # Compute the weights for the regressor
    if args.model_type == "reg" or args.model_type == "all" and "quantized" in args.loss_fn:
        # This should be put in a function
        bins = np.arange(np.log1p(threshold), np.log1p(200), np.log1p(0.5))
        if args.model_type == "all":
            bins = np.insert(bins, 0, np.log1p(0))
        values_unif_log, edges_unif_log = np.histogram(target_train.numpy(), bins=bins, density=False)
        # Assign bins to targets
        target_bins = np.digitize(target_train.numpy(), edges_unif_log, right=False).astype(float) - 1

        nbins = (np.nanmax(target_bins) + 1).astype(int)
        if nbins > len(values_unif_log):
            write_log(f"\nBins min: {np.nanmin(target_bins).astype(int)}, bins max: {np.nanmax(target_bins).astype(int)}, nbins: {nbins}, len weights: {len(values_unif_log)}", args, accelerator, 'a')
            target_bins[target_bins == nbins -1] = nbins - 2
            nbins = nbins - 1
            write_log("\nUpdating last bin...", args, accelerator, 'a')
        write_log(f"\nbins min: {np.nanmin(target_bins).astype(int)}, bins max: {np.nanmax(target_bins).astype(int)}, nbins: {nbins}", args, accelerator, 'a')
        target_bins = torch.tensor(target_bins)
        target_bins[mask_nan] = torch.nan

    means_low, stds_low, means_high, stds_high = compute_input_statistics(
        low_high_graph['low'].x[:,train_idxs,:], low_high_graph['high'].x, args, accelerator)
    
    low_high_graph['low'].x, low_high_graph['high'].x = standardize_input(
        low_high_graph['low'].x, low_high_graph['high'].x, means_low, stds_low, means_high, stds_high, args, accelerator) # num_nodes, time, vars, levels
    
    vars_names = ['q', 't', 'u', 'v', 'z']
    levels = ['200', '500', '700', '850', '1000']
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
    if low_high_graph['high'].x.size()[1] > 1:
        write_log(f"\nHigh land_use: mean={low_high_graph['high'].x[:,1:].mean()}, std={low_high_graph['high'].x[:,1:].std()}",
              args, accelerator, 'a')
    
    if args.target_type == "temperature":
        low_high_graph['low'].x = torch.cat((low_high_graph['low'].x[:,:,:1,:], low_high_graph['low'].x[:,:,2:,:]), dim=2)

    low_high_graph['low'].x = torch.flatten(low_high_graph['low'].x, start_dim=2, end_dim=-1)   # num_nodes, time, vars*levels

    #-----------------------------------------------------
    #-------------- DATASET AND DATALOADER ---------------
    #-----------------------------------------------------
    
    Dataset_Graph = getattr(dataset, args.dataset_name)
    
    if "quantized" in args.loss_fn:
        dataset_graph = Dataset_Graph(targets=target_train,
            w=target_bins, graph=low_high_graph, model_name=args.model_name)
    else:
        dataset_graph = Dataset_Graph(targets=target_train,
            graph=low_high_graph, model_name=args.model_name)

    # Define the custom collate function
    custom_collate_fn = getattr(dataset, args.collate_name)
        
    # Define the custom samplers
    sampler_graph_train = Iterable_Graph(dataset_graph=dataset_graph, shuffle=True, idxs_vector=train_idxs)
    sampler_graph_val = Iterable_Graph(dataset_graph=dataset_graph, shuffle=False, idxs_vector=val_idxs, t_offset=val_idxs.min())

    write_log(f'\nTrainset size = {train_idxs.shape[0]}, validationset size = {val_idxs.shape[0]}.', args, accelerator, 'a')

    # Define the dataloaders
    dataloader_train = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph_train, collate_fn=custom_collate_fn)

    dataloader_val = torch.utils.data.DataLoader(dataset_graph, batch_size=10, num_workers=0,
                    sampler=sampler_graph_val, collate_fn=custom_collate_fn)

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        write_log(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %", args, accelerator, 'a')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001, last_epoch=-1)

#-----------------------------------------------------
#------------------ LOAD PARAMETERS ------------------
#-----------------------------------------------------

    epoch_start=0
    
    if accelerator is not None:
        model, optimizer, dataloader_train, dataloader_val, lr_scheduler, loss_fn = accelerator.prepare(
            model, optimizer, dataloader_train, dataloader_val, lr_scheduler, loss_fn)
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

    
    write_log(f"\nUsing pct_trainset={args.pct_trainset}, lr={optimizer.param_groups[0]['lr']:.8f}, " +
                f"weight decay = {args.weight_decay} and epochs={args.epochs}." + 
                f"loss: {loss_fn}", args, accelerator, 'a') 
    if accelerator is None:
        write_log(f"\nModel = {args.model_name}, batch size = {args.batch_size}", args, accelerator, 'a') 
    else:
        write_log(f"\nModel = {args.model_name}, batch size = {args.batch_size*torch.cuda.device_count()}", args, accelerator, 'a')

    start = time.time()

    trainer = Trainer()
    if args.model_type == "cl":
        trainer.train_cl(model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=epoch_start)
    elif args.model_type == "reg" or args.model_type == "all":
        trainer.train_reg(model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=epoch_start)
    end = time.time()

    write_log(f"\nCompleted in {end - start} seconds.\nDONE!", args, accelerator, 'a')
    

