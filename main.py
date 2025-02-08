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

import utils.utils
from utils.utils import write_log, date_to_idxs, check_freezed_layers, set_seed_everything, find_not_all_nan_times, derive_train_val_idxs
from utils.utils import Trainer
from accelerate import Accelerator

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--target_file', type=str, default=None)
parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--weights_file', type=str, default=None) 

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

#-- start and end training dates
parser.add_argument('--train_year_start', type=float)
parser.add_argument('--train_month_start', type=float)
parser.add_argument('--train_day_start', type=float)
parser.add_argument('--train_year_end', type=float)
parser.add_argument('--train_month_end', type=float)
parser.add_argument('--train_day_end', type=float)
parser.add_argument('--first_year', type=float)
parser.add_argument('--validation_year', type=float, default=None)

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
    model = Model()

    # Loss
    if args.loss_fn == 'sigmoid_focal_loss':
        loss_fn = getattr(torchvision.ops, args.loss_fn)
    elif args.loss_fn == 'quantized_loss':
        loss_fn = getattr(utils.utils, args.loss_fn)(alpha=args.alpha)
    elif args.loss_fn == 'weighted_mse_loss':
        loss_fn = getattr(utils.utils, args.loss_fn)()
    elif args.loss_fn == 'weighted_mae_loss':
        loss_fn = getattr(utils.utils, args.loss_fn)()
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

    idxs_not_all_nan = find_not_all_nan_times(target_train)

    write_log(f"\nAfter removing all nan time indexes, {idxs_not_all_nan.sum()}" +
            f" time indexes are considered ({(idxs_not_all_nan.sum() / target_train.shape[1] * 100):.1f} % of initial ones).",
            args, accelerator, 'a')

    # We assume that the validtion set is a single year within the dataset
    train_idxs, val_idxs = derive_train_val_idxs(args.train_year_start, args.train_month_start, args.train_day_start, args.train_year_end,
                                                 args.train_month_end, args.train_day_end, args.first_year, idxs_not_all_nan, args.validation_year)

    train_start_idx = train_idxs.min()
    train_end_idx = train_idxs.max()
    val_start_idx = val_idxs.min()
    val_end_idx = val_idxs.max()
    
    write_log(f"\nTrain from {args.train_day_start}/{args.train_month_start}/{args.train_year_start} to " +
              f"{args.train_day_end}/{args.train_month_end}/{args.train_year_end} with validation year {args.validation_year}",
              args, accelerator, 'a')

    # Define input and target
    low_high_graph['low'].x = low_high_graph['low'].x[:,min(train_start_idx, val_start_idx):max(train_end_idx, val_end_idx),:] # nodes, time, var*lev
    target_train = target_train[:,min(train_start_idx, val_start_idx):max(train_end_idx, val_end_idx)] # 0, 140256
    
    #-----------------------------------------------------
    #-------------- DATASET AND DATALOADER ---------------
    #-----------------------------------------------------
    
    Dataset_Graph = getattr(dataset, args.dataset_name)
    
    if args.loss_fn == 'weighted_mse_loss' or args.loss_fn == 'weighted_mae_loss' or args.loss_fn == "quantized_loss":
        with open(args.input_path+args.weights_file, 'rb') as f:
            weights_reg = pickle.load(f)
        weights_reg = weights_reg[:,min(train_start_idx, val_start_idx):max(train_end_idx, val_end_idx)]

        write_log("\nUsing weights in the loss.", args, accelerator, 'a')

        dataset_graph = Dataset_Graph(targets=target_train,
            w=weights_reg, graph=low_high_graph, model_name=args.model_name)
    else:
        if accelerator is None or accelerator.is_main_process:
            write_log("\nNot using weights in the loss.", args, accelerator, 'a')
        dataset_graph = Dataset_Graph(targets=target_train,
            graph=low_high_graph, model_name=args.model_name)

    custom_collate_fn = getattr(dataset, args.collate_name)
        
    #-- split into trainset and testset
    n_gpu = 4
    tail_train_idxs = len(train_idxs) % n_gpu
    tail_val_idxs = len(val_idxs) % n_gpu
    if tail_train_idxs != 0:
        train_idxs = train_idxs[:-tail_train_idxs]
    if tail_val_idxs != 0:
        val_idxs = val_idxs[:-tail_val_idxs]
    dataset_graph_train = torch.utils.data.Subset(dataset_graph, train_idxs)
    dataset_graph_val = torch.utils.data.Subset(dataset_graph, val_idxs)

    sampler_graph_train = Iterable_Graph(dataset_graph=dataset_graph_train, shuffle=True)
    sampler_graph_val = Iterable_Graph(dataset_graph=dataset_graph_val, shuffle=True)

    write_log(f'\nTrainset size = {len(dataset_graph_train)}, validationset size = {len(dataset_graph_val)}.', args, accelerator, 'a')

    dataloader_train = torch.utils.data.DataLoader(dataset_graph_train, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph_train, collate_fn=custom_collate_fn)

    dataloader_val = torch.utils.data.DataLoader(dataset_graph_val, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph_val, collate_fn=custom_collate_fn)

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        write_log(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %", args, accelerator, 'a')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0, last_epoch=-1)

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
        model = trainer.train_cl(model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=epoch_start)
    elif args.model_type == "reg":
        model = trainer.train_reg(model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=epoch_start)      
    end = time.time()

    write_log(f"\nCompleted in {end - start} seconds.\nDONE!")
    

