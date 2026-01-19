import torch

import torchvision.ops
import pickle
import time
import argparse
import os
import importlib
from dataset import Iterable_Graph
import dataset

import utils.loss_functions
from utils.tools import write_log, set_seed_everything
from utils.tools import prepare_target_Rall, prepare_target_T_Rall, find_not_all_nan_times_new, derive_train_val_idxs_new, derive_train_val_idxs_CORDEX
from utils.tools import derive_qmse_bins, compute_input_statistics, standardize_input
from utils.train_test import Trainer
from accelerate import Accelerator

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--target_file', type=str, default=None)
parser.add_argument('--target_var', type=str, default='tasmax')
parser.add_argument('--procedure', type=str, default='z-score')
parser.add_argument('--graph_file', type=str, default=None) 

parser.add_argument('--out_checkpoint_file', type=str, default="checkpoint.pth")

parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')
parser.add_argument('--make_val_plots', action='store_true')
parser.add_argument('--no-make_val_plots', dest='make_val_plots', action='store_false')

#-- training hyperparameters
parser.add_argument('--experiment', type=str, default='ESD_pseudo_reality')
parser.add_argument('--epochs', type=int, default=15, help='number of total training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size (global)')
parser.add_argument('--step_size', type=int, default=10, help='scheduler step size (global)')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (wd)')
parser.add_argument('--lr_scheduler', type=str, default="StepLR")

parser.add_argument('--loss_fn', type=str, default="mse_loss")
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--seed', type=int)
parser.add_argument('--n_gpu', type=int, default=4)

parser.add_argument('--model_type', type=str)
parser.add_argument('--model_name', type=str, default='P_GNN4CD_model')
parser.add_argument('--dataset_name', type=str, default='Dataset_Graph')
parser.add_argument('--collate_name', type=str)
parser.add_argument('--seq_l', type=int)

#-- start and end training dates
parser.add_argument('--train_month_start', type=int)
parser.add_argument('--train_day_start', type=int)
parser.add_argument('--train_month_end', type=int)
parser.add_argument('--train_day_end', type=int)


#-- wandb configuration
parser.add_argument('--wandb_api_key', type=str)
parser.add_argument('--wandb_project_name', type=str)
parser.add_argument('--wandb_username', type=str)
parser.add_argument('--wandb_team', type=str)


if __name__ == '__main__':

    args = parser.parse_args()

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

    training_experiment= args.experiment
    if training_experiment == 'ESD_pseudo_reality':
        period_training = '1961-1980'
        train_year_start=1961
        train_year_end=1980
        first_year=1961
        validation_years=1975#change with CORDEX validation years [1965,1975]
    elif training_experiment == 'Emulator_hist_future':
        period_training = '1961-1980_2080-2099'

    
    number_valyears=1 #len(validation_years)

    
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

    #-----------------------------------------------------
    #-------------------- LOAD THE DATA ------------------
    #-----------------------------------------------------

    # Load the graph and target files
    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    with open(args.input_path+args.target_file, 'rb') as f:
        target_train = pickle.load(f)
        
    target_train = target_train.T

    #-----------------------------------------------------
    #--------------- MODEL, LOSS, OPTIMIZER --------------
    #-----------------------------------------------------

    models = importlib.import_module(f"models.{args.model_name}")
    Model = getattr(models, args.model_name)
    model = Model(seq_l=args.seq_l+1)

    # Loss
 
    loss_fn = getattr(utils.loss_functions, args.loss_fn)()

    #-----------------------------------------------------
    #--------------- TARGET AND INDEXES ------------------
    #-----------------------------------------------------

    # Prepapre the target for the Rall model
    if args.target_var=="precipitation":
        target_train = prepare_target_Rall(target_train, threshold = 0.1)
    else:
        target_train = prepare_target_T_Rall(target_train, procedure=args.procedure)
    #CHECKING NAN values 

    # Identify the indexes for which at least one node value is not nan
    idxs_not_all_nan = find_not_all_nan_times_new(target_train)

    #write_log(f"\nAfter removing all nan time indexes, {len(idxs_not_all_nan)}" +
    #        f" time indexes are considered ({(len(idxs_not_all_nan) / target_train.shape[1] * 100):.1f} " +
    #        "% of initial ones).", args, accelerator, 'a')
    
    
    # Derive the train and validation indexes

    train_idxs, val_idxs = derive_train_val_idxs_new(train_year_start, args.train_month_start, args.train_day_start, train_year_end, args.train_month_end,
                         args.train_day_end, first_year, args.model_name, None, validation_years, args=args, accelerator=accelerator)
                         
    #train_idxs, val_idxs = derive_train_val_idxs_CORDEX(
    #    train_year_start, args.train_month_start, args.train_day_start, train_year_end,
    #    args.train_month_end, args.train_day_end, first_year, args.model_name, None,
    #    validation_years, number_valyears, args=args, accelerator=accelerator)
    
    
    write_log(f"\nTrain from {args.train_day_start}/{args.train_month_start}/{train_year_start} to " +
                f"{args.train_day_end}/{args.train_month_end}/{train_year_end} with validation years " +
                f"{validation_years}", args, accelerator, 'a')
    
    
    # Check that the size of the train and val idxs is multiple of n_gpu
    # to avoid issues with accelerator.gather_for_metrics; if not, simply
    # discard the last idxs to obtain a multiple of n_gpu
    #tail_train_idxs = len(train_idxs) % args.n_gpu
    #tail_val_idxs = len(val_idxs) % args.n_gpu
    #if tail_train_idxs != 0:
    #    train_idxs = train_idxs[:-tail_train_idxs]
    #if tail_val_idxs != 0:
    #    val_idxs = val_idxs[:-tail_val_idxs]
        
    # Compute the weights for the regressor
    
    target_bins = derive_qmse_bins(target_train, train_idxs, args, accelerator, threshold=0.1)
    write_log(f'\nPrecipitation bins = {target_bins}', args, accelerator, 'a')


    # Compute the training statistics and standardize the training data
    means_low, stds_low, means_high, stds_high = compute_input_statistics(
        low_high_graph['low'].x[:,train_idxs,:], low_high_graph['high'].x, args, accelerator)
    
    low_high_graph['low'].x, low_high_graph['high'].x = standardize_input(
        low_high_graph['low'].x, low_high_graph['high'].x, means_low, stds_low, means_high, stds_high, args, accelerator) # num_nodes, time, vars, levels

    # Flatten the training tensor
    low_high_graph['low'].x = torch.flatten(low_high_graph['low'].x, start_dim=2, end_dim=-1)   # num_nodes, time, vars*levels

    #-----------------------------------------------------
    #-------------- DATASET AND DATALOADER ---------------
    #-----------------------------------------------------
    
    Dataset_Graph = getattr(dataset, args.dataset_name)
    
    if "quantized" in args.loss_fn:
        dataset_graph = Dataset_Graph(targets=target_train,
            w=target_bins, graph=low_high_graph, model_name=args.model_name, seq_l=args.seq_l)
    else:
        dataset_graph = Dataset_Graph(targets=target_train,
            graph=low_high_graph, model_name=args.model_name, seq_l=args.seq_l)

    
    
    
    # Define the custom collate function
    custom_collate_fn = getattr(dataset, args.collate_name)
        
    # Define the custom samplers
    sampler_graph_train = Iterable_Graph(dataset_graph=dataset_graph, shuffle=True, idxs_vector=train_idxs)
    sampler_graph_val = Iterable_Graph(dataset_graph=dataset_graph, shuffle=False, idxs_vector=val_idxs, t_offset=val_idxs.min())

    write_log(f'\nTrainset size = {train_idxs.shape[0]}, validation set size = {val_idxs.shape[0]}.', args, accelerator, 'a')

    # Define the dataloaders
    dataloader_train = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph_train, collate_fn=custom_collate_fn)

    dataloader_val = torch.utils.data.DataLoader(dataset_graph, batch_size=10, num_workers=0,
                    sampler=sampler_graph_val, collate_fn=custom_collate_fn)

    if accelerator is None or accelerator.is_main_process:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        write_log(f"\nRAM memory {round((used_memory/total_memory) * 100, 2)} %", args, accelerator, 'a')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)

    
    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    elif args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001, last_epoch=-1)
    else:
        lr_scheduler = None

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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_log(f"\nTotal number of trainable parameters: {total_params}.", args, accelerator, 'a')

#-----------------------------------------------------
#----------------------- TRAIN -----------------------
#-----------------------------------------------------

    write_log(f"\nUsing lr={optimizer.param_groups[0]['lr']:.8f}, " +
                f"weight decay = {args.weight_decay} and epochs={args.epochs}." + 
                f"loss: {loss_fn}", args, accelerator, 'a') 
    if accelerator is None:
        write_log(f"\nModel = {args.model_name}, batch size = {args.batch_size}", args, accelerator, 'a') 
    else:
        write_log(f"\nModel = {args.model_name}, batch size = {args.batch_size*torch.cuda.device_count()}", args, accelerator, 'a')

    start = time.time()

    trainer = Trainer()
    
    trainer.train_R_Rall(model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=epoch_start)
    end = time.time()

    write_log(f"\nCompleted in {end - start} seconds.\nDONE!", args, accelerator, 'a')
    

