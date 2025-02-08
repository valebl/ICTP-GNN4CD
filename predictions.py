import numpy as np
import pickle
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
import sys

from accelerate import Accelerator

from torch_geometric.data import HeteroData

import models
import dataset
from dataset import Dataset_Graph, Iterable_Graph

from utils import date_to_idxs, Tester, set_seed_everything
from torch_geometric.utils import degree

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--checkpoint', type=str)
parser.add_argument('--checkpoint_reg', type=str, default=None)
parser.add_argument('--checkpoint_cl', type=str, default=None)
parser.add_argument('--output_file', type=str, default="G_predictions.pkl")

parser.add_argument('--graph_file', type=str, default=None) 
parser.add_argument('--target_file', type=str, default="pr_target.pkl") 
parser.add_argument('--model_type', type=str, default=None)
parser.add_argument('--model', type=str, default=None) 
parser.add_argument('--model_reg', type=str, default=None) 
parser.add_argument('--model_cl', type=str, default=None) 
parser.add_argument('--dataset_name', type=str, default=None) 
parser.add_argument('--mode', type=str, default="cl_reg") 

#-- start and end training dates
parser.add_argument('--test_year_start', type=int)
parser.add_argument('--test_month_start', type=int)
parser.add_argument('--test_day_start', type=int)
parser.add_argument('--test_year_end', type=int)
parser.add_argument('--test_month_end', type=int)
parser.add_argument('--test_day_end', type=int)
parser.add_argument('--first_year', type=int)
parser.add_argument('--first_year_input', type=int)

parser.add_argument('--batch_size', type=int)
parser.add_argument('--seed', type=int)

parser.add_argument('--use_accelerate',  action='store_true')
parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')


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

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'w') as f:
            f.write(f"Starting!")

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path+args.log_file, 'w') as f:
            f.write("Starting the testing...")
            f.write(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.")

    test_start_idx, test_end_idx = date_to_idxs(args.test_year_start, args.test_month_start,
                                                args.test_day_start, args.test_year_end, args.test_month_end,
                                                args.test_day_end, args.first_year)
    
    test_start_idx_input, test_end_idx_input = date_to_idxs(args.test_year_start, args.test_month_start,
                                                args.test_day_start, args.test_year_end, args.test_month_end,
                                                args.test_day_end, args.first_year_input)

    #correction for start idxs
    if test_start_idx >= 24:
        test_start_idx = test_start_idx-24
        test_start_idx_input = test_start_idx_input-24
    else:
        with open(args.output_path+args.log_file, 'a') as f:
            f.write(f"\ntest_start_idx={test_start_idx} < 24, thus testing will start from idx {test_start_idx+24}")

    with open(args.input_path+args.target_file, 'rb') as f:
        pr_target = pickle.load(f)

    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    pr_target = pr_target[:,test_start_idx:test_end_idx]

    low_high_graph['low'].x = low_high_graph['low'].x[:,test_start_idx_input:test_end_idx_input,:] 

    Dataset_Graph = getattr(dataset, args.dataset_name)
    
    if args.mode == "cl_reg":
        dataset_graph = Dataset_Graph(targets=None, graph=low_high_graph, model_name=args.model_reg)
    else:
        dataset_graph = Dataset_Graph(targets=None, graph=low_high_graph, model_name=args.model)

    custom_collate_fn = getattr(dataset, 'custom_collate_fn_graph')
        
    sampler_graph = Iterable_Graph(dataset_graph=dataset_graph, shuffle=False)
        
    dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph, collate_fn=custom_collate_fn)
    
    if args.mode == "cl_reg":
        Model_cl = getattr(models, args.model_cl)
        Model_reg = getattr(models, args.model_reg)
        model_cl = Model_cl()
        model_reg = Model_reg()
    else:
        Model = getattr(models, args.model)
        model = Model()

    if accelerator is None:
        if args.mode == "cl_reg":
            checkpoint_cl = torch.load(args.checkpoint_cl, map_location=torch.device('cpu'))
            checkpoint_reg = torch.load(args.checkpoint_reg, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        device = 'cpu'
    else:
        if args.mode == "cl_reg":
            checkpoint_cl = torch.load(args.checkpoint_cl)
            checkpoint_reg = torch.load(args.checkpoint_reg)
        else:
            checkpoint = torch.load(args.checkpoint)
        device = accelerator.device
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write("\nLoading state dict.")
    if args.mode == "cl_reg":
        model_cl.load_state_dict(checkpoint_cl)
        model_reg.load_state_dict(checkpoint_reg)
    else:
        model.load_state_dict(checkpoint)

    if accelerator is not None:
        if args.mode == "cl_reg":
            model_cl, model_reg, dataloader = accelerator.prepare(model_cl, model_reg, dataloader)
        else:
            model, dataloader = accelerator.prepare(model, dataloader)

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f"\nStarting the test, from {int(args.test_day_start)}/{int(args.test_month_start)}/{int(args.test_year_start)} to " +
                    f"{int(args.test_day_end)}/{int(args.test_month_end)}/{int(args.test_year_end)} (from idx {test_start_idx} to idx {test_end_idx}).")

    tester = Tester()

    start = time.time()

    if args.mode == "encoding":
        pr, times = tester.test_encoding(model, dataloader, low_high_graph=low_high_graph, args=args, accelerator=accelerator)
    elif args.mode == "cl_reg":
        pr_cl, pr_reg, times = tester.test_cl_reg(model_cl, model_reg, dataloader, low_high_graph=low_high_graph, args=args, accelerator=accelerator)
    elif args.mode == "reg":
        pr_reg, times = tester.test(model, dataloader, low_high_graph=low_high_graph, args=args, accelerator=accelerator)
    elif args.mode == "cl":
        pr_cl, times = tester.test(model, dataloader, low_high_graph=low_high_graph, args=args, accelerator=accelerator)
    else:
        raise Exception("mode should be: 'reg', 'cl', 'encoding' or 'cl_reg'")

    end = time.time()

    if accelerator is not None:
        accelerator.wait_for_everyone()

        # Gather the values in *tensor* across all processes and concatenate them on the first dimension. Useful to
        # regroup the predictions from all processes when doing evaluation.

        times = accelerator.gather(times).squeeze()
        times, indices = torch.sort(times)

        if args.mode == "encoding":
            pr = accelerator.gather(pr).squeeze().swapaxes(0,1)[:,indices,:]
        elif args.mode == "cl_reg":
            pr_cl = accelerator.gather(pr_cl).squeeze().swapaxes(0,1)[:,indices]
            pr_reg = accelerator.gather(pr_reg).squeeze().swapaxes(0,1)[:,indices]
            #pr_combined = accelerator.gather(pr_combined).squeeze().swapaxes(0,1)[:,indices]
        elif args.mode == "reg":
            pr_reg = accelerator.gather(pr_reg).squeeze().swapaxes(0,1)[:,indices]
        elif args.mode == "cl":
            pr_cl = accelerator.gather(pr_cl).squeeze().swapaxes(0,1)[:,indices]

    data = HeteroData()

    if args.mode == "encoding":
        data.encodings = pr.cpu().numpy()
    elif args.mode == "cl_reg":
        data.pr_cl = pr_cl.cpu().numpy()
        data.pr_reg = pr_reg.cpu().numpy()
        #data.pr_combined = pr_combined.cpu().numpy()  
    elif args.mode == "reg":
        data.pr_reg = pr_reg.cpu().numpy()
    elif args.mode == "cl":
        data.pr_cl = pr_cl.cpu().numpy()
    
    data.pr_target = pr_target[:,24:].cpu().numpy()
    data.times = times.cpu().numpy()
    data["low"].lat = low_high_graph["low"].lat.cpu().numpy()
    data["low"].lon = low_high_graph["low"].lon.cpu().numpy()
    data["high"].lat = low_high_graph["high"].lat.cpu().numpy()
    data["high"].lon = low_high_graph["high"].lon.cpu().numpy()

    degree = degree(low_high_graph['high', 'within', 'high'].edge_index[0], low_high_graph['high'].num_nodes)
    data["high"].degree = degree.cpu().numpy()
    
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, 'a') as f:
            f.write(f"\nDone. Testing concluded in {end-start} seconds.")
            f.write("\nWrite the files.")

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(data, f)

    

  
