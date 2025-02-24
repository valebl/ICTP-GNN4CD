import numpy as np
import pickle
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
import sys
import importlib

from accelerate import Accelerator

from torch_geometric.data import HeteroData

import dataset
from dataset import Dataset_Graph, Iterable_Graph

from utils.utils import date_to_idxs, Tester, set_seed_everything
from torch_geometric.utils import degree

from utils.utils_plots import create_zones, plot_italy, extremes_cmap
from utils.utils_plots import plot_maps, plot_single_map, plot_mean_time_series, plot_seasonal_maps
from utils.utils import date_to_idxs, write_log

import matplotlib
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patches as patches
        

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str, help='path to output directory')
parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

parser.add_argument('--train_path', type=str)
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
parser.add_argument('--test_idxs_file', type=str, default="")

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

parser.add_argument('--make_plots',  action='store_true')
parser.add_argument('--no-make_plots', dest='make_plots', action='store_false')


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

    write_log("Starting the testing...", args, accelerator, 'w')
    write_log(f"Cuda is available: {torch.cuda.is_available()}. There are {torch.cuda.device_count()} available GPUs.", args, accelerator, 'a')

    if args.test_idxs_file == "":
        test_start_idx, test_end_idx = date_to_idxs(args.test_year_start, args.test_month_start,
            args.test_day_start, args.test_year_end, args.test_month_end,
            args.test_day_end, args.first_year)        
        if test_start_idx < 24:
            test_start_idx = 24
        test_idxs = torch.tensor([*range(test_start_idx, test_end_idx)])
        write_log(f"\nUsing the provided start and end test times to derive the test idxs.", args, accelerator, 'a')
    else:
        with open(args.train_path+args.test_idxs_file, 'rb') as f:
            test_idxs = pickle.load(f)
        write_log(f"Using the provided test idxs vector.", args, accelerator, 'a')

    with open(args.input_path+args.target_file, 'rb') as f:
        pr_target = pickle.load(f)

    with open(args.input_path+args.graph_file, 'rb') as f:
        low_high_graph = pickle.load(f)

    Dataset_Graph = getattr(dataset, args.dataset_name)
    
    if args.mode == "cl_reg":
        dataset_graph = Dataset_Graph(targets=None, graph=low_high_graph, model_name=args.model_reg)
    else:
        dataset_graph = Dataset_Graph(targets=None, graph=low_high_graph, model_name=args.model)

    custom_collate_fn = getattr(dataset, 'custom_collate_fn_graph')
        
    sampler_graph = Iterable_Graph(dataset_graph=dataset_graph, shuffle=False, idxs_vector=test_idxs)
        
    dataloader = torch.utils.data.DataLoader(dataset_graph, batch_size=args.batch_size, num_workers=0,
                    sampler=sampler_graph, collate_fn=custom_collate_fn)

    if args.mode == "cl_reg":
        model_file_cl = importlib.import_module(f"models.{args.model_cl}")
        model_file_reg = importlib.import_module(f"models.{args.model_reg}")
        Model_cl = getattr(model_file_cl, args.model_cl)
        Model_reg = getattr(model_file_reg, args.model_reg)
        model_cl = Model_cl()
        model_reg = Model_reg()
    else:
        model_file = importlib.import_module(f"models.{args.model}")
        Model = getattr(model_file, args.model)
        model = Model()

    if accelerator is None:
        if args.mode == "cl_reg":
            checkpoint_cl = torch.load(args.train_path+args.checkpoint_cl, map_location=torch.device('cpu'))
            checkpoint_reg = torch.load(args.train_path+args.checkpoint_reg, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(args.train_path+args.checkpoint, map_location=torch.device('cpu'))
        device = 'cpu'
    else:
        if args.mode == "cl_reg":
            checkpoint_cl = torch.load(args.train_path+args.checkpoint_cl)
            checkpoint_reg = torch.load(args.train_path+args.checkpoint_reg)
        else:
            checkpoint = torch.load(args.train_path+args.checkpoint)
        device = accelerator.device
    
    write_log("\nLoading state dict.", args, accelerator, 'a')
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

    # write_log(f"\nStarting the test, from idx {test_start_idx} to idx {test_end_idx}.", args, accelerator, 'a')

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
    
    data.pr_target = pr_target[:,test_idxs].cpu().numpy()
    data.times = times.cpu().numpy()
    data["low"].lat = low_high_graph["low"].lat.cpu().numpy()
    data["low"].lon = low_high_graph["low"].lon.cpu().numpy()
    data["high"].lat = low_high_graph["high"].lat.cpu().numpy()
    data["high"].lon = low_high_graph["high"].lon.cpu().numpy()

    degree = degree(low_high_graph['high', 'within', 'high'].edge_index[0], low_high_graph['high'].num_nodes)
    data["high"].degree = degree.cpu().numpy()
    
    write_log(f"\nDone. Testing concluded in {end-start} seconds.\nWrite the files.", args, accelerator, 'a')

    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.output_file, 'wb') as f:
            pickle.dump(data, f)


    ###################
    ###--- PLOTS ---###
    ###################
    
    if args.make_plots:

        G = data

        # for compatibility with previous versions of the code
        try:
            G.pr_target
        except:
            G.pr_target = G.pr_gripho
            del G.pr_gripho

        # Remove the nodes with <= 2 nodes (e.g. small islands) where the predictions may be inaccurate
        mask = G["high"].degree > 2 * np.array([~np.isnan(G.pr_target[i,:]).all() for i in range(G.pr_target.shape[0])])

        # Define the vector of lon-lat coordinaes
        pos = np.concatenate((np.expand_dims(G['high'].lon,-1), np.expand_dims(G['high'].lat,-1)), axis=-1)

        # To plot the countourns of Italy
        zones_file='/leonardo_work/ICT24_ESP/SHARED/HiResPrecipNet/Italia.txt'
        zones = create_zones(zones_file=zones_file)

        cmap = 'jet'

        day_start=1
        month_start=1
        year_start=2016
        day_end=31
        month_end=12
        year_end=2016
        
        x_size = 38
        y_size = 44
        xlim = [6.75, 18.50]
        ylim = [36.50, 47.00]
        font_size = 100
        font_size_title = 120
        cbar_y = 0.95
        cbar_title_size = 100
        dpi = 120
        
        threshold = 0.1
        mask_nan = np.isnan(G.pr_target)
        
        G.pr_target[G.pr_target < threshold] = 0
        G.pr_target = np.round(G.pr_target, decimals=1)
        G.pr_target[mask_nan] = np.nan        
        
        # Define the true classifier labels
        y_target_cl = np.where(G.pr_target < threshold, 0.0, 1.0)
        y_target_cl[mask_nan] = np.nan
        
        mask_N = y_target_cl==0
        mask_P = y_target_cl==1
        
        pr_reg = np.expm1(G.pr_reg)
        G.pr_cl = y_target_cl
        
        G.pr = pr_reg * G.pr_cl
        # G.pr[G.pr >= 200] = 200
        G.pr[G.pr < threshold] = 0
        
        G.pr_cl[mask_nan] = np.nan
        G.pr[mask_nan] = np.nan

        mask_all = ~mask_nan
        mask_all[0] = mask_all[0] * mask[0]

        y_true_cl = y_target_cl[mask_all].flatten()
        y_pred_cl = G.pr_cl[mask_all].flatten()
        
        corrects = (y_true_cl == y_pred_cl)
        acc = np.nansum(corrects) / len(corrects) * 100
        acc_0 = np.nansum(corrects[y_true_cl==0]) / np.nansum(y_true_cl==0) * 100
        acc_1 = np.nansum(corrects[y_true_cl==1]) / np.nansum(y_true_cl==1) * 100            
        write_log(f"\nClassifier\nAccuracy: {acc:.2f}\nAccuracy on class 0: {acc_0:.2f}\nAccuracy on class 1: {acc_1:.2f}", args, accelerator, 'a')

        # Apply the mask
        G.pr_cl = G.pr_cl[mask,:]
        G.pr = G.pr[mask,:]
        G.pr_target = G.pr_target[mask,:]
        pos = pos[mask,:]
        y_target_cl = y_target_cl[mask,:]

        make_seasonal_plots = True
        if args.test_idxs_file is not None:  
            djf_start, djf_end = date_to_idxs(year_start=2015, month_start=12, day_start=1, year_end=2016,
                                              month_end=2, day_end=29, first_year=2015, first_month=12, first_day=1)
            mam_start, mam_end = date_to_idxs(year_start=2016, month_start=3, day_start=1, year_end=2016,
                                              month_end=5, day_end=31, first_year=2015, first_month=12, first_day=1)
            jja_start, jja_end = date_to_idxs(year_start=2016, month_start=6, day_start=1, year_end=2016,
                                              month_end=8, day_end=31, first_year=2015, first_month=12, first_day=1)
            son_start, son_end = date_to_idxs(year_start=2016, month_start=9, day_start=1, year_end=2016,
                                              month_end=11, day_end=30, first_year=2015, first_month=12, first_day=1)
        else:
            djf_start = 0
            if test_idxs.shape[0] == 8760:
                djf_end = (31 + 31 + 28) * 24
            elif test_idxs.shape[0] == 8784:
                djf_end = (31 + 31 + 29) * 24
            else:
                write_log("Cannot identify the months, thus seasonal plots are skipped.")
                make_seasonal_plots = False
            mam_start = djf_end
            mam_end = mam_start + (31 + 30 + 31) * 24
            jja_start = mam_end
            jja_end = jja_start + (31 + 31 + 30) * 24
            son_start = jja_end
            son_end = son_start + (30 + 31 + 30) * 24

        if make_seasonal_plots:
            pr_pred_seasons = []
            pr_target_seasons = []
            pr_pred_seasons.append(G.pr[:,djf_start: djf_end])
            pr_pred_seasons.append(G.pr[:,mam_start: mam_end])
            pr_pred_seasons.append(G.pr[:,jja_start: jja_end])
            pr_pred_seasons.append(G.pr[:,son_start: son_end])
            
            pr_target_seasons.append(G.pr_target[:,djf_start: djf_end])
            pr_target_seasons.append(G.pr_target[:,mam_start: mam_end])
            pr_target_seasons.append(G.pr_target[:,jja_start: jja_end])
            pr_target_seasons.append(G.pr_target[:,son_start: son_end])

        G.pr_cl = G.pr_cl[:,24*31:]
        G.pr = G.pr[:,24*31:]
        G.pr_target = G.pr_target[:,24*31:]
        y_target_cl = y_target_cl[:,24*31:]

        # Classifier
        plot_maps(pos, G.pr_cl, y_target_cl, pr_min=0, aggr=np.nansum, pr_max=1500,
            title=f"Classifier: number of hours with pr>=0.1mm - Year 2016", legend_title="[h]", cmap='jet', save_path=None, save_file_name=None, zones=zones,
            x_size=x_size, y_size=y_size, font_size_title=font_size_title, font_size=font_size, cbar_title_size=cbar_title_size,
            ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}classifier.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # Cumulative precipitation
        plot_maps(pos, G.pr, G.pr_target, pr_min=0, aggr=np.nansum, pr_max=2750,
            title=f"Cumulative precipitation - Year 2016", legend_title="[mm]", cmap='jet', save_path=None, save_file_name=None, zones=zones,
            x_size=x_size, y_size=y_size, font_size_title=font_size_title, font_size=font_size, cbar_title_size=cbar_title_size,
            ylim=ylim, xlim=xlim, cbar_y=cbar_y, subtitle_x=0.55)
        plt.savefig(f'{args.output_path}cumulative.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        pr_reg_bias = np.nansum(G.pr, axis=1) - np.nansum(G.pr_target, axis=1)
        pr_reg_bias_percentage = pr_reg_bias / np.nansum(G.pr_target, axis=1) * 100

        plot_single_map(pos, pr_reg_bias, pr_min=-1000, aggr=None, pr_max=1000,
            title=f"Cumulative precipitation bias - Year 2016", legend_title="[mm/h]", subtitle_y=0.98, subtitle_x=0.5,
            cmap='BrBG', save_path=None, save_file_name=None, zones=zones, x_size=x_size, y_size=y_size, 
            font_size_title=85, font_size=font_size, cbar_title_size=100, ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}cumulative_bias.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        plot_single_map(pos, pr_reg_bias_percentage, pr_min=-100, aggr=None, pr_max=100,
            title=f"Cumulative precipitation percentage bias - Year 2016", legend_title="[%]", subtitle_y=0.98, subtitle_x=0.5,
            cmap='BrBG', save_path=None, save_file_name=None, zones=zones, x_size=x_size, y_size=y_size, 
            font_size_title=85, font_size=font_size, cbar_title_size=100, ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}cumulative_bias_percentage.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # Average precipitation
        plot_maps(pos, G.pr, G.pr_target, pr_min=0, aggr=np.nanmean, pr_max=0.3,
            title=f"Average precipitation - Year 2016", legend_title="[mm/h]", cmap='jet', save_path=None, save_file_name=None, zones=zones,
            x_size=x_size, y_size=y_size, font_size_title=font_size_title, font_size=font_size, cbar_title_size=cbar_title_size,
            ylim=ylim, xlim=xlim, cbar_y=cbar_y, subtitle_x=0.55)
        plt.savefig(f'{args.output_path}average.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        pr_bias_avg = np.nanmean(G.pr, axis=1) - np.nanmean(G.pr_target, axis=1)
        pr_bias_percentage_avg = pr_bias_avg / np.nanmean(G.pr_target, axis=1) * 100

        plot_single_map(pos, pr_bias_avg, pr_min=-0.15, aggr=None, pr_max=0.15,
            title=f"Average precipitation bias - Year 2016", legend_title="[mm/h]", subtitle_y=0.98, subtitle_x=0.5,
            cmap='BrBG', save_path=None, save_file_name=None, zones=zones, x_size=x_size, y_size=y_size, 
            font_size_title=85, font_size=font_size, cbar_title_size=100, ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}average_bias.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        plot_single_map(pos, pr_bias_percentage_avg, pr_min=-100, aggr=None, pr_max=100,
            title=f"Average precipitation percentage bias - Year 2016", legend_title="[%]", subtitle_y=0.98, subtitle_x=0.5,
            cmap='BrBG', save_path=None, save_file_name=None, zones=zones, x_size=x_size, y_size=y_size, 
            font_size_title=85, font_size=font_size, cbar_title_size=100, ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}average_bias_percentage.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # Daily maps

        # Derive the daily aggregated data
        shape_24 = (G.pr.shape[0], G.pr.shape[1] // 24, 24)
        pr_daily = G.pr.reshape(shape_24)
        pr_target_daily = G.pr_target.reshape(shape_24)

        pr_daily_avg = np.nanmean(np.sum(pr_daily, axis=2).squeeze(), axis=1)
        pr_target_daily_avg = np.nanmean(np.sum(pr_target_daily, axis=2).squeeze(), axis=1)
        
        pr_daily_avg_geq3mm = np.nanmean(np.sum(pr_daily*(pr_target_daily>=3), axis=2).squeeze(), axis=1)
        pr_target_daily_avg_geq3mm = np.nanmean(np.sum(pr_target_daily*(pr_target_daily>=3), axis=2).squeeze(), axis=1)
        
        plot_maps(pos, pr_daily_avg, pr_target_daily_avg, aggr=None, title=f"Average daily precipitation - Year 2016",
                  legend_title="[mm/d]", cmap_type="custom_jet_discrete_avg", save_path=None,
                  save_file_name=None, zones=zones, x_size=x_size, y_size=y_size, font_size_title=font_size_title,
                  font_size=font_size, cbar_title_size=cbar_title_size, ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}average_daily.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        plot_maps(pos, pr_daily_avg_geq3mm, pr_target_daily_avg_geq3mm, aggr=None, title=f"Average daily precipitation - Year 2016",
                  legend_title="[mm/d]", cmap_type="custom_jet_discrete_avg_limits", save_path=None, pr_max=5,
                  bounds=np.array([0.0, 0.1, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
                  save_file_name=None, zones=zones, x_size=x_size, y_size=y_size, font_size_title=font_size_title,
                  font_size=font_size, cbar_title_size=cbar_title_size, ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}average_daily_geq3mm.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        # ### Extreme event - North Italy 22 Nov 2016 - 25 Nov 2016
        if args.test_idxs_file is None:
            cmap_extreme = extremes_cmap()
    
            start, end = date_to_idxs(year_start=2016, month_start=11, day_start=22, year_end=2016, month_end=11, day_end=25,
                             first_year=2016, first_month=1, first_day=1)
            
            plot_maps(pos, G.pr[:,start:end], G.pr_target[:,start:end], pr_min=0.0, pr_max=400,  legend_title="[mm]",
                aggr=np.nansum, title=f"Cumulative precipitation (from {22}/{11}/{2016} to {25}/{11}/{2016})", cmap=cmap_extreme, subtitle_x=0.55,
                save_path=None, save_file_name=None, zones=zones, x_size=x_size, y_size=y_size, font_size_title=font_size_title, font_size=font_size)
            plt.savefig(f'{args.output_path}extreme.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
            plt.close()
            
            write_log(f"Exreme event - GNN4CD max={np.nanmax(G.pr[:,start:end])}, GRIPHO max={np.nanmax(G.pr_target[:,start:end])}", args, accelerator, 'a')

        # Seasonal results

        if make_seasonal_plots: 
            plot_seasonal_maps(pos, pr_pred_seasons, zones=zones, pr_min=0.1, pr_max=500, aggr=np.nansum,
                               title='Cumulative precipitation for 2016 seasons - GNN4CD')
            plt.savefig(f'{args.output_path}seasons_gnn4cd.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
            plt.close()
            plot_seasonal_maps(pos, pr_target_seasons, zones=zones, pr_min=0.1, pr_max=500, aggr=np.nansum,
                               title='Cumulative precipitation for 2016 seasons - OBSERVATION')
            plt.savefig(f'{args.output_path}seasons_gripho.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
            plt.close()
            
            # Distributions
            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(28,7))
            ax_list = [ax[0], ax[1], ax[2], ax[3]]
            text_list = ['DJF', 'MAM', 'JJA', 'SON']
            plt.rcParams.update({'font.size': 14})
            
            for s in range(4):
            
                axi = ax_list[s]
                y = pr_target_seasons[s].flatten()
                pr = pr_pred_seasons[s].flatten()
                binwidth = 1
            
                y = y[y>0]
                pr = pr[pr>0]
            
                bins_max_y = 160 #min(200, int(np.nanmax(y)))
                bins_max_pr = 160 #bins_max_y #min(200, int(np.nanmax(pr)))
                                
                hist_y, bin_edges_y = np.histogram(y, bins=np.arange(0,150,0.5), range=[int(np.nanmin(y)), bins_max_y+binwidth], density=True)
                hist_pr, bin_edges_pr = np.histogram(pr, bins=np.arange(0,150,0.5), range=[int(np.nanmin(y)), bins_max_y+binwidth], density=True)
                
                axi.scatter(bin_edges_y[:-1], hist_y, label='GRIPHO', color="darkturquoise", alpha=0.5)
                axi.scatter(bin_edges_pr[:-1], hist_pr, label='DL-MODEL', color="indigo", alpha=0.5)
                axi.set_yscale('log')
                axi.set_xscale('log')
                axi.set_xlabel('precipitation intensity [mm/hr]')
                axi.set_ylabel('pdf')
                axi.set_title(text_list[s], fontsize=18)
                axi.legend()
            
            plt.savefig(f'{args.output_path}seasonal_pdfs.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
            plt.close()
    
        # Time series
        rmse, rmse_perc = plot_mean_time_series(pos, G.pr_target, G.pr, points=np.arange(G.pr_target.shape[0]), aggr=np.nanmean, title="")
        plt.savefig(f'{args.output_path}time_series.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # Diurnal Cycles

        # Precipitation average

        if make_seasonal_plots:
            pr_pred_seasons_daily_cycle = np.zeros((4,24))
            for s in range(4):
                pr_season = pr_pred_seasons[s]
                for i in range(0,24):
                    pr_pred_seasons_daily_cycle[s,i] = np.nanmean(pr_season[:,i::24])
    
            pr_gripho_seasons_daily_cycle = np.zeros((4,24))
            for s in range(4):
                pr_season = pr_target_seasons[s]
                for i in range(0,24):
                    pr_gripho_seasons_daily_cycle[s,i] = np.nanmean(pr_season[:,i::24])
    
            points = np.arange(G.pr_target.shape[0])
            
            text_list = ['DJF', 'MAM', 'JJA', 'SON']
            plt.rcParams.update({'font.size': 25})
            
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,18))
            
            ax_list = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]
            
            for s in range(4):
            
                pr_mean = pr_gripho_seasons_daily_cycle[s]
                pr_pred_mean = pr_pred_seasons_daily_cycle[s]
            
                n = 25
                ax_list[s].plot(range(1,n), pr_pred_mean, label='GNN4CD R', linestyle='-', linewidth=2, color='red')
                ax_list[s].plot(range(1,n), pr_mean, label='GRIPHO', linestyle=':', linewidth=2, color='black')
                ax_list[s].set_title(text_list[s], fontsize=45)
                ax_list[s].set_ylabel("pr [mm/h]", fontsize=40)
                ax_list[s].set_xlabel("time [h]", fontsize=40)
                ax_list[s].set_ylim([0,0.30])
                # ax_list[s].set_xlim([0,24])
                ax_list[s].set_xticks(ticks=range(0,n,6))
                ax_list[s].grid(which='major', color='lightgrey')
            
            plt.suptitle("Average", y=1, fontsize=40)
            plt.legend(loc='upper left', prop={'size': 30})
            plt.tight_layout()
            plt.savefig(f'{args.output_path}diurnal_average.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
            plt.close()
           
            pr_pred_seasons_daily_cycle_intensity = np.zeros((4,24))
            pr_pred_seasons_daily_cycle_frequency = np.zeros((4,24))
            for s in range(4):
                pr_season = pr_pred_seasons[s]
                for i in range(0,24):
                    pr_pred_seasons_daily_cycle_intensity[s,i] = np.nanmean(pr_season[:,i::24][pr_season[:,i::24]>=0.1])
                    pr_pred_seasons_daily_cycle_frequency[s,i] = (pr_season[:,i::24]>=0.1).sum() / pr_season[:,i::24].flatten().shape[0] * 100
            
            pr_gripho_seasons_daily_cycle_intensity = np.zeros((4,24))
            pr_gripho_seasons_daily_cycle_frequency = np.zeros((4,24))
            for s in range(4):
                pr_season = pr_target_seasons[s]
                for i in range(0,24):
                    pr_gripho_seasons_daily_cycle_intensity[s,i] = np.nanmean(pr_season[:,i::24][pr_season[:,i::24]>=0.1])
                    pr_gripho_seasons_daily_cycle_frequency[s,i] = (pr_season[:,i::24]>=0.1).sum() / pr_season[:,i::24].flatten().shape[0] * 100
            
            # Precipitation intensity
            
            points = np.arange(G.pr_target.shape[0])
            
            text_list = ['DJF', 'MAM', 'JJA', 'SON']
            plt.rcParams.update({'font.size': 25})
            
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,18))
            
            ax_list = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]
            
            for s in range(4):
            
                pr_mean = pr_gripho_seasons_daily_cycle_intensity[s]
                pr_pred_mean = pr_pred_seasons_daily_cycle_intensity[s]
            
                n = 25
                ax_list[s].plot(range(1,n), pr_pred_mean, label='GNN4CD R', linestyle='-', linewidth=2, color='red')
                ax_list[s].plot(range(1,n), pr_mean, label='GRIPHO', linestyle=':', linewidth=2, color='black')
                ax_list[s].set_title(text_list[s], fontsize=45)
                ax_list[s].set_ylabel("pr [mm/h]", fontsize=40)
                ax_list[s].set_xlabel("time [h]", fontsize=40)
                ax_list[s].set_ylim([0.5,3.5])
                # ax_list[s].set_xlim([0,25])
                ax_list[s].set_xticks(ticks=range(0,n,6))
                ax_list[s].grid(which='major', color='lightgrey')
            
            plt.suptitle("Intensity", y=1, fontsize=40)
            plt.legend(loc='upper left', prop={'size': 30})
            plt.tight_layout()
            plt.savefig(f'{args.output_path}diurnal_intensity.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
            plt.close()

            # Precipitation frequency
            
            points = np.arange(G.pr_target.shape[0])
            
            text_list = ['DJF', 'MAM', 'JJA', 'SON']
            plt.rcParams.update({'font.size': 25})
            
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,18))
            
            ax_list = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]
            
            for s in range(4):
            
                pr_mean = pr_gripho_seasons_daily_cycle_frequency[s]
                pr_pred_mean = pr_pred_seasons_daily_cycle_frequency[s]
            
                n = 25
                ax_list[s].plot(range(1,n), pr_pred_mean, label='GNN4CD RC', linestyle='-', linewidth=2, color='red')
                ax_list[s].plot(range(1,n), pr_mean, label='GRIPHO', linestyle=':', linewidth=2, color='black')
                ax_list[s].set_title(text_list[s], fontsize=45)
                ax_list[s].set_ylabel("pr [mm/h]", fontsize=40)
                ax_list[s].set_xlabel("time [h]", fontsize=40)
                ax_list[s].set_ylim([0,20])
                # ax_list[s].set_xlim([0,25])
                ax_list[s].set_xticks(ticks=range(0,n,6))
                ax_list[s].grid(which='major', color='lightgrey')
            
            plt.suptitle("Frequency", y=1, fontsize=40)
            plt.legend(loc='upper left', prop={'size': 30})
            plt.tight_layout()
            plt.savefig(f'{args.output_path}diurnal_frequency.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
            plt.close()

        # PDF
        
        plt.rcParams.update({'font.size': 18})
        y = (G.pr_target).flatten()
        pr = (G.pr).flatten()
        binwidth = 0.5
        
        bins_max_y = min(200, int(np.nanmax(y)))
        bins_max_pr = bins_max_y
                         
        fig, ax = plt.subplots(figsize=(8,8))
        _ = plt.hist(y, bins=np.arange(int(np.nanmin(y)), bins_max_y+binwidth, binwidth), facecolor='darkturquoise',
                     alpha=0.4, density=True, label='OBSERVATIONS', edgecolor='k') 
        _ = plt.hist(pr, bins=np.arange(int(np.nanmin(y)), bins_max_y+binwidth, binwidth), facecolor='indigo',
                     alpha=0.4, density=True, label='PREDICTIONS', edgecolor='k')  # arguments are passed to np.histogram
        ax.set_yscale('log')
        ax.set_xlabel('precipitation intensity [mm/hr]')
        ax.set_ylabel('count (normalized)')
        plt.legend()

        y = (G.pr_target).flatten()
        pr = (G.pr).flatten()

        p99 = np.nanpercentile(y, q=99)
        p999 = np.nanpercentile(y, q=99.9)
        
        p99_pred = np.nanpercentile(pr, q=99)
        p999_pred = np.nanpercentile(pr, q=99.9)
        
        plt.rcParams.update({'font.size': 18})
        
        hist_y, bin_edges_y = np.histogram(y, bins=np.arange(0,200,0.5).astype(np.float32), density=False)
        hist_pr, bin_edges_pr = np.histogram(pr, bins=np.arange(0,200,0.5).astype(np.float32), density=False)
        
        Ntot_y = hist_y.sum()
        Ntot_pr = hist_pr.sum()
        
        bin_edges_y_centre = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
        bin_edges_pr_centre = (bin_edges_pr[:-1] + bin_edges_pr[1:]) / 2
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.scatter(bin_edges_y_centre, hist_y/Ntot_y, color='darkturquoise', s=80, label="GRIPHO", alpha=0.4, zorder=2)
        ax.scatter(bin_edges_pr_centre, hist_pr/Ntot_pr, color='indigo', s=80, label="GNN4CD R", alpha=0.4, zorder=2)
        ax.plot([p99, p99], [0,50], '-', color='gold', label='GRIPHO - p99', zorder=1)
        ax.plot([p999, p999], [0,50], '-r', label='GRIPHO - p99.9', zorder=1)
        ax.plot([p99_pred, p99_pred], [0,50], ':', color='gold', label='GNN4CD R - p99', zorder=1)
        ax.plot([p999_pred, p999_pred], [0,50], ':r', label='GNN4CD R - p99.9', zorder=1)
        l = plt.legend(loc='upper right', facecolor='white', framealpha=1, fontsize=16)
        ax.set_ylim([10**(-10),5])
        # ax.set_xlim([0,5])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', zorder=0)
        ax.set_xlabel('precipitation [mm/h]', fontsize=22)
        ax.set_ylabel('frequency', fontsize=22)
        plt.savefig(f'{args.output_path}pdf.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # binsize of 0.1mm
        plt.rcParams.update({'font.size': 18})
        
        hist_y, bin_edges_y = np.histogram(y, bins=np.arange(0,200,0.1).astype(np.float32), density=False)
        hist_pr, bin_edges_pr = np.histogram(pr, bins=np.arange(0,200,0.1).astype(np.float32), density=False)
        
        Ntot_y = hist_y.sum()
        Ntot_pr = hist_pr.sum()
        
        bin_edges_y_centre = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
        bin_edges_pr_centre = (bin_edges_pr[:-1] + bin_edges_pr[1:]) / 2
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.scatter(bin_edges_pr_centre, hist_pr/Ntot_pr, color='indigo', s=20, label="GNN4CD", alpha=0.2, zorder=2)
        ax.scatter(bin_edges_y_centre, hist_y/Ntot_y, color='darkturquoise', s=20, label="GRIPHO", alpha=0.2, zorder=2)
        ax.plot([p99, p99], [0,50], '-', color='gold', label='GRIPHO - 99th', zorder=1)
        ax.plot([p999, p999], [0,50], '-r', label='GRIPHO - 99.9th', zorder=1)
        ax.plot([p99_pred, p99_pred], [0,50], ':', color='gold', label='GNN4CD - 99th', zorder=1)
        ax.plot([p999_pred, p999_pred], [0,50], ':r', label='GNN4CD - 99.9th', zorder=1)
        # ax.plot([0.2, 0.2], [0,50], '-', color='green', label='0.2 mm', zorder=1)
        l = plt.legend(loc='upper right', facecolor='white', framealpha=1, fontsize=12)
        ax.set_ylim([10**(-10),5])
        # ax.set_xlim([0,5])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', zorder=0)
        ax.set_xlabel('precipitation [mm/hr]')
        ax.set_ylabel('frequency')
        plt.savefig(f'{args.output_path}pdf_01mm.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # binsize of 0.5mm and step histogram

        hist_y, bin_edges_y = np.histogram(y, bins=np.arange(0,200,0.5).astype(np.float32), density=False)
        hist_pr, bin_edges_pr = np.histogram(pr, bins=np.arange(0,200,0.5).astype(np.float32), density=False)
        
        Ntot_y = hist_y.sum()
        Ntot_pr = hist_pr.sum()
        
        bin_edges_y_centre = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
        bin_edges_pr_centre = (bin_edges_pr[:-1] + bin_edges_pr[1:]) / 2
        
        plt.rcParams.update({'font.size': 18})
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.step(bin_edges_pr_centre[1:], hist_pr[1:], color='indigo', where="mid", linewidth=1, label=f"GNN4CD")
        ax.step(bin_edges_y_centre[1:], hist_y[1:], color='darkturquoise', where="mid", linewidth=1, label=f"GRIPHO")
        l = plt.legend(loc='upper right', facecolor='white', framealpha=1, fontsize=10)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', zorder=0)
        ax.set_xlabel('precipitation intensity [mm/hr]')
        ax.set_ylabel('frequency')
        plt.savefig(f'{args.output_path}pdf_05mm_step.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # 99 and 99.9 percentiles

        p99_y = np.nanpercentile(G.pr_target, q=99, axis=1)
        p99_pred = np.nanpercentile(G.pr, q=99, axis=1)

        plot_maps(pos, p99_pred, p99_y, pr_min=0, aggr=None, pr_max=7, subtitle_x=0.5,
            title=f"99 percentile - Year 2016", legend_title="[mm/h]", cmap='jet', save_path=None, save_file_name=None, zones=zones,
            x_size=x_size, y_size=y_size, font_size_title=font_size_title, font_size=font_size, cbar_title_size=cbar_title_size,
            ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}p99.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        p99_bias = p99_pred - p99_y
        p99_bias_percentile = p99_bias / p99_y * 100
        
        plot_single_map(pos, p99_bias, pr_min=-3, aggr=None, pr_max=3,
            title=f"P99 precipitation bias - Year 2016", legend_title="[mm/h]", subtitle_y=0.98, subtitle_x=0.5,
            cmap='BrBG', save_path=None, save_file_name=None, zones=zones, x_size=x_size, y_size=y_size, 
            font_size_title=85, font_size=font_size, cbar_title_size=100, ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}p99_bias.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        plot_single_map(pos, p99_bias_percentile, pr_min=-100, aggr=None, pr_max=100,
            title=f"P99 precipitation percentage bias - Year 2016", legend_title="[%]", subtitle_y=0.98, subtitle_x=0.5,
            cmap='BrBG', save_path=None, save_file_name=None, zones=zones, x_size=x_size, y_size=y_size, 
            font_size_title=85, font_size=font_size, cbar_title_size=100, ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}p99_bias_percentage.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        p999_y = np.nanpercentile(G.pr_target, q=99.9, axis=1)
        p999_pred = np.nanpercentile(G.pr, q=99.9, axis=1)

        plot_maps(pos, p999_pred, p999_y, pr_min=0, aggr=None, pr_max=20,  subtitle_x=0.5,
            title=f"99.9 percentile - Year 2016", legend_title="[mm/h]", cmap='jet', save_path=None, save_file_name=None, zones=zones,
            x_size=x_size, y_size=y_size, font_size_title=font_size_title, font_size=font_size, cbar_title_size=cbar_title_size,
            ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}p999.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        p999_bias = p999_pred - p999_y
        p999_bias_percentile = p999_bias / p999_y * 100

        plot_single_map(pos, p999_bias, pr_min=-10, aggr=None, pr_max=10,
            title=f"P99.9 precipitation bias - Year 2016", legend_title="[mm/h]", subtitle_y=0.98, subtitle_x=0.5,
            cmap='BrBG', save_path=None, save_file_name=None, zones=zones, x_size=x_size, y_size=y_size, 
            font_size_title=85, font_size=font_size, cbar_title_size=100, ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}p999_bias.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        plot_single_map(pos, p999_bias_percentile, pr_min=-100, aggr=None, pr_max=100,
            title=f"P99.9 precipitation percentage bias - Year 2016", legend_title="[%]", subtitle_y=0.98, subtitle_x=0.5,
            cmap='BrBG', save_path=None, save_file_name=None, zones=zones, x_size=x_size, y_size=y_size, 
            font_size_title=85, font_size=font_size, cbar_title_size=100, ylim=ylim, xlim=xlim, cbar_y=cbar_y)
        plt.savefig(f'{args.output_path}p999_bias_percentage.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()

        # Additional plots

        mask_north = pos[:,1] >= 43.75
        mask_centre_sud = pos[:,1] < 43.75
        
        y_north = (G.pr_target[mask_north,:]).flatten()
        pr_north = (G.pr[mask_north,:]).flatten()
        
        y_centre_sud = (G.pr_target[mask_centre_sud,:]).flatten()
        pr_centre_sud = (G.pr[mask_centre_sud,:]).flatten()
        
        p99_north = np.nanpercentile(y_north, q=99)
        p999_north = np.nanpercentile(y_north, q=99.9)
        p99_pred_north = np.nanpercentile(pr_north, q=99)
        p999_pred_north = np.nanpercentile(pr_north, q=99.9)

        p99_centre_sud = np.nanpercentile(y_centre_sud, q=99)
        p999_centre_sud = np.nanpercentile(y_centre_sud, q=99.9)
        p99_pred_centre_sud = np.nanpercentile(pr_centre_sud, q=99)
        p999_pred_centre_sud = np.nanpercentile(pr_centre_sud, q=99.9)

        plt.rcParams.update({'font.size': 18})
        
        hist_y_north, bin_edges_y_north = np.histogram(y_north, bins=np.arange(0,200,0.5).astype(np.float32), density=False)
        hist_pr_north, bin_edges_pr_north = np.histogram(pr_north, bins=np.arange(0,200,0.5).astype(np.float32), density=False)
        
        Ntot_y_north = hist_y_north.sum()
        Ntot_pr_north = hist_pr_north.sum()
              
        bin_edges_y_centre_north = (bin_edges_y_north[:-1] + bin_edges_y_north[1:]) / 2
        bin_edges_pr_centre_north = (bin_edges_pr_north[:-1] + bin_edges_pr_north[1:]) / 2
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.scatter(bin_edges_y_centre_north, hist_y_north/Ntot_y, color='darkturquoise', s=80, label="GRIPHO", alpha=0.4, zorder=2)
        ax.scatter(bin_edges_pr_centre_north, hist_pr_north/Ntot_pr, color='indigo', s=80, label="GNN4CD R-all", alpha=0.4, zorder=2)
        ax.plot([p99_north, p99_north], [0,50], '-', color='gold', label='GRIPHO - p99', zorder=1)
        ax.plot([p999_north, p999_north], [0,50], '-r', label='GRIPHO - p99.9', zorder=1)
        ax.plot([p99_pred_north, p99_pred_north], [0,50], ':', color='gold', label='GNN4CD R-all - p99', zorder=1)
        ax.plot([p999_pred_north, p999_pred_north], [0,50], ':r', label='GNN4CD R-all - p99.9', zorder=1)
        l = plt.legend(loc='upper right', facecolor='white', framealpha=1, fontsize=16)
        ax.set_ylim([10**(-10),5])
        # ax.set_xlim([0,5])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', zorder=0)
        ax.set_xlabel('precipitation [mm/h]', fontsize=20)
        ax.set_ylabel('frequency', fontsize=20)
        plt.savefig(f'{args.output_path}pdf_north.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        plt.rcParams.update({'font.size': 18})
        
        hist_y_centre_sud, bin_edges_y_centre_sud = np.histogram(y_centre_sud, bins=np.arange(0,200,0.5).astype(np.float32), density=False)
        hist_pr_centre_sud, bin_edges_pr_centre_sud = np.histogram(pr_centre_sud, bins=np.arange(0,200,0.5).astype(np.float32), density=False)
        
        Ntot_y_centre_sud = hist_y_centre_sud.sum()
        Ntot_pr_centre_sud = hist_pr_centre_sud.sum()
        
        bin_edges_y_centre_centre_sud = (bin_edges_y_centre_sud[:-1] + bin_edges_y_centre_sud[1:]) / 2
        bin_edges_pr_centre_centre_sud = (bin_edges_pr_centre_sud[:-1] + bin_edges_pr_centre_sud[1:]) / 2
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.scatter(bin_edges_y_centre_centre_sud, hist_y_centre_sud/Ntot_y_centre_sud, color='darkturquoise', s=80, label="GRIPHO", alpha=0.4, zorder=2)
        ax.scatter(bin_edges_pr_centre_centre_sud, hist_pr_centre_sud/Ntot_pr_centre_sud, color='indigo', s=80, label="GNN4CD RC", alpha=0.4, zorder=2)
        ax.plot([p99_centre_sud, p99_centre_sud], [0,50], '-', color='gold', label='GRIPHO - p99', zorder=1)
        ax.plot([p999_centre_sud, p999_centre_sud], [0,50], '-r', label='GRIPHO - p99.9', zorder=1)
        ax.plot([p99_pred_centre_sud, p99_pred_centre_sud], [0,50], ':', color='gold', label='GNN4CD RC - p99', zorder=1)
        ax.plot([p999_pred_centre_sud, p999_pred_centre_sud], [0,50], ':r', label='GNN4CD RC - p99.9', zorder=1)
        l = plt.legend(loc='upper right', facecolor='white', framealpha=1, fontsize=16)
        ax.set_ylim([10**(-10),5])
        # ax.set_xlim([0,5])
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.grid(visible=True, which='both', axis='both', color='lightgrey', zorder=0)
        ax.set_xlabel('precipitation [mm/h]', fontsize=22)
        ax.set_ylabel('frequency', fontsize=22)
        plt.savefig(f'{args.output_path}pdf_centre_south.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close()        
        
        # Spatial correlation
        from scipy import stats
       
        method = stats.BootstrapMethod(method='BCa')

        spatial_corr_avg = stats.pearsonr(np.nanmean(G.pr, axis=1).flatten(), np.nanmean(G.pr_target, axis=1).flatten())
        spatial_corr_p99 = stats.pearsonr(np.nanpercentile(G.pr, q=99, axis=1).flatten(), np.nanpercentile(G.pr_target,  q=99, axis=1).flatten())
        spatial_corr_p999 = stats.pearsonr(np.nanpercentile(G.pr, q=99.9, axis=1).flatten(), np.nanpercentile(G.pr_target,  q=99.9, axis=1).flatten())

        write_log(f"Italy spatial corr - avg:\t{spatial_corr_avg}\np99:\t{spatial_corr_p99}\np99.9:\t{spatial_corr_p999}", args, accelerator, 'a')
        
        # spatial_corr_avg.confidence_interval(confidence_level=0.68,method=method)

        spatial_corr_avg = stats.pearsonr(np.nanmean(G.pr[mask_north,:], axis=1).flatten(), 
                                          np.nanmean(G.pr_target[mask_north,:], axis=1).flatten())
        spatial_corr_p99 = stats.pearsonr(np.nanpercentile(G.pr[mask_north,:], q=99, axis=1).flatten(),
                                          np.nanpercentile(G.pr_target[mask_north,:],  q=99, axis=1).flatten())
        spatial_corr_p999 = stats.pearsonr(np.nanpercentile(G.pr[mask_north,:], q=99.9, axis=1).flatten(),
                                           np.nanpercentile(G.pr_target[mask_north,:],  q=99.9, axis=1).flatten())

        write_log(f"North spatial corr - avg:\t{spatial_corr_avg}\np99:\t{spatial_corr_p99}\np99.9:\t{spatial_corr_p999}", args, accelerator, 'a')

        spatial_corr_avg = stats.pearsonr(np.nanmean(G.pr[mask_centre_sud,:], axis=1).flatten(),
                                          np.nanmean(G.pr_target[mask_centre_sud,:], axis=1).flatten())
        spatial_corr_p99 = stats.pearsonr(np.nanpercentile(G.pr[mask_centre_sud,:], q=99, axis=1).flatten(),
                                          np.nanpercentile(G.pr_target[mask_centre_sud,:],  q=99, axis=1).flatten())
        spatial_corr_p999 = stats.pearsonr(np.nanpercentile(G.pr[mask_centre_sud,:], q=99.9, axis=1).flatten(), 
                                           np.nanpercentile(G.pr_target[mask_centre_sud,:],  q=99.9, axis=1).flatten())
        
        write_log(f"Centre-south spatial corr - avg:\t{spatial_corr_avg}\np99:\t{spatial_corr_p99}\np99.9:\t{spatial_corr_p999}", args, accelerator, 'a')
        

        # # QQ plot (nice but slow!)
        # import statsmodels.api as sm
        
        # x = sm.ProbPlot(G.pr.flatten())
        # y = sm.ProbPlot(G.pr_target.flatten())

        # plt.rcParams.update({'font.size': 30})
        # fig, ax = plt.subplots(figsize=(20,20))
        # sm.qqplot_2samples(x,y, xlabel="GNN4CD [mm/h]", ylabel="OBSERVATION [mm/h]", ax=ax, line="45")
        # plt.savefig(f'{args.output_path}qqplot.png', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        # plt.close()
