import os
import torch
import numpy as np
import pickle
import time
import wandb
from utils.metrics.metrics import AverageMeter
from utils.helpers.tools import write_log, invert_normalization
from utils.plotting.plots import plot_maps, plot_pdf, get_cmap_dict
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib
import json
from utils.helpers.tools import convert_dict
from torch_geometric.data import HeteroData

#-----------------------------------------------------
#---------------------- TRAIN ------------------------
#-----------------------------------------------------

class NLL_Trainer(object):

    def __init__(self):
        super(NLL_Trainer, self).__init__()

    def train_reg(
            self,
            model,
            dataloader_train,
            dataloader_val,
            optimizer,
            loss_fn,
            lr_scheduler,
            val_size,
            times,
            accelerator,
            args,
            epoch_start=0,
            log_val_plots=False):
        
        write_log(f"\nStart training the regressor.", args, accelerator, 'a')

        step = 0
        
        for epoch in range(epoch_start, epoch_start+args.epochs):

            model.train()
            write_log(f"\nEpoch {epoch} --- learning rate {optimizer.param_groups[0]['lr']:.8f}", args, accelerator, 'a')
            
            # Define objects to track meters
            loss_meter = AverageMeter()
            val_loss_meter = AverageMeter()

            start = time.time()
            
            # TRAIN
            for i, graph in enumerate(dataloader_train):

                # Get target and mask from graph
                train_mask = graph['high'].train_mask
                y = graph["high"].y

                mu_pred, sigma_pred = model(graph)
                if train_mask.sum() == 0:
                    continue  # skip batch with all-NaN targets
                loss = loss_fn(mu_pred[train_mask].flatten(), sigma_pred[train_mask].flatten(), y[train_mask].flatten())
                if torch.isnan(loss):
                    write_log(f"\nWARNING: NaN loss at epoch {epoch} batch {i}, skipping.", args, accelerator, 'a')
                    continue

                optimizer.zero_grad()
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                step += 1
                
                # Log values to wandb
                loss_meter.update(val=loss.item(), n=mu_pred.shape[0])
                
                accelerator.log({
                    'epoch':epoch,
                    'train loss iteration': loss_meter.val,
                    'train loss avg': loss_meter.avg,
                }, step=step)

            end = time.time()

            accelerator.log({
                'epoch':epoch,
                'train loss avg': loss_meter.avg,
                'lr': optimizer.param_groups[0]['lr']
            }, step=step)

            write_log(
                f"\nEpoch {epoch} completed in {end - start:.4f} seconds." +
                f"Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.10f}. ", args, accelerator, 'a'
            )
                    
            accelerator.save_state(output_dir=args.output_path+f"checkpoints/checkpoint_{epoch}/", safe_serialization=False)
            torch.save({"epoch": epoch}, args.output_path+f"checkpoints/checkpoint_{epoch}/epoch")

            # VALIDATION
            if dataloader_val is not None:
                model.eval()

                do_plots = log_val_plots and (epoch % 25 == 0 or epoch == epoch_start + args.epochs - 1)

                if do_plots:
                    mu_pred_list = []
                    y_list = []
                    idxs_list = []

                with torch.no_grad():
                    for graph in dataloader_val:

                        # Get target and mask from graph
                        train_mask = graph['high'].train_mask
                        y = graph["high"].y

                        mu_pred, sigma_pred = model(graph) # mu, phi if tweedie loss
                        if train_mask.sum() > 0:
                            loss = loss_fn(mu_pred[train_mask].flatten(), sigma_pred[train_mask].flatten(), y[train_mask].flatten())
                        else:
                            loss = torch.tensor(float('nan'))

                        # retrieve graphs for individual time instances
                        n_nodes_per_graph = mu_pred.shape[0] // graph.num_graphs
                        y_split = torch.split(y, n_nodes_per_graph)
                        mu_pred_split = torch.split(mu_pred, n_nodes_per_graph)
                        train_mask_split = torch.split(train_mask, n_nodes_per_graph)
                        y = torch.stack(y_split, dim=0).squeeze(-1)  # (B, n_nodes)
                        mu_pred = torch.stack(mu_pred_split, dim=0).squeeze(-1)  # (B, n_nodes)
                        train_mask = torch.stack(train_mask_split, dim=0).squeeze(-1) # (B, n_nodes)

                        val_loss_meter.update(val=loss.item(), n=mu_pred.shape[0])
                        accelerator.log({
                            'epoch':epoch,
                            'val loss iteration': val_loss_meter.val,
                            'val loss avg': val_loss_meter.avg
                        }, step=step)

                        if do_plots:
                            mu_pred = torch.atleast_2d(mu_pred) # from (N,) to (1,N)
                            y = torch.atleast_2d(y)
                            idxs = torch.atleast_2d(torch.tensor(graph.idxs, device=accelerator.device))
                            mu_pred_list.append(mu_pred) # time, nodes
                            y_list.append(y)
                            idxs_list.append(idxs)

                    ###### PLOTS ######
                    if do_plots:
                        mu_pred_all = accelerator.gather(torch.stack(mu_pred_list)).swapaxes(0,1)[:,:val_size]
                        y_all = accelerator.gather(torch.stack(y_list)).swapaxes(0,1)[:,:val_size]
                        idxs_all = accelerator.gather(torch.stack(idxs_list)).squeeze()[:val_size]

                        # Create a few plots to compare
                        self._create_plots_reg(mu_pred_all, y_all, idxs_all, times, graph, accelerator, step, epoch, args)
                            
                accelerator.log({
                    'epoch':epoch,
                    'val loss avg': val_loss_meter.avg,
                }, step=step)
                    
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if args.validation_year is not None:
                        lr_scheduler.step(val_loss_meter.avg)
                else:
                    lr_scheduler.step()

    def _create_plots_reg(self, y_pred, y, t, times, graph, accelerator, step, epoch, args):

        if "CORDEXML" in args.run_type:
            run_type = "CORDEXML"
        else:
            run_type = args.run_type

        utils_dir = os.path.join(os.path.dirname(__file__), '..')
        json_path = os.path.join(utils_dir, f"{run_type}_plot_params.json")
        with open(json_path) as f:
            meta = json.load(f)

        meta = convert_dict(meta)
        target_type = args.target_type

        lon = graph['high'].lon.cpu().numpy()
        lat = graph['high'].lat.cpu().numpy()
        # Reconstruct 2D coordinates if graph stores 1D unique axes
        # y_pred shape: (n_gpus, T, n_nodes) → n_nodes is the last dim
        n_nodes_total = y_pred.shape[-1]
        if len(lat) != n_nodes_total and len(lat) * len(lon) == n_nodes_total:
            lon_2d, lat_2d = np.meshgrid(lon, lat)
            lat = lat_2d.flatten()
            lon = lon_2d.flatten()

        # convert to cpu and numpy
        _, indices = torch.sort(t)
        indices = indices.cpu().numpy()
        times = times[indices]

        y_pred_plot = y_pred.squeeze().swapaxes(0,1).cpu().numpy()[:,indices]
        y_plot = y.squeeze().swapaxes(0,1).cpu().numpy()[:,indices]

        if target_type == "precipitation":
            # no log transformations, training directly in linear space
            y_pred_pdf = y_pred_plot.flatten()
            y_pdf = y_plot.flatten()
            if "CERRA" in args.run_type:
                y_pred_plot *= 24 # mm/day
                y_plot *= 24 # mm/day
                bins = np.arange(0,40,0.5).astype(np.float32)
            elif "CORDEXML" in args.run_type:
                bins = np.arange(0,1000,1).astype(np.float32)
        elif target_type == "temperature":
            y_pred_plot = invert_normalization(y_pred_plot, stats_path=args.output_path)
            y_plot = invert_normalization(y_plot, stats_path=args.output_path)
            y_pred_pdf = y_pred_plot.flatten()
            y_pdf = y_plot.flatten()
            bins = np.arange(int(np.nanmin(y_plot)), int(np.nanmax(y_plot)) + 2, 1).astype(np.float32)

        cmap_dict = get_cmap_dict()
        bounds_avg = [0, 1, 1.5, 2, 4, 6, 8, 10, 12] #, 15, 20] #, 25, 30, 35]
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds_avg, ncolors=256)

        pr_gnn4cd_avg = np.nanmean(y_pred_plot, axis=-1)
        pr_target_avg = np.nanmean(y_plot, axis=-1)

        bias =  pr_gnn4cd_avg - pr_target_avg

        if meta[target_type]["cmap"] == "cmap_dict['avg']['cmap']":
            cmap = cmap_dict['avg']['cmap']
        else:
            cmap = meta[target_type]["cmap"]

        fig_avg = plot_maps(
            [lon, lon],
            [lat, lat],
            [pr_gnn4cd_avg, pr_target_avg],
            aggr=None,
            s=meta[target_type]["s"],
            legend_title=meta[target_type]["map_unit"],
            cmap=cmap,
            sub_titles=["GNN4CD", "TARGET"],
            x_size=meta["general"]["figsize"][0],
            y_size=meta["general"]["figsize"][1],
            font_size_title=meta["general"]["fontsize_title"],
            font_size=meta["general"]["fontsize"],
            cbar_title_size=meta["general"]["fontsize_cbar_title"],
            pr_max=meta[target_type]["vmax"],
            pr_min=meta[target_type]["vmin"],
            cbar_pad=20,
            suptitle_y=0.87,
            suptitle_x=0.72,
            show_ticks=False,
            plot_func="scatter",
            xlim=meta["general"]["xlim"],
            ylim=meta["general"]["ylim"],
            proj=ccrs.PlateCarree(),
            cbar_ax_lim=[0.93,0.23,0.015,0.55]
        )

        fig_bias = plot_maps(
            lon,
            lat,
            bias,
            aggr=None,
            s=meta[target_type]["s"],
            legend_title=meta[target_type]["map_unit"],
            cmap="BrBG",
            sub_titles=["GNN4CD - TARGET"],
            x_size=meta["general"]["figsize"][0]/2,
            y_size=meta["general"]["figsize"][1],
            font_size_title=25,
            font_size=20,
            cbar_title_size=20,
            pr_max=meta[target_type]["vmax_bias"],
            pr_min=meta[target_type]["vmin_bias"],
            cbar_pad=20,
            suptitle_y=0.87,
            suptitle_x=0.72,
            show_ticks=False,
            plot_func="scatter",
            xlim=meta["general"]["xlim"],
            ylim=meta["general"]["ylim"],
            proj=ccrs.PlateCarree(),
            cbar_ax_lim=[0.93,0.23,0.015,0.55]
        )

        hist_vals, bins = np.histogram(y_pred_pdf, bins=bins, density=False)
        bins_mid = (bins[:-1] + bins[1:]) / 2
        Ntot = np.nansum(hist_vals)
        hist_vals_target, bins_target = np.histogram(y_pdf, bins=bins, density=False)
        bins_target_mid = (bins_target[:-1] + bins_target[1:]) / 2
        Ntot_target = np.nansum(hist_vals_target)

        if meta[target_type]["xlim_pdf"] is None:
            meta[target_type]["xlim_pdf"] = [0.2, bins.max()+10]

        fig_pdf = plot_pdf(
            bin_list=[bins_target_mid, bins_mid],
            hist_list=[hist_vals_target/Ntot_target, hist_vals/Ntot],
            label_list=["TARGET", "GNN4CD"],
            xlabel=meta[target_type]["pdf_unit"],
            color_list=["black", "darkorange"],
            tail_lim=meta[target_type]["tail_lim"],
            ylim=meta[target_type]["ylim_pdf"],
            title=meta[target_type]["pdf_title"],
            xlim=meta[target_type]["xlim_pdf"],
            plot_func=meta[target_type]["plot_func_pdf"],
            fontsize=20,
            suptitle="",
            tail_ylim=meta[target_type]["tail_ylim"],
            log_xy=meta[target_type]["log_xy"],
            tail_zoom=meta[target_type]["tail_zoom"],
            legend_outside=meta[target_type]["legend_outside"]
        )

        if args.target_type == "precipitation":
            accelerator.log({
                meta[target_type]["map_title"]: [wandb.Image(fig_avg)],
                "bias":  [wandb.Image(fig_bias)],
                "pdf": [wandb.Image(fig_pdf)],
                }, step=step)
        else:
            accelerator.log({
                meta[target_type]["map_title"]: [wandb.Image(fig_avg)],
                "bias":  [wandb.Image(fig_bias)],
                "pdf": [wandb.Image(fig_pdf)],
                }, step=step)
        
        plt.close(fig_avg)
        plt.close(fig_bias)
        plt.close(fig_pdf)

        if epoch == (args.epochs-1): # last epoch
            data = HeteroData()
            if args.target_type == "precipitation":
                data.pr_gnn4cd = y_pred_plot
            elif args.target_type == "temperature":
                if args.run_type == "CERRA":
                    data.t2m_gnn4cd = y_pred_plot
                elif args.run_type == "CORDEXML":
                    data.tasmax_gnn4cd = y_pred_plot
            
            data.target = y_plot

            data.times = times
            data.times_target = times
            data["high"].lat = lat
            data["high"].lon = lon

            with open(args.output_path + f"output_graph_{args.validation_year}.pkl", 'wb') as f:
                pickle.dump(data, f)