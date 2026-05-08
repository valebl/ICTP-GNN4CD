import torch
import numpy as np
import pickle
import time
import wandb
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
import json

from utils.metrics.average_meter import AverageMeter
from utils.helpers.tools import write_log, convert_dict
from utils.plotting.validation import create_validation_plots
from utils.extractors.extract_prediction import extract_prediction
from utils.predictand_transforms.inverse_transform_predictand import inverse_transform_predictand

#-----------------------------------------------------
#---------------------- TRAIN ------------------------
#-----------------------------------------------------

class Trainer(object):

    def __init__(self):
        super().__init__()

    def train(
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
            epoch_start=0):
        
        write_log(f"\nStart training the regressor.", args, accelerator, 'a')

        step = 0
        
        for epoch in range(epoch_start, epoch_start+args.epochs):

            model.train()
            write_log(f"\nEpoch {epoch} --- learning rate {optimizer.param_groups[0]['lr']:.8f}", args, accelerator, 'a')
            
            # Define objects to track meters
            loss_meter = AverageMeter()
            val_loss_meter = AverageMeter()

            if getattr(loss_fn, "components", False):
                loss_meter_components = {}
                val_loss_meter_components = {}
                for i, component in enumerate(loss_fn.components):
                    loss_meter_components[component] = AverageMeter()
                    val_loss_meter_components[component] = AverageMeter()

            start = time.time()
            
            # TRAIN
            for i, graph in enumerate(dataloader_train):

                # Get target and mask from graph
                train_mask = graph['high'].train_mask
                y = graph["high"].y

                y_out = model(graph)

                if getattr(loss_fn, "use_bins", False):
                    bins = graph['high'].w
                    loss, loss_components = loss_fn(y_out, y, bins)
                else:
                    if getattr(loss_fn, "components", False):
                        loss, loss_components = loss_fn(y_out, y)  # the loss internally handles the different y_out cases
                    else:
                        loss = loss_fn(y_out, y)  # the loss internally handles the different y_out cases
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                #accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                step += 1
                
                # Log values to wandb
                loss_meter.update(val=loss.item(), n=y.shape[0])

                if getattr(loss_fn, "components", False):
                    for i, component in enumerate(loss_fn.components):
                        loss_meter_components[component].update(val=loss_components[i].item(), n=y.shape[0])
                
                accelerator.log({
                    'epoch':epoch,
                    'train loss iteration': loss_meter.val,
                    'train loss avg': loss_meter.avg,
                }, step=step)

            end = time.time()

            accelerator.log({
                'epoch':epoch,
                'train loss avg': loss_meter.avg,
                'lr': np.mean(lr_scheduler.get_last_lr())
            }, step=step)

            if getattr(loss_fn, "components", False):
                for i, component in enumerate(loss_fn.components):
                    accelerator.log({
                        'epoch':epoch,
                        f'train loss {component}': loss_meter_components[component].avg,
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

                # if epoch%5==0:
                if args.make_val_plots and epoch % args.val_plot_frequency==0:
                    y_list = []
                    y_pred_list = []
                    idxs_list = []

                with torch.no_grad():    
                    for graph in dataloader_val:
                        
                        # Get target and mask from graph
                        train_mask = graph['high'].train_mask
                        y = graph["high"].y

                        y_out = model(graph)
                        
                        if getattr(loss_fn, "use_bins", False):
                            bins = graph['high'].w
                            loss, loss_components = loss_fn(y_out, y, bins)
                        else:
                            if getattr(loss_fn, "components", False):
                                loss, loss_components = loss_fn(y_out, y)  # the loss internally handles the different y_out cases
                            else:
                                loss = loss_fn(y_out, y)  # the loss internally handles the different y_out cases

                        val_loss_meter.update(val=loss.item(), n=y.shape[0])
                        if getattr(loss_fn, "components", False):
                            for i, component in enumerate(loss_fn.components):
                                val_loss_meter_components[component].update(val=loss_components[i].item(), n=y.shape[0])

                        accelerator.log({
                            'epoch':epoch,
                            'val loss iteration': val_loss_meter.val,
                            'val loss avg': val_loss_meter.avg
                        }, step=step)
                        
                        if args.make_val_plots:
                            y_pred = extract_prediction(y_out, args.loss_name)

                            # Retrieve graphs for individual time instances
                            n_nodes = graph["high"].num_nodes
                            B = y.shape[0] // n_nodes
                            y = y.view(B, n_nodes)
                            y_pred = y_pred.view(B, n_nodes, -1)
                            train_mask = train_mask.view(B, n_nodes)

                            y_pred = torch.atleast_2d(y_pred) # from (N,) to (1,N)
                            y = torch.atleast_2d(y)
                            idxs = torch.atleast_2d(torch.tensor(graph.idxs, device=accelerator.device))

                            y_pred_list.append(y_pred) # (time, nodes)
                            y_list.append(y)
                            idxs_list.append(idxs)     

                    ###### PLOTS ######
                    if args.make_val_plots:

                        # Gather from GPUs and remove duplicated values due to gather
                        y_pred_all = accelerator.gather(torch.stack(y_pred_list)).squeeze()[:val_size, :] # (time, nodes)
                        y_all = accelerator.gather(torch.stack(y_list)).squeeze()[:val_size, :]
                        idxs_all = accelerator.gather(torch.stack(idxs_list)).squeeze()[:val_size]

                        # Squeeze, swapaxes and , convert to cpu and numpy
                        y_pred_all = y_pred_all.swapaxes(0,1).cpu().numpy() # (nodes, time)
                        y_all = y_all.swapaxes(0,1).cpu().numpy()

                        # Indices to ensure data are sorted correctly
                        _, indices = torch.sort(idxs_all)
                        indices = indices.cpu().numpy()

                        y_pred_all = y_pred_all[:, indices]
                        y_all = y_all[:, indices]
                        times = times[indices]

                        print(f"y_pred_all.shape: {y_pred_all.shape}, y_all.shape: {y_all.shape}, indices.shape: {indices.shape}")

                        # Get the actual precipitation/temperature prediction
                        stats = np.load(args.output_path+"predictand_stats.npz", allow_pickle=True)
                        y_all = inverse_transform_predictand(y_all, stats)
                        y_pred_all = inverse_transform_predictand(y_pred_all, stats)

                        # Load validation plots metadata
                        metadata_file_path = args.val_plot_config
                        with open(metadata_file_path) as f:
                            meta = json.load(f)

                        # Arguments for the plot function                        
                        meta = convert_dict(meta)
                        target_type = args.target_type
                        lon = graph['high'].lon.cpu().numpy()
                        lat = graph['high'].lat.cpu().numpy()

                        # Create a few plots to compare
                        fig_avg, fig_bias, fig_pdf = create_validation_plots(
                            y_pred_all, 
                            y_all,
                            lon,
                            lat,
                            args.target_type,
                            meta
                        )

                        accelerator.log({
                            "average": [wandb.Image(fig_avg)],
                            "bias":  [wandb.Image(fig_bias)],
                            "pdf": [wandb.Image(fig_pdf)],
                            }, step=step)

                        plt.close(fig_avg)
                        plt.close(fig_bias)
                        plt.close(fig_pdf)

                        if epoch == (args.epochs-1): # last epoch
                            data = HeteroData()
                            if args.target_type == "precipitation":
                                data.pr_gnn4cd = y_pred
                            elif args.target_type == "temperature":
                                data.tasmax_gnn4cd = y_pred
                            
                            data.target = y

                            data.times = times
                            data.times_target = times
                            data["high"].lat = lat
                            data["high"].lon = lon

                            with open(args.output_path + f"output_graph_{args.validation_year}.pkl", 'wb') as f:
                                pickle.dump(data, f)
                            
                accelerator.log({
                    'epoch':epoch,
                    'val loss avg': val_loss_meter.avg,
                }, step=step)


                if getattr(loss_fn, "components", False):
                    for i, component in enumerate(loss_fn.components):
                        accelerator.log({
                            'epoch':epoch,
                            f'train loss {component}': val_loss_meter_components[component].avg,
                        }, step=step)
                    
            if lr_scheduler is not None:
                lr_scheduler.step()  