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
from utils.helpers.target_variable_style import resolve_plot_meta
from utils.plotting.validation import create_validation_plots, create_multivariable_validation_plots
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

        # Load stats for validation
        target_variables = args.target_variables.split(",")
        is_multivariable = len(target_variables) > 1
        stats_per_var = {
            var_name: np.load(args.output_path + f"predictand_stats_{var_name}.npz", allow_pickle=True)
            for var_name in target_variables
        }

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
                accelerator.clip_grad_norm_(model.parameters(), 1)
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
            torch.save({"lr_scheduler": lr_scheduler.state_dict()}, args.output_path+f"checkpoints/checkpoint_{epoch}/lr_scheduler_state")

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

                        # use generate_ensemble() when the model supports it (e.g. the
                        # noisy CRPS model) since a plain model(graph) call in eval mode
                        # returns a single deterministic sample, which CRPS can't score --
                        # unwrap .module first in case the model is DDP/accelerate-wrapped
                        model_ = model.module if hasattr(model, "module") else model
                        is_ensemble = hasattr(model_, "generate_ensemble")
                        if is_ensemble:
                            y_out = model_.generate_ensemble(graph)
                        else:
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
                            y_pred = extract_prediction(y_out, args.loss_name, args=args)
                            if isinstance(y_pred, (tuple, list)):
                                y_pred = y_pred[0]

                            # Retrieve graphs for individual time instances
                            n_nodes = graph["high"].num_nodes
                            B = y.shape[0] // n_nodes  # y.shape[0] is always N regardless of a trailing n_vars axis
                            if is_multivariable:
                                # y, train_mask: (N, n_target_variables) -> (B, n_nodes, n_target_variables)
                                y = y.view(B, n_nodes, -1)
                                train_mask = train_mask.view(B, n_nodes, -1)
                            else:
                                # y, train_mask: (N,) -> (B, n_nodes), unchanged from before
                                y = y.view(B, n_nodes)
                                train_mask = train_mask.view(B, n_nodes)

                            if is_ensemble:
                                # y_pred is (M, B*n_nodes[, output_dim]): M is
                                # the model's native leading axis. Split the
                                # flattened batch*nodes axis
                                M = y_pred.shape[0]
                                y_pred = y_pred.reshape(M, B, n_nodes, *y_pred.shape[2:])
                            else:
                                y_pred = y_pred.view(B, n_nodes, -1)

                            y_pred = torch.atleast_2d(y_pred) # from (N,) to (1,N)
                            y = torch.atleast_2d(y)
                            idxs = torch.atleast_2d(torch.tensor(graph.idxs, device=accelerator.device))

                            y_pred_list.append(y_pred) # (time, nodes)
                            y_list.append(y)
                            idxs_list.append(idxs)     

                    ###### PLOTS ######
                    if args.make_val_plots:

                        # Gather from GPUs and remove duplicated values due to gather
                        y_pred_all = accelerator.gather(torch.stack(y_pred_list)) # (time, nodes)
                        y_all = accelerator.gather(torch.stack(y_list))
                        idxs_all = accelerator.gather(torch.stack(idxs_list))

                        # Indices to ensure data are sorted correctly
                        idxs_all = idxs_all.squeeze()[:val_size]
                        _, indices = torch.sort(idxs_all)
                        indices = indices.cpu().numpy()

                        # Squeeze, swapaxes and , convert to cpu and numpy
                        y_pred_all = y_pred_all.cpu().numpy()
                        y_all = y_all.cpu().numpy()
                        times = times[indices]

                        # Squeeze, swapaxes and convert to cpu and numpy
                        y_pred_all = y_pred_all.squeeze()[:val_size, :][indices, :] # (time, nodes[, n_vars])
                        y_all = y_all.squeeze()[:val_size, :][indices, :]

                        if is_ensemble and is_multivariable:
                            # (time, M, nodes, n_vars) -> (nodes, time, n_vars, M)
                            y_pred_all = y_pred_all.transpose(2, 0, 3, 1)
                        elif is_ensemble and y_pred_all.ndim == 3:
                            # (time, M, nodes) -> (nodes, time, M)
                            y_pred_all = y_pred_all.transpose(2, 0, 1)
                        else:
                            # single-variable deterministic: (nodes, time)
                            # multivariable deterministic: (nodes, time, n_vars)
                            y_pred_all = y_pred_all.swapaxes(0,1)
                        y_all = y_all.swapaxes(0,1)

                        print(f"y_pred_all.shape: {y_pred_all.shape}, y_all.shape: {y_all.shape}, indices.shape: {indices.shape}")

                        # Load validation plots metadata (once, shared across variables)
                        metadata_file_path = args.val_plot_config
                        with open(metadata_file_path) as f:
                            meta = json.load(f)
                        meta = convert_dict(meta)

                        lon = graph['high'].lon.cpu().numpy()
                        lat = graph['high'].lat.cpu().numpy()

                        data_to_save = {} if epoch == (args.epochs-1) else None

                        y_pred_dict = {}
                        y_dict = {}
                        meta_dict = {"general": meta["general"]}

                        for i, var_name in enumerate(target_variables):
                            # Slice out this variable's (nodes, time[, M]) array.
                            # Layouts (set up by the transpose block above):
                            #   single-variable, deterministic: (nodes, time)         -- nothing to slice
                            #   single-variable, ensemble:       (nodes, time, M)      -- nothing to slice
                            #   multivariable, deterministic:    (nodes, time, n_vars) -- n_vars is last
                            #   multivariable, ensemble:         (nodes, time, n_vars, M) -- n_vars is 2nd-to-last
                            if is_multivariable and is_ensemble:
                                y_pred_var = y_pred_all[:, :, i, :]  # (nodes, time, M)
                                y_var = y_all[..., i]                 # (nodes, time)
                            elif is_multivariable:
                                y_pred_var = y_pred_all[..., i]       # (nodes, time)
                                y_var = y_all[..., i]                 # (nodes, time)
                            else:
                                y_pred_var = y_pred_all               # (nodes, time) or (nodes, time, M)
                                y_var = y_all                         # (nodes, time)

                            # Get the actual physical-units prediction, using this
                            # variable's own transform stats (each variable was
                            # transformed independently in train.py)
                            stats = stats_per_var[var_name]
                            y_var = inverse_transform_predictand(y_var, stats)
                            y_pred_var = inverse_transform_predictand(y_pred_var, stats)

                            if y_pred_var.ndim == 3:

                                y_pred_var = np.mean(y_pred_var, axis=-1)
                                print(f"After averaging y_pred_var.shape: {y_pred_var.shape}.", flush=True)

                            y_pred_dict[var_name] = y_pred_var
                            y_dict[var_name] = y_var
                            # Resolve this variable's plot style
                            meta_dict[var_name] = resolve_plot_meta(var_name, meta)[var_name]

                            if data_to_save is not None:
                                data_to_save[var_name] = (y_pred_var, y_var)

                        if is_multivariable:
                            fig_avg, fig_bias, fig_pdf = create_multivariable_validation_plots(
                                y_pred_dict,
                                y_dict,
                                lon,
                                lat,
                                target_variables,
                                meta_dict
                            )
                        else:
                            var_name = target_variables[0]
                            fig_avg, fig_bias, fig_pdf = create_validation_plots(
                                y_pred_dict[var_name],
                                y_dict[var_name],
                                lon,
                                lat,
                                var_name,
                                meta_dict
                            )

                        accelerator.log({
                            "average": [wandb.Image(fig_avg)],
                            "bias":  [wandb.Image(fig_bias)],
                            "pdf": [wandb.Image(fig_pdf)],
                            }, step=step)

                        plt.close(fig_avg)
                        plt.close(fig_bias)
                        plt.close(fig_pdf)

                        if data_to_save is not None: # last epoch
                            data = HeteroData()
                            for var_name, (y_pred_var, y_var) in data_to_save.items():
                                setattr(data, f"{var_name}_gnn4cd", y_pred_var)
                                setattr(data, f"{var_name}_target", y_var)

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