import torch
import numpy as np
import pickle
import time
import wandb
from utils.metrics import AverageMeter, accuracy_binary_one, accuracy_binary_one_classes
from utils.tools import write_log
from utils.plots import create_zones, plot_maps, plot_pdf, plot_diurnal_cycles
import matplotlib.pyplot as plt

# target_type = "temperature"
target_type = "precipitation"

#-----------------------------------------------------
#---------------------- TRAIN ------------------------
#-----------------------------------------------------

class Trainer(object):

    def __init__(self):
        super(Trainer, self).__init__()

    #--- CLASSIFIER
    def train_cl(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args,
                        epoch_start, alpha=0.75, gamma=2):
        
        write_log(f"\nStart training the classifier.", args, accelerator, 'a')

        step = 0
        
        for epoch in range(epoch_start, epoch_start+args.epochs):
            
            model.train()
            write_log(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}", args, accelerator, 'a')

            # Define objects to track meters durng training
            all_loss_meter = AverageMeter()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            acc_class0_meter = AverageMeter()
            acc_class1_meter = AverageMeter()

            start = time.time()

            for graph in dataloader_train:
                
                optimizer.zero_grad()             
                y_pred = model(graph).squeeze()

                train_mask = graph["high"].train_mask      
                y = graph['high'].y   

                # Gather from all processes for metrics
                all_y_pred, all_y, all_train_mask = accelerator.gather((y_pred, y, train_mask))

                # Apply mask
                y_pred, y = y_pred[train_mask], y[train_mask]
                all_y_pred, all_y = all_y_pred[all_train_mask], all_y[all_train_mask]
                
                loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
                all_loss = loss_fn(all_y_pred, all_y, alpha, gamma, reduction='mean')
                
                accelerator.backward(loss)
                #accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                step += 1
                
                loss_meter.update(val=loss.item(), n=y_pred.shape[0])   
                all_loss_meter.update(val=all_loss.item(), n=all_y_pred.shape[0])   
                
                acc = accuracy_binary_one(all_y_pred, all_y)
                acc_class0, acc_class1 = accuracy_binary_one_classes(all_y_pred, all_y)

                acc_meter.update(val=acc.item(), n=all_y_pred.shape[0])
                acc_class0_meter.update(val=acc_class0.item(), n=(all_y==0).sum().item())
                acc_class1_meter.update(val=acc_class1.item(), n=(all_y==1).sum().item())

                accelerator.log({'epoch':epoch, 'accuracy iteration': acc_meter.val, 'loss avg': all_loss_meter.avg,
                                 'loss avg (1GPU)': loss_meter.avg, 'accuracy avg': acc_meter.avg,
                                 'accuracy class0 avg': acc_class0_meter.avg, 'accuracy class1 avg': acc_class1_meter.avg}, step=step)
                
            end = time.time()

            # End of epoch --> write log and save checkpoint
            accelerator.log({'epoch':epoch, 'loss epoch': all_loss_meter.avg, 'loss epoch (1GPU)': loss_meter.avg,  'accuracy epoch': acc_meter.avg,
                             'accuracy class0 epoch': acc_class0_meter.avg, 'accuracy class1 epoch': acc_class1_meter.avg}, step=step)
            write_log(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {all_loss_meter.sum:.4f} - average: {all_loss_meter.avg:.10f}; "
                      + f"acc: {acc_meter.avg:.4f}; acc class 0: {acc_class0_meter.avg:.4f}; acc class 1: {acc_class1_meter.avg:.4f}.", args, accelerator, 'a')
            
            # if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.00001:
            #     lr_scheduler.step()

            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/")
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")

            # Perform the validation step
            model.eval()

            y_pred_val = []
            y_val = []
            train_mask_val = []
                
            with torch.no_grad():    
                for i, graph in enumerate(dataloader_val):
                    # Append the data for the current epoch
                    train_mask_val.append(graph["high"].train_mask)            
                    y_pred_val.append(model(graph).squeeze())
                    y_val.append(graph['high'].y)

                # Create tensors
                train_mask_val = torch.stack(train_mask_val)
                y_pred_val = torch.stack(y_pred_val)
                y_val = torch.stack(y_val)

                # Validation metrics for 1GPU
                loss_val_1gpu = loss_fn(y_pred_val[train_mask_val], y_val[train_mask_val], alpha, gamma, reduction="mean")

                # Gather from all processes for metrics
                y_pred_val, y_val, train_mask_val = accelerator.gather((y_pred_val, y_val, train_mask_val))

                # Apply mask
                y_pred_val, y_val = y_pred_val[train_mask_val], y_val[train_mask_val]

                # Compute metrics on all validation dataset            
                loss_val = loss_fn(y_pred_val, y_val, alpha, gamma, reduction="mean")

                acc_class0_val, acc_class1_val = accuracy_binary_one_classes(y_pred_val, y_val)
                acc_val = accuracy_binary_one(y_pred_val, y_val)
            
                        
            if lr_scheduler is not None:
                lr_scheduler.step(loss_val.item())
           
            accelerator.log({'epoch':epoch, 'validation loss': loss_val.item(), 'validation loss (1GPU)': loss_val_1gpu.item(),
                             'validation accuracy': acc_val.item(),
                             'validation accuracy class0': acc_class0_val.item(),
                             'validation accuracy class1': acc_class1_val.item(),
                             'lr': np.mean(lr_scheduler._last_lr)}, step=step)
                
    #--- REGRESSOR
    def train_reg(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=0):
        
        write_log(f"\nStart training the regressor.", args, accelerator, 'a')

        step = 0
        
        for epoch in range(epoch_start, epoch_start+args.epochs):

            model.train()
            write_log(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}", args, accelerator, 'a')
            
            # Define objects to track meters
            loss_meter = AverageMeter()
            all_loss_meter = AverageMeter()
            
            if "quantized_loss" in args.loss_fn:
                loss_term1_meter = AverageMeter()
                loss_term2_meter = AverageMeter()

            start = time.time()
            
            # TRAIN
            for i, graph in enumerate(dataloader_train):

                optimizer.zero_grad()
                y_pred = model(graph).squeeze()

                train_mask = graph['high'].train_mask
                y = graph['high'].y


                # Gather from all processes for metrics
                all_y_pred, all_y, all_train_mask = accelerator.gather((y_pred, y, train_mask))

                # Apply mask
                y_pred, y = y_pred[train_mask], y[train_mask]
                all_y_pred, all_y = all_y_pred[all_train_mask], all_y[all_train_mask]

                if "quantized" in args.loss_fn:
                    w = graph['high'].w
                    all_w =accelerator.gather((w))
                    w = w[train_mask]
                    all_w = all_w[all_train_mask]

                # print(f"{accelerator.device} - all_y_pred.shape: {all_y_pred.shape}, all_y.shape: {all_y.shape}, all_w.shape: {all_w.shape}")
                
                if "quantized_loss" in args.loss_fn:
                    loss, _, _ = loss_fn(y_pred, y, w)
                    all_loss, loss_term1, loss_term2 = loss_fn(all_y_pred, all_y, all_w)
                # elif args.loss_fn == "quantized_loss_scaled":
                #     loss = loss_fn(y_pred, y, w, epoch)
                #     all_loss = loss_fn(all_y_pred, all_y, all_y, epoch)
                else:
                    loss = loss_fn(y_pred, y)
                    all_loss = loss_fn(all_y_pred, all_y)
                
                accelerator.backward(loss)
                #accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                step += 1
                
                # Log values to wandb
                loss_meter.update(val=loss.item(), n=y_pred.shape[0])    
                all_loss_meter.update(val=all_loss.item(), n=all_y_pred.shape[0])
                
                if "quantized_loss" in args.loss_fn:
                    loss_term1_meter.update(val=loss_term1.item(), n=all_y_pred.shape[0])
                    loss_term2_meter.update(val=loss_term2.item(), n=all_y_pred.shape[0])
                    
                accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg, 'loss all avg': all_loss_meter.avg}, step=step)

            end = time.time()

            if "quantized_loss" in args.loss_fn:
                accelerator.log({'epoch':epoch, 'train loss (1GPU)': loss_meter.avg, 'train loss': all_loss_meter.avg,
                                 'train mse loss': loss_term1_meter.avg, 'train quantized loss': loss_term2_meter.avg}, step=step)
            else:
                accelerator.log({'epoch':epoch, 'train loss (1GPU)': loss_meter.avg, 'train loss': all_loss_meter.avg}, step=step)

            write_log(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds." +
                      f"Loss - total: {all_loss_meter.sum:.4f} - average: {all_loss_meter.avg:.10f}. ", args, accelerator, 'a')
                    
            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/", safe_serialization=False)
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")

            # VALIDATION
            # Validation is performed on all the validation dataset at once
            model.eval()

            y_pred_val = []
            y_val = []
            train_mask_val = []
            t = []

            if "quantized" in args.loss_fn:
                w_val = []

            with torch.no_grad():    
                for graph in dataloader_val:
                    # Append the data for the current epoch
                    y_pred_val.extend(model(graph,inference=True)) # num_nodes, time
                    graph = graph.to_data_list()
                    [train_mask_val.append(graph_i["high"].train_mask) for graph_i in graph]
                    [y_val.append(graph_i['high'].y) for graph_i in graph]
                    [t.append(graph_i.t) for graph_i in graph]
                    if "quantized" in args.loss_fn:
                        [w_val.append(graph_i['high'].w) for graph_i in graph]

                # write_log(f"\n{y_pred_val[0].shape}, {train_mask_val[0].shape}, {y_val[0].shape}", args, accelerator, 'a')

                # Create tensors
                train_mask_val = torch.stack(train_mask_val, dim=-1).squeeze().swapaxes(0,1) # time, nodes
                y_pred_val = torch.stack(y_pred_val, dim=-1).squeeze().swapaxes(0,1)
                y_val = torch.stack(y_val, dim=-1).squeeze().swapaxes(0,1)
                t = torch.stack(t, dim=-1).squeeze()
                if "quantized" in args.loss_fn:
                    w_val = torch.stack(w_val, dim=-1).squeeze().swapaxes(0,1)

                # write_log(f"\n{y_pred_val.shape}, {train_mask_val.shape}, {y_val.shape}", args, accelerator, 'a')

                # Log validation metrics for 1GPU
                if "quantized_loss" in args.loss_fn:
                    loss_val_1gpu,  _, _ = loss_fn(y_pred_val.flatten()[train_mask_val.flatten()],
                                                   y_val.flatten()[train_mask_val.flatten()],
                                                   w_val.flatten()[train_mask_val.flatten()])
                # elif args.loss_fn == "quantized_loss_scaled":
                #     loss_val_1gpu = loss_fn(y_pred_val.flatten()[train_mask_val.flatten()],
                #                                    y_val.flatten()[train_mask_val.flatten()],
                #                                    w_val.flatten()[train_mask_val.flatten()],
                #                                    epoch) 
                else:
                    loss_val_1gpu = loss_fn(y_pred_val.flatten()[train_mask_val.flatten()],
                                            y_val.flatten()[train_mask_val.flatten()])

                # Gather from all processes for metrics
                y_pred_val, y_val, train_mask_val, t = accelerator.gather((y_pred_val, y_val, train_mask_val, t))

                # nodes, time
                y_pred_val, y_val, train_mask_val = y_pred_val.swapaxes(0,1), y_val.swapaxes(0,1), train_mask_val.swapaxes(0,1) # nodes, time


                if "quantized" in args.loss_fn:
                    w_val = accelerator.gather((w_val))
                    w_val = w_val.swapaxes(0,1)

                ###### PLOTS ######
                # Create a few plots to compare
                if "fvg" in args.input_path:
                    p = {"xsize": 8, "ysize": 12, "ylim": [45.45, 46.8], "xlim": [12.70, 14.05], "s": 250}
                else:
                    p = {"xsize": 16, "ysize": 12, "ylim": [43.75, 47.05], "xlim": [6.70, 14.05], "s": 150}
                pos = np.stack((graph[0]['high'].lon.cpu().numpy(), graph[0]['high'].lat.cpu().numpy()),axis=-1)
                zones_file='./utils/Italia.txt'
                zones = create_zones(zones_file=zones_file)
                if target_type == "precipitation":
                    y_pred_plot = torch.expm1(y_pred_val)
                    y_plot = torch.expm1(y_val)
                else:
                    min_val = 250
                    max_val= 350
                    y_pred_plot = y_pred_val * (max_val - min_val) + min_val
                    y_plot = y_val * (max_val - min_val) + min_val
                y_pred_plot[~train_mask_val] = torch.nan
                y_plot[~train_mask_val] = torch.nan
                # convert to cpu and numpy
                _, indices = torch.sort(t)
                indices = indices.cpu().numpy()
                y_pred_plot = y_pred_plot.cpu().numpy()[:,indices]
                y_plot = y_plot.cpu().numpy()[:,indices]
                with open(args.output_path+"indices.pkl", 'wb') as f:
                    pickle.dump(indices, f)
                if target_type == "temperature":
                    v_min=270
                    v_max=290
                    range_bins=[250,350,1]
                    ylim_pdf=None
                    ylim_diurnal_cycles=[270,300]
                    cmap="coolwarm"
                    unit="[K]"
                    map_title="average temperature"
                    diurnal_cycle_title="temperature diurnal cycle"
                    aggr=np.nanmean
                else:
                    v_min=0
                    v_max=2750
                    range_bins=[0,75,1]
                    ylim_pdf=None
                    ylim_diurnal_cycles=[0.5,3.5]
                    cmap="jet"
                    unit="[mm/h]"
                    map_title="cumulative precipitation"
                    diurnal_cycle_title="intensity diurnal cycle"
                    aggr=np.nansum
                fig_avg = plot_maps(pos, y_pred_plot, y_plot, pr_min=v_min, aggr=aggr, pr_max=v_max,
                    title="", legend_title=unit, cmap=cmap, zones=zones, x_size=p["xsize"], y_size=p["ysize"],
                    font_size_title=20, font_size=20, cbar_title_size=20, s=p["s"], ylim=p["ylim"], xlim=p["xlim"], cbar_y=0.95, subtitle_x=0.55)
                fig_pdf = plot_pdf(y_pred_plot, y_plot, range=range_bins, ylim=ylim_pdf, xlabel=unit)
                fig_avg_dc = plot_diurnal_cycles(y_pred_plot, y_plot, ylim=ylim_diurnal_cycles, ylablel=unit)
                
                # Apply mask
                y_pred_val, y_val = y_pred_val[train_mask_val], y_val[train_mask_val]
                    
                if "quantized" in args.loss_fn:
                    w_val = w_val[train_mask_val]
                    loss_val, loss_term1_val, loss_term2_val = loss_fn(y_pred_val.flatten(), y_val.flatten(), w_val.flatten())
                # elif args.loss_fn == "quantized_loss_scaled":
                #     w_val = w_val[train_mask_val]
                #     loss_val = loss_fn(y_pred_val.flatten(), y_val.flatten(), w_val.flatten(), epoch)
                else:
                    loss_val = loss_fn(y_pred_val.flatten(), y_val.flatten())

                accelerator.log({map_title: [wandb.Image(fig_avg)], "pdf": [wandb.Image(fig_pdf)], diurnal_cycle_title: [wandb.Image(fig_avg_dc)]}, step=step)
                plt.close()                

            if lr_scheduler is not None:
                # lr_scheduler.step(loss_val.item())
                lr_scheduler.step()
            
            if "quantized" in args.loss_fn:
                accelerator.log({'epoch':epoch, 'validation loss (1GPU)': loss_val_1gpu.item(), 'validation loss': loss_val.item(),
                                 'validation mse loss': loss_term1_val.item(),'validation quantized loss': loss_term2_val.item(),
                                 'lr': np.mean(lr_scheduler.get_last_lr())}, step=step)
            else:
                accelerator.log({'epoch':epoch, 'validation loss (1GPU)': loss_val_1gpu.item(), 'validation loss': loss_val.item(),
                                 'lr': np.mean(lr_scheduler.get_last_lr())}, step=step)
                

#-----------------------------------------------------
#----------------------- TEST ------------------------
#-----------------------------------------------------


class Tester(object):

    def test(self, model, dataloader, args, accelerator=None):
        model.eval()
        step = 0 

        pr = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                # Regressor
                y_pred = model(graph)
                #pr.append(torch.expm1(y_pred)) 
                pr.append(y_pred)
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        pr = torch.stack(pr)
        times = torch.stack(times)

        return pr, times
    
    def test_encoding(self, model_reg, dataloader, args, accelerator=None):
        model_reg.eval()
        step = 0 

        pr_reg = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                # Regressor
                y_pred_reg = model_reg(graph)
                pr_reg.append(y_pred_reg)
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        pr_reg = torch.stack(pr_reg)
        times = torch.stack(times)

        return pr_reg, times
    
    def test_cl_reg(self, model_cl, model_reg, dataloader, args, accelerator=None):
        model_cl.eval()
        model_reg.eval()
        step = 0 

        pr_cl = []
        pr_reg = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                y_pred_cl = model_cl(graph)
                y_pred_reg = model_reg(graph)
                
                # Classifier
                pr_cl.append(y_pred_cl)
                
                # Regressor
                pr_reg.append(y_pred_reg)

                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1

        pr_cl = torch.stack(pr_cl)
        pr_reg = torch.stack(pr_reg)
        times = torch.stack(times)

        return pr_cl, pr_reg, times


#-----------------------------------------------------
#-------------------- VALIDATION ---------------------
#-----------------------------------------------------
    

class Validator(object):

    def validate_cl(self, model, dataloader, loss_fn, accelerator, alpha=0.75, gamma=2):

        model.eval()

        # loss_meter = AverageMeter()
        # acc_meter = AverageMeter()
        # acc_class0_meter = AverageMeter()
        # acc_class1_meter = AverageMeter()

        y_pred_val = []
        y_val = []
        train_mask_val = []
    
        with torch.no_grad():    
            for graph in dataloader:

                train_mask = graph["high"].train_mask

                if train_mask.sum() == 0:
                    continue

                # Classifier
                y_pred = model(graph).squeeze()
                y = graph['high'].y

                # Gather results from other GPUs
                accelerator.wait_for_everyone()
                all_y_pred, all_y, all_train_mask = accelerator.gather_for_metrics((y_pred, y, train_mask))

                # Append the batch results
                y_pred_val.append(all_y_pred)
                y_val.append(all_y)
                train_mask_val.append(all_train_mask)

            # Create tensors
            train_mask_val = torch.stack(train_mask_val)
            y_pred_val = torch.stack(y_pred_val)[train_mask_val]
            y_val = torch.stack(y_val)[train_mask_val]

            print(train_mask_val.shape, train_mask_val.sum(), y_pred_val.shape, y_val.shape)

            # Compute metrics on all validation dataset            
            loss = loss_fn(y_pred_val, y_val, alpha, gamma, reduction="mean")
            acc = accuracy_binary_one(y_pred_val, y_val)
            acc_class0, acc_class1 = accuracy_binary_one_classes(y_pred_val, y_val)

        return loss.item(), acc.item(), acc_class0.item(), acc_class1.item()

            # loss_meter.update(val=loss.item(), n=y_pred.shape[0])
            # acc_meter.update(val=acc.item(), n=y_pred.shape[0])
            # if not acc_class0.isnan():
            #     acc_class0_meter.update(val=acc_class0.item(), n=y_pred.shape[0])
            # if not acc_class1.isnan():
            #     acc_class1_meter.update(val=acc_class1.item(), n=y_pred.shape[0])

        # return loss_meter.avg, acc_meter.avg, acc_class0_meter.avg, acc_class1_meter.avg

    def validate_reg(self, model, dataloader, loss_fn, accelerator, args):

        all_loss_meter_val = AverageMeter()

        model.eval()
        
        with torch.no_grad(): 
            for i, graph in enumerate(dataloader):

                train_mask = graph["high"].train_mask
                if train_mask.sum() == 0:
                    continue
                y = graph['high'].y
                w = graph['high'].w
                    
                # Regressor
                y_pred = model(graph).squeeze()

                # Gather from all processes for metrics
                all_y_pred, all_y, all_w, all_train_mask = accelerator.gather((y_pred, y, w, train_mask))

                # Apply mask
                y_pred, y, w = y_pred[train_mask], y[train_mask], w[train_mask]
                all_y_pred, all_y, all_w = all_y_pred[all_train_mask], all_y[all_train_mask], all_w[all_train_mask]

                #loss, _, _ = loss_fn(y_pred, y, w)
                all_loss, _, _ = loss_fn(all_y_pred, all_y, all_w)    
                    
                all_loss_meter_val.update(val=all_loss.item(), n=all_y_pred.shape[0])

        return all_loss_meter_val.avg


    def validate_reg_all(self, model, dataloader, accelerator, args):
        
        train_mask = []
        y_pred = []
        y = []
        w = []

        model.eval()
        
        with torch.no_grad(): 
            for i, graph in enumerate(dataloader):

                train_mask.append(graph["high"].train_mask)                    
                y.append(graph['high'].y)
                w.append(graph['high'].w)
                y_pred.append(model(graph).squeeze())

            # Create tensors
            train_mask = torch.stack(train_mask)
            y_pred = torch.stack(y_pred)
            y = torch.stack(y)
            w = torch.stack(w)
                    
        return y_pred, y, w, train_mask
    
        
