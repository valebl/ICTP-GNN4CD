import torch
from utils.extractors.extract_prediction import extract_prediction

class Predictor(object):

    def predict(self, model, dataloader, pred_size, args, accelerator=None):
        model.eval()
        step = 0

        y_pred_list = []
        idxs_list = []
        with torch.no_grad():
            for graph in dataloader:
                
                out = model(graph)
                y_pred = extract_prediction(out, loss_name=args.loss_name)

                device = accelerator.device if accelerator is not None else y_pred.device
                idxs = torch.as_tensor(graph.idxs, device=device).reshape(-1)
                batch_size = idxs.numel()
                n_nodes = y_pred.shape[0] // batch_size
                y_pred = y_pred.view(batch_size, n_nodes, -1)

                y_pred_list.append(y_pred) # (time, nodes)
                idxs_list.append(idxs)     

                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        # Concatenate batches into (time, nodes, samples)
        y_pred = torch.cat(y_pred_list, dim=0)
        idxs = torch.cat(idxs_list, dim=0)

        if accelerator is not None:
            accelerator.wait_for_everyone()
            y_pred_all = accelerator.gather(y_pred)
            idxs_all = accelerator.gather(idxs)


        # Indices to ensure data are sorted correctly
        idxs_all = idxs_all.reshape(-1)[:pred_size]
        _, idxs_sorted = torch.sort(idxs_all)
        idxs_sorted = idxs_sorted.cpu().numpy()

        # Squeeze, swapaxes, convert to cpu and numpy
        y_pred_all = y_pred_all.cpu().numpy()
        y_pred_all = y_pred_all[:pred_size, :, :][idxs_sorted, :, :] # (time, nodes, samples)
        y_pred_all = y_pred_all.swapaxes(0,1) # (nodes, time)

        print(f"\ny_pred_all.shape: {y_pred_all.shape}, idxs_sorted.shape: {idxs_sorted.shape}")

        return y_pred_all, idxs_sorted
