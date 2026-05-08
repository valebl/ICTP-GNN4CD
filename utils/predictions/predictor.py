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

                idxs = torch.atleast_2d(torch.tensor(graph.idxs, device=accelerator.device))

                if args.batch_size > 1:
                    # Retrieve graphs for individual time instances
                    n_nodes = graph["high"].num_nodes
                    B = y_pred.shape[0] // n_nodes
                    y_pred = y_pred.view(B, n_nodes, -1)
                    y_pred = torch.atleast_2d(y_pred) # from (N,) to (1,N)
                    idxs = torch.atleast_2d(idxs)

                y_pred_list.append(y_pred) # (time, nodes)
                idxs_list.append(idxs)     

                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        # Stack lists into tensors
        y_pred = torch.stack(y_pred_list).squeeze()
        idxs = torch.stack(idxs_list).squeeze()

        if accelerator is not None:
            accelerator.wait_for_everyone()
            y_pred_all = accelerator.gather(y_pred)
            idxs_all = accelerator.gather(idxs)
            
        # Squeeze, swapaxes, convert to cpu and numpy
        y_pred_all = y_pred_all.squeeze().swapaxes(0,1).cpu().numpy()[:, :pred_size] # (nodes, time)
        idxs_all = idxs_all.squeeze().squeeze()[:pred_size]

        # Indices to ensure data are sorted correctly
        _, idxs_sorted = torch.sort(idxs_all)
        idxs_sorted = idxs_sorted.cpu().numpy()

        y_pred_all = y_pred_all[:, idxs_sorted]

        print(f"\ny_pred_all.shape: {y_pred_all.shape}, idxs_sorted.shape: {idxs_sorted.shape}")

        return y_pred_all, idxs_sorted