import torch
from utils.extractors.extract_prediction import extract_prediction
from utils.helpers.tools import write_log


class Predictor(object):

    def predict(self, model, dataloader, pred_size, args, accelerator=None):

        if accelerator is not None:
            device = accelerator.device
        else:
            device = 'cpu'

        model.eval()
        step = 0

        y_pred_lists = None  # one accumulation list per output stream, built on first batch
        idxs_list = []
        with torch.no_grad():
            for graph in dataloader:

                out = model(graph)
                y_pred = extract_prediction(out, loss_name=args.loss_name, args=args)

                # normalize to a tuple: works the same whether extract_prediction
                # returns a single tensor (e.g. MSE) or several (e.g. p, shape,
                # scale for the Bernoulli-Gamma NLL)
                if not isinstance(y_pred, (tuple, list)):
                    y_pred = (y_pred,)
                if y_pred_lists is None:
                    y_pred_lists = [[] for _ in y_pred]

                idxs = torch.atleast_2d(torch.tensor(graph.idxs, device=device))

                if args.batch_size > 1:
                    # Retrieve graphs for individual time instances
                    n_nodes = graph["high"].num_nodes
                    B = y_pred[0].shape[0] // n_nodes
                    y_pred = tuple(torch.atleast_2d(yp.view(B, n_nodes, -1)) for yp in y_pred)
                    idxs = torch.atleast_2d(idxs)

                for lst, yp in zip(y_pred_lists, y_pred):
                    lst.append(yp)  # (time, nodes)
                idxs_list.append(idxs)

                if step % 100 == 0:
                    write_log(f"\nStep {step} done.", args, accelerator, 'a')
                step += 1

        idxs = torch.stack(idxs_list).squeeze()
        if accelerator is not None:
            accelerator.wait_for_everyone()
            idxs_all = accelerator.gather(idxs)
        else:
            idxs_all = idxs

        # Indices to ensure data are sorted correctly (shared across all output streams)
        _, idxs_sorted = torch.sort(idxs_all)
        idxs_sorted = idxs_sorted.cpu().numpy()
        idxs_sorted = idxs_sorted.squeeze()[:pred_size]

        outputs = []
        for lst in y_pred_lists:
            # Stack list into a tensor
            y_pred = torch.stack(lst).squeeze()

            if accelerator is not None:
                accelerator.wait_for_everyone()
                y_pred_all = accelerator.gather(y_pred)
            else:
                y_pred_all = y_pred

            # Squeeze, swapaxes, convert to cpu and numpy
            y_pred_all = y_pred_all.cpu().numpy()
            y_pred_all = y_pred_all.squeeze()[:pred_size, :][idxs_sorted, :]
            y_pred_all = y_pred_all.swapaxes(0, 1)  # (nodes, time)
            outputs.append(y_pred_all)

        print(f"\ny_pred_all shapes: {[o.shape for o in outputs]}, idxs_sorted.shape: {idxs_sorted.shape}")

        # backward compatible: single output -> single array
        # multiple outputs -> tuple of arrays, in the same order extract_prediction returned them
        if len(outputs) == 1:
            return outputs[0], idxs_sorted
        return outputs, idxs_sorted