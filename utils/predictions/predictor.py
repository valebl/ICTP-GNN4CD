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

        model_ = model.module if hasattr(model, "module") else model
        is_ensemble = hasattr(model_, "generate_ensemble")

        y_pred_lists = None  # one accumulation list per output stream, built on first batch
        idxs_list = []
        with torch.no_grad():
            for graph in dataloader:

                # use generate_ensemble() when the model supports it (e.g. the
                # noisy CRPS model) since a plain model(graph) call in eval mode
                # returns a single deterministic sample, which CRPS can't score.
                # generate_ensemble() returns (M, N_high, output_dim) -- M
                # first, matching the model's native convention, preserved
                # as-is through everything below.
                if is_ensemble:
                    out = model_.generate_ensemble(graph)  # (M, batch*n_nodes, output_dim)
                else:
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
                    if is_ensemble:
                        # each yp is (M, batch_size*n_nodes, ...): M is the
                        # model's native leading axis and is left untouched
                        # here -- only the flattened batch*nodes axis gets
                        # split into (B, n_nodes). No permute at this stage;
                        # M stays exactly where the model put it.
                        def _split_batch_nodes(yp):
                            M = yp.shape[0]
                            B = yp.shape[1] // n_nodes
                            return torch.atleast_2d(yp.reshape(M, B, n_nodes, *yp.shape[2:]))
                        y_pred = tuple(_split_batch_nodes(yp) for yp in y_pred)
                    else:
                        B = y_pred[0].shape[0] // n_nodes
                        y_pred = tuple(torch.atleast_2d(yp.view(B, n_nodes, -1)) for yp in y_pred)
                    idxs = torch.atleast_2d(idxs)

                for lst, yp in zip(y_pred_lists, y_pred):
                    lst.append(yp)  # (M, B, nodes) for ensembles, (B, nodes, output_dim) otherwise
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
            # Stack list into a tensor. dim=0 inserts the time/step axis in
            # front of whatever each entry's shape already is -- required so
            # the time axis is axis 0 for accelerator.gather() (which
            # concatenates across processes along axis 0, assuming that's
            # the sharded axis). For ensembles this gives (time, M, B, nodes);
            # M deliberately is NOT axis 0 of this intermediate tensor, since
            # time has to be for distributed gather to work correctly -- M is
            # moved to its final position explicitly below, after gathering.
            y_pred = torch.stack(lst, dim=0).squeeze()

            if accelerator is not None:
                accelerator.wait_for_everyone()
                y_pred_all = accelerator.gather(y_pred)
            else:
                y_pred_all = y_pred

            y_pred_all = y_pred_all.cpu().numpy()
            y_pred_all = y_pred_all.squeeze()[:pred_size, :][idxs_sorted, :]

            if is_ensemble and y_pred_all.ndim == 3:
                # currently (time, M, nodes) -- move to (nodes, time, M) so
                # axis 0 is node-indexed (matching mask/plotting conventions
                # elsewhere) and M is last (matching the axis=-1 mean used
                # downstream in predict.py / the validation loop)
                y_pred_all = y_pred_all.transpose(2, 0, 1)
            else:
                y_pred_all = y_pred_all.swapaxes(0, 1)  # (nodes, time)

            outputs.append(y_pred_all)

        print(f"\ny_pred_all shapes: {[o.shape for o in outputs]}, idxs_sorted.shape: {idxs_sorted.shape}")

        # backward compatible: single output -> single array
        # multiple outputs -> tuple of arrays, in the same order extract_prediction returned them
        if len(outputs) == 1:
            return outputs[0], idxs_sorted
        return outputs, idxs_sorted