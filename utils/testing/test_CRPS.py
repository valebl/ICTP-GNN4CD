import torch


class CRPS_Tester(object):
    """
    Inference for noisy models trained with CRPS.

    In eval mode + no_grad, the noisy model returns a single tensor (one noisy
    pass with fresh Gaussian noise). To build a probabilistic ensemble, we run
    the dataloader `ensemble_size` times.
    """

    def test_CRPS(self, model, dataloader, args, ensemble_size: int = 10,
                  accelerator=None):
        model.eval()

        ensemble_preds = []
        idxs_stacked = None

        for member_idx in range(ensemble_size):
            y_pred_list = []
            idxs_list = []
            step = 0
            with torch.no_grad():
                for graph in dataloader:
                    device = accelerator.device if accelerator is not None else graph['high'].y.device
                    idx = torch.atleast_2d(torch.tensor(graph.idxs, device=device))
                    idxs_list.append(idx)

                    y_pred = model(graph)
                    y_pred_list.append(y_pred)

                    if step % 100 == 0:
                        if accelerator is None or accelerator.is_main_process:
                            with open(args.output_path + args.log_file, 'a') as f:
                                f.write(f"\nMember {member_idx+1}/{ensemble_size} - step {step} done.")
                    step += 1

            y_pred_member = torch.stack(y_pred_list).squeeze()
            ensemble_preds.append(y_pred_member)

            if idxs_stacked is None:
                idxs_stacked = torch.stack(idxs_list).squeeze()

            if accelerator is None or accelerator.is_main_process:
                with open(args.output_path + args.log_file, 'a') as f:
                    f.write(f"\nMember {member_idx+1}/{ensemble_size} done.")

        # (ensemble_size, n_test, N_high)
        ensemble = torch.stack(ensemble_preds, dim=0)
        return ensemble, idxs_stacked
