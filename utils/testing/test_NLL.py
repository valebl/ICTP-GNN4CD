import torch

class NLL_Tester(object):

    def test_GaussianNLL(self, model, dataloader, args, accelerator=None):
        model.eval()
        step = 0

        mu_pred_list = []
        sigma_pred_list = []
        idxs_list = []
        with torch.no_grad():
            for graph in dataloader:

                idx = torch.atleast_2d(torch.tensor(graph.idxs, device=accelerator.device))
                idxs_list.append(idx)
                
                mu_pred, sigma_pred = model(graph)
                mu_pred_list.append(mu_pred)
                sigma_pred_list.append(sigma_pred)
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        mu = torch.stack(mu_pred_list).squeeze()
        sigma = torch.stack(sigma_pred_list).squeeze()
        idxs = torch.stack(idxs_list).squeeze()

        return mu, sigma, idxs

    def test_BernoulliGammaNLL(self, model, dataloader, args, accelerator=None):
        model.eval()
        step = 0

        y_pred_list = []
        idxs_list = []
        with torch.no_grad():
            for graph in dataloader:

                idx = torch.atleast_2d(torch.tensor(graph.idxs, device=accelerator.device))
                idxs_list.append(idx)
                
                p_pred, shape_pred, scale_pred = model(graph)
                y_pred = p_pred * (shape_pred * scale_pred)
                y_pred_list.append(y_pred)
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        y_pred = torch.stack(y_pred_list).squeeze()
        idxs = torch.stack(idxs_list).squeeze()

        return y_pred, idxs
