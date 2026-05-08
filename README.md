## GNN4CD-CORDEXML - branch Dev_flow_matching

This folder builds upon the branch Refactory_Valentina, therefore the new models are added using the **REGISTRY** logic.

Specifically, two new models are being tested:
- GNN4CD_CFM_Model
- GNN4CD_GraphCFM_Model

To comply with the rest of the code, each models always return a single object `out` which is different when the model is used for training, validation or prediction:
- training: `out = torch.cat([v_pred, v_target], dim=1)`
- validation: `out = torch.cat([v_pred, v_target, samples_mean], dim=1)`
- prediction: `out = torch.tensor([samples_mean])`

For this reason, samples_mean is directly computed by the model._sample method when the model.forward method is called in inference mode and returned as explained below.

The new `utils/losses/CFM_Loss` takes `v_pred, v_target` from `out` and computes `MSE(v_pred, v_target)`.

The new extractor `utils/extractors/cfm_extractor` distinguishes between validation and prediction to extract `samples_mean` from `out`.
