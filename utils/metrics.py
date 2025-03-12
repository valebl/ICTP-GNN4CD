import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, ks_2samp, entropy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

#-----------------------------------------------------
#---------------------- METRICS ----------------------
#-----------------------------------------------------

class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    (from the Deep Learning tutorials of DSSC)
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_binary_one(prediction, target, reduction="mean"):
    prediction_class = torch.where(prediction > 0.0, 1.0, 0.0)
    correct_items = (prediction_class == target).float()
    if reduction is None:
        return correct_items
    elif reduction == "mean":
        return torch.mean(correct_items)
        # return correct_items.sum() / prediction.shape[0]


def accuracy_binary_one_classes(prediction, target, reduction="mean"):
    prediction_class = torch.where(prediction > 0.0, 1.0, 0.0)
    correct_items = prediction_class == target
    correct_items_class0 = correct_items[target==0.0]
    correct_items_class1 = correct_items[target==1.0]
    if reduction is None:
        return correct_items_class0, correct_items_class1
    elif reduction == "mean":
        if correct_items_class0.shape[0] > 0:
            acc_class0 = correct_items_class0.sum() / correct_items_class0.shape[0]
        else:
            acc_class0 = torch.tensor(torch.nan)
        if correct_items_class1.shape[0] > 0:
            acc_class1 = correct_items_class1.sum() / correct_items_class1.shape[0]
        else:
            acc_class1 = torch.tensor(torch.nan)
        return acc_class0, acc_class1


def compute_metrics(y_pred, y_true, threshold=0.1):
    metrics = {}

    # Compute precipitation
    pr_pred = np.expm1(y_pred)
    pr_true = np.expm1(y_true)
    pr_true[pr_true < threshold] = 0
    pr_true = np.round(pr_true, decimals=1)
    pr_pred[pr_pred < threshold] = 0

    pr_pred_spatial_avg = np.nanmean(pr_pred, axis=1)
    pr_true_spatial_avg = np.nanmean(pr_true, axis=1)
    pr_pred_spatial_p99 = np.nanpercentile(pr_pred, q=99, axis=1)
    pr_true_spatial_p99 = np.nanpercentile(pr_true, q=99, axis=1)
    pr_pred_spatial_p999 = np.nanpercentile(pr_pred, q=99.9, axis=1)
    pr_true_spatial_p999 = np.nanpercentile(pr_true, q=99.9, axis=1)

    # spatial biases
    spatial_bias_percentage = (pr_pred_spatial_avg - pr_true_spatial_avg) / (pr_true_spatial_avg + 1e-6) * 100
    spatial_p99_bias_percentage = (pr_pred_spatial_p99 - pr_true_spatial_p99) / (pr_true_spatial_p99 + 1e-6) * 100
    spatial_p999_bias_percentage = (pr_pred_spatial_p999 - pr_true_spatial_p999) / (pr_true_spatial_p999 + 1e-6) * 100

    mask_not_nan_y = ~np.isnan(y_true.flatten())
    mask_not_nan = ~np.isnan(pr_true.flatten())
    y_true = y_true.flatten()[mask_not_nan_y]
    y_pred = y_pred.flatten()[mask_not_nan_y]
    
    # Basic error metrics
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))

    metrics['Avg spatial Bias (over)'] = np.mean(spatial_bias_percentage[spatial_bias_percentage>0])
    metrics['Avg spatial Bias (under)'] = np.mean(spatial_bias_percentage[spatial_bias_percentage<=0])
    metrics['Avg spatial p99 Bias (over)'] = np.nanmean(spatial_p99_bias_percentage[spatial_p99_bias_percentage>0])
    metrics['Avg spatial p99 Bias (under)'] = np.nanmean(spatial_p99_bias_percentage[spatial_p99_bias_percentage<=0])
    metrics['Avg spatial p99.9 Bias (over)'] = np.nanmean(spatial_p999_bias_percentage[spatial_p999_bias_percentage>0])
    metrics['Avg spatial p99.9 Bias (under)'] = np.nanmean(spatial_p999_bias_percentage[spatial_p999_bias_percentage<=0])
    
    # Spatial correlation
    metrics['Pearson Corr'], _ = pearsonr(y_pred, y_true)
    metrics['Spearman Corr'], _ = spearmanr(y_pred, y_true)
    
    # Probability of Detection and False Alarm Ratio for extremes
    pr_true_p99 = np.nanpercentile(pr_true, q=99)
    hits = np.nansum((pr_pred >= pr_true_p99) & (pr_true >= pr_true_p99))
    false_alarms = np.nansum((pr_pred >= pr_true_p99) & (pr_true < pr_true_p99))
    actual_extremes = np.nansum(pr_true >= pr_true_p99)
    predicted_extremes = np.nansum(pr_pred >= pr_true_p99)
    
    metrics['POD (p99)'] = hits / (actual_extremes + 1e-6) # Probability of Detection
    metrics['FAR (p99)'] = false_alarms / (predicted_extremes + 1e-6)  # False Alarm Ratio
    
    # To avoid numerical issues due to unrealistic predictions
    pr_pred[np.isinf(pr_pred)] = np.nan

    # distributions comparison
    metrics['Earth Mover Distance'] = wasserstein_distance(pr_true.flatten()[mask_not_nan], pr_pred.flatten()[mask_not_nan])
    metrics['KL Divergence'] = entropy(pr_true.flatten()[mask_not_nan] + 1e-6, pr_pred.flatten()[mask_not_nan] + 1e-6)
    ks_stat, p_value = ks_2samp(pr_true.flatten()[mask_not_nan], pr_pred.flatten()[mask_not_nan])
    metrics['KS Statistic'] = ks_stat
    metrics['KS p-value'] = p_value

    # PDF comparison
    hist_y_true, _ = np.histogram(pr_true.flatten()[mask_not_nan], bins=np.arange(0,200,0.1).astype(np.float32), density=False)
    hist_y_pred, _ = np.histogram(pr_pred.flatten()[mask_not_nan], bins=np.arange(0,200,0.1).astype(np.float32), density=False)

    metrics["PDF Cos Sim"] = cosine_similarity((hist_y_true/hist_y_true.sum()).reshape(1, -1), (hist_y_pred/hist_y_pred.sum()).reshape(1, -1))
    metrics["PDF Chi Squared"] = 0.5 * np.sum((hist_y_true/hist_y_true.sum() - hist_y_pred/hist_y_pred.sum()) ** 2 / (hist_y_true/hist_y_true.sum() + hist_y_pred/hist_y_pred.sum() + 1e-6))
    
    return metrics