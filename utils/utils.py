import time
import sys
import pickle
import torch.nn as nn
import numpy as np
import random

import torch
import torch.nn.functional as F
from datetime import datetime, date
from calendar import monthrange

import numpy as np
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, ks_2samp, entropy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

######################################################
#------------------ GENERAL UTILITIES ---------------
######################################################


def write_log(s, args, accelerator=None, mode='a'):
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, mode) as f:
            f.write(s)


def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def set_seed_everything(seed):
    r"""sets the seed for generating random numbers
    Args:
        seed (int): the desired seed
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


######################################################
#--------------- PREPROCESSING UTILITIES -------------
######################################################


def cut_window(lon_min, lon_max, lat_min, lat_max, lon, lat, pr, z, mask_land, *argv):
    r'''
    Derives a new version of the longitude, latitude and precipitation
    tensors, by only retaining the values inside the specified lon-lat rectangle
    Args:
        lon_min, lon_max, lat_min, lat_max: integers
        lon, lat, z, pr: tensors
    Returns:
        The new tensors with the selected values
    '''

    bool_lon = torch.logical_and(lon >= lon_min, lon <= lon_max)
    bool_lat = torch.logical_and(lat >= lat_min, lat <= lat_max)
    bool_both = torch.logical_and(bool_lon, bool_lat)
    lon_sel = lon[bool_both]
    lat_sel = lat[bool_both]
    z_sel = z[bool_both]
    pr_sel = pr[:,bool_both]
    v = []
    for arg in argv:
        v.append(arg[bool_both])
    if mask_land is None:
        return lon_sel, lat_sel, pr_sel, z_sel, None, *v
    else:
        mask_land = mask_land[bool_both]
        return lon_sel, lat_sel, pr_sel, z_sel, mask_land, *v


def retain_valid_nodes(lon, lat, pr, e, mask_land=None, *argv):
    r'''
    Selects only the nodes for which precipitation is not
    nan in at least one timestamp. All the other nodes are
    ignored (e.g. the sea values in GRIPHO). If a land mask
    is provided, non-land points are also ignored.
    Args:
        lon (torch.tensor): longitude for each spatial point
        lat (torch.tensor): latitude for each spatial point
        pr (torch.tensor): precipitation for each spatial point
        e (torch.tensor): elevation for each spatial point
        mask_land (torch.tensor, optional): a mask for the land points
    Returns:
        The valid points for each input tensor
    '''

    valid_nodes = ~torch.isnan(pr).all(dim=0)
    if mask_land is not None:
        valid_nodes = np.logical_and(valid_nodes, ~torch.isnan(mask_land))
    lon = lon[valid_nodes]
    lat = lat[valid_nodes]
    pr = pr[:,valid_nodes]
    e = e[valid_nodes]
    v = []
    for arg in argv:
        v.append(arg[valid_nodes])
    return lon, lat, pr, e, *v


def derive_edge_indexes_within(lon_radius, lat_radius, lon_n1 ,lat_n1, lon_n2, lat_n2):
    r'''
    Derives edge_indexes within two sets of nodes based on specified lon, lat distances
    Args:
        lon_n1 (torch.tensor): longitudes of all first nodes in the edges
        lat_n1 (torch.tensor): latitudes of all fisrt nodes in the edges
        lon_n2 (torch.tensor): longitudes of all second nodes in the edges
        lat_n2 (torch.tensor): latitudes of all second nodes in the edges
    Return:
        The edge_indexes tensor
    '''

    edge_indexes = []

    lonlat_n1 = torch.concatenate((lon_n1.unsqueeze(-1), lat_n1.unsqueeze(-1)),dim=-1)
    lonlat_n2 = torch.concatenate((lon_n2.unsqueeze(-1), lat_n2.unsqueeze(-1)),dim=-1)

    for ii, xi in enumerate(lonlat_n1):
        
        bool_lon = abs(lon_n2 - xi[0]) < lon_radius
        bool_lat = abs(lat_n2 - xi[1]) < lat_radius
        bool_both = torch.logical_and(bool_lon, bool_lat).bool()
        jj_list = torch.nonzero(bool_both)
        xj_list = lonlat_n2[bool_both]
        for jj, xj in zip(jj_list, xj_list):
            if not torch.equal(xi, xj):
                edge_indexes.append(torch.tensor([ii, jj]))

    edge_indexes = torch.stack(edge_indexes)

    return edge_indexes


def derive_edge_indexes_low2high(lon_n1 ,lat_n1, lon_n2, lat_n2, k, undirected=False, use_edge_weight=True):
    '''
    Derives edge_indexes between two sets of nodes based on specified number of neighbours k
    Args:
        lon_n1 (torch.tensor): longitudes of all first nodes in the edges
        lat_n1 (torch.tensor): latitudes of all fisrt nodes in the edges
        lon_n2 (torch.tensor): longitudes of all second nodes in the edges
        lat_n2 (torch.tensor): latitudes of all second nodes in the edges
        k (int): the number of neighbours
    Return:
        The edge_indexes tensor
    '''
    edge_index = []
    edge_weight = []

    lonlat_n1 = torch.concatenate((lon_n1.unsqueeze(-1), lat_n1.unsqueeze(-1)),dim=-1)
    lonlat_n2 = torch.concatenate((lon_n2.unsqueeze(-1), lat_n2.unsqueeze(-1)),dim=-1)

    dist = torch.cdist(lonlat_n2.double(), lonlat_n1.double(), p=2, compute_mode='donot_use_mm_for_euclid_dist')
    _ , neighbours = dist.topk(k, largest=False, dim=-1)

    for n_n2 in range(lonlat_n2.shape[0]):
        for n_n1 in neighbours[n_n2,:]:
            edge_index.append(torch.tensor([n_n1, n_n2]))
            edge_weight.append(dist[n_n2, n_n1])
            if undirected:
                edge_index.append(torch.tensor([n_n2, n_n1]))

    edge_index = torch.stack(edge_index)
    edge_weight = torch.stack(edge_weight)
    
    if use_edge_weight:
        return edge_index, edge_weight
    else:
        return edge_index


def date_to_idxs(year_start, month_start, day_start, year_end, month_end, day_end,
                 first_year, first_month=1, first_day=1):
    r'''
    Computes the start and end idxs crrespnding to the specified period, with respect to a
    reference date.
    Args:
        year_start (int): year at which period starts
        month_start (int): month at which period starts
        day_start (int): day at which period starts
        year_end (int): year at which period ends
        month_end (int): month at which period ends
        day_end (int): day at which period ends
        first_year (int): reference year to compute the idxs
    Returns:
        The start and end idxs for the period
    '''

    start_idx = (date(int(year_start), int(month_start), int(day_start)) - date(int(first_year), int(first_month), int(first_day))).days * 24
    end_idx = (date(int(year_end), int(month_end), int(day_end)) - date(int(first_year), int(first_month), int(first_day))).days * 24 + 24

    return start_idx, end_idx
    

def find_not_all_nan_times(target_train):
    r'''
    Define a mask to ignore time indexes with all nan values
    Args:
        target_train (tensor)
    Returns:
        train_idxs (tensor)
        val_idxs (tensor)
    '''
    mask_not_all_nan = [torch.tensor(True) for i in range(24)]
    initial_time_dim = target_train.shape[1]
    for t in range(initial_time_dim):
        nan_sum = target_train[:,t].isnan().sum()
        mask_not_all_nan.append(nan_sum < target_train.shape[0])
    mask_not_all_nan = torch.stack(mask_not_all_nan)
    idxs_not_all_nan = torch.argwhere(mask_not_all_nan)

    return idxs_not_all_nan
    

def derive_train_val_idxs(train_year_start, train_month_start, train_day_start, train_year_end, train_month_end,
                         train_day_end, first_year, idxs_not_all_nan=None, validation_year=None):
    r'''
    Computes the train and validation indexes
    Args:
        train_year_start (int): year at which period starts
        train_month_start (int): month at which period starts
        train_day_start (int): day at which period starts
        train_year_end (int): year at which period ends
        train_month_end (int): month at which period ends
        train_day_end (int): day at which period ends
        first_year (int): reference year to compute the idxs
        validation_year (int): year considered for validation
    Returns:
        train_idxs (tensor)
        val_idxs (tensor)
    '''
    # Derive the idxs corresponding to the training period
    train_start_idx, train_end_idx = date_to_idxs(train_year_start, train_month_start, train_day_start,
                                                  train_year_end, train_month_end, train_day_end, first_year)

    # Derive the idxs corresponding to the training period
    if validation_year is None:
        pass
    else:
        # To make the seasonal plots we consider from 01/12/YYYY-1 to 30/11/YYYY
        val_start_idx, val_end_idx = date_to_idxs(validation_year-1, 12, 1, validation_year, 11, 30, first_year)

    # We need the previous 24h to make the prediction at time t
    if train_start_idx < 24:
        train_start_idx = 24
        
    if val_start_idx < 24:
        val_start_idx = 24

    if train_start_idx >= train_end_idx:
        raise Exception("Train start idxs is not larger than train end idx.")
    if val_start_idx >= val_end_idx:
        raise Exception("Val start idxs is not larger than val end idx.")
            
    # Val year before or after train years
    if train_start_idx >= val_end_idx or train_end_idx <= val_start_idx:
        train_idxs_list = [*range(train_start_idx, train_end_idx)]
        val_idxs_list = [*range(val_start_idx, val_end_idx)]
    # Val year inside train years
    elif val_start_idx > train_start_idx and val_end_idx < train_end_idx:
        train_idxs_list = [*range(train_start_idx, val_start_idx)] + [*range(val_end_idx,  train_end_idx)]
        val_idxs_list = [*range(val_start_idx, val_end_idx)]
    else:
        raise Exception("Partially overlapping train and validation periods are not supported." +
                        "Val must be before, after or completely inside train years.")

    # Remove the idxs for which all graph nodes have nan target
    if idxs_not_all_nan is not None:
        train_idxs_list = [i for i in train_idxs_list if i in idxs_not_all_nan]
        val_idxs = [i for i in val_idxs_list if i in idxs_not_all_nan]
    
    train_idxs = torch.tensor(train_idxs_list)
    val_idxs = torch.tensor(val_idxs_list)

    return train_idxs, val_idxs


def derive_train_val_test_idxs_random_months(train_year_start, train_month_start, train_day_start, train_year_end, train_month_end,
                         train_day_end, first_year, idxs_not_all_nan=None, args=None, accelerator=None):
    r'''
    Computes the train, validation and test indeces, assuming that validation and test are periods
    of one year where the months are chosen randomly among the whole period. All remaining months
    are part of the training datset
    Args:
        train_year_start (int): year at which period starts
        train_month_start (int): month at which period starts
        train_day_start (int): day at which period starts
        train_year_end (int): year at which period ends
        train_month_end (int): month at which period ends
        train_day_end (int): day at which period ends
        first_year (int): reference year to compute the idxs
        validation_year (int): year considered for validation
    Returns:
        train_idxs (tensor)
        val_idxs (tensor)
        test_idxs (tensor)
    '''
    # Derive the idxs corresponding to the period
    train_start_idx, train_end_idx = date_to_idxs(train_year_start, train_month_start, train_day_start,
                                                  train_year_end, train_month_end, train_day_end, first_year)
    
    if train_start_idx < 24:
        train_start_idx = 24
    
    # lists where for each month I save the year from which I will take it from
    val_year_per_month={}
    test_year_per_month={}

    train_idxs_list = []
    val_idxs_list = []
    test_idxs_list = []

    years_val = [*range(train_year_start, train_year_end+1)]
    years_test = [*range(train_year_start, train_year_end+1)]
    
    for month in range(1,13):
        if years_val == []:
            years_val = [*range(train_year_start, train_year_end+1)]
        if years_test == []:
            years_test = [*range(train_year_start, train_year_end+1)]
        if month > train_month_start: # we always ignore the first month
            if month <= train_month_end:
                available_years = [*range(train_year_start, train_year_end+1)]
            else:
                available_years = [*range(train_year_start, train_year_end)]
        else:
            available_years = [*range(train_year_start+1, train_year_end)]

        available_years_val = [y for y in available_years if y in years_val]
        chosen_year_val = random.sample(available_years_val, 1)[0]
        
        available_years_test = [y for y in available_years if y in years_test]
        if chosen_year_val in available_years_test:
            available_years_test.remove(chosen_year_val)
        
        chosen_year_test = random.sample(available_years_test, 1)[0]

        years_val.remove(chosen_year_val)
        years_test.remove(chosen_year_test)

        # Save for later log
        val_year_per_month[str(month)] = chosen_year_val
        test_year_per_month[str(month)] = chosen_year_test

        val_month_start_idx, val_month_end_idx = date_to_idxs(chosen_year_val, month, 1, chosen_year_val,
                                            month, monthrange(chosen_year_val, month)[1], first_year)
        test_month_start_idx, test_month_end_idx = date_to_idxs(chosen_year_test, month, 1, chosen_year_test,
                                            month, monthrange(chosen_year_test, month)[1], first_year)
        
        val_idxs_list += [*range(val_month_start_idx, val_month_end_idx)]
        test_idxs_list += [*range(test_month_start_idx, test_month_end_idx)]

    train_idxs_list = [*range(train_start_idx, train_end_idx)]
    train_idxs_list = [i for i in train_idxs_list if i not in val_idxs_list and i not in test_idxs_list]

    train_idxs = torch.tensor(train_idxs_list)
    val_idxs = torch.tensor(val_idxs_list)
    test_idxs = torch.tensor(test_idxs_list)

    if args is not None:
        with open(args.output_path+"log.txt", 'a') as f:
            f.write(f"\nValidation year: {val_year_per_month}\nTest year: {test_year_per_month}")
        if accelerator is None or accelerator.is_main_process:
            write_log(f"\nValidation year: {val_year_per_month}\nTest year: {test_year_per_month}", args, accelerator, 'a')
            with open(args.output_path + "train_idxs.pkl", 'wb') as f:
                pickle.dump(train_idxs, f)
            with open(args.output_path + "val_idxs.pkl", 'wb') as f:
                pickle.dump(val_idxs, f)
            with open(args.output_path + "test_idxs.pkl", 'wb') as f:
                pickle.dump(test_idxs, f)
    
    return train_idxs, val_idxs

    
    

######################################################
#------------------- TRAIN UTILITIES -----------------
######################################################


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

#-----------------------------------------------------
#--------------- CUSTOM LOSS FUNCTIONS ---------------
#-----------------------------------------------------


class weighted_mse_loss():
    def __call__(input_batch, target_batch, weights):
        e = (input_batch - target_batch) ** 2
        return torch.sum(weights * e) / torch.sum(weights)

class weighted_mae_loss():
    def __call__(input_batch, target_batch, weights):
        e = torch.abs(input_batch - target_batch)
        return torch.sum(weights * e) / torch.sum(weights)

class quantized_loss():
    '''
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, alpha=0.025):
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        print(f"alpha: {self.alpha}")

    def __call__(self, prediction_batch, target_batch, bins):
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        loss_quantized = 0
        bins = bins.int()
        for b in torch.unique(bins):
            mask_b = (bins == b)
            loss_quantized += self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])
        return loss_mse + self.alpha * loss_quantized, loss_mse, loss_quantized
    

class quantized_loss_scaled():
    '''
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, gamma=0.5, scale=0.001):
        self.gamma = gamma
        self.scale = scale
        print(f"gamma: {self.gamma}, scale: {self.scale}")

    def __call__(self, prediction_batch, target_batch, bins):
        loss_quantized = 0
        bins = bins.int()
        for b in torch.unique(bins):
            mask_b = (bins == b)
            loss_quantized += torch.sum((prediction_batch[mask_b] - target_batch[mask_b])**2) * (1/torch.sum(mask_b))**self.gamma
        return self.scale * loss_quantized
    

class quantized_loss_mod():
    '''
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, alpha=1):
        self.mse_loss = nn.MSELoss(reduction="none")
        self.alpha = alpha
        print(f"alpha: {self.alpha}")

    def __call__(self, prediction_batch, target_batch, bins):
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        loss_quantized = []
        bins = bins.int()
        alpha_vector = []
        for b in torch.unique(bins):
            mask_b = (bins == b)
            loss_quantized += alpha_vector * self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])
        loss_quantized = torch.sum(torch.stack(loss_quantized))
        return loss_mse + self.alpha * loss_quantized, loss_mse, loss_quantized


class quantized_loss_bins():
    '''
    Used in inference to derive the QMSE term for the individual bins
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, alpha=0.025):
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        print(f"alpha: {self.alpha}")

    def __call__(self, prediction_batch, target_batch, bins, accelerator, nbins=12):
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        loss_quantized = 0
        bins = bins.int()
        losses = torch.ones((nbins)).to(accelerator.device) * torch.nan
        for b in torch.unique(bins):
            mask_b = (bins == b)
            losses[b] = self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])
        return losses, None, None


#-----------------------------------------------------
#-------------------- MODEL PARAMETERS ---------------
#-----------------------------------------------------

def check_freezed_layers(model, log_path, log_file, accelerator):
    for name, param in model.named_parameters():
        n_param = param.numel() 
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nLayer {name} requires_grad = {param.requires_grad} and has {n_param} parameters") 


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

            # Define objects to track meters
            all_loss_meter = AverageMeter()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            acc_class0_meter = AverageMeter()
            acc_class1_meter = AverageMeter()

            # Validation
            loss_meter_val = AverageMeter()
            loss_meter_val_1_gpu = AverageMeter()
            acc_meter_val = AverageMeter()
            acc_class0_meter_val = AverageMeter()
            acc_class1_meter_val = AverageMeter()

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
            
            if args.loss_fn == "quantized_loss":
                loss_term1_meter = AverageMeter()
                loss_term2_meter = AverageMeter()

            start = time.time()
            
            # TRAIN
            for graph in dataloader_train:
            
                optimizer.zero_grad()
                y_pred = model(graph).squeeze()

                train_mask = graph['high'].train_mask
                y = graph['high'].y
                w = graph['high'].w

                # Gather from all processes for metrics
                all_y_pred, all_y, all_w, all_train_mask = accelerator.gather((y_pred, y, w, train_mask))

                # Apply mask
                y_pred, y, w = y_pred[train_mask], y[train_mask], w[train_mask]
                all_y_pred, all_y, all_w = all_y_pred[all_train_mask], all_y[all_train_mask], all_w[all_train_mask]

                # print(f"{accelerator.device} - all_y_pred.shape: {all_y_pred.shape}, all_y.shape: {all_y.shape}, all_w.shape: {all_w.shape}")
                
                if args.loss_fn == "quantized_loss":
                    loss, _, _ = loss_fn(y_pred, y, w)
                    all_loss, loss_term1, loss_term2 = loss_fn(all_y_pred, all_y, all_w)
                else:
                    loss = loss_fn(y_pred, y, w)
                    all_loss = loss_fn(all_y_pred, all_y, all_w)
                
                accelerator.backward(loss)
                #accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                step += 1
                
                # Log values to wandb
                loss_meter.update(val=loss.item(), n=y_pred.shape[0])    
                all_loss_meter.update(val=all_loss.item(), n=all_y_pred.shape[0])
                
                if args.loss_fn == "quantized_loss":
                    loss_term1_meter.update(val=loss_term1.item(), n=all_y_pred.shape[0])
                    loss_term2_meter.update(val=loss_term2.item(), n=all_y_pred.shape[0])
                    
                accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg, 'loss all avg': all_loss_meter.avg}, step=step)

            end = time.time()

            if args.loss_fn == "quantized_loss":
                accelerator.log({'epoch':epoch, 'train loss (1GPU)': loss_meter.avg, 'train loss': all_loss_meter.avg,
                                 'train mse loss': loss_term1_meter.avg, 'train quantized loss': loss_term2_meter.avg}, step=step)
            else:
                accelerator.log({'epoch':epoch, 'train loss (1GPU)': loss_meter.avg, 'train loss': all_loss_meter.avg}, step=step)

            write_log(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds." +
                      f"Loss - total: {all_loss_meter.sum:.4f} - average: {all_loss_meter.avg:.10f}. ", args, accelerator, 'a')
                    
            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/")
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")

            # VALIDATION
            # Validation is performed on all the validation dataset at once
            model.eval()

            y_pred_val = []
            y_val = []
            w_val = []
            train_mask_val = []

            with torch.no_grad():    
                for graph in dataloader_val:
                    # Append the data for the current epoch
                    train_mask_val.append(graph["high"].train_mask)            
                    y_pred_val.append(model(graph).squeeze())
                    y_val.append(graph['high'].y)
                    w_val.append(graph['high'].w)
                
                # Create tensors
                train_mask_val = torch.stack(train_mask_val)
                y_pred_val = torch.stack(y_pred_val)
                y_val = torch.stack(y_val)
                w_val = torch.stack(w_val)

                # Log validation metrics for 1GPU
                if args.loss_fn == "quantized_loss":
                    loss_val_1gpu,  _, _ = loss_fn(y_pred_val[train_mask_val], y_val[train_mask_val], w_val[train_mask_val])
                else:
                    loss_val_1gpu = loss_fn(y_pred_val[train_mask_val], y_val[train_mask_val], w_val[train_mask_val])

                # Gather from all processes for metrics
                y_pred_val, y_val, w_val, train_mask_val = accelerator.gather((y_pred_val, y_val, w_val, train_mask_val))

                # Apply mask
                y_pred_val, y_val, w_val = y_pred_val[train_mask_val], y_val[train_mask_val], w_val[train_mask_val]
                    
                if args.loss_fn == "quantized_loss":
                    loss_val, loss_term1_val, loss_term2_val = loss_fn(y_pred_val, y_val, w_val)
                else:
                    loss_val = loss_fn(y_pred_val, y_val, w_val)

            if lr_scheduler is not None:
                lr_scheduler.step(loss_val.item())
            
            if args.loss_fn == "quantized_loss":
                accelerator.log({'epoch':epoch, 'validation loss (1GPU)': loss_val_1gpu.item(), 'validation loss': loss_val.item(),
                                 'validation mse loss': loss_term1_val.item(),'validation quantized loss': loss_term2_val.item(),
                                 'lr': np.mean(lr_scheduler._last_lr)}, step=step)
            else:
                accelerator.log({'epoch':epoch, 'validation loss (1GPU)': loss_val_1gpu.item(), 'validation loss': loss_val.item(),
                                 'lr': np.mean(lr_scheduler._last_lr)}, step=step)
        


#-----------------------------------------------------
#----------------------- TEST ------------------------
#-----------------------------------------------------


class Tester(object):

    def test(self, model, dataloader,low_high_graph, args, accelerator=None):
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
    
    def test_encoding(self, model_reg, dataloader,low_high_graph, args, accelerator=None):
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
    
    def test_cl_reg(self, model_cl, model_reg, dataloader,low_high_graph, args, accelerator=None):
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
    
        

