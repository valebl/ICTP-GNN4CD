import pickle
import numpy as np
import random
import sys
import torch
from datetime import date


#-----------------------------------------------------
#------------------ GENERAL UTILITIES ----------------
#-----------------------------------------------------

def printf(format, *args):
    sys.stdout.write(format % args)

def write_log(s, args=None, accelerator=None, mode='a'):
    r'''
    Writes the given string to the log file
    Args:
        s (str): the string
    Returns:
        None
    '''
    if accelerator is None or accelerator.is_main_process:
        if args is not None:
            with open(args.output_path + args.log_file, mode) as f:
                f.write(s)
        else:
            print(s)

def use_gpu_if_possible():
    r'''
    Checks if the gpu is available
    Returns:
        True is cuda is available, False otherwise
    '''    
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def set_seed_everything(seed):
    r'''
    Sets the seed for generating random numbers
    Args:
        seed (int): the desired seed
    Returns:
        None
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#-----------------------------------------------------
#--------------- PREPROCESSING UTILITIES -------------
#-----------------------------------------------------


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


def date_to_idxs_new(year_start, month_start, day_start, year_end, month_end, day_end,
                 first_year, first_month=1, first_day=2):
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

    start_idx = (date(int(year_start), int(month_start), int(day_start)) - date(int(first_year), int(first_month), int(first_day))).days 
    end_idx = (date(int(year_end), int(month_end), int(day_end)) - date(int(first_year), int(first_month), int(first_day))).days 

    return start_idx, end_idx



def find_not_all_nan_times(target_train):
    r'''
    Define a mask to ignore time indexes with all nan values
    Args:
        target_train (tensor)
    Returns:
        train_idxs (tensor)
    '''
    mask_not_all_nan = []
    initial_time_dim = target_train.shape[1]
    for t in range(initial_time_dim):
        nan_sum = target_train[:,t].isnan().sum()
        mask_not_all_nan.append(nan_sum < target_train.shape[0])
    mask_not_all_nan = torch.stack(mask_not_all_nan)
    mask_not_all_nan[:24] = True
    idxs_not_all_nan = torch.argwhere(mask_not_all_nan)

    return idxs_not_all_nan
    

def find_not_all_nan_times_new(target_train):
    r'''
    Define a mask to ignore time indexes with all nan values
    Args:
        target_train (tensor)
    Returns:
        train_idxs (tensor)
    '''
    mask_not_all_nan = []
    initial_time_dim = target_train.shape[1]
    for t in range(initial_time_dim):
        nan_sum = target_train[:,t].isnan().sum()
        mask_not_all_nan.append(nan_sum < target_train.shape[0])
    mask_not_all_nan = torch.stack(mask_not_all_nan)
    mask_not_all_nan[:24] = True
    idxs_not_all_nan = torch.argwhere(mask_not_all_nan)

    return idxs_not_all_nan
    

def derive_train_val_idxs(train_year_start, train_month_start, train_day_start, train_year_end, train_month_end,
                         train_day_end, first_year, model_name, idxs_not_all_nan=None, validation_year=None, args=None, accelerator=None):
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
        val_start_idx, val_end_idx = date_to_idxs(validation_year, 1, 1, validation_year, 12, 31, first_year)

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
        if "3h" in model_name:
            train_idxs_list = [i for i in train_idxs_list if i in idxs_not_all_nan and i % 3 == 0]
            val_idxs_list = [i for i in val_idxs_list if i in idxs_not_all_nan and i % 3 == 0]
        else:    
            train_idxs_list = [i for i in train_idxs_list if i in idxs_not_all_nan]
            val_idxs_list = [i for i in val_idxs_list if i in idxs_not_all_nan]
    
    train_idxs = torch.tensor(train_idxs_list)
    val_idxs = torch.tensor(val_idxs_list)

    if args is not None:
        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path + "train_idxs.pkl", 'wb') as f:
                pickle.dump(torch.tensor(train_idxs), f)
            with open(args.output_path + "val_idxs.pkl", 'wb') as f:
                pickle.dump(torch.tensor(val_idxs), f)
                
    return train_idxs, val_idxs


def derive_train_val_idxs_new(train_year_start, train_month_start, train_day_start, train_year_end, train_month_end,
                         train_day_end, first_year, model_name, idxs_not_all_nan=None, validation_year=None, args=None, accelerator=None):
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
    train_start_idx, train_end_idx = date_to_idxs_new(train_year_start, train_month_start, train_day_start,
                                                  train_year_end, train_month_end, train_day_end, first_year)

    # Derive the idxs corresponding to the training period
    if validation_year is None:
        pass
    else:
        val_start_idx, val_end_idx = date_to_idxs_new(validation_year, 1, 1, validation_year, 12, 31, first_year)

    # We need the previous 2 days to make the prediction at time t
    if train_start_idx < 2:
        train_start_idx = 2
        

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
            val_idxs_list = [i for i in val_idxs_list if i in idxs_not_all_nan]
    
    train_idxs = torch.tensor(train_idxs_list)
    val_idxs = torch.tensor(val_idxs_list)

    if args is not None:
        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path + "train_idxs.pkl", 'wb') as f:
                pickle.dump(torch.tensor(train_idxs), f)
            with open(args.output_path + "val_idxs.pkl", 'wb') as f:
                pickle.dump(torch.tensor(val_idxs), f)
                
    return train_idxs, val_idxs



def _merge_intervals(intervals):
    """Merge overlapping/contiguous [start, end) intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:  # overlap or touch
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _subtract_intervals(base_interval, cut_intervals):
    """
    Subtract a set of [start, end) cut_intervals from base_interval [bs, be).
    Returns list of remaining intervals.
    """
    bs, be = base_interval
    if bs >= be:
        return []

    cut_intervals = _merge_intervals([(max(bs, s), min(be, e)) for s, e in cut_intervals if e > bs and s < be])
    if not cut_intervals:
        return [(bs, be)]

    remaining = []
    cur = bs
    for s, e in cut_intervals:
        if s > cur:
            remaining.append((cur, s))
        cur = max(cur, e)
    if cur < be:
        remaining.append((cur, be))
    return remaining

def derive_train_val_idxs_CORDEX(
    train_year_start, train_month_start, train_day_start,
    train_year_end, train_month_end, train_day_end,
    first_year,
    model_name,
    idxs_not_all_nan=None,
    validation_years=None,          # NEW: list[int]
    validation_year_start=None,     # NEW: int (optional)
    K=None,                         # NEW: int (optional)
    args=None,
    accelerator=None
):
    r'''
    Computes the train and validation indexes.

    Validation can be provided as:
      - validation_years: list of years, OR
      - validation_year_start + K: consecutive years [start, start+K-1]

    STRICT RULE (old behavior): validation must be fully outside training.
    That is, every validation interval must be entirely before or entirely after
    the training interval (no overlap allowed).
    Returns:
        train_idxs (tensor)
        val_idxs (tensor)
    '''
    # ---- Train interval ----
    train_start_idx, train_end_idx = date_to_idxs(
        train_year_start, train_month_start, train_day_start,
        train_year_end, train_month_end, train_day_end,
        first_year
    )

    # Need previous 24h
    train_start_idx = max(train_start_idx, 2)

    if train_start_idx >= train_end_idx:
        raise Exception("Train start idx is not smaller than train end idx.")

    # ---- Resolve validation years ----
    if validation_years is not None:
        years = list(validation_years)
    elif validation_year_start is not None and K is not None:
        years = list(range(validation_year_start, validation_year_start + K))
    else:
        years = None

    if not years:
        raise ValueError("Provide validation_years OR (validation_year_start and K).")

    # ---- Build validation intervals (one per year) as [start, end) ----
    val_intervals = []
    for y in years:
        vs, ve = date_to_idxs(y, 1, 1, y, 12, 31, first_year)

        # Need previous 24h
        vs = max(vs, 2)

        if vs >= ve:
            raise Exception(f"Val start idx is not smaller than val end idx for year {y}.")

        val_intervals.append((vs, ve))

    # Merge contiguous/overlapping validation years into larger blocks
    val_intervals = _merge_intervals(val_intervals)

    # Sanity check train interval
    if train_start_idx >= train_end_idx:
        raise Exception("Train start idx is not smaller than train end idx.")

    # Compute train = train_interval minus val_intervals
    train_intervals = _subtract_intervals((train_start_idx, train_end_idx), val_intervals)

        # Expand intervals into index lists
    train_idxs_list = []
    for s, e in train_intervals:
        train_idxs_list.extend(range(s, e))


    val_idxs_list = []
    for s, e in val_intervals:
        val_idxs_list.extend(range(s, e))

    # ---- Remove idxs for which all graph nodes have nan target ----
    if idxs_not_all_nan is not None:
        idxs_set = set(idxs_not_all_nan)
        train_idxs_list = [i for i in train_idxs_list if i in idxs_set]
        val_idxs_list   = [i for i in val_idxs_list   if i in idxs_set]

    train_idxs = torch.tensor(train_idxs_list, dtype=torch.long)
    val_idxs   = torch.tensor(val_idxs_list, dtype=torch.long)

    # ---- Save if requested ----
    if args is not None:
        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path + "train_idxs.pkl", "wb") as f:
                pickle.dump(train_idxs, f)
            with open(args.output_path + "val_idxs.pkl", "wb") as f:
                pickle.dump(val_idxs, f)

    return train_idxs, val_idxs

                   
# def compute_input_statistics(x_low, x_high, args, accelerator=None):

#     r'''
#     input:
#      -x_low:
#      -x_high:
#      -args
#      -accelerator
#     returns:
#      a tuple containing 4 objects:
#        - means_low, stds_low: mean values and standard deviation of low-resol predictors
#        - means_high, stds_high: mean values and standard deviation  of high-resol predictors
#     '''

#     write_log(f'\nComputing statistics for the low-res input data.', args, accelerator, 'a')

#     # Low-res data
#     means_low = np.zeros((5))
#     stds_low = np.zeros((5))
#     for var in range(5):
#         m = np.nanmean(x_low[:,:,var,:]) # num_nodes, time, vars, levels
#         s = np.nanstd(x_low[:,:,var,:])  # num_nodes, time, vars, levels
#         means_low[var] = m
#         stds_low[var] = s

#     write_log(f'\nComputing statistics for the high-res input data.', args, accelerator, 'a')

#     # High-res data
#     if x_high.size()[1] > 1:
#         means_high = torch.tensor([x_high[:,0].mean(), x_high[:,1:].mean()])
#         stds_high = torch.tensor([x_high[:,0].std(), x_high[:,1:].std()])
#     else:
#         means_high = torch.tensor(x_high.mean())
#         stds_high = torch.tensor(x_high.std())        

#     # Write the standardized data to disk
#     with open(args.output_path + "means_low.pkl", 'wb') as f:
#         pickle.dump(means_low, f)
#     with open(args.output_path + "stds_low.pkl", 'wb') as f:
#         pickle.dump(stds_low, f)
#     with open(args.output_path + "means_high.pkl", 'wb') as f:
#         pickle.dump(means_high, f)
#     with open(args.output_path + "stds_high.pkl", 'wb') as f:
#         pickle.dump(stds_high, f)

#     return means_low, stds_low, means_high, stds_high




# #this function standardize all the predictors
# def standardize_input(x_low, x_high, means_low, stds_low, means_high, stds_high, args=None, accelerator=None):

#     write_log(f'\nStandardizing the low-res input data.', args, accelerator, 'a')

#     # Preallocate memory efficiently
#     x_low_standard = torch.empty_like(x_low, dtype=torch.float32)
    
#     # Standardize the data
#     for var in range(5):
#         x_low_standard[:,:,var,:] = (x_low[:,:,var,:]-means_low[var])/stds_low[var]  # num_nodes, time, vars, levels
    
#     write_log(f'\nStandardizing the high-res input data.', args, accelerator, 'a')

#     # Standardize the data
#     x_high_standard = torch.zeros((x_high.size()), dtype=torch.float32)
    
#     #in case we have the land use fields
#     if x_high.size()[1] > 1:
#         x_high_standard[:,0] = (x_high[:,0] - means_high[0]) / stds_high[0]
#         x_high_standard[:,1:] = (x_high[:,1:] - means_high[1]) / stds_high[1]
#     else:
#     #or if we have only orography
#         x_high_standard = (x_high - means_high) / stds_high

#     return x_low_standard, x_high_standard


#Try 0201
def compute_input_statistics(x_low, x_high, args, accelerator=None):
    r'''
    input: 
        -x_low: shape (num_nodes, time, vars, levels) - numpy array or torch tensor
        -x_high: shape (num_nodes, features) - torch tensor
        -args
        -accelerator
    returns: a tuple containing 4 objects:
        - means_low, stds_low: mean values and standard deviation of low-resol predictors
        - means_high, stds_high: mean values and standard deviation of high-resol predictors
    '''
    write_log(f'\nComputing statistics for the low-res input data.', args, accelerator, 'a')
    
    # tensor to ndarray
    if isinstance(x_low, torch.Tensor):
        x_low_np = x_low.numpy()
    else:
        x_low_np = x_low
    
    # Low-res data - for each variable and each level separately
    num_vars = x_low_np.shape[2]  # 5 variables
    num_levels = x_low_np.shape[3]  # 3 levels
    
    means_low = np.zeros((num_vars, num_levels))
    stds_low = np.zeros((num_vars, num_levels))
    
    for var in range(num_vars):
        for level in range(num_levels):
            m = np.nanmean(x_low_np[:, :, var, level])
            s = np.nanstd(x_low_np[:, :, var, level])
            s = max(s, 1e-6)  # avoid division by zero
            means_low[var, level] = m
            stds_low[var, level] = s
            
            write_log(f'  Var {var}, Level {level}: mean={m:.4f}, std={s:.4f}', 
                     args, accelerator, 'a')
    
    write_log(f'\nComputing statistics for the high-res input data.', args, accelerator, 'a')
    
    # High-res data
    if x_high.size()[1] > 1:
        means_high = torch.tensor([x_high[:,0].mean().item(), x_high[:,1:].mean().item()])
        stds_high = torch.tensor([x_high[:,0].std().item(), x_high[:,1:].std().item()])
        stds_high = torch.clamp(stds_high, min=1e-6)  # avoid division by zero
    else:
        means_high = torch.tensor(x_high.mean().item())
        stds_high = torch.tensor(x_high.std().item())
        stds_high = torch.clamp(stds_high, min=torch.tensor(1e-6))
    
    # Write the standardized data to disk
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + "means_low.pkl", 'wb') as f:
            pickle.dump(means_low, f)
        with open(args.output_path + "stds_low.pkl", 'wb') as f:
            pickle.dump(stds_low, f)
        with open(args.output_path + "means_high.pkl", 'wb') as f:
            pickle.dump(means_high, f)
        with open(args.output_path + "stds_high.pkl", 'wb') as f:
            pickle.dump(stds_high, f)
    
    return means_low, stds_low, means_high, stds_high


def standardize_input(x_low, x_high, means_low, stds_low, means_high, stds_high, 
                      args=None, accelerator=None):
    r'''
    Standardize input data using provided statistics
    
    Args:
        x_low: low-resolution input, shape (num_nodes, time, vars, levels)
        x_high: high-resolution input
        means_low: mean values for low-res, shape (vars, levels)
        stds_low: std values for low-res, shape (vars, levels)
        means_high: mean values for high-res
        stds_high: std values for high-res
    
    Returns:
        x_low_standard, x_high_standard: standardized inputs
    '''
    write_log(f'\nStandardizing the low-res input data.', args, accelerator, 'a')
    
    # Standardize low-res data
    x_low_standard = torch.empty_like(x_low, dtype=torch.float32)
    
    num_vars = x_low.shape[2]
    num_levels = x_low.shape[3]
    
    # Standardize low-res data for each variable and each level separately
    for var in range(num_vars):
        for level in range(num_levels):
            mean_val = float(means_low[var, level])
            std_val = float(stds_low[var, level])
            x_low_standard[:, :, var, level] = (x_low[:, :, var, level] - mean_val) / std_val
    
    write_log(f'\nStandardizing the high-res input data.', args, accelerator, 'a')
    
    # Standardize high-res data
    x_high_standard = torch.zeros((x_high.size()), dtype=torch.float32)
    
    if x_high.size()[1] > 1:
        # If there is a land use field
        x_high_standard[:, 0] = (x_high[:, 0] - means_high[0]) / stds_high[0]
        x_high_standard[:, 1:] = (x_high[:, 1:] - means_high[1]) / stds_high[1]
    else:
        # Or only terrain
        x_high_standard = (x_high - means_high) / stds_high
    
    # Print standardized data statistics for verification
    write_log(f'\nStandardized low-res data statistics:', args, accelerator, 'a')
    # Calculate statistics of finite values
    finite_mask = torch.isfinite(x_low_standard)
    if finite_mask.any():
        valid_data = x_low_standard[finite_mask]
        write_log(f'  Mean: {valid_data.mean():.6f}, Std: {valid_data.std():.6f}', 
                 args, accelerator, 'a')
        write_log(f'  Min: {valid_data.min():.6f}, Max: {valid_data.max():.6f}', 
                 args, accelerator, 'a')
    
    return x_low_standard, x_high_standard
#0201 try end```



def prepare_target(target_train, model_type, threshold = 0.1):
    
    # derive two masks:
    # - mask_nan, i.e. where the target is nan
    # - mask_geq_threshold, i.e. where the target is larger than the preferred threshold (now 0.1mm)
    mask_threshold = target_train < threshold #mm
    mask_nan = torch.isnan(target_train)

    # set to 0.0 everything below sensitivity threshold
    target_train[mask_threshold] = 0.0
    # round to comply with instrument sensitivity
    target_train = torch.round(target_train, decimals=1)

    if model_type == "C":
        #-- CLASSIFIER --#        
        target_train = torch.where(target_train >= threshold, 1, 0).float()
    elif model_type == "R":
        #-- REGRESSOR ON pr >=threshold --#    
        target_train = torch.log1p(target_train)
        target_train[mask_threshold] = torch.nan
    elif model_type == "Rall":
        #-- REGRESSOR ON ALL --#
        target_train = torch.log1p(target_train)

    target_train[mask_nan] = torch.nan

    return target_train

def prepare_target_Rall(target_train, threshold = 0.1):
    
    # derive two masks:
    # - mask_nan, i.e. where the target is nan
    # - mask_geq_threshold, i.e. where the target is larger than the preferred threshold (now 0.1mm)
    mask_threshold = target_train < threshold #mm
    mask_nan = torch.isnan(target_train)

    # set to 0.0 everything below sensitivity threshold
    target_train[mask_threshold] = 0.0
    # round to comply with instrument sensitivity
    target_train = torch.round(target_train, decimals=2)

   
    #-- target for REGRESSOR ON ALL --#
    target_train = torch.log1p(target_train)

    target_train[mask_nan] = torch.nan

    return target_train


# def prepare_target_T_Rall(target_train, procedure = 'z-score'):
    
#     # derive one mask:
#     # - mask_nan, i.e. where the target is nan
#     mask_nan = torch.isnan(target_train)
 
#     target_train[mask_nan] = torch.nan
    
#     # round to comply with instrument sensitivity
#     target_train = torch.round(target_train, decimals=4)

   
#     #-- target normalization for REGRESSOR ON ALL --#
#     if procedure=='z-score':
#         mean_tensor= target_train.mean(axis=0)
#         std_tensor= target_train.std(axis=0)
#         mean_absolute= mean_tensor.mean()
#         std_absolute= target_train.std()
#         target_train= (target_train - mean_absolute)/std_absolute
        
#     else: #procedure normalization
#         max_tensor=torch.amax(target_train.T,0)
#         min_tensor=torch.amin(target_train.T,0)
#         target_train = (target_train - target_train.min())/(target_train.max()-target_train.min())

#     return target_train


#0201 try
# Modified normalization, and stats were added for inverse transform
def prepare_target_T_Rall(target_train, procedure='z-score'):
    r'''
    Prepare target for temperature regression
    
    Args:
        target_train: raw target values
        procedure: 'z-score' or 'minmax'
    
    Returns:
        target_train: normalized target
        stats: dict containing normalization statistics for inverse transform
    '''
    mask_nan = torch.isnan(target_train)
    #target_train = torch.round(target_train, decimals=4)    
    
    # Target normalization
    if procedure == 'z-score':
        valid_data = target_train[~mask_nan]
        mean_val = valid_data.mean()
        std_val = valid_data.std()
        std_val = torch.clamp(std_val, min=1e-6)  # avoid division by zero
        
        target_train = (target_train - mean_val) / std_val
        stats = {'mean': mean_val.item(), 'std': std_val.item(), 'procedure': 'z-score'}
        
    else:  # minmax normalization
        valid_data = target_train[~mask_nan]
        min_val = valid_data.min()
        max_val = valid_data.max()
        range_val = max_val - min_val
        range_val = torch.clamp(range_val, min=1e-6)
        
        target_train = (target_train - min_val) / range_val
        stats = {'min': min_val.item(), 'max': max_val.item(), 'procedure': 'minmax'}
    
    target_train[mask_nan] = torch.nan
    
    return target_train, stats
#0201 try end


def derive_qmse_bins(target_train, train_idxs, args, accelerator, threshold=0.1):

    bins = np.arange(np.log1p(threshold), np.log1p(350), np.log1p(0.25))
    if args.model_type == "Rall":
        bins = np.insert(bins, 0, np.log1p(0))
    # consider only the time indices that are part of the training set
    values_unif_log, edges_unif_log = np.histogram(target_train[:,train_idxs].numpy(), bins=bins, density=False)
    # Assign bins to targets
    target_bins = np.digitize(target_train.numpy(), edges_unif_log, right=False).astype(float) - 1

    nbins = (np.nanmax(target_bins) + 1).astype(int)
    if nbins > len(values_unif_log):
        write_log(f"\nBins min: {np.nanmin(target_bins).astype(int)}, bins max: {np.nanmax(target_bins).astype(int)}, nbins: {nbins}, len weights: {len(values_unif_log)}", args, accelerator, 'a')
        target_bins[target_bins == nbins -1] = nbins - 2
        nbins = nbins - 1
        write_log("\nUpdating last bin...", args, accelerator, 'a')
    write_log(f"\nbins min: {np.nanmin(target_bins).astype(int)}, bins max: {np.nanmax(target_bins).astype(int)}, nbins: {nbins}", args, accelerator, 'a')
    target_bins = torch.tensor(target_bins)
    target_bins[torch.isnan(target_train)] = torch.nan

    return target_bins


    

#-----------------------------------------------------
#------------------- TRAIN UTILITIES -----------------
#-----------------------------------------------------

def check_freezed_layers(model, log_path, log_file, accelerator):
    for name, param in model.named_parameters():
        n_param = param.numel() 
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nLayer {name} requires_grad = {param.requires_grad} and has {n_param} parameters") 



