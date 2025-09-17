import time
import sys
import pickle
import numpy as np
import random

import torch
import torch.nn.functional as F
from datetime import date
from calendar import monthrange

import numpy as np


######################################################
#------------------ GENERAL UTILITIES ---------------
######################################################


def write_log(s, args=None, accelerator=None, mode='a'):
    if accelerator is None or accelerator.is_main_process:
        if args is not None:
            with open(args.output_path + args.log_file, mode) as f:
                f.write(s)
        else:
            print(s)


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
    # mask_not_all_nan = [torch.tensor(True) for i in range(24)
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
    
    # train_idxs = torch.tensor(train_idxs_list)
    # val_idxs = torch.tensor(val_idxs_list)

    if args is not None:
        if accelerator is None or accelerator.is_main_process:
            with open(args.output_path + "train_idxs.pkl", 'wb') as f:
                pickle.dump(torch.tensor(train_idxs_list), f)
            with open(args.output_path + "val_idxs.pkl", 'wb') as f:
                pickle.dump(torch.tensor(val_idxs_list), f)
                
    return train_idxs_list, val_idxs_list


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


def remove_extremes_idxs_from_training(
    train_idxs,
    extremes_dict={1: [22,11,2016,25,11,2016], 2:[9,10,2015,15,10,2015], 3:[31,10,2015,1,11,2015], 4:[13,9,2015,14,9,2015], 5:[9,11,2014,13,11,2014],
                   6: [9,10,2014,13,10,2014], 7:[11,11,2012,12,11,2012], 8:[4,11,2011,4,11,2011], 9:[22,11,2011,22,11,2011], 10:[22,10,2011,22,10,2011]}):
        
    extremes_idxs_list = []

    for _, l in extremes_dict.items():
        day_start=l[0]
        month_start=l[1]
        year_start=l[2]
        day_end=l[3]
        month_end=l[4]
        year_end=l[5]

        start_idx, end_idx = date_to_idxs(day_start=day_start, month_start=month_start, year_start=year_start,
                                                  day_end=day_end, month_end=month_end, year_end=year_end, first_year=2001)
        
        _ = [extremes_idxs_list.append(i) for i in range(start_idx, end_idx)]

    print(extremes_idxs_list)

    train_idxs_upd = [i for i in train_idxs if i not in extremes_idxs_list]

    print(len(train_idxs), len(train_idxs_upd))
    return train_idxs_upd

                   
def compute_input_statistics(x_low, x_high, args, accelerator=None):

    write_log(f'\nComputing statistics for the low-res input data.', args, accelerator, 'a')

    # Low-res data
    if args.stats_mode == "var":
        means_low = np.zeros((5))
        stds_low = np.zeros((5))
        for var in range(5):
            m = np.nanmean(x_low[:,:,var,:]) # num_nodes, time, vars, levels
            s = np.nanstd(x_low[:,:,var,:])  # num_nodes, time, vars, levels
            means_low[var] = m
            stds_low[var] = s
    elif args.stats_mode == "field":
        means_low = np.zeros((5,5))
        stds_low = np.zeros((5,5))
        for var in range(5):
            for lev in range(5):
                m = np.nanmean(x_low[:,:,var,lev])  # num_nodes, time, vars, levels
                s = np.nanstd(x_low[:,:,var,lev])   # num_nodes, time, vars, levels
                means_low[var, lev] = m
                stds_low[var, lev] = s
    else:
        raise Exception("Arg 'stats_mode' should be either 'var' or 'field'")

    write_log(f'\nComputing statistics for the high-res input data.', args, accelerator, 'a')

    # High-res data
    if x_high.size()[1] > 1:
        means_high = torch.tensor([x_high[:,0].mean(), x_high[:,1:].mean()])
        stds_high = torch.tensor([x_high[:,0].std(), x_high[:,1:].std()])
    else:
        means_high = x_high.mean()
        stds_high = x_high.std()     

    # Write the standardized data to disk
    with open(args.output_path + "means_low.pkl", 'wb') as f:
        pickle.dump(means_low, f)
    with open(args.output_path + "stds_low.pkl", 'wb') as f:
        pickle.dump(stds_low, f)
    with open(args.output_path + "means_high.pkl", 'wb') as f:
        pickle.dump(means_high, f)
    with open(args.output_path + "stds_high.pkl", 'wb') as f:
        pickle.dump(stds_high, f)

    return means_low, stds_low, means_high, stds_high


def standardize_input(x_low, x_high, means_low, stds_low, means_high, stds_high, args=None, accelerator=None, stats_mode_default="var"):

    write_log(f'\nStandardizing the low-res input data.', args, accelerator, 'a')

    # Preallocate memory efficiently
    x_low_standard = torch.empty_like(x_low, dtype=torch.float32)

    if args is not None:
        stats_mode = args.stats_mode
    else:
        stats_mode = stats_mode_default

    # Standardize the data
    if stats_mode == "var":
        for var in range(5):
            x_low_standard[:,:,var,:] = (x_low[:,:,var,:]-means_low[var])/stds_low[var]  # num_nodes, time, vars, levels
    elif stats_mode == "field":
        for var in range(5):
            for lev in range(5):
                x_low_standard[:,:,var,lev] = (x_low[:,:,var,lev]-means_low[var, lev])/stds_low[var, lev]  # num_nodes, time, vars, levels
    else:
        raise Exception("Arg 'stats_mode' should be either 'var' or 'field'")

    write_log(f'\nStandardizing the high-res input data.', args, accelerator, 'a')

    # Standardize the data
    x_high_standard = torch.zeros((x_high.size()), dtype=torch.float32)
    
    if x_high.size()[1] > 1:
        x_high_standard[:,0] = (x_high[:,0] - means_high[0]) / stds_high[0]
        x_high_standard[:,1:] = (x_high[:,1:] - means_high[1]) / stds_high[1]
    else:
        x_high_standard = x_high - means_high / stds_high

    return x_low_standard, x_high_standard
    

######################################################
#------------------- TRAIN UTILITIES -----------------
######################################################





#-----------------------------------------------------
#-------------------- MODEL PARAMETERS ---------------
#-----------------------------------------------------

def check_freezed_layers(model, log_path, log_file, accelerator):
    for name, param in model.named_parameters():
        n_param = param.numel() 
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nLayer {name} requires_grad = {param.requires_grad} and has {n_param} parameters") 



