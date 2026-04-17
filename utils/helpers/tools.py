import numpy as np
import random
import torch
import numpy as np
import torch
import ast
import datetime
import cftime

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


def date_to_idxs_from_timeindex(
    year_start, month_start, day_start, time_index,
    year_end=None, month_end=None, day_end=None    
):
    """
    Compute start/end indices using the actual time_index array.
    Works with numpy.datetime64, Python datetime, and cftime calendars.
    """

    # Detect reference type from the first element
    ref = time_index[0]

    # Build start timestamp in Python datetime ----
    start_dt = datetime.datetime(year_start, month_start, day_start, 0, 0, 0)

    # Build end timestamp if needed ----
    if year_end is not None:
        end_dt = datetime.datetime(year_end, month_end, day_end, 23, 59, 59)

    # Convert Python datetime to same type as time_index
    def convert(ts, ref):
        # Case 1: time_index uses numpy.datetime64
        if isinstance(ref, np.datetime64):
            return np.datetime64(ts)

        # Case 2: time_index uses Python datetime
        if isinstance(ref, datetime.datetime):
            return ts

        # Case 3: time_index uses CFTime
        if isinstance(ref, cftime.datetime):
            return cftime.datetime(
                ts.year, ts.month, ts.day,
                ts.hour, ts.minute, ts.second,
                calendar=ref.calendar
            )

        raise TypeError(f"Unsupported time index type: {type(ref)}")

    start_ts = convert(start_dt, ref)
    if year_end is not None:
        end_ts = convert(end_dt, ref)

    # Searchsorted
    start_idx = int(np.searchsorted(time_index, start_ts, side="left"))

    if year_end is not None:
        end_idx = int(np.searchsorted(time_index, end_ts, side="right"))
        return start_idx, end_idx

    return start_idx

    

def prepare_target_for_train(target, target_type, train_idxs, stats_path="", mode="minmax"):
    target_train = target[:, train_idxs]
    if target_type == "precipitation":
        return np.log1p(target)
    elif target_type == "temperature":
        if mode == "minmax":
            min_val_temp = int(np.floor(np.nanmin(target)))
            max_val_temp = int(np.ceil(np.nanmax(target)))
            np.savez(stats_path+"tasmax_norm_stats.npz", mode="minmax", min=min_val_temp, max=max_val_temp)
            return (target - min_val_temp) / (max_val_temp - min_val_temp)
        elif mode == "z_score":
            mean_all = np.nanmean(target_train)
            std_all = np.nanstd(target_train)
            np.savez(stats_path+"tasmax_norm_stats.npz", mode=mode, mean=mean_all, std=std_all)
            return (target - mean_all) / std_all
        elif mode == "z_score_gp":
            mean_grid_points = np.nanmean(target_train, axis=1)[:, None]
            std_grid_points = (np.nanstd(target_train, axis=1) + 1e-6)[:, None]
            np.savez(stats_path+"tasmax_norm_stats.npz", mode=mode, mean=mean_grid_points, std=std_grid_points)
            return (target - mean_grid_points) / std_grid_points
    else:
        raise ValueError(f"Unknown target type: {target_type}")


def invert_normalization(y_norm, sigma_norm=None, stats_path=None):
    stats = np.load(stats_path+"tasmax_norm_stats.npz")
    if stats["mode"] == "minmax":
        min_val_temp = stats["min"]
        max_val_temp = stats["max"]
        y_pred = y_norm * (max_val_temp - min_val_temp) + min_val_temp
        if sigma_norm is not None:
            sigma_pred = sigma_norm * (max_val_temp - min_val_temp)
    elif stats["mode"] == "z_score" or stats["mode"] == "z_score_gp":
        mean = stats["mean"]
        std  = stats["std"]
        y_pred = y_norm * std + mean
        if sigma_norm is not None:
            sigma_pred = sigma_norm * std
    if sigma_norm is not None:
        return y_pred, sigma_pred
    else:
        return y_pred


def find_not_all_nan_times(target_train, skip=24):
    """
    Define a mask to ignore time indexes with all NaN values.
    Args:
        target_train (np.ndarray) with shape (nodes, time, ...)
    Returns:
        idxs_not_all_nan (np.ndarray) of shape (k, 1)
    """

    initial_time_dim = target_train.shape[1]
    mask_not_all_nan = []

    for t in range(initial_time_dim):
        nan_sum = np.isnan(target_train[:, t]).sum()
        mask_not_all_nan.append(nan_sum < target_train.shape[0])

    mask_not_all_nan = np.array(mask_not_all_nan, dtype=bool)

    # Force first L time steps to be valid
    mask_not_all_nan[:skip] = True
    idxs_not_all_nan = np.argwhere(mask_not_all_nan)

    return idxs_not_all_nan


def derive_train_val_idxs_years_list(
    train_years,
    val_years,
    history_length,
    time_index,
    args=None,
    accelerator=None
    ):
    
    train_idxs = []
    train_idxs_valid = []
    val_idxs = []
    val_idxs_valid = []

    write_log(f"\nTrain years: {train_years}", args, accelerator, 'a')
    for y in train_years:
        start_idx, end_idx = date_to_idxs_from_timeindex(
            year_start=y, month_start=1, day_start=1,
            year_end=y, month_end=12, day_end=31,
            time_index=time_index
        )
        if start_idx < history_length:
            start_idx = history_length
        if end_idx == len(time_index):
            end_idx -= 1
        if y in val_years: # no overlapping between training and validation
            end_idx -= history_length
        train_idxs.append(np.arange(start_idx - history_length, end_idx))
        train_idxs_valid.append(np.arange(start_idx, end_idx))

    train_idxs = np.concatenate(train_idxs)
    train_idxs_valid = np.concatenate(train_idxs_valid)
    # Filter the train_idxs that are valid and return their positions inside train_idxs
    train_idxs_valid = np.where(np.isin(train_idxs, train_idxs_valid))[0]

    write_log(f"\nValidation years: {val_years}", args, accelerator, 'a')
    for y in val_years:
        start_idx, end_idx = date_to_idxs_from_timeindex(
            year_start=y, month_start=1, day_start=1,
            year_end=y, month_end=12, day_end=31,
            time_index=time_index
        )
        if start_idx < history_length:
            start_idx = history_length
        if end_idx == len(time_index):
            end_idx -= 1
        val_idxs.append(np.arange(start_idx - history_length, end_idx))
        val_idxs_valid.append(np.arange(start_idx, end_idx))
    
    val_idxs = np.concatenate(val_idxs)
    val_idxs_valid = np.concatenate(val_idxs_valid)
    # Filter the val_idxs that are valid and return their positions inside val_idxs
    val_idxs_valid = np.where(np.isin(val_idxs, val_idxs_valid))[0]
        
    return train_idxs, train_idxs_valid, val_idxs, val_idxs_valid


def derive_train_val_idxs(
    train_year_start, train_month_start, train_day_start,
    train_year_end, train_month_end, train_day_end,
    history_length,
    time_index,
    idxs_not_all_nan=None,
    validation_year=None,
    args=None,
    accelerator=None
    ):
    """
    Computes the train and validation indexes.
    Returns:
        train_idxs_list (list of ints)
        val_idxs_list (list of ints)
    """

    # --- Compute training period indices ---
    train_start_idx, train_end_idx = date_to_idxs_from_timeindex(
        year_start=train_year_start, month_start=train_month_start, day_start=train_day_start,
        year_end=train_year_end, month_end=train_month_end, day_end=train_day_end,
        time_index=time_index
    )

    if train_end_idx == len(time_index):
        train_end_idx -= 1

    # --- Compute validation period indices ---
    if validation_year is not None:
        val_start_idx, val_end_idx = date_to_idxs_from_timeindex(
            year_start=validation_year, month_start=1, day_start=1,
            year_end=validation_year, month_end=12, day_end=31,
            time_index=time_index
        )
    else:
        val_start_idx, val_end_idx = None, None

    # --- Sanity checks ---
    if train_start_idx >= train_end_idx:
        raise Exception("Train start idx is not smaller than train end idx.")
    if validation_year is not None and val_start_idx >= val_end_idx:
        raise Exception("Val start idx is not smaller than val end idx.")

    # --- Build train/val index lists ---
    if validation_year is None:
        # No validation → only training
        train_idxs = np.arange(train_start_idx - history_length, train_end_idx)
        train_idxs_valid = np.arange(train_start_idx, train_end_idx)
        val_idxs = np.array([])
        val_idxs_valid_subset = np.array([])
        if train_start_idx - history_length < 0:
            raise ValueError(f"Train start date: {train_year_start}/{train_month_start}/{train_day_start} " +
                             f"not valid with history length {history_length}.")
        else:
            write_log(f"\nValidation year not provided.", args, accelerator, 'a')
            write_log(f"\nTraining from {time_index[train_start_idx-1]} to {time_index[train_end_idx-1]}",
                      args, accelerator, 'a')
    else:
        # Validation before training
        if train_start_idx >= val_end_idx:
            # We want a complete year for validation and no overlapping
            if train_start_idx - val_end_idx <= history_length:
                train_start_idx = val_start_idx + history_length
            # train
            train_idxs = np.arange(train_start_idx - history_length, train_end_idx)
            train_idxs_valid = np.arange(train_start_idx, train_end_idx)
            # val
            val_idxs = np.arange(val_start_idx - history_length, val_end_idx)
            val_idxs_valid = np.arange(val_start_idx + history_length, val_end_idx)
            if val_start_idx - history_length < 0:
                raise ValueError(f"Validation year {validation_year} " +
                                f"not valid with history length {history_length}.")
            else:
                write_log(f"\nValidation before training", args, accelerator, 'a')
                write_log(f"\nTraining from {time_index[train_start_idx-1]} to {time_index[train_end_idx-1]}",
                      args, accelerator, 'a')
                write_log(f"\nValidation from {time_index[val_start_idx-1]} to {time_index[val_end_idx-1]}",
                      args, accelerator, 'a')
                
        # Validation after training
        elif train_end_idx <= val_start_idx:
            # We want a complete year for validation and no overlapping
            if val_start_idx - train_end_idx <= history_length:
                train_end_idx = val_start_idx - history_length
            # train
            train_idxs = np.arange(train_start_idx - history_length, train_end_idx)
            train_idxs_valid = np.arange(train_start_idx, train_end_idx)
            # val
            val_idxs = np.arange(val_start_idx - history_length, val_end_idx)
            val_idxs_valid = np.arange(val_start_idx, val_end_idx)
            write_log(f"\nValidation after training", args, accelerator, 'a')
            write_log(f"\nTraining from {time_index[train_start_idx]} to {time_index[train_end_idx-1]}",
                    args, accelerator, 'a')
            write_log(f"\nValidation from {time_index[val_start_idx]} to {time_index[val_end_idx-1]}",
                    args, accelerator, 'a')

        # Validation inside training period
        elif val_start_idx > train_start_idx and val_end_idx < train_end_idx:
            # train
            train_start_idx_1 = train_start_idx
            train_end_idx_1 = val_start_idx - history_length
            train_start_idx_2 = val_end_idx + history_length
            train_end_idx_2 = train_end_idx
            train_idxs = np.array(
                list(np.arange(train_start_idx_1 - history_length, train_end_idx_1)) +
                list(np.arange(train_start_idx_2 - history_length, train_end_idx_2))
            )
            train_idxs_valid = np.array(
                list(np.arange(train_start_idx_1, train_end_idx_1)) +
                    list(np.arange(train_start_idx_2, train_end_idx_2))
            )
            # val
            val_idxs = np.arange(val_start_idx - history_length, val_end_idx)
            val_idxs_valid = np.arange(val_start_idx, val_end_idx)

            print(val_start_idx, val_end_idx, train_start_idx_1, train_end_idx_1, train_start_idx_2, train_end_idx_2, history_length)

            if train_start_idx_1 - history_length < 0:
                raise ValueError(f"Train start date: {train_year_start}/{train_month_start}/{train_day_start} " +
                                 f"not valid with history length {history_length}.")
            else:
                write_log(f"\nValidation inside training", args, accelerator, 'a')
                write_log(f"\nTraining from {time_index[train_start_idx_1]} to {time_index[train_end_idx_1]} " + 
                          f"\nand from {time_index[train_start_idx_2]} to {time_index[train_end_idx_2]}",
                        args, accelerator, 'a')
                write_log(f"\nValidation from {time_index[val_start_idx]} to {time_index[val_end_idx]}",
                        args, accelerator, 'a')
        else:
            raise Exception(
                "Partially overlapping train and validation periods are not supported. "
                "Validation must be before, after, or fully inside training years."
            )

    # # --- Remove indices where all nodes are NaN ---
    # if idxs_not_all_nan is not None:
    #     # Ensure idxs_not_all_nan is a flat NumPy array
    #     idxs_not_all_nan = np.array(idxs_not_all_nan).flatten()
    #     train_idxs_list = [i for i in train_idxs_list if i in idxs_not_all_nan]
    #     val_idxs_list = [i for i in val_idxs_list if i in idxs_not_all_nan]

    if idxs_not_all_nan is not None:
        mask_valid = np.isin(train_idxs_valid, idxs_not_all_nan)
        train_idxs_valid_not_all_nan = train_idxs_valid[mask_valid]
        train_idxs_valid_subset = np.where(np.isin(train_idxs, train_idxs_valid_not_all_nan))[0]
        if validation_year is not None:
            mask_valid = np.isin(val_idxs_valid, idxs_not_all_nan)
            val_idxs_valid_not_all_nan = val_idxs_valid[mask_valid]
            val_idxs_valid_subset = np.where(np.isin(val_idxs, val_idxs_valid_not_all_nan))[0]
    else:
        train_idxs_valid_subset = train_idxs_valid
        if validation_year is not None:
            val_idxs_valid_subset = val_idxs_valid

    # --- Save to disk if requested ---
    if args is not None:
        if accelerator is None or accelerator.is_main_process:
            np.save(args.output_path+"train_time_index.npy", time_index[train_idxs])
            if validation_year is not None:
                np.save(args.output_path+"val_time_index.npy", time_index[val_idxs])

    return train_idxs, train_idxs_valid_subset, val_idxs, val_idxs_valid_subset


def derive_qmse_bins(target, train_idxs, args, accelerator, binmin=np.log1p(0.1), binmax=np.log1p(200), binwidth=np.log1p(0.5), bins=None):

    # Precompute bins
    if bins is None:
        bins = np.arange(binmin, binmax, binwidth)
        if args.model_type == "all":
            bins = np.insert(bins, 0, 0.0)  # log1p(0)

    # Histogram only on training subset (flatten for speed)
    train_vals = target[:, train_idxs].ravel()
    values_unif_log, edges_unif_log = np.histogram(train_vals, bins=bins)

    # searchsorted returns indices in [0, len(edges)-1]
    target_bins = np.searchsorted(edges_unif_log, target, side="left").astype(float, copy=False)

    # Handle NaNs in target
    nan_mask = np.isnan(target)
    if nan_mask.any():
        target_bins[nan_mask] = np.nan

    # Determine number of bins
    if nan_mask.any():
        max_bin = int(np.nanmax(target_bins))
    else:
        max_bin = int(target_bins.max())

    nbins = max_bin + 1

    # Fix case with an out-of-range bin
    if nbins > len(values_unif_log):
        write_log(
            f"\nBins min: {int(np.nanmin(target_bins))}, "
            f"bins max: {max_bin}, nbins: {nbins}, "
            f"len weights: {len(values_unif_log)}",
            args, accelerator, 'a'
        )

        # Clamp last bin
        target_bins[target_bins == nbins - 1] = nbins - 2
        nbins -= 1

        write_log("\nUpdating last bin...", args, accelerator, 'a')

    write_log(
        f"\nbins min: {int(np.nanmin(target_bins))}, "
        f"bins max: {int(np.nanmax(target_bins))}, nbins: {nbins}",
        args, accelerator, 'a'
    )

    write_log(f"\nBins: {bins}", args, accelerator, 'a')

    target_bins[target < 0.1] = np.nan

    return target_bins


def compute_input_statistics_and_standardize(
        x_low,
        x_high,
        train_idxs,
        n_vars,
        args,
        accelerator=None,
        apply_stats=True,
        high_independent_vars=False):

    write_log(f"\nComputing statistics for the low-res input data", args, accelerator, 'a')

    train_slice = x_low[:, train_idxs, :, :]  # (nodes, time_train, vars, levels)

    # if args.stats_mode == "var":
    means_low = np.mean(train_slice, axis=(0,1,3))   # (n_vars,)
    stds_low  = np.std(train_slice,  axis=(0,1,3))
    # elif args.stats_mode == "field":
    #     means_low = np.nanmean(train_slice, axis=(0,1))     # (n_vars, n_levels)
    #     stds_low  = np.nanstd(train_slice,  axis=(0,1))
    # else:
    #     raise Exception("Arg 'stats_mode' should be either 'var' or 'field'")

    write_log(f"\nComputing statistics for the high-res input data", args, accelerator, 'a')

    if x_high.shape[1] > 1: # (num_nodes, num_node_features)
        if not high_independent_vars:
            means_high = np.array([
                np.nanmean(x_high[:,0]),
                np.nanmean(x_high[:,1:])
            ])
            stds_high = np.array([
                np.nanstd(x_high[:,0]),
                np.nanstd(x_high[:,1:])
            ])
        else:
            means_high = np.array([np.nanmean(x_high[:,i]) for i in range(x_high.shape[1])])
            stds_high = np.array([np.nanstd(x_high[:,i]) for i in range(x_high.shape[1])])
    else:
        means_high = np.nanmean(x_high)
        stds_high  = np.nanstd(x_high)

    # save stats
    np.save(args.output_path + "means_low.npy", means_low)
    np.save(args.output_path + "stds_low.npy", stds_low)
    np.save(args.output_path + "means_high.npy", means_high)
    np.save(args.output_path + "stds_high.npy", stds_high)

    if not apply_stats:
        return means_low, stds_low, means_high, stds_high

    x_low_std, x_high_std = standardize_input(
        x_low, x_high,
        means_low, stds_low,
        means_high, stds_high,
        n_vars=n_vars,
        high_independent_vars=high_independent_vars
    )

    return x_low_std, x_high_std


def standardize_input(
    x_low,
    x_high,
    means_low,
    stds_low,
    means_high,
    stds_high,
    n_vars,
    high_independent_vars=False
):
    # ---- LOW-RES STANDARDIZATION ----
    means_b = means_low.reshape(1, 1, n_vars, 1) # from (n_vars,)
    stds_b  = stds_low.reshape(1, 1, n_vars, 1) # from (n_vars,)

    x_low = (x_low - means_b) / stds_b

    # ---- HIGH-RES STANDARDIZATION ----
    if x_high.shape[1] > 1:
        if not high_independent_vars:
            # First channel
            x_high[:, 0]  = (x_high[:, 0]  - means_high[0]) / stds_high[0]
            # Remaining channels
            x_high[:, 1:] = (x_high[:, 1:] - means_high[1]) / stds_high[1]
        else:
            for i in range(x_high.shape[1]):
                x_high[:, i]  = (x_high[:, i]  - means_high[i]) / stds_high[i]
    else:
        x_high = (x_high - means_high) / stds_high

    return x_low, x_high


def check_freezed_layers(model, log_path, log_file, accelerator):
    for name, param in model.named_parameters():
        n_param = param.numel() 
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nLayer {name} requires_grad = {param.requires_grad} and has {n_param} parameters") 


def try_eval(value):
    """Safely evaluate Python literals or expressions."""
    if not isinstance(value, str):
        return value

    # Normalize common literal strings
    if value in {"None", "none"}:
        return None
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False

    # Try literal_eval for lists, numbers, tuples, dicts
    try:
        return ast.literal_eval(value)
    except Exception:
        pass

    # Try evaluating simple math expressions like "10**-8"
    try:
        return eval(value, {"__builtins__": {}}, {})
    except Exception:
        pass

    # Otherwise return the original string
    return value


def convert_dict(d):
    """Recursively convert all values in a nested dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = convert_dict(v)
        else:
            out[k] = try_eval(v)
    return out
