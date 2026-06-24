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
    """Writes a string to args.log_file. If args is None, defaults to print.

    Parameters
    ----------
    s : str
        The string to write
    args : parser.parse_args() object
        (Default value = None)
    accelerator : Accelerator object
        (Default value = None)
    mode : str
        The writing mode, 'w'=write, 'a'=append (Default value = 'a')

    Returns
    -------

    """
    if accelerator is None or accelerator.is_main_process:
        if args is not None:
            with open(args.output_path + args.log_file, mode) as f:
                f.write(s)
        else:
            print(s)


def use_gpu_if_possible():
    """Checks wheather cuda is available
    """
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def set_seed_everything(seed):
    """Sets the seed for generating random numbers

    Parameters
    ----------
    seed : int
        The integer seed

    Returns
    -------
        None

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_to_time_index_type(ts, ref):
    """Convert a Python ``datetime`` object into the same temporal type used by a
    reference time index.

    Parameters
    ----------
    ts : datetime.datetime
        A Python ``datetime`` object representing the timestamp to convert.
    ref : datetime-like
        A reference timestamp whose type determines the target conversion.
        This is usually an element of ``time_index`` and may be one of:
        - ``numpy.datetime64``
        - ``datetime.datetime``
        - ``cftime.datetime`` (e.g., ``DatetimeNoLeap``, ``Datetime360Day``)

    Returns
    -------
        The transformed data

    """
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


def date_to_idxs_from_timeindex(
    year_start, month_start, day_start, time_index,
    year_end=None, month_end=None, day_end=None    
):
    """Compute start/end indices using the actual time_index array.
    Works with numpy.datetime64, Python datetime, and cftime calendars.

    Parameters
    ----------
    year_start : int
        
    month_start : int
        
    day_start : int
        
    time_index : int
        
    year_end : int
         (Default value = None)
    month_end : int
         (Default value = None)
    day_end : int
         (Default value = None)

    Returns
    -------
    start_idx : int

    end_idx : int (optional)

    """

    # Detect reference type from the first element
    ref = time_index[0]

    # Build start timestamp in Python datetime ----
    start_dt = datetime.datetime(year_start, month_start, day_start, 0, 0, 0)

    # Build end timestamp if needed ----
    if year_end is not None:
        end_dt = datetime.datetime(year_end, month_end, day_end, 23, 59, 59)

    start_ts = convert_to_time_index_type(start_dt, ref)
    if year_end is not None:
        end_ts = convert_to_time_index_type(end_dt, ref)

    # Searchsorted
    start_idx = int(np.searchsorted(time_index, start_ts, side="left"))

    if year_end is not None:
        end_idx = int(np.searchsorted(time_index, end_ts, side="right"))
        return start_idx, end_idx

    return start_idx


def find_not_all_nan_times(data, L=24, args=None, accelerator=None):
    """Define a temporal mask, 1 if at least one spatial value is not nan.

    Parameters
    ----------
    data : np.ndarray
        The data of shape (num_nodes, time)
    L : int
        The first L values to be considered valid. (Default value = 24)

    Returns
    -------
        idxs_not_all_nan (np.ndarray) of shape (k, 1)

    """
    initial_time_dim = data.shape[1]
    mask_not_all_nan = []

    for t in range(initial_time_dim):
        nan_sum = np.isnan(data[:, t]).sum()
        mask_not_all_nan.append(nan_sum < data.shape[0])

    mask_not_all_nan = np.array(mask_not_all_nan, dtype=bool)

    # Force first L time steps to be valid
    mask_not_all_nan[:L] = True
    idxs_not_all_nan = np.argwhere(mask_not_all_nan)

    write_log(f"\nNot all nan time indexes: {len(idxs_not_all_nan)} " +
        f"({(len(idxs_not_all_nan) / data.shape[1] * 100):.1f} % of initial ones).",
        args, accelerator, 'a')

    return idxs_not_all_nan


def derive_train_val_idxs_from_years_list(
    train_years,
    val_years,
    history_length,
    time_index,
    args=None,
    accelerator=None
    ):
    """

    Parameters
    ----------
    train_years :
        
    val_years :
        
    history_length :
        
    time_index :
        
    args :
         (Default value = None)
    accelerator :
         (Default value = None)

    Returns
    -------

    """
    
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
    """Computes the train and validation indexes.

    Parameters
    ----------
    train_year_start :
        
    train_month_start :
        
    train_day_start :
        
    train_year_end :
        
    train_month_end :
        
    train_day_end :
        
    history_length :
        
    time_index :
        
    idxs_not_all_nan :
         (Default value = None)
    validation_year :
         (Default value = None)
    args :
         (Default value = None)
    accelerator :
         (Default value = None)

    Returns
    -------
    
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


def compute_predictions_idxs(cfg, time_index, history_length, date_to_idxs_fn):
    """
    Given a normalized test config, compute:
    - test_idxs
    - test_idxs_valid
    - test_idxs_valid_subset
    """

    # Case 1 — list of years
    if cfg["mode"] == "years_list":
        test_idxs_list = []
        test_idxs_valid_list = []

        for year in cfg["years"]:
            start_idx, end_idx = date_to_idxs_fn(
                year_start=year, month_start=1, day_start=1,
                year_end=year, month_end=12, day_end=31,
                time_index=time_index
            )

            # ensure history_length safety
            if start_idx - history_length < 0:
                start_idx = history_length

            test_idxs_list.append(np.arange(start_idx - history_length, end_idx))
            test_idxs_valid_list.append(np.arange(start_idx, end_idx))

        test_idxs = np.concatenate(test_idxs_list)
        test_idxs_valid = np.concatenate(test_idxs_valid_list)

        test_idxs_valid_subset = np.where(np.isin(test_idxs, test_idxs_valid))[0]

        return test_idxs, test_idxs_valid, test_idxs_valid_subset

    # Case 2 — date range
    start_idx, end_idx = date_to_idxs_fn(
        year_start=cfg["start"].year, month_start=cfg["start"].month, day_start=cfg["start"].day,
        year_end=cfg["end"].year,   month_end=cfg["end"].month,   day_end=cfg["end"].day,
        time_index=time_index
    )

    if start_idx - history_length < 0:
        raise ValueError(f"Invalid test start date: {cfg['start']}")

    test_idxs = np.arange(start_idx - history_length, end_idx)
    test_idxs_valid = np.arange(start_idx, end_idx)
    test_idxs_valid_subset = np.where(np.isin(test_idxs, test_idxs_valid))[0]

    return test_idxs, test_idxs_valid, test_idxs_valid_subset


def inspect_model(model, args, accelerator):
    """

    Parameters
    ----------
    model : object (of nn.Module)
        The model instance
        
    args :
        
    accelerator :

    Returns
    -------

    """
    for name, param in model.named_parameters():
        n_param = param.numel() 
        if accelerator is None or accelerator.is_main_process:
            if args is not None:
                with open(args.output_path + args.log_file, 'a') as f:
                    f.write(f"\nLayer {name} requires_grad = {param.requires_grad} and has {n_param} parameters") 


def try_eval(value):
    """Safely evaluate Python literals or expressions.

    Parameters
    ----------
    value : str
        The string to be evaluated

    Returns
    -------
    The converted value

    """
    if not isinstance(value, str):
        return value

    # Normalise common literal strings
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
    """Recursively convert all values in a nested dict.

    Parameters
    ----------
    d : dict
        The dictionary, whose elements will be converted
        using the try_eval function

    Returns
    -------
    The dictionary with converted values

    """
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = convert_dict(v)
        else:
            out[k] = try_eval(v)
    return out

def get_backend(x):
    if isinstance(x, torch.Tensor):
        return torch
    elif isinstance(x, np.ndarray):
        return np
    else:
        raise TypeError(f"Unsupported type for predictor transform: {type(x)}")
