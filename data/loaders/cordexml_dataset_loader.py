import numpy as np
from utils.helpers.tools import write_log
from data.loaders.read_dataset import read_dataset
from .registry import register_dataset_loader


@register_dataset_loader("CORDEXML")
def load_dataset_CORDEXML(
    file_path,
    file,
    args,
    params=['q', 't', 'u', 'v', 'z'],
    levels=['850', '700', '500']
    ):
    
    n_params = len(params)
    n_levels = len(levels)

    # Predictor data are all stored in the same file        
    ds, low_time_index, low_native_time_res = read_dataset(file_path+file)

    for var_idx, var in enumerate(params):
        for lev_idx, lev in enumerate(levels):

            var_lev = f"{var}_{lev}"
            write_log(f"\nPreprocessing {var_lev} ... ", args, accelerator=None, mode='a')

            # processed is an xarray.Dataset → extract the variable
            data = ds[var_lev].values # np.array

            # 3. Initialize input_ds on first variable
            if var_idx == 0 and lev_idx == 0:
                
                lat_low = ds["lat"].values
                lon_low = ds["lon"].values

                lat_dim = len(lat_low)
                lon_dim = len(lon_low)

                time_dim = len(ds["time"])

                input_ds = np.zeros(
                    (time_dim, n_params, n_levels, lat_dim, lon_dim),
                    dtype=np.float32
                )

            # 4. Store processed variable
            input_ds[:, var_idx, lev_idx, :, :] = data

    return input_ds, lat_low, lon_low, low_time_index, low_native_time_res, low_native_time_res
    