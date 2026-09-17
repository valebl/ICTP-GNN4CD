import xarray as xr
import numpy as np

def read_dataset(path):
    """
    Load a NetCDF dataset and extract time information if present.
    Returns:
        ds: xarray.Dataset
        time: pandas.DatetimeIndex or None
        native_res_hours: int or None
    """

    ds = xr.open_dataset(path, engine='netcdf4')

    # Case 1: Dataset has NO time dimension
    if "time" not in ds.dims and "time" not in ds.coords and "valid_time" not in ds.dims and "valid_time" not in ds.coords:
        return ds, None, None

    # Case 2: Dataset has a time dimension
    try:
        time = ds["time"].to_index()
    except:
        time = ds["valid_time"].to_index()

    # Infer native resolution (handles both numpy timedelta64 and cftime timedelta)
    if len(time) > 1:
        deltas = np.diff(time)
        delta = deltas[0]
        if hasattr(delta, 'total_seconds'): # cftime or Python timedelta
            native_res_hours = int(delta.total_seconds() / 3600)
        else:
            native_res_hours = int(delta / np.timedelta64(1, "h"))
    else:
        native_res_hours = None # single timestep → no resolution

    native_res_hours = str(f"{native_res_hours}h")
    if native_res_hours == "24h":
        native_res_hours = "1d"

    return ds, time, native_res_hours
