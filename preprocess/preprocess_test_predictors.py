import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess CORDEX-ML test predictor NetCDF files for the axial-attention model."
    )
    parser.add_argument("--raw-test-path", required=True)
    parser.add_argument("--output-test-path", required=True)
    parser.add_argument("--periods", default="historical,mid_century,end_century")
    parser.add_argument("--realizations", default="perfect,imperfect")
    parser.add_argument("--params", default="q,t,u,v,z")
    parser.add_argument("--levels", default="500,700,850")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"Could not find any coordinate among {candidates}")


def _standardize_dataset(ds, params, levels):
    lat_name = _coord_name(ds, ("lat", "latitude"))
    lon_name = _coord_name(ds, ("lon", "longitude"))
    time_name = _coord_name(ds, ("time",))

    rename = {}
    if lat_name != "lat":
        rename[lat_name] = "lat"
    if lon_name != "lon":
        rename[lon_name] = "lon"
    if time_name != "time":
        rename[time_name] = "time"
    if rename:
        ds = ds.rename(rename)

    if ds["lat"].ndim != 1 or ds["lon"].ndim != 1:
        raise ValueError(
            "Attention test preprocessing expects 1D low-resolution lat/lon axes. "
            f"Got lat={ds['lat'].shape}, lon={ds['lon'].shape}."
        )

    if float(ds["lat"][0]) > float(ds["lat"][-1]):
        ds = ds.sortby("lat")
    if float(ds["lon"][0]) > float(ds["lon"][-1]):
        ds = ds.sortby("lon")

    keep_vars = []
    missing = []
    for var in params:
        for lev in levels:
            name = f"{var}_{lev}"
            if name in ds.data_vars:
                keep_vars.append(name)
            else:
                missing.append(name)

    if missing:
        raise KeyError(f"Missing required predictor variables: {missing}")

    # NEW TEST PREPROCESSING FOR AXIAL ATTENTION:
    # Keep only the variables used by the model and force the loader's expected
    # dimension order: time, lat, lon. This avoids accidentally reusing older
    # preprocessed files with incompatible coordinate orientation or metadata.
    out = ds[keep_vars].transpose("time", "lat", "lon")

    # Preserve bounds if present, but they are not used by the model.
    if "time_bnds" in ds.data_vars:
        out["time_bnds"] = ds["time_bnds"]

    for name in keep_vars:
        out[name] = out[name].astype(np.float32)

    return out


def _netcdf_encoding(ds):
    encoding = {}
    for name in ds.data_vars:
        if name == "time_bnds":
            continue
        encoding[name] = {"zlib": True, "complevel": 1, "dtype": "float32"}
    return encoding


def main():
    args = parse_args()
    raw_root = Path(args.raw_test_path)
    output_root = Path(args.output_test_path)
    periods = _split_csv(args.periods)
    realizations = _split_csv(args.realizations)
    params = _split_csv(args.params)
    levels = _split_csv(args.levels)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw test path does not exist: {raw_root}")

    print(f"[preprocess_test_predictors] raw:    {raw_root}")
    print(f"[preprocess_test_predictors] output: {output_root}")

    processed = 0
    for period in periods:
        for realization in realizations:
            in_dir = raw_root / period / "predictors" / realization
            out_dir = output_root / period / "predictors" / realization
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                print(f"[preprocess_test_predictors] skipping missing {in_dir}")
                continue

            for src in sorted(in_dir.glob("*.nc")):
                dst = out_dir / src.name
                if dst.exists() and not args.overwrite:
                    print(f"[preprocess_test_predictors] exists, skip: {dst}")
                    continue

                print(f"[preprocess_test_predictors] {src} -> {dst}")
                with xr.open_dataset(src) as ds:
                    out = _standardize_dataset(ds, params=params, levels=levels)
                    out.to_netcdf(dst, encoding=_netcdf_encoding(out))
                processed += 1

    print(f"[preprocess_test_predictors] done. processed={processed}")


if __name__ == "__main__":
    main()
