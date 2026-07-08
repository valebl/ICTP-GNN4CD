import numpy as np
import xarray as xr
import pickle
import time
import argparse
import torch
import json

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
transform = T.AddLaplacianEigenvectorPE(k=2)

from utils.helpers.tools import write_log
from data.structures.graph import (
    derive_edge_index_within,
    derive_edge_index_multiscale,
    compute_dist_scale,
    save_edge_norm_constants,
    load_edge_norm_constants
)

from data.loaders.read_dataset import read_dataset

from args.preprocess.add_base_args import add_base_args
from data.loaders.registry import get_dataset_loader


#-----------------------------------------------------
#-------------- PRELIMINARY OPERATIONS ---------------
#-----------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = add_base_args(parser)

args = parser.parse_args()

time_start = time.time()

write_log(f"Input path: {args.input_path_predictors}", args, accelerator=None, mode='w')

#-----------------------------------------------------
#----------- PREPROCESSING LOW RES DATA --------------
#-----------------------------------------------------

write_log(f"\n\n#### Preprocessing of the low resolution data.", args, accelerator=None, mode='a')

# Load the input dataset
params = args.params.split(",")
levels = [int(l) for l in args.levels.split(",")]
load_dataset = get_dataset_loader(args.dataset_name)

input_ds, lat_low, lon_low, low_time_index, low_native_time_res, low_time_res = load_dataset(
    params=params, levels=levels, file_path=args.input_path_predictors, file=args.predictors_file, args=args)

if lat_low[0] > lat_low[-1]:
    write_log(f"\nFlipping the lat axes to have the origin in the bottom left corner", args, accelerator=None, mode='a')
    lat_low = np.flip(lat_low, axis=0)  # Flip the latitude array along the first axis
    input_ds = np.flip(input_ds, axis=3) # time, var, lev, lat, lon

if lon_low[0] > lon_low[-1]:
    write_log(f"\nFlipping the lon axes to have the origin in the bottom left corner", args, accelerator=None, mode='a')
    lon_low = np.flip(lon_low, axis=0)  # Flip the latitude array along the first axis
    input_ds = np.flip(input_ds, axis=4) # time, var, lev, lat, lon

lat_low, lon_low = np.meshgrid(lat_low, lon_low, indexing='ij')

lat_low = lat_low.flatten()
lon_low = lon_low.flatten()

input_ds = np.transpose(input_ds, (3, 4, 0, 1, 2)) #torch.permute(input_ds, (3,4,0,1,2)) # lat, lon, time, vars, levels
input_ds = input_ds.reshape(-1, *input_ds.shape[2:]) #input_ds = torch.flatten(input_ds, end_dim=1)   # num_nodes, time, vars, levels
# input_ds = torch.flatten(input_ds, start_dim=2, end_dim=-1)

write_log(f'\nPreprocessing of low resolution data finished.', args, accelerator=None, mode='a')

#-----------------------------------------------------
#---------- PREPROCESSING HIGH RES DATA --------------
#-----------------------------------------------------

write_log(f'\n\n#### Preprocessing of the high resolution data.', args, accelerator=None, mode='a')

# 1. Target
write_log(f"\n-- 1. TARGET ", args, accelerator=None, mode='a')
if args.target_file == "":
    target_high = None
    write_log(f"- ignoring", args, accelerator=None, mode='a')
else:
    target_ds, high_new_time_index, high_time_res = read_dataset(args.input_path_target + args.target_file)
    write_log(f"Target time-resolution is {high_time_res} ... ", args, accelerator=None, mode='a')

    if args.target_type == "precipitation":
        try:
            target_high = target_ds.pr.to_numpy()
        except:
            target_high = target_ds.tp.to_numpy()
    elif args.target_type == "temperature":
        try:
            target_high = target_ds.tasmax.to_numpy()
        except:
            target_high = target_ds.t2m.to_numpy()

    if args.target_multiplier is not None: 
        target_high *= args.target_multiplier
        write_log(f'\n\tMultiplying pr by {args.target_multiplier} to get the correct unit.', args, accelerator=None, mode='a')

    try:
        lon_high = target_ds.lon.to_numpy()
        lat_high = target_ds.lat.to_numpy()
    except:
        lon_high = target_ds.longitude.to_numpy()
        lat_high = target_ds.latitude.to_numpy()

    write_log(f'\n\tLon and lat high derived from target and have shape: lon {lon_high.shape}, lat {lat_high.shape}.', args, accelerator=None, mode='a')

# 2. Orography
write_log(f"\n-- 2. OROGRAPHY ", args, accelerator=None, mode='a')
orog_ds = xr.open_dataset(args.input_path_topo + args.topo_file)

try:
    orog = orog_ds.orog.to_numpy()
except:
    orog = orog_ds.z.to_numpy()

if target_high is None:
    lon_high = orog_ds.lon.to_numpy()
    lat_high = orog_ds.lat.to_numpy()
    write_log(f'\n\tLon and lat high derived from orog and have shape: lon {lon_high.shape}, lat {lat_high.shape}.', args, accelerator=None, mode='a')

if lon_high.ndim == 1 and lat_high.ndim == 1:
    lat_high, lon_high = np.meshgrid(
        lat_high,
        lon_high,
        indexing="ij"
    )

# 3. Mask sea-land
write_log(f"\n-- 3. MASK SEA-LAND ", args, accelerator=None, mode='a')
mask_sealand_ds = xr.open_dataset(args.input_path_mask_sealand + args.mask_sealand_file)

try:
    mask_sealand = mask_sealand_ds.lsm.to_numpy()
except:
    mask_sealand = mask_sealand_ds.z.to_numpy()

# 4. Matrix coordinates ij
write_log(f"\n-- 4. MATRIX COORDINATES ij ", args, accelerator=None, mode='a')
i, j = np.meshgrid(
    np.arange(lon_high.shape[0]),
    np.arange(lon_high.shape[1]),
    indexing="ij"
)

i = 2.0 * i / (lon_high.shape[0] - 1) - 1.0 # normalise in [-1,1]
j = 2.0 * j / (lon_high.shape[1] - 1) - 1.0

coords = np.stack([i, j], axis=-1).reshape(-1, 2)

# 5. Land use
write_log(f"\n-- 5. LAND USE - ignoring", args, accelerator=None, mode='a')

write_log(f"\n\nDone! Spatial domain is [{lon_high.min()}, {lon_high.max()}] x [{lat_high.min()}, {lat_high.max()}].", args, accelerator=None, mode='a')
write_log(f"\nlon shape {lon_high.shape}, lat shape {lat_high.shape}", args, accelerator=None, mode='a')

## Now reshape for PYG compatibility but keep track of original dims
y_dim = lon_high.shape[0] # time, x, y
x_dim = lon_high.shape[1]

if target_high is not None:
    target_high = target_high.reshape(target_high.shape[0],-1) # (time, num_nodes)
    target_high = target_high.swapaxes(0,1) # (num_nodes, time)

lon_high = lon_high.flatten()
lat_high = lat_high.flatten()
orog = np.expand_dims(orog.flatten(), axis=-1)
mask_sealand = np.expand_dims(mask_sealand.flatten(), axis=-1)

num_nodes_high = lon_high.shape[0]

write_log(f"\nThe high resolution graph has {num_nodes_high} nodes.", args, accelerator=None, mode='a')

#-----------------------------------------------------
#--------------- BUILDING THE GRAPH ------------------
#-----------------------------------------------------

write_log(f"\n\n#### Creating the graph object.", args, accelerator=None, mode='a')

low_high_graph = HeteroData()

#-- EDGES --#
use_edge_attr_high = True
use_edge_attr_low2high = True
use_edge_attr_low = False

if args.low2high_norm_constants is None:
    dist_scale_low2high = compute_dist_scale(lon_senders=lon_low, lat_senders=lat_low)
    save_edge_norm_constants(args.output_path+"low2high_norm_constants.json", dist_scale_low2high)
else:
    dist_scale_low2high = load_edge_norm_constants(args.input_path+"low2high_norm_constants.json")["dist_scale"]

write_log(f"\n-- 1. Derive low-to-high edges", args, accelerator=None, mode='a')

# 1. Low-to-high edges
edges_low2high, edges_low2high_attr = derive_edge_index_multiscale(
    lon_senders=lon_low,
    lat_senders=lat_low,
    lon_receivers=lon_high,
    lat_receivers=lat_high,
    k=args.k_low2high, undirected=False,
    use_edge_attr=use_edge_attr_low2high,
    dist_scale=dist_scale_low2high)

edges_low2high = torch.tensor(edges_low2high)

#### Remove the low nodes that are not connected to any high node
src = edges_low2high[0]              # shape (2,num_edges)
unique_src = torch.unique(src)       # sorted unique low-node indices

num_low = input_ds.shape[0]
# Initialize all as -1 (meaning: removed)
new_index = -torch.ones(num_low, dtype=torch.long)

# Fill only the kept nodes with new compacted indices
new_index[unique_src] = torch.arange(unique_src.size(0))

# Filter the low features
low_input_upd = input_ds[unique_src]
lon_low_upd = lon_low[unique_src]
lat_low_upd = lat_low[unique_src]

# Remap the edge index
edges_low2high_upd = edges_low2high.clone()
edges_low2high_upd[0] = new_index[edges_low2high[0]]

# Sanity check
assert (edges_low2high_upd[0] >= 0).all()
assert low_input_upd.shape[0] == unique_src.shape[0]

# 2. high-within-high edges
write_log(f"\n-- 2. Derive high-within-high edges", args, accelerator=None, mode='a')

if args.high_norm_constants is None:
    dist_scale_high = compute_dist_scale(lon_senders=lon_low, lat_senders=lat_low)
    save_edge_norm_constants(args.output_path+"high_norm_constants.json", dist_scale_high)
else:
    dist_scale_high = load_edge_norm_constants(args.input_path+"high_norm_constants.json")["dist_scale"]

edges_high, edges_high_attr = derive_edge_index_within(
    lon_radius=args.lon_grid_radius_high,
    lat_radius=args.lat_grid_radius_high,
    lon_senders=lon_high,
    lat_senders=lat_high,
    orog_senders=None, 
    lon_receivers=lon_high,
    lat_receivers=lat_high,
    orog_receivers=None,
    use_edge_attr=use_edge_attr_high,
    dist_scale=dist_scale_high
    )

edges_high = torch.tensor(edges_high)

# # 3. Low-within-low edges
# write_log(f"\n-- 3. Derive low-within-low edges", args, accelerator=None, mode='a')

# edges_low, edges_low_attr = derive_edge_index_within(
#     lon_radius=args.lon_grid_radius_low,
#     lat_radius=args.lat_grid_radius_low,
#     lon_senders=lon_low_upd,
#     lat_senders=lat_low_upd,
#     lon_receivers=lon_low_upd,
#     lat_receivers=lat_low_upd,
#     use_edge_attr=use_edge_attr_low
#     )

# edges_low = torch.tensor(edges_low)

#-- TO GRAPH ATTRIBUTES --#

write_log(f"\n-- 3. Create the graph attributes", args, accelerator=None, mode='a')

low_high_graph['low'].lat = torch.tensor(lat_low_upd)
low_high_graph['low'].lon = torch.tensor(lon_low_upd)
low_high_graph['low'].num_nodes = low_high_graph["low"].lon.shape[0]

low_high_graph['high'].lat = torch.tensor(lat_high)
low_high_graph['high'].lon = torch.tensor(lon_high)
low_high_graph['high'].num_nodes = low_high_graph["high"].lon.shape[0]

# 1. Low to High
low_high_graph['low', 'to', 'high'].edge_index = edges_low2high_upd
if use_edge_attr_low2high:
    low_high_graph['low', 'to', 'high'].edge_attr = torch.tensor(edges_low2high_attr).float()

# 2. High within High
low_high_graph['high', 'within', 'high'].edge_index = edges_high
if use_edge_attr_high:
    low_high_graph['high', 'within', 'high'].edge_attr = torch.tensor(edges_high_attr).float()

# # 3. Low within Low
# low_high_graph['low', 'within', 'low'].edge_index = edges_low
# if use_edge_attr_low:
#     low_high_graph['low', 'within', 'low'].edge_attr = torch.tensor(edges_low_attr).float()

# # 4. High to Low
# low_high_graph['high', 'to', 'low'].edge_index = low_high_graph['low', 'to', 'high'].edge_index.flip(0)

#-- SAVE METADATA --#

write_log(f"\n\n#### Save metadata", args, accelerator=None, mode='a')

metadata_low = {
    "variables": params,
    "levels": levels,
    "time_range": [str(low_time_index[0]), str(low_time_index[-1])],
    "time_dim": low_time_index.shape[0],
    "time_res": low_time_res,
    "native_time_res": low_native_time_res,
    "lat_range": [float(lat_low.min()), float(lat_low.max())],
    "lon_range": [float(lon_low.min()), float(lon_low.max())],
    "lon-lat_dim (n_points)": lat_low_upd.shape[0],
}
write_log(f"\n-- 1. Writing the low input metadata file.", args, accelerator=None, mode='a')
with open(args.output_path+"low_input_metadata.json", "w") as f:
    json.dump(metadata_low, f, indent=4)

if target_high is not None:
    metadata_target = {
        "variables": "pr",
        "time_res": high_time_res,
        "time_range": [str(high_new_time_index[0]), str(high_new_time_index[-1])],
        "native_time_res": high_time_res,
        "lon-lat_dim (n_points)": len(lat_high),
        "time_dim": len(high_new_time_index),
        "lat_range": [float(lat_high.min()), float(lat_high.max())],
        "lon_range": [float(lon_high.min()), float(lon_high.max())],
    }

    write_log(f"\n-- 2. Writing the target metadata file.", args, accelerator=None, mode='a')
    with open(args.output_path+"target_metadata.json", "w") as f:
        json.dump(metadata_target, f, indent=4)

#-- WRITING --#
write_log(f"\n\n#### Save processed files", args, accelerator=None, mode='a')

write_log(f"\n-- 1. Low resolution input", args, accelerator=None, mode='a')
np.save(args.output_path+"low_input.npy", low_input_upd)

write_log(f"\n-- 2. Low time index file.", args, accelerator=None, mode='a')
np.save(args.output_path+"time_index.npy", low_time_index)

if target_high is not None:
    write_log(f"\n-- 3. Target", args, accelerator=None, mode='a')
    np.save(args.output_path+"target.npy", target_high)
else:
    write_log(f"\n-- 3. Target - ignoring", args, accelerator=None, mode='a')

write_log(f"\n-- 4. Orography", args, accelerator=None, mode='a')
np.save(args.output_path+"orog.npy", orog)

write_log(f"\n-- 5. Mask sea-land", args, accelerator=None, mode='a')
np.save(args.output_path+"mask_sealand.npy", mask_sealand)

write_log(f"\n-- 6. Coords ij", args, accelerator=None, mode='a')
np.save(args.output_path+"coords_ij.npy", coords)

write_log(f"\n-- 7. Graph", args, accelerator=None, mode='a')
with open(args.output_path + args.output_file, 'wb') as f:
    pickle.dump(low_high_graph, f)
if target_high is not None:
    write_log(f"\n-- 8. High time index.", args, accelerator=None, mode='a')
    np.save(args.output_path+"high_time_index.npy", high_new_time_index)
else:
    write_log(f"\n-- 8. High time index - ignoring", args, accelerator=None, mode='a')

write_log(f"\n-- 9. unique_src", args, accelerator=None, mode='a')
np.save(args.output_path+"unique_src.npy", unique_src)

write_log(f"\n\n#### DONE!\nIn total, preprocessing took {time.time() - time_start} seconds", args, accelerator=None, mode='a')
