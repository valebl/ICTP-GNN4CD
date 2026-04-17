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
from data.structures.graph import derive_edge_index_within, derive_edge_index_multiscale
from data.loaders.complete_loader import load_dataset_CORDEXML
from data.loaders.netcdf_loader import read_dataset

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--output_path', type=str)
parser.add_argument('--log_file', type=str)
parser.add_argument('--input_path_target', type=str)
parser.add_argument('--input_path_predictors', type=str)
parser.add_argument('--input_path_topo', type=str)
parser.add_argument('--target_file', type=str)
parser.add_argument('--predictors_file', type=str)
parser.add_argument('--topo_file', type=str)
# No --input_path_mask_sealand / --mask_sealand_file: for SA the land-sea mask
# is derived directly from the topo file (orog > 0 => land, orog == 0 => sea).

#-- lat lon grid values
parser.add_argument('--lon_min', type=float)
parser.add_argument('--lon_max', type=float)
parser.add_argument('--lat_min', type=float)
parser.add_argument('--lat_max', type=float)
parser.add_argument('--lon_grid_radius_high', type=float)
parser.add_argument('--lat_grid_radius_high', type=float)
parser.add_argument('--lon_grid_radius_low', type=float, default=0.36)
parser.add_argument('--lat_grid_radius_low', type=float, default=0.36)

#-- other
parser.add_argument('--mask_path', type=str, default=None)
parser.add_argument('--mask_file', type=str, default=None)
parser.add_argument('--predictors_dataset', type=str)
parser.add_argument('--target_dataset', type=str)
parser.add_argument('--target_type', type=str, default="precipitation")
parser.add_argument('--target_multiplier', type=float, default=1)

parser.add_argument('--low_transformed_time_res', type=str, default="1h")
parser.add_argument('--high_transformed_time_res', type=str, default="1h")

#-- era5
parser.add_argument('--input_files_suffix_low', type=str, help='suffix for the input files (convenction: {parameter}{suffix}.nc)', default='')
parser.add_argument('--n_levels_low', type=int, help='number of pressure levels considered', default=5)


######################################################
##------------- PRELIMINARY OPERATIONS -------------##
######################################################

args = parser.parse_args()

time_start = time.time()

params = ['q', 't', 'u', 'v', 'z']
levels = ['850', '700', '500']
load_dataset = load_dataset_CORDEXML

######################################################
##---------- PREPROCESSING LOW RES DATA ------------##
######################################################

write_log(f"#### Preprocessing of the low resolution data.", args, accelerator=None, mode='w')

# Load the input dataset
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

#### IMPORTANT CHANGE - NORMALIZATION NOW IN MAIN AND PREDICTION ####
input_ds = np.transpose(input_ds, (3, 4, 0, 1, 2)) # lat, lon, time, vars, levels
input_ds = input_ds.reshape(-1, *input_ds.shape[2:]) # num_nodes, time, vars, levels

write_log(f'\nPreprocessing of low resolution data finished.', args, accelerator=None, mode='a')

######################################################
##--------- PREPROCESSING HIGH RES DATA ------------##
######################################################

write_log(f'\n\n#### Preprocessing of the high resolution data.', args, accelerator=None, mode='a')

# 1. Target
write_log(f"\n-- 1. TARGET ", args, accelerator=None, mode='a')
target_ds, high_new_time_index, high_time_res = read_dataset(args.input_path_target + args.target_file)
write_log(f"Target time-resolution is {high_time_res} ... ", args, accelerator=None, mode='a')

lon_high = target_ds.lon.to_numpy()
lat_high = target_ds.lat.to_numpy()

if args.target_type == "precipitation":
    target_high = target_ds.pr.to_numpy()
elif args.target_type == "temperature":
    target_high = target_ds.tasmax.to_numpy()

if args.target_multiplier is not None:
    target_high *= args.target_multiplier
    write_log(f'\n\tMultiplying pr by {args.target_multiplier} to get the correct unit.', args, accelerator=None, mode='a')

# 2. Orography
# Static_fields.nc may cover a larger domain than the target file.
# Select orog at the target's lat/lon grid points to ensure shape consistency.
write_log(f"\n-- 2. OROGRAPHY ", args, accelerator=None, mode='a')
orog_ds = xr.open_dataset(args.input_path_topo + args.topo_file)

# Get the unique 1D lat/lon values from the target (handles both 1D and 2D target grids)
lat_high_1d = np.unique(lat_high.flatten()) if lat_high.ndim > 1 else lat_high
lon_high_1d = np.unique(lon_high.flatten()) if lon_high.ndim > 1 else lon_high

orog_ds_sel = orog_ds.sel(lat=lat_high_1d, lon=lon_high_1d, method='nearest', tolerance=0.01)
orog = orog_ds_sel.orog.to_numpy()        # shape: (n_lat, n_lon) matching target
lon_orog = orog_ds_sel.lon.to_numpy()
lat_orog = orog_ds_sel.lat.to_numpy()

write_log(f"\n\tTopo selected shape: {orog.shape} (from full topo to match target domain)", args, accelerator=None, mode='a')

# Verify that the selected topo grid matches the target grid (tolerance 0.01 deg)
assert np.allclose(lat_orog, lat_high_1d, atol=0.01), \
    f"Topo lat does not match target lat (max diff={np.abs(lat_orog - lat_high_1d).max():.4f})"
assert np.allclose(lon_orog, lon_high_1d, atol=0.01), \
    f"Topo lon does not match target lon (max diff={np.abs(lon_orog - lon_high_1d).max():.4f})"
write_log(f"\n\tTopo/target lat-lon alignment check passed (atol=0.01).", args, accelerator=None, mode='a')

# 3. Mask sea-land (derived from topo: land where orog > 0, sea where orog == 0)
write_log(f"\n-- 3. MASK SEA-LAND (derived from topo: orog > 0 => land) ", args, accelerator=None, mode='a')
mask_sealand = (orog > 0).astype(np.float32)

# 4. Matrix coordinates ij
write_log(f"\n-- 4. MATRIX COORDINATES ij ", args, accelerator=None, mode='a')
i, j = np.meshgrid(
    np.arange(lon_high.shape[0]),
    np.arange(lat_high.shape[0]),
    indexing="ij"
)

i = 2.0 * i / (lon_high.shape[0] - 1) - 1.0 # normalise in [-1,1]
j = 2.0 * j / (lat_high.shape[0] - 1) - 1.0

coords = np.stack([i, j], axis=-1).reshape(-1, 2)
np.save(args.output_path+"coords_ij.npy", coords)

# 5. Land use
write_log(f"\n-- 5. LAND USE - ignoring.", args, accelerator=None, mode='a')

write_log(f"\n\nDone! Spatial domain is [{lon_high.min()}, {lon_high.max()}] x [{lat_high.min()}, {lat_high.max()}] with {target_high.shape[1]} nodes.", args, accelerator=None, mode='a')
write_log(f"\nlon shape {lon_high.shape}, lat shape {lat_high.shape}, pr shape {target_high.shape}, orog shape {orog.shape}", args, accelerator=None, mode='a')

## Now reshape for PYG compatibility but keep track of original dims
y_dim = target_high.shape[1] # time, lat, lon
x_dim = target_high.shape[2]

target_high = target_high.reshape(target_high.shape[0],-1) # (time, num_nodes=lat*lon)
target_high = target_high.swapaxes(0,1) # (num_nodes, time)

# SA domain: lon/lat from xarray are 1D unique axes (unlike ALPS rotated-pole where
# they are already 2D). Build the full 2D node grid matching C-order target reshape:
# target reshape is (lat outer, lon inner) → node m = (lat[m//n_lon], lon[m%n_lon])
lon_high_2d, lat_high_2d = np.meshgrid(lon_high, lat_high)  # both (n_lat, n_lon)
lon_high = lon_high_2d.flatten()   # (n_lat*n_lon,) = (16384,)
lat_high = lat_high_2d.flatten()   # (n_lat*n_lon,) = (16384,)

orog = np.expand_dims(orog.flatten(), axis=-1)
mask_sealand = np.expand_dims(mask_sealand.flatten(), axis=-1)

num_nodes_high = target_high.shape[0]

write_log(f"\nThe high resolution graph has {num_nodes_high} nodes.", args, accelerator=None, mode='a')

######################################################
##-------------- BUILDING THE GRAPH ----------------##
######################################################

write_log(f"\n\n#### Creating the graph object.", args, accelerator=None, mode='a')

low_high_graph = HeteroData()

#-- EDGES --#
use_edge_attr_high = False
use_edge_attr_low = False

write_log(f"\n-- 1. Derive low-to-high edges", args, accelerator=None, mode='a')

# 1. Low-to-high edges
edges_low2high, edges_low2high_attr = derive_edge_index_multiscale(
    lon_senders=lon_low,
    lat_senders=lat_low,
    lon_receivers=lon_high,
    lat_receivers=lat_high,
    k=9, undirected=False,
    use_edge_attr=use_edge_attr_low)

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

# 2. High-within-high edges
write_log(f"\n-- 2. Derive high-within-high edges", args, accelerator=None, mode='a')

edges_high, edges_high_attr = derive_edge_index_within(
    lon_radius=args.lon_grid_radius_high,
    lat_radius=args.lat_grid_radius_high,
    lon_senders=lon_high,
    lat_senders=lat_high,
    orog_senders=orog.squeeze(),
    lon_receivers=lon_high,
    lat_receivers=lat_high,
    orog_receivers=orog.squeeze(),
    use_edge_attr=use_edge_attr_high
    )

edges_high = torch.tensor(edges_high)

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
if use_edge_attr_low:
    low_high_graph['low', 'to', 'high'].edge_attr = torch.tensor(edges_low2high_attr).float()

# 2. High within High
low_high_graph['high', 'within', 'high'].edge_index = edges_high
if use_edge_attr_high:
    low_high_graph['high', 'within', 'high'].edge_attr = torch.tensor(edges_high_attr).float()

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

write_log(f"\n-- 3. Target", args, accelerator=None, mode='a')
np.save(args.output_path+"target.npy", target_high)

write_log(f"\n-- 4. Orography", args, accelerator=None, mode='a')
np.save(args.output_path+"orog.npy", orog)

write_log(f"\n-- 5. Mask sea-land", args, accelerator=None, mode='a')
np.save(args.output_path+"mask_sealand.npy", mask_sealand)

write_log(f"\n-- 6. Coords ij", args, accelerator=None, mode='a')
np.save(args.output_path+"coords_ij.npy", coords)

write_log(f"\n-- 7. Graph", args, accelerator=None, mode='a')
with open(args.output_path + 'low_high_graph.pkl', 'wb') as f:
    pickle.dump(low_high_graph, f)

write_log(f"\n-- 8. High time index file.", args, accelerator=None, mode='a')
np.save(args.output_path+"high_time_index.npy", high_new_time_index)

write_log(f"\n\n#### DONE!\nIn total, preprocessing took {time.time() - time_start} seconds", args, accelerator=None, mode='a')

write_log(f"\n-- 9. unique_src", args, accelerator=None, mode='a')
np.save(args.output_path+"unique_src.npy", unique_src)