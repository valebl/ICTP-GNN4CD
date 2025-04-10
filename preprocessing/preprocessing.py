import numpy as np
import xarray as xr
import pickle
import time
import argparse
import sys
import torch
import netCDF4 as nc
from torch_geometric.utils import degree

from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
transform = T.AddLaplacianEigenvectorPE(k=2)
# sys.path.append("/leonardo_work/ICT25_ESP/vblasone/ICTP-GNN4CD")

from utils.tools import write_log
from utils.graph import cut_window, retain_valid_nodes, derive_edge_index_within, derive_edge_index_multiscale

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path_phase_2', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str)
parser.add_argument('--log_file', type=str)
parser.add_argument('--input_path_gripho', type=str)
parser.add_argument('--input_path_topo', type=str)
parser.add_argument('--gripho_file', type=str)
parser.add_argument('--topo_file', type=str)
parser.add_argument('--land_use_path', type=str)
parser.add_argument('--land_use_file', type=str)

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
parser.add_argument('--suffix_phase_2', type=str, default='')
parser.add_argument('--mask_path', type=str)
parser.add_argument('--mask_file', type=str)
parser.add_argument('--predictors_type', type=str)

#-- era5
parser.add_argument('--input_files_prefix_low', type=str, help='prefix for the input files (convenction: {prefix}{parameter}.nc)', default='sliced_')
parser.add_argument('--n_levels_low', type=int, help='number of pressure levels considered', default=5)

target_type = "temperature"

######################################################
##------------- PRELIMINARY OPERATIONS -------------##
######################################################

args = parser.parse_args()

write_log("\nStart!", args, accelerator=None, mode='w')

time_start = time.time()

######################################################
##-------------------- PHASE 2A --------------------##
##---------- PREPROCESSING LOW RES DATA ------------##
######################################################

if args.predictors_type == "era5":
    params = ['q', 't', 'u', 'v', 'z']
elif args.predictors_type == "regcm":
    params = ['hus', 'ta', 'ua', 'va', 'zg']
else:
    raise Exception("args.predictors_type should be either era5 or regcm")

n_params = len(params)

#-------------------------#
# INPUT TENSOR FROM FILES #
#-------------------------#

write_log('\nStarting the preprocessing of the low resolution data.', args, accelerator=None, mode='a')

for p_idx, p in enumerate(params):
    if args.predictors_type == "era5":
        write_log(f'\nPreprocessing {args.input_files_prefix_low}{p}.nc ...', args, accelerator=None, mode='a')
        with nc.Dataset(f'{args.input_path_phase_2}{args.input_files_prefix_low}{p}.nc') as ds:
            data = ds[p][:]
            if p_idx == 0: # first parameter being processed -> get dimensions and initialize the input dataset
                lat_low = ds['latitude'][:]
                lon_low = ds['longitude'][:]
                lat_dim = len(lat_low)
                lon_dim = len(lon_low)
                time_dim = len(ds['time'])
                input_ds = np.zeros((time_dim, n_params, args.n_levels_low, lat_dim, lon_dim), dtype=np.float32) # time, variables, levels, lat, lon
        input_ds[:, p_idx,:,:,:] = data
            
    elif args.predictors_type == "regcm":
        with nc.Dataset(f'{args.input_path_phase_2}{args.input_files_prefix_low}{p}.nc') as ds:
            for l_idx, level in enumerate(['200', '500', '700', '850', '1000']):
                write_log(f'\nPreprocessing {args.input_files_prefix_low}{p}.nc for level {level}', args, accelerator=None, mode='a')
                var_name = f"{p}{level}"
                _data = ds[var_name][:]
                if "zg" in var_name:
                    _data *= 9.81
                    write_log(f'\nMultiplying {var_name} by 9.81 to get kg*m^2/s^2.', args, accelerator=None, mode='a')
                if p_idx == 0 and l_idx == 0: # first parameter being processed -> get dimensions and initialize the input dataset
                    lat_low = ds['latitude'][:]
                    lon_low = ds['longitude'][:]
                    lat_dim = len(lat_low)
                    lon_dim = len(lon_low)
                    time_dim = len(ds['time'])
                    input_ds = np.zeros((time_dim, n_params, args.n_levels_low, lat_dim, lon_dim), dtype=np.float32) # time, variables, levels, lat, lon
                data = torch.from_numpy(_data)
                mask = torch.from_numpy(_data.mask.astype(bool))
                data[mask] = torch.nan
                input_ds[:, p_idx,l_idx,:,:] = data.numpy()

lat_low = np.flip(lat_low, axis=0)  # Flip the latitude array along the first axis
lat_low, lon_low = np.meshgrid(lat_low, lon_low, indexing='ij')

lat_low = lat_low.flatten()
lon_low = lon_low.flatten()

#--------------------------#
# POST-PROCESSING OF INPUT #
#--------------------------#

input_ds = torch.tensor(input_ds)
    
#----- Flip the dataset -----#
# the origin in the input files is in the top left corner, while we use the bottom left corner    
input_ds = torch.flip(input_ds, [3])

#### IMPORTANT CHANGE - NORMALIZATION NOW IN MAIN AND PREDICTION #### 
 
input_ds = torch.permute(input_ds, (3,4,0,1,2)) # lat, lon, time, vars, levels
input_ds = torch.flatten(input_ds, end_dim=1)   # num_nodes, time, vars, levels
# input_ds = torch.flatten(input_ds, start_dim=2, end_dim=-1)

with open(args.output_path + args.log_file, 'a') as f:
    f.write(f'\nPreprocessing of low resolution data finished.')


######################################################
##-------------------- PHASE 2B --------------------##    
##--------- PREPROCESSING HIGH RES DATA ------------##
######################################################

write_log(f"\n\nStarting the preprocessing of high resolution data.", args, accelerator=None, mode='a')

#-------------------------------#
# CUT LON, LAT, PR, Z TO WINDOW #
#-------------------------------#

dataset_high = xr.open_dataset(args.input_path_gripho + args.gripho_file)
topo = xr.open_dataset(args.input_path_topo + args.topo_file)

#lon = torch.tensor(gripho.longitude.to_numpy())
#lat = torch.tensor(gripho.latitude.to_numpy())
#lat, lon = torch.meshgrid(lat, lon)
lon = dataset_high.lon.to_numpy()
lat = dataset_high.lat.to_numpy()
lon, lat = np.meshgrid(lon, lat)
if target_type == "precipitation":
    target_high = dataset_high.pr.to_numpy()
elif target_type == "temperature":
    target_high = dataset_high.t2m.to_numpy()

if args.predictors_type == "regcm":
    z = topo.orog.to_numpy()
    mask_land = xr.open_dataset(args.mask_path + args.mask_file)
    mask_land = mask_land.pr.to_numpy().squeeze()
    lon_z = topo.lon.to_numpy()
    lat_z = topo.lat.to_numpy()
else:
    z = topo.z.to_numpy()
    # mask_land = None
    mask_land = xr.open_dataset(args.mask_path + args.mask_file)
    mask_land = mask_land.pr.to_numpy().squeeze()
    lon_z = topo.lon.to_numpy()
    lat_z = topo.lat.to_numpy()
lon_z, lat_z = np.meshgrid(lon_z, lat_z)
    
if args.predictors_type == "regcm":
    target_high *= 3600
    write_log(f'\nMultiplying pr by 3600 to get mm.', args, accelerator=None, mode='a')

# Reading LAND USE data

landU  = xr.open_dataset(args.land_use_path+args.land_use_file,  engine='netcdf4') #open nc file by default with netcdf4, if avail
water = landU.water.to_numpy()
coast = landU.coast.to_numpy()
urban_MD = landU.urban_MD.to_numpy()
urban_HD = landU.urban_HD.to_numpy()
forest = landU.forest.to_numpy()
ucrop = landU.ucrop.to_numpy()
lon_landU = landU.lon.to_numpy()
lat_landU = landU.lat.to_numpy()
lon_landU, lat_landU = np.meshgrid(lon_landU, lat_landU)

write_log("\nCutting the window...", args, accelerator=None, mode='a')

#-- Cut gripho and topo to the desired window --#
# lon_high, lat_high, target_high, z_high, mask_land_high, water_high, coast_high, urban_MD_high, urban_HD_high, forest_high, ucrop_high = cut_window(
#         args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon, lat, target_high, z, mask_land, water, coast, urban_MD, urban_HD, forest, ucrop)

lon_high, lat_high, target_high = cut_window(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon, lat, target_high)

print("target done!")

if mask_land is not None:
    lon_high_z, lat_high_z, z_high, mask_land_high = cut_window(
            args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon_z, lat_z, z, mask_land)
else:
    lon_high_z, lat_high_z, z_high = cut_window(
            args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon_z, lat_z, z)
    mask_land_high = None

print("z done!")

lon_high_landU, lat_high_landU, water_high, coast_high, urban_MD_high, urban_HD_high, forest_high, ucrop_high = cut_window(
        args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon_landU, lat_landU, water, coast, urban_MD, urban_HD, forest, ucrop)

print("land use done!")

assert np.array_equal(lon_high, lon_high_z)
assert np.array_equal(lon_high, lon_high_landU)
assert np.array_equal(lat_high, lat_high_z)
assert np.array_equal(lat_high, lat_high_landU)

write_log(f"\nDone! Window is [{lon_high.min()}, {lon_high.max()}] x [{lat_high.min()}, {lat_high.max()}] with {target_high.shape[1]} nodes.", args, accelerator=None, mode='a')

write_log(f"\nlon shape {lon_high.shape}, lat shape {lat_high.shape}, pr shape {target_high.shape}, z shape {z_high.shape}, land vars shape {water_high.shape}", args, accelerator=None, mode='a')

#------------------------------------#
# REMOVE NODES NOT IN LAND TERRITORY #
#------------------------------------#

lon_high, lat_high, target_high, z_high, water_high, coast_high, urban_MD_high, urban_HD_high, forest_high, ucrop_high = retain_valid_nodes(
        lon_high, lat_high, target_high, z_high, mask_land_high, water_high, coast_high, urban_MD_high, urban_HD_high, forest_high, ucrop_high)

target_high = target_high.swapaxes(0,1) # (num_nodes, time)

land_vars_high = np.stack([water_high, coast_high, urban_MD_high, urban_HD_high, forest_high, ucrop_high], axis=-1)

# print(lon_high.shape, lat_high.shape, pr_high.shape, z_high.shape, land_vars_high.shape)

num_nodes_high = target_high.shape[0]

write_log(f"\nAfter removing the non land territory nodes, the high resolution graph has {num_nodes_high} nodes.", args, accelerator=None, mode='a')


#---------------------------------------#
# CLASSIFICATION AND REGRESSION TARGETS #
#---------------------------------------#

#-- ROUND THE TARGET --#   
threshold = 0.1 # mm
target_high[target_high < 0.1] = 0.0
if args.predictors_type == "era5":
    target_high = np.round(target_high, decimals=1)

target_high = torch.tensor(target_high)

write_log("Writing some files...", args, accelerator=None, mode='a')

#-- WRITE THE FILES --#       
with open(args.output_path + 'pr_target.pkl', 'wb') as f:
    pickle.dump(target_high, f)

#### IMPORTANT CHANGE - NORMALIZATION NOW IN MAIN AND PREDICTION #### 

#-----------------#
# BUILD THE GRAPH #
#-----------------#

low_high_graph = HeteroData()
high_graph = Data()

#-- EDGES --#

edges_low2high, edges_low2high_attr = derive_edge_index_multiscale(lon_senders=lon_low, lat_senders=lat_low,
                                lon_receivers=lon_high, lat_receivers=lat_high, k=9, undirected=False)

edges_high2low, edges_high2low_attr = derive_edge_index_multiscale(lon_senders=lon_high, lat_senders=lat_high,
                                lon_receivers=lon_low, lat_receivers=lat_low, k=9, undirected=False)

edges_high, edges_high_attr = derive_edge_index_within(lon_radius=args.lon_grid_radius_high, lat_radius=args.lat_grid_radius_high,
                                lon_senders=lon_high, lat_senders=lat_high, lon_receivers=lon_high, lat_receivers=lat_high)

edges_low, edges_low_attr = derive_edge_index_within(lon_radius=args.lon_grid_radius_low, lat_radius=args.lat_grid_radius_low,
                                lon_senders=lon_low, lat_senders=lat_low, lon_receivers=lon_low, lat_receivers=lat_low)

# edges_low, edges_low_attr = derive_edge_index_within(lon_radius=0.251, lat_radius=0.251,
#                                 lon_senders=lon_low, lat_senders=lat_low, lon_receivers=lon_low, lat_receivers=lat_low)

# edges_high, edges_high_attr = derive_edge_index_multiscale(lon_senders=lon_high, lat_senders=lat_high,
#                                 lon_receivers=lon_high, lat_receivers=lat_high, k=4, undirected=True)

# edges_low, edges_low_attr = derive_edge_index_multiscale(lon_senders=lon_low, lat_senders=lat_low,
#                                 lon_receivers=lon_low, lat_receivers=lat_low, k=4, undirected=False)


#-- TO GRAPH ATTRIBUTES --#

low_high_graph['low'].x = input_ds
low_high_graph['low'].lat = torch.tensor(lat_low)
low_high_graph['low'].lon = torch.tensor(lon_low)

low_high_graph['high'].lat = torch.tensor(lat_high)
low_high_graph['high'].lon = torch.tensor(lon_high)
low_high_graph['high'].z_std = torch.tensor(z_high).unsqueeze(-1)
low_high_graph['high'].land_std = torch.tensor(land_vars_high).float()
# low_high_graph['high'].node_type = torch.where(degree(low_high_graph['high','within','high'].edge_index[0], low_high_graph['high'].num_nodes) == 8, 1, 0)
low_high_graph['high'].x = torch.cat((low_high_graph['high'].z_std, low_high_graph['high'].land_std),dim=-1)

# High within High
low_high_graph['high', 'within', 'high'].edge_index = torch.tensor(edges_high)
low_high_graph['high', 'within', 'high'].edge_attr = torch.tensor(edges_high_attr).float()

# Low to High
low_high_graph['low', 'to', 'high'].edge_index = torch.tensor(edges_low2high)
low_high_graph['low', 'to', 'high'].edge_attr = torch.tensor(edges_low2high_attr).float()

# Low within Low
low_high_graph['low', 'within', 'low'].edge_index = torch.tensor(edges_low)
low_high_graph['low', 'within', 'low'].edge_attr = torch.tensor(edges_low_attr).float()

# Low to High
low_high_graph['high', 'to', 'low'].edge_index = torch.tensor(edges_high2low)
low_high_graph['high', 'to', 'low'].edge_attr = torch.tensor(edges_high2low_attr).float()


#-- WRITE THE GRAPH --#

with open(args.output_path + 'low_high_graph' + args.suffix_phase_2 + '.pkl', 'wb') as f:
    pickle.dump(low_high_graph, f)

write_log(f"\nIn total, preprocessing took {time.time() - time_start} seconds", args, accelerator=None, mode='a')  



