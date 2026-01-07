import numpy as np
import xarray as xr
import pickle
import time
import argparse
import sys
import torch
#import netCDF4 as nc
from torch_geometric.utils import degree
from torch_geometric.data import Data, HeteroData


from utils.tools import write_log
from utils.graph import cut_window, derive_edge_index_within, derive_edge_index_multiscale

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#-- paths
parser.add_argument('--input_path', type=str, help='path to input directory')
parser.add_argument('--output_path', type=str)
parser.add_argument('--log_file', type=str)
parser.add_argument('--input_path_target', type=str)
parser.add_argument('--input_path_topo', type=str)
parser.add_argument('--target_file', type=str)
parser.add_argument('--topo_file', type=str)
parser.add_argument('--domain', type=str)
parser.add_argument('--experiment', type=str)

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
parser.add_argument('--target_multiplier', type=float, default=1)

#-- era5
parser.add_argument('--input_files_prefix_low', type=str, help='prefix for the input files (convenction: {prefix}{parameter}.nc)', default='')
parser.add_argument('--n_levels_low', type=int, help='number of pressure levels considered', default=5)


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

training_experiment= args.experiment
if training_experiment == 'ESD_pseudo_reality':
    period_training = '1961-1980'
elif training_experiment == 'Emulator_hist_future':
    period_training = '1961-1980_2080-2099'


domain = args.domain
if domain == 'ALPS':
    gcm_name = 'CNRM-CM5'

predictor_filename = f'/content/{gcm_name}_{period_training}.nc'
predictor = xr.open_dataset(predictor_filename)


n_params=5
n_levels_low=3
levels=np.array([850,700,500]) 
#write_log('\nStarting the preprocessing of the low resolution data.', args, accelerator=None, mode='a')
lat_low = predictor['lat'].values[:]
lon_low = predictor['lon'].values[:]
lat_dim = len(lat_low)
lon_dim = len(lon_low)
time_dim = len(predictor['time'])
input_ds = np.zeros((time_dim, n_params, n_levels_low, lat_dim, lon_dim), dtype=np.float32)



        

#lat_low = np.flip(lat_low, axis=0)  # Flip the latitude array along the first axis
lat_low, lon_low = np.meshgrid(lat_low, lon_low, indexing='ij')

lat_low = lat_low.flatten()
lon_low = lon_low.flatten()

#--------------------------#
# POST-PROCESSING OF INPUT #
#--------------------------#

input_ds = torch.tensor(input_ds)
    
#----- Flip the dataset -----#
# the origin in the input files is in the top left corner, while we use the bottom left corner    
#input_ds = torch.flip(input_ds, [3])

#### IMPORTANT CHANGE - NORMALIZATION NOW IN MAIN AND PREDICTION #### 
 
input_ds = torch.permute(input_ds, (3,4,0,1,2)) # lat, lon, time, vars, levels
input_ds = torch.flatten(input_ds, end_dim=1)   # num_nodes, time, vars, levels

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

write_log(f"\nLoading target and topography.", args, accelerator=None, mode='a')
dataset_high = xr.open_dataset(args.input_path_target + args.target_file, engine="netcdf4")
topo = xr.open_dataset(args.input_path_topo + args.topo_file, engine="netcdf4")

lon = dataset_high.lon.to_numpy()
lat = dataset_high.lat.to_numpy()
if lon.shape != lat.shape:
    lon, lat = np.meshgrid(lon, lat)
try:
    target_high = dataset_high.pr.to_numpy()
except:
    target_high = dataset_high.tp.to_numpy()




z = topo.orog.to_numpy()
lon_z = topo.lon.to_numpy()
lat_z = topo.lat.to_numpy()

if lon_z.shape != lat_z.shape:
    lon_z, lat_z = np.meshgrid(lon_z, lat_z)
    
if args.target_multiplier is not None:
    target_high *= args.target_multiplier
    write_log(f'\nMultiplying pr by {args.target_multiplier} to get the correct unit.', args, accelerator=None, mode='a')



write_log("\nCutting the window...", args, accelerator=None, mode='a')

#-- Cut gripho and topo to the desired window --#
lon_high, lat_high, target_high = cut_window(args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon, lat, target_high)

print("target done!")


lon_high_z, lat_high_z, z_high = cut_window(
            args.lon_min, args.lon_max, args.lat_min, args.lat_max, lon_z, lat_z, z)
    

assert (np.allclose(lon_high, lon_high_z, atol=0.01) and lon_high.shape == lon_high_z.shape)

print("z done!")



write_log(f"\nDone! Window is [{lon_high.min()}, {lon_high.max()}] x [{lat_high.min()}, {lat_high.max()}] with {target_high.shape[1]} nodes.", args, accelerator=None, mode='a')

write_log(f"\nlon shape {lon_high.shape}, lat shape {lat_high.shape}, pr shape {target_high.shape}, z shape {z_high.shape}", args, accelerator=None, mode='a')



#---------------------------------------#
# TARGETS #
#---------------------------------------#

target_high = torch.tensor(target_high)

write_log("\nWriting some files...", args, accelerator=None, mode='a')

#-- WRITE THE FILES --#       
with open(args.output_path + 'target.pkl', 'wb') as f:
    pickle.dump(target_high, f)

#### IMPORTANT CHANGE - NORMALIZATION NOW IN MAIN AND PREDICTION #### 

#-----------------#
# BUILD THE GRAPH #
#-----------------#

low_high_graph = HeteroData()
high_graph = Data()

#-- EDGES --#

edges_low2high, _ = derive_edge_index_multiscale(lon_senders=lon_low, lat_senders=lat_low,
                                lon_receivers=lon_high, lat_receivers=lat_high, k=9, undirected=False)

edges_high, _ = derive_edge_index_within(lon_radius=args.lon_grid_radius_high, lat_radius=args.lat_grid_radius_high,
                                lon_senders=lon_high, lat_senders=lat_high, lon_receivers=lon_high, lat_receivers=lat_high)

#-- TO GRAPH ATTRIBUTES --#
#writing the predictors field inside the low_high graph
low_high_graph['low'].x = input_ds
#saving the latitude/longitude info for the low_resolution grid
low_high_graph['low'].lat = torch.tensor(lat_low)
low_high_graph['low'].lon = torch.tensor(lon_low)
#saving the latitude/longitude info for the high_resolution grid
low_high_graph['high'].lat = torch.tensor(lat_high)
low_high_graph['high'].lon = torch.tensor(lon_high)
#saving the static fields info
low_high_graph['high'].z_std = torch.tensor(z_high).unsqueeze(-1)
low_high_graph['high'].x = low_high_graph['high'].z_std

# High within High
low_high_graph['high', 'within', 'high'].edge_index = torch.tensor(edges_high)

# Low to High
low_high_graph['low', 'to', 'high'].edge_index = torch.tensor(edges_low2high)

#-- WRITE THE GRAPH --#

with open(args.output_path + 'low_high_graph' + args.suffix_phase_2 + '.pkl', 'wb') as f:
    pickle.dump(low_high_graph, f)

write_log(f"\nIn total, preprocessing took {time.time() - time_start} seconds", args, accelerator=None, mode='a')  
