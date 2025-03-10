import time
import sys
import pickle
import torch.nn as nn
import numpy as np
import random

import torch
import torch.nn.functional as F
from datetime import datetime, date
from calendar import monthrange

import numpy as np
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, ks_2samp, entropy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial import transform
from scipy.spatial.distance import cdist

######################################################
#------------------ GENERAL UTILITIES ---------------
######################################################


def write_log(s, args, accelerator=None, mode='a'):
    if accelerator is None or accelerator.is_main_process:
        with open(args.output_path + args.log_file, mode) as f:
            f.write(s)


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

def cut_window(lon_min, lon_max, lat_min, lat_max, lon, lat, pr, z, mask_land, *argv):
    r'''
    Derives a new version of the longitude, latitude and precipitation
    tensors, by only retaining the values inside the specified lon-lat rectangle
    Args:
        lon_min, lon_max, lat_min, lat_max: integers
        lon, lat, z, pr: np.arrays
    Returns:
        The new tensors with the selected values
    '''

    bool_lon = np.logical_and(lon >= lon_min, lon <= lon_max)
    bool_lat = np.logical_and(lat >= lat_min, lat <= lat_max)
    bool_both = np.logical_and(bool_lon, bool_lat)
    lon_sel = lon[bool_both]
    lat_sel = lat[bool_both]
    z_sel = z[bool_both]
    pr_sel = pr[:,bool_both]
    v = []
    for arg in argv:
        v.append(arg[bool_both])
    if mask_land is None:
        return lon_sel, lat_sel, pr_sel, z_sel, None, *v
    else:
        mask_land = mask_land[bool_both]
        return lon_sel, lat_sel, pr_sel, z_sel, mask_land, *v


def retain_valid_nodes(lon, lat, pr, e, mask_land=None, *argv):
    r'''
    Selects only the nodes for which precipitation is not
    nan in at least one timestamp. All the other nodes are
    ignored (e.g. the sea values in GRIPHO). If a land mask
    is provided, non-land points are also ignored.
    Args:
        lon (np.array): longitude for each spatial point
        lat (np.array): latitude for each spatial point
        pr (np.array): precipitation for each spatial point
        e (np.array): elevation for each spatial point
        mask_land (np.array, optional): a mask for the land points
    Returns:
        The valid points for each input tensor
    '''

    valid_nodes = ~np.isnan(pr).all(axis=0)
    if mask_land is not None:
        valid_nodes = np.logical_and(valid_nodes, ~np.isnan(mask_land))
    lon = lon[valid_nodes]
    lat = lat[valid_nodes]
    pr = pr[:,valid_nodes]
    e = e[valid_nodes]
    v = []
    for arg in argv:
        v.append(arg[valid_nodes])
    return lon, lat, pr, e, *v


def derive_edge_index_within(lon_radius, lat_radius, lon_senders ,lat_senders, lon_receivers, lat_receivers):
    r'''
    Derives edge_indexes within two sets of nodes based on specified lon, lat distances
    Args:
        lon_senders (np.array): longitudes of all first nodes in the edges
        lat_senders (np.array): latitudes of all fisrt nodes in the edges
        lon_receivers (np.array): longitudes of all second nodes in the edges
        lat_receivers (np.array): latitudes of all second nodes in the edges
    Return:
        The edge_indexes tensor
    '''

    edge_index = []

    lonlat_senders = np.column_stack((lon_senders, lat_senders))
    lonlat_receivers = np.column_stack((lon_receivers,lat_receivers))

    for ii, xi in enumerate(lonlat_senders):
        
        bool_lon = np.abs(lon_receivers - xi[0]) < lon_radius
        bool_lat = np.abs(lat_receivers - xi[1]) < lat_radius
        bool_both = np.logical_and(bool_lon, bool_lat)

        jj_list = np.nonzero(bool_both)[0] # to get indices
        xj_list = lonlat_receivers[bool_both]

        for jj, xj in zip(jj_list, xj_list):
            if not np.array_equal(xi, xj):
                edge_index.append(np.array([ii, jj]))
    
    edge_index = np.array(edge_index).T
    print(edge_index.shape)

    return edge_index


def derive_edge_index_low2high(lon_low ,lat_low, lon_high, lat_high, k, undirected=False, use_edge_attr=True):
    '''
    Derives edge_indexes between two sets of nodes based on specified number of neighbours k
    Args:
        lon_low (np.array): longitudes of all first nodes in the edges
        lat_low (np.array): latitudes of all fisrt nodes in the edges
        lon_high (np.array): longitudes of all second nodes in the edges
        lat_high (np.array): latitudes of all second nodes in the edges
        k (int): the number of neighbours
    Return:
        The edge_indexes tensor
    '''
    edge_index = []
    edge_attr = []

    lonlat_low = np.concatenate((np.expand_dims(lon_low,-1), np.expand_dims(lat_low,-1)), axis=-1)
    lonlat_high = np.concatenate((np.expand_dims(lon_high,-1), np.expand_dims(lat_high,-1)),axis=-1)

    dist = cdist(lonlat_high, lonlat_low, metric='euclidean')
    neighbours = np.argsort(dist, axis=-1)[:, :k]
    # _ , neighbours = dist.topk(k, largest=False, dim=-1)

    for n_n2 in range(lonlat_high.shape[0]):
        for n_n1 in neighbours[n_n2,:]:
            edge_index.append(np.array([n_n1, n_n2]))
            # edge_attr.append(dist[n_n2, n_n1])
            if undirected:
                edge_index.append(np.array([n_n2, n_n1]))

    edge_index = np.array(edge_index).T
    # edge_attr = np.array(edge_attr).T
    
    if use_edge_attr:
        senders = edge_index[0]
        receivers = edge_index[1]
        edge_attr = get_edge_features(lon_low, lat_low, lon_high, lat_high, senders, receivers)
        return edge_index, edge_attr
    else:
        return edge_index
    
def get_edge_features(node_lon_senders, node_lat_senders, node_lon_receivers, node_lat_receivers,
                      senders, receivers, rotate_latitude=True, rotate_longitude=True):
    '''
    lon_n1, lat_n1, lon_n2, lat_n2, edge_index [2, n_edges]
    '''
    
    node_phi_senders, node_theta_senders = lat_lon_deg_to_spherical(node_lon_senders, node_lat_senders)
    node_phi_receivers, node_theta_receivers = lat_lon_deg_to_spherical(node_lon_receivers, node_lat_receivers)
    
    relative_position = get_relative_position_in_receiver_local_coordinates(
        node_phi_senders, node_theta_senders, node_phi_receivers, node_theta_receivers, senders, receivers,
        latitude_local_coordinates=rotate_latitude,
        longitude_local_coordinates=rotate_longitude)
    
    # Note this is L2 distance in 3d space, rather than geodesic distance.
    relative_edge_distances = np.linalg.norm(
        relative_position, axis=-1, keepdims=True)

    # Normalize to the maximum edge distance. Note that we expect to always
    # have an edge that goes in the opposite direction of any given edge
    # so the distribution of relative positions should be symmetric around
    # zero. So by scaling by the maximum length, we expect all relative
    # positions to fall in the [-1., 1.] interval, and all relative distances
    # to fall in the [0., 1.] interval.
    max_edge_distance = relative_edge_distances.max()
    relative_edge_distances = relative_edge_distances / max_edge_distance
    relative_position = relative_position / max_edge_distance

    return np.concatenate((relative_position, relative_edge_distances), axis=-1)


def spherical_to_cartesian(phi, theta):
    '''
    Adapted from GraphCast
    '''
    # Assuming unit radius.
    return (np.cos(phi)*np.sin(theta),
            np.sin(phi)*np.sin(theta),
            np.cos(theta))

def lat_lon_deg_to_spherical(node_lat, node_lon):
    '''
    Adapted from GraphCast
    '''
    phi = np.deg2rad(node_lon)
    theta = np.deg2rad(90 - node_lat)
    return phi, theta

def get_relative_position_in_receiver_local_coordinates(
    node_phi_senders, node_theta_senders, node_phi_receivers, node_theta_receivers, senders, receivers,
    latitude_local_coordinates=True, longitude_local_coordinates=True):
    """Returns relative position features for the edges.

    The relative positions will be computed in a rotated space for a local
    coordinate system as defined by the receiver. The relative positions are
    simply obtained by subtracting sender position minues receiver position in
    that local coordinate system after the rotation in R^3.

    Args:
        node_phi: [num_nodes] with polar angles.
        node_theta: [num_nodes] with azimuthal angles.
        senders: [num_edges] with indices.
        receivers: [num_edges] with indices.
        latitude_local_coordinates: Whether to rotate edges such that in the
            positions are computed such that the receiver is always at latitude 0.
        longitude_local_coordinates: Whether to rotate edges such that in the
            positions are computed such that the receiver is always at longitude 0.

    Returns:
        Array of relative positions in R3 [num_edges, 3]
    """

    node_pos_senders = np.stack(spherical_to_cartesian(node_phi_senders, node_theta_senders), axis=-1)
    node_pos_receivers = np.stack(spherical_to_cartesian(node_phi_receivers, node_theta_receivers), axis=-1)

    # No rotation in this case.
    if not (latitude_local_coordinates or longitude_local_coordinates):
        return node_pos_senders[senders] - node_pos_receivers[receivers]

    # Get rotation matrices for the local space space for every node.
    rotation_matrices = get_rotation_matrices_to_local_coordinates(
        reference_phi=node_phi_receivers,
        reference_theta=node_theta_receivers,
        rotate_latitude=latitude_local_coordinates,
        rotate_longitude=longitude_local_coordinates)

    # Each edge will be rotated according to the rotation matrix of its receiver
    # node.
    edge_rotation_matrices = rotation_matrices[receivers]

    # Rotate all nodes to the rotated space of the corresponding edge.
    # Note for receivers we can also do the matmul first and the gather second:
    # ```
    # receiver_pos_in_rotated_space = rotate_with_matrices(
    #    rotation_matrices, node_pos)[receivers]
    # ```
    # which is more efficient, however, we do gather first to keep it more
    # symmetric with the sender computation.
    receiver_pos_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, node_pos_receivers[receivers])
    sender_pos_in_in_rotated_space = rotate_with_matrices(
        edge_rotation_matrices, node_pos_senders[senders])
    # Note, here, that because the rotated space is chosen according to the
    # receiver, if:
    # * latitude_local_coordinates = True: latitude for the receivers will be
    #   0, that is the z coordinate will always be 0.
    # * longitude_local_coordinates = True: longitude for the receivers will be
    #   0, that is the y coordinate will be 0.

    # Now we can just subtract.
    # Note we are rotating to a local coordinate system, where the y-z axes are
    # parallel to a tangent plane to the sphere, but still remain in a 3d space.
    # Note that if both `latitude_local_coordinates` and
    # `longitude_local_coordinates` are True, and edges are short,
    # then the difference in x coordinate between sender and receiver
    # should be small, so we could consider dropping the new x coordinate if
    # we wanted to the tangent plane, however in doing so
    # we would lose information about the curvature of the mesh, which may be
    # important for very coarse meshes.
    return sender_pos_in_in_rotated_space - receiver_pos_in_rotated_space


def get_rotation_matrices_to_local_coordinates(reference_phi, reference_theta,
                                               rotate_latitude, rotate_longitude):

    """Returns a rotation matrix to rotate to a point based on a reference vector.

    The rotation matrix is build such that, a vector in the
    same coordinate system at the reference point that points towards the pole
    before the rotation, continues to point towards the pole after the rotation.

    Args:
        reference_phi: [leading_axis] Polar angles of the reference.
        reference_theta: [leading_axis] Azimuthal angles of the reference.
        rotate_latitude: Whether to produce a rotation matrix that would rotate
            R^3 vectors to zero latitude.
        rotate_longitude: Whether to produce a rotation matrix that would rotate
            R^3 vectors to zero longitude.

    Returns:
        Matrices of shape [leading_axis] such that when applied to the reference
            position with `rotate_with_matrices(rotation_matrices, reference_pos)`

            * phi goes to 0. if "rotate_longitude" is True.

            * theta goes to np.pi / 2 if "rotate_latitude" is True.

            The rotation consists of:
            * rotate_latitude = False, rotate_longitude = True:
                Latitude preserving rotation.
            * rotate_latitude = True, rotate_longitude = True:
                Latitude preserving rotation, followed by longitude preserving
                rotation.
            * rotate_latitude = True, rotate_longitude = False:
                Latitude preserving rotation, followed by longitude preserving
                rotation, and the inverse of the latitude preserving rotation. Note
                this is computationally different from rotating the longitude only
                and is. We do it like this, so the polar geodesic curve, continues
                to be aligned with one of the axis after the rotation.

    """

    if rotate_longitude and rotate_latitude:

        # We first rotate around the z axis "minus the azimuthal angle", to get the
        # point with zero longitude
        azimuthal_rotation = - reference_phi

        # One then we will do a polar rotation (which can be done along the y
        # axis now that we are at longitude 0.), "minus the polar angle plus 2pi"
        # to get the point with zero latitude.
        polar_rotation = - reference_theta + np.pi/2

        return transform.Rotation.from_euler(
            "zy", np.stack([azimuthal_rotation, polar_rotation],
                        axis=1)).as_matrix()
    elif rotate_longitude:
        # Just like the previous case, but applying only the azimuthal rotation.
        azimuthal_rotation = - reference_phi
        return transform.Rotation.from_euler("z", -reference_phi).as_matrix()
    elif rotate_latitude:
        # Just like the first case, but after doing the polar rotation, undoing
        # the azimuthal rotation.
        azimuthal_rotation = - reference_phi
        polar_rotation = - reference_theta + np.pi/2

        return transform.Rotation.from_euler(
            "zyz", np.stack(
                [azimuthal_rotation, polar_rotation, -azimuthal_rotation]
                , axis=1)).as_matrix()
    else:
        raise ValueError(
            "At least one of longitude and latitude should be rotated.")


def rotate_with_matrices(rotation_matrices, positions):
    return np.einsum("bji,bi->bj", rotation_matrices, positions)


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
    mask_not_all_nan = [torch.tensor(True) for i in range(24)]
    initial_time_dim = target_train.shape[1]
    for t in range(initial_time_dim):
        nan_sum = target_train[:,t].isnan().sum()
        mask_not_all_nan.append(nan_sum < target_train.shape[0])
    mask_not_all_nan = torch.stack(mask_not_all_nan)
    idxs_not_all_nan = torch.argwhere(mask_not_all_nan)

    return idxs_not_all_nan
    

def derive_train_val_idxs(train_year_start, train_month_start, train_day_start, train_year_end, train_month_end,
                         train_day_end, first_year, idxs_not_all_nan=None, validation_year=None):
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
        # To make the seasonal plots we consider from 01/12/YYYY-1 to 30/11/YYYY
        val_start_idx, val_end_idx = date_to_idxs(validation_year-1, 12, 1, validation_year, 11, 30, first_year)

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
        train_idxs_list = [i for i in train_idxs_list if i in idxs_not_all_nan]
        val_idxs = [i for i in val_idxs_list if i in idxs_not_all_nan]
    
    train_idxs = torch.tensor(train_idxs_list)
    val_idxs = torch.tensor(val_idxs_list)

    return train_idxs, val_idxs


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
    means_high = torch.tensor([x_high[:,0].mean(), x_high[:,1:].mean()])
    stds_high = torch.tensor([x_high[:,0].std(), x_high[:,1:].std()])

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


def standardize_input(x_low, x_high, means_low, stds_low, means_high, stds_high, args, accelerator=None):

    write_log(f'\nStandardizing the low-res input data.', args, accelerator, 'a')

    x_low_standard = torch.zeros((x_low.size()), dtype=torch.float32)

    # Standardize the data
    if args.stats_mode == "var":
        for var in range(5):
            x_low_standard[:,:,var,:] = (x_low[:,:,var,:]-means_low[var])/stds_low[var]  # num_nodes, time, vars, levels
    elif args.stats_mode == "field":
        for var in range(5):
            for lev in range(5):
                x_low_standard[:,:,var,lev] = (x_low[:,:,var,lev]-means_low[var, lev])/stds_low[var, lev]  # num_nodes, time, vars, levels
    else:
        raise Exception("Arg 'stats_mode' should be either 'var' or 'field'")

    write_log(f'\nStandardizing the high-res input data.', args, accelerator, 'a')

    # Standardize the data
    x_high_standard = torch.zeros((x_high.size()), dtype=torch.float32)
    
    x_high_standard[:,0] = (x_high[:,0] - means_high[0]) / stds_high[0]
    x_high_standard[:,1:] = (x_high[:,1:] - means_high[1]) / stds_high[1]

    return x_low_standard, x_high_standard
    

######################################################
#------------------- TRAIN UTILITIES -----------------
######################################################


#-----------------------------------------------------
#---------------------- METRICS ----------------------
#-----------------------------------------------------


class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    (from the Deep Learning tutorials of DSSC)
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_binary_one(prediction, target, reduction="mean"):
    prediction_class = torch.where(prediction > 0.0, 1.0, 0.0)
    correct_items = (prediction_class == target).float()
    if reduction is None:
        return correct_items
    elif reduction == "mean":
        return torch.mean(correct_items)
        # return correct_items.sum() / prediction.shape[0]


def accuracy_binary_one_classes(prediction, target, reduction="mean"):
    prediction_class = torch.where(prediction > 0.0, 1.0, 0.0)
    correct_items = prediction_class == target
    correct_items_class0 = correct_items[target==0.0]
    correct_items_class1 = correct_items[target==1.0]
    if reduction is None:
        return correct_items_class0, correct_items_class1
    elif reduction == "mean":
        if correct_items_class0.shape[0] > 0:
            acc_class0 = correct_items_class0.sum() / correct_items_class0.shape[0]
        else:
            acc_class0 = torch.tensor(torch.nan)
        if correct_items_class1.shape[0] > 0:
            acc_class1 = correct_items_class1.sum() / correct_items_class1.shape[0]
        else:
            acc_class1 = torch.tensor(torch.nan)
        return acc_class0, acc_class1


def compute_metrics(y_pred, y_true, threshold=0.1):
    metrics = {}

    # Compute precipitation
    pr_pred = np.expm1(y_pred)
    pr_true = np.expm1(y_true)
    pr_true[pr_true < threshold] = 0
    pr_true = np.round(pr_true, decimals=1)
    pr_pred[pr_pred < threshold] = 0

    pr_pred_spatial_avg = np.nanmean(pr_pred, axis=1)
    pr_true_spatial_avg = np.nanmean(pr_true, axis=1)
    pr_pred_spatial_p99 = np.nanpercentile(pr_pred, q=99, axis=1)
    pr_true_spatial_p99 = np.nanpercentile(pr_true, q=99, axis=1)
    pr_pred_spatial_p999 = np.nanpercentile(pr_pred, q=99.9, axis=1)
    pr_true_spatial_p999 = np.nanpercentile(pr_true, q=99.9, axis=1)

    # spatial biases
    spatial_bias_percentage = (pr_pred_spatial_avg - pr_true_spatial_avg) / (pr_true_spatial_avg + 1e-6) * 100
    spatial_p99_bias_percentage = (pr_pred_spatial_p99 - pr_true_spatial_p99) / (pr_true_spatial_p99 + 1e-6) * 100
    spatial_p999_bias_percentage = (pr_pred_spatial_p999 - pr_true_spatial_p999) / (pr_true_spatial_p999 + 1e-6) * 100

    mask_not_nan_y = ~np.isnan(y_true.flatten())
    mask_not_nan = ~np.isnan(pr_true.flatten())
    y_true = y_true.flatten()[mask_not_nan_y]
    y_pred = y_pred.flatten()[mask_not_nan_y]
    
    # Basic error metrics
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))

    metrics['Avg spatial Bias (over)'] = np.mean(spatial_bias_percentage[spatial_bias_percentage>0])
    metrics['Avg spatial Bias (under)'] = np.mean(spatial_bias_percentage[spatial_bias_percentage<=0])
    metrics['Avg spatial p99 Bias (over)'] = np.nanmean(spatial_p99_bias_percentage[spatial_p99_bias_percentage>0])
    metrics['Avg spatial p99 Bias (under)'] = np.nanmean(spatial_p99_bias_percentage[spatial_p99_bias_percentage<=0])
    metrics['Avg spatial p99.9 Bias (over)'] = np.nanmean(spatial_p999_bias_percentage[spatial_p999_bias_percentage>0])
    metrics['Avg spatial p99.9 Bias (under)'] = np.nanmean(spatial_p999_bias_percentage[spatial_p999_bias_percentage<=0])
    
    # Spatial correlation
    metrics['Pearson Corr'], _ = pearsonr(y_pred, y_true)
    metrics['Spearman Corr'], _ = spearmanr(y_pred, y_true)
    
    # Probability of Detection and False Alarm Ratio for extremes
    pr_true_p99 = np.nanpercentile(pr_true, q=99)
    hits = np.nansum((pr_pred >= pr_true_p99) & (pr_true >= pr_true_p99))
    false_alarms = np.nansum((pr_pred >= pr_true_p99) & (pr_true < pr_true_p99))
    actual_extremes = np.nansum(pr_true >= pr_true_p99)
    predicted_extremes = np.nansum(pr_pred >= pr_true_p99)
    
    metrics['POD (p99)'] = hits / (actual_extremes + 1e-6) # Probability of Detection
    metrics['FAR (p99)'] = false_alarms / (predicted_extremes + 1e-6)  # False Alarm Ratio
    
    # To avoid numerical issues due to unrealistic predictions
    pr_pred[np.isinf(pr_pred)] = np.nan

    # distributions comparison
    metrics['Earth Mover Distance'] = wasserstein_distance(pr_true.flatten()[mask_not_nan], pr_pred.flatten()[mask_not_nan])
    metrics['KL Divergence'] = entropy(pr_true.flatten()[mask_not_nan] + 1e-6, pr_pred.flatten()[mask_not_nan] + 1e-6)
    ks_stat, p_value = ks_2samp(pr_true.flatten()[mask_not_nan], pr_pred.flatten()[mask_not_nan])
    metrics['KS Statistic'] = ks_stat
    metrics['KS p-value'] = p_value

    # PDF comparison
    hist_y_true, _ = np.histogram(pr_true.flatten()[mask_not_nan], bins=np.arange(0,200,0.1).astype(np.float32), density=False)
    hist_y_pred, _ = np.histogram(pr_pred.flatten()[mask_not_nan], bins=np.arange(0,200,0.1).astype(np.float32), density=False)

    metrics["PDF Cos Sim"] = cosine_similarity((hist_y_true/hist_y_true.sum()).reshape(1, -1), (hist_y_pred/hist_y_pred.sum()).reshape(1, -1))
    metrics["PDF Chi Squared"] = 0.5 * np.sum((hist_y_true/hist_y_true.sum() - hist_y_pred/hist_y_pred.sum()) ** 2 / (hist_y_true/hist_y_true.sum() + hist_y_pred/hist_y_pred.sum() + 1e-6))
    
    return metrics

#-----------------------------------------------------
#--------------- CUSTOM LOSS FUNCTIONS ---------------
#-----------------------------------------------------


class weighted_mse_loss():
    def __call__(input_batch, target_batch, weights):
        e = (input_batch - target_batch) ** 2
        return torch.sum(weights * e) / torch.sum(weights)

class weighted_mae_loss():
    def __call__(input_batch, target_batch, weights):
        e = torch.abs(input_batch - target_batch)
        return torch.sum(weights * e) / torch.sum(weights)

class quantized_loss():
    '''
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, alpha=0.025):
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        print(f"alpha: {self.alpha}")

    def __call__(self, prediction_batch, target_batch, bins):
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        loss_quantized = 0
        bins = bins.int()
        for b in torch.unique(bins):
            mask_b = (bins == b)
            loss_quantized += self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])
        return loss_mse + self.alpha * loss_quantized, loss_mse, loss_quantized
    

class quantized_loss_scaled():
    '''
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, gamma=0.5, scale=0.001):
        self.gamma = gamma
        self.scale = scale
        print(f"gamma: {self.gamma}, scale: {self.scale}")

    def __call__(self, prediction_batch, target_batch, bins):
        loss_quantized = 0
        bins = bins.int()
        for b in torch.unique(bins):
            mask_b = (bins == b)
            loss_quantized += torch.sum((prediction_batch[mask_b] - target_batch[mask_b])**2) * (1/torch.sum(mask_b))**self.gamma
        return self.scale * loss_quantized
    

class quantized_loss_mod():
    '''
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, alpha=1):
        self.mse_loss = nn.MSELoss(reduction="none")
        self.alpha = alpha
        print(f"alpha: {self.alpha}")

    def __call__(self, prediction_batch, target_batch, bins):
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        loss_quantized = []
        bins = bins.int()
        alpha_vector = []
        for b in torch.unique(bins):
            mask_b = (bins == b)
            loss_quantized += alpha_vector * self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])
        loss_quantized = torch.sum(torch.stack(loss_quantized))
        return loss_mse + self.alpha * loss_quantized, loss_mse, loss_quantized


class quantized_loss_bins():
    '''
    Used in inference to derive the QMSE term for the individual bins
    bins:   array containing the bin number for each of the nodes
            shape = (n_nodes)
    '''
    def __init__(self, alpha=0.025):
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        print(f"alpha: {self.alpha}")

    def __call__(self, prediction_batch, target_batch, bins, accelerator, nbins=12):
        loss_mse = self.mse_loss(prediction_batch, target_batch)
        loss_quantized = 0
        bins = bins.int()
        losses = torch.ones((nbins)).to(accelerator.device) * torch.nan
        for b in torch.unique(bins):
            mask_b = (bins == b)
            losses[b] = self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])
        return losses, None, None


#-----------------------------------------------------
#-------------------- MODEL PARAMETERS ---------------
#-----------------------------------------------------

def check_freezed_layers(model, log_path, log_file, accelerator):
    for name, param in model.named_parameters():
        n_param = param.numel() 
        if accelerator is None or accelerator.is_main_process:
            with open(log_path+log_file, 'a') as f:
                f.write(f"\nLayer {name} requires_grad = {param.requires_grad} and has {n_param} parameters") 


#-----------------------------------------------------
#---------------------- TRAIN ------------------------
#-----------------------------------------------------

class Trainer(object):

    def __init__(self):
        super(Trainer, self).__init__()

    #--- CLASSIFIER
    def train_cl(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args,
                        epoch_start, alpha=0.75, gamma=2):
        
        write_log(f"\nStart training the classifier.", args, accelerator, 'a')

        step = 0
        
        for epoch in range(epoch_start, epoch_start+args.epochs):
            
            model.train()
            write_log(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}", args, accelerator, 'a')

            # Define objects to track meters
            all_loss_meter = AverageMeter()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            acc_class0_meter = AverageMeter()
            acc_class1_meter = AverageMeter()

            # Validation
            loss_meter_val = AverageMeter()
            loss_meter_val_1_gpu = AverageMeter()
            acc_meter_val = AverageMeter()
            acc_class0_meter_val = AverageMeter()
            acc_class1_meter_val = AverageMeter()

            start = time.time()

            for graph in dataloader_train:
                
                optimizer.zero_grad()             
                y_pred = model(graph).squeeze()

                train_mask = graph["high"].train_mask      
                y = graph['high'].y   

                # Gather from all processes for metrics
                all_y_pred, all_y, all_train_mask = accelerator.gather((y_pred, y, train_mask))

                # Apply mask
                y_pred, y = y_pred[train_mask], y[train_mask]
                all_y_pred, all_y = all_y_pred[all_train_mask], all_y[all_train_mask]
                
                loss = loss_fn(y_pred, y, alpha, gamma, reduction='mean')
                all_loss = loss_fn(all_y_pred, all_y, alpha, gamma, reduction='mean')
                
                accelerator.backward(loss)
                #accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                step += 1
                
                loss_meter.update(val=loss.item(), n=y_pred.shape[0])   
                all_loss_meter.update(val=all_loss.item(), n=all_y_pred.shape[0])   
                
                acc = accuracy_binary_one(all_y_pred, all_y)
                acc_class0, acc_class1 = accuracy_binary_one_classes(all_y_pred, all_y)

                acc_meter.update(val=acc.item(), n=all_y_pred.shape[0])
                acc_class0_meter.update(val=acc_class0.item(), n=(all_y==0).sum().item())
                acc_class1_meter.update(val=acc_class1.item(), n=(all_y==1).sum().item())

                accelerator.log({'epoch':epoch, 'accuracy iteration': acc_meter.val, 'loss avg': all_loss_meter.avg,
                                 'loss avg (1GPU)': loss_meter.avg, 'accuracy avg': acc_meter.avg,
                                 'accuracy class0 avg': acc_class0_meter.avg, 'accuracy class1 avg': acc_class1_meter.avg}, step=step)
                
            end = time.time()

            # End of epoch --> write log and save checkpoint
            accelerator.log({'epoch':epoch, 'loss epoch': all_loss_meter.avg, 'loss epoch (1GPU)': loss_meter.avg,  'accuracy epoch': acc_meter.avg,
                             'accuracy class0 epoch': acc_class0_meter.avg, 'accuracy class1 epoch': acc_class1_meter.avg}, step=step)
            write_log(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds. Loss - total: {all_loss_meter.sum:.4f} - average: {all_loss_meter.avg:.10f}; "
                      + f"acc: {acc_meter.avg:.4f}; acc class 0: {acc_class0_meter.avg:.4f}; acc class 1: {acc_class1_meter.avg:.4f}.", args, accelerator, 'a')
            
            # if lr_scheduler is not None and lr_scheduler.get_last_lr()[0] > 0.00001:
            #     lr_scheduler.step()

            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/")
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")

            # Perform the validation step
            model.eval()

            y_pred_val = []
            y_val = []
            train_mask_val = []
                
            with torch.no_grad():    
                for i, graph in enumerate(dataloader_val):
                    # Append the data for the current epoch
                    train_mask_val.append(graph["high"].train_mask)            
                    y_pred_val.append(model(graph).squeeze())
                    y_val.append(graph['high'].y)

                # Create tensors
                train_mask_val = torch.stack(train_mask_val)
                y_pred_val = torch.stack(y_pred_val)
                y_val = torch.stack(y_val)

                # Validation metrics for 1GPU
                loss_val_1gpu = loss_fn(y_pred_val[train_mask_val], y_val[train_mask_val], alpha, gamma, reduction="mean")

                # Gather from all processes for metrics
                y_pred_val, y_val, train_mask_val = accelerator.gather((y_pred_val, y_val, train_mask_val))

                # Apply mask
                y_pred_val, y_val = y_pred_val[train_mask_val], y_val[train_mask_val]

                # Compute metrics on all validation dataset            
                loss_val = loss_fn(y_pred_val, y_val, alpha, gamma, reduction="mean")

                acc_class0_val, acc_class1_val = accuracy_binary_one_classes(y_pred_val, y_val)
                acc_val = accuracy_binary_one(y_pred_val, y_val)
            
                        
            if lr_scheduler is not None:
                lr_scheduler.step(loss_val.item())
           
            accelerator.log({'epoch':epoch, 'validation loss': loss_val.item(), 'validation loss (1GPU)': loss_val_1gpu.item(),
                             'validation accuracy': acc_val.item(),
                             'validation accuracy class0': acc_class0_val.item(),
                             'validation accuracy class1': acc_class1_val.item(),
                             'lr': np.mean(lr_scheduler._last_lr)}, step=step)
                
    #--- REGRESSOR
    def train_reg(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, accelerator, args, epoch_start=0):
        
        write_log(f"\nStart training the regressor.", args, accelerator, 'a')

        step = 0
        
        for epoch in range(epoch_start, epoch_start+args.epochs):

            model.train()
            write_log(f"\nEpoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.8f}", args, accelerator, 'a')
            
            # Define objects to track meters
            loss_meter = AverageMeter()
            all_loss_meter = AverageMeter()
            
            if args.loss_fn == "quantized_loss":
                loss_term1_meter = AverageMeter()
                loss_term2_meter = AverageMeter()

            start = time.time()
            
            # TRAIN
            for graph in dataloader_train:
            
                optimizer.zero_grad()
                y_pred = model(graph).squeeze()

                train_mask = graph['high'].train_mask
                y = graph['high'].y
                w = graph['high'].w

                # Gather from all processes for metrics
                all_y_pred, all_y, all_w, all_train_mask = accelerator.gather((y_pred, y, w, train_mask))

                # Apply mask
                y_pred, y, w = y_pred[train_mask], y[train_mask], w[train_mask]
                all_y_pred, all_y, all_w = all_y_pred[all_train_mask], all_y[all_train_mask], all_w[all_train_mask]

                # print(f"{accelerator.device} - all_y_pred.shape: {all_y_pred.shape}, all_y.shape: {all_y.shape}, all_w.shape: {all_w.shape}")
                
                if args.loss_fn == "quantized_loss":
                    loss, _, _ = loss_fn(y_pred, y, w)
                    all_loss, loss_term1, loss_term2 = loss_fn(all_y_pred, all_y, all_w)
                else:
                    loss = loss_fn(y_pred, y, w)
                    all_loss = loss_fn(all_y_pred, all_y, all_w)
                
                accelerator.backward(loss)
                #accelerator.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                step += 1
                
                # Log values to wandb
                loss_meter.update(val=loss.item(), n=y_pred.shape[0])    
                all_loss_meter.update(val=all_loss.item(), n=all_y_pred.shape[0])
                
                if args.loss_fn == "quantized_loss":
                    loss_term1_meter.update(val=loss_term1.item(), n=all_y_pred.shape[0])
                    loss_term2_meter.update(val=loss_term2.item(), n=all_y_pred.shape[0])
                    
                accelerator.log({'epoch':epoch, 'loss iteration': loss_meter.val, 'loss avg': loss_meter.avg, 'loss all avg': all_loss_meter.avg}, step=step)

            end = time.time()

            if args.loss_fn == "quantized_loss":
                accelerator.log({'epoch':epoch, 'train loss (1GPU)': loss_meter.avg, 'train loss': all_loss_meter.avg,
                                 'train mse loss': loss_term1_meter.avg, 'train quantized loss': loss_term2_meter.avg}, step=step)
            else:
                accelerator.log({'epoch':epoch, 'train loss (1GPU)': loss_meter.avg, 'train loss': all_loss_meter.avg}, step=step)

            write_log(f"\nEpoch {epoch+1} completed in {end - start:.4f} seconds." +
                      f"Loss - total: {all_loss_meter.sum:.4f} - average: {all_loss_meter.avg:.10f}. ", args, accelerator, 'a')
                    
            accelerator.save_state(output_dir=args.output_path+f"checkpoint_{epoch}/")
            torch.save({"epoch": epoch}, args.output_path+f"checkpoint_{epoch}/epoch")

            # VALIDATION
            # Validation is performed on all the validation dataset at once
            model.eval()

            y_pred_val = []
            y_val = []
            w_val = []
            train_mask_val = []

            with torch.no_grad():    
                for graph in dataloader_val:
                    # Append the data for the current epoch
                    train_mask_val.append(graph["high"].train_mask)            
                    y_pred_val.append(model(graph).squeeze())
                    y_val.append(graph['high'].y)
                    w_val.append(graph['high'].w)
                
                # Create tensors
                train_mask_val = torch.cat(train_mask_val, dim=0)
                y_pred_val = torch.cat(y_pred_val, dim=0)
                y_val = torch.cat(y_val, dim=0)
                w_val = torch.cat(w_val, dim=0)

                # Log validation metrics for 1GPU
                if args.loss_fn == "quantized_loss":
                    loss_val_1gpu,  _, _ = loss_fn(y_pred_val[train_mask_val], y_val[train_mask_val], w_val[train_mask_val])
                else:
                    loss_val_1gpu = loss_fn(y_pred_val[train_mask_val], y_val[train_mask_val], w_val[train_mask_val])

                # Gather from all processes for metrics
                y_pred_val, y_val, w_val, train_mask_val = accelerator.gather((y_pred_val, y_val, w_val, train_mask_val))

                # Apply mask
                y_pred_val, y_val, w_val = y_pred_val[train_mask_val], y_val[train_mask_val], w_val[train_mask_val]
                    
                if args.loss_fn == "quantized_loss":
                    loss_val, loss_term1_val, loss_term2_val = loss_fn(y_pred_val, y_val, w_val)
                else:
                    loss_val = loss_fn(y_pred_val, y_val, w_val)

            if lr_scheduler is not None:
                lr_scheduler.step(loss_val.item())
            
            if args.loss_fn == "quantized_loss":
                accelerator.log({'epoch':epoch, 'validation loss (1GPU)': loss_val_1gpu.item(), 'validation loss': loss_val.item(),
                                 'validation mse loss': loss_term1_val.item(),'validation quantized loss': loss_term2_val.item(),
                                 'lr': np.mean(lr_scheduler._last_lr)}, step=step)
            else:
                accelerator.log({'epoch':epoch, 'validation loss (1GPU)': loss_val_1gpu.item(), 'validation loss': loss_val.item(),
                                 'lr': np.mean(lr_scheduler._last_lr)}, step=step)
        


#-----------------------------------------------------
#----------------------- TEST ------------------------
#-----------------------------------------------------


class Tester(object):

    def test(self, model, dataloader,low_high_graph, args, accelerator=None):
        model.eval()
        step = 0 

        pr = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                # Regressor
                y_pred = model(graph)
                #pr.append(torch.expm1(y_pred)) 
                pr.append(y_pred)
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        pr = torch.stack(pr)
        times = torch.stack(times)

        return pr, times
    
    def test_encoding(self, model_reg, dataloader,low_high_graph, args, accelerator=None):
        model_reg.eval()
        step = 0 

        pr_reg = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                # Regressor
                y_pred_reg = model_reg(graph)
                pr_reg.append(y_pred_reg)
                
                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        pr_reg = torch.stack(pr_reg)
        times = torch.stack(times)

        return pr_reg, times
    
    def test_cl_reg(self, model_cl, model_reg, dataloader,low_high_graph, args, accelerator=None):
        model_cl.eval()
        model_reg.eval()
        step = 0 

        pr_cl = []
        pr_reg = []
        times = []
        with torch.no_grad():    
            for graph in dataloader:

                t = graph.t
                times.append(t)
                
                y_pred_cl = model_cl(graph)
                y_pred_reg = model_reg(graph)
                
                # Classifier
                pr_cl.append(y_pred_cl)
                
                # Regressor
                pr_reg.append(y_pred_reg)

                if step % 100 == 0:
                    if accelerator is None or accelerator.is_main_process:
                        with open(args.output_path+args.log_file, 'a') as f:
                            f.write(f"\nStep {step} done.")
                step += 1 

        pr_cl = torch.stack(pr_cl)
        pr_reg = torch.stack(pr_reg)
        times = torch.stack(times)

        return pr_cl, pr_reg, times


#-----------------------------------------------------
#-------------------- VALIDATION ---------------------
#-----------------------------------------------------
    

class Validator(object):

    def validate_cl(self, model, dataloader, loss_fn, accelerator, alpha=0.75, gamma=2):

        model.eval()

        # loss_meter = AverageMeter()
        # acc_meter = AverageMeter()
        # acc_class0_meter = AverageMeter()
        # acc_class1_meter = AverageMeter()

        y_pred_val = []
        y_val = []
        train_mask_val = []
    
        with torch.no_grad():    
            for graph in dataloader:

                train_mask = graph["high"].train_mask

                if train_mask.sum() == 0:
                    continue

                # Classifier
                y_pred = model(graph).squeeze()
                y = graph['high'].y

                # Gather results from other GPUs
                accelerator.wait_for_everyone()
                all_y_pred, all_y, all_train_mask = accelerator.gather_for_metrics((y_pred, y, train_mask))

                # Append the batch results
                y_pred_val.append(all_y_pred)
                y_val.append(all_y)
                train_mask_val.append(all_train_mask)

            # Create tensors
            train_mask_val = torch.stack(train_mask_val)
            y_pred_val = torch.stack(y_pred_val)[train_mask_val]
            y_val = torch.stack(y_val)[train_mask_val]

            print(train_mask_val.shape, train_mask_val.sum(), y_pred_val.shape, y_val.shape)

            # Compute metrics on all validation dataset            
            loss = loss_fn(y_pred_val, y_val, alpha, gamma, reduction="mean")
            acc = accuracy_binary_one(y_pred_val, y_val)
            acc_class0, acc_class1 = accuracy_binary_one_classes(y_pred_val, y_val)

        return loss.item(), acc.item(), acc_class0.item(), acc_class1.item()

            # loss_meter.update(val=loss.item(), n=y_pred.shape[0])
            # acc_meter.update(val=acc.item(), n=y_pred.shape[0])
            # if not acc_class0.isnan():
            #     acc_class0_meter.update(val=acc_class0.item(), n=y_pred.shape[0])
            # if not acc_class1.isnan():
            #     acc_class1_meter.update(val=acc_class1.item(), n=y_pred.shape[0])

        # return loss_meter.avg, acc_meter.avg, acc_class0_meter.avg, acc_class1_meter.avg

    def validate_reg(self, model, dataloader, loss_fn, accelerator, args):

        all_loss_meter_val = AverageMeter()

        model.eval()
        
        with torch.no_grad(): 
            for i, graph in enumerate(dataloader):

                train_mask = graph["high"].train_mask
                if train_mask.sum() == 0:
                    continue
                y = graph['high'].y
                w = graph['high'].w
                    
                # Regressor
                y_pred = model(graph).squeeze()

                # Gather from all processes for metrics
                all_y_pred, all_y, all_w, all_train_mask = accelerator.gather((y_pred, y, w, train_mask))

                # Apply mask
                y_pred, y, w = y_pred[train_mask], y[train_mask], w[train_mask]
                all_y_pred, all_y, all_w = all_y_pred[all_train_mask], all_y[all_train_mask], all_w[all_train_mask]

                #loss, _, _ = loss_fn(y_pred, y, w)
                all_loss, _, _ = loss_fn(all_y_pred, all_y, all_w)    
                    
                all_loss_meter_val.update(val=all_loss.item(), n=all_y_pred.shape[0])

        return all_loss_meter_val.avg


    def validate_reg_all(self, model, dataloader, accelerator, args):
        
        train_mask = []
        y_pred = []
        y = []
        w = []

        model.eval()
        
        with torch.no_grad(): 
            for i, graph in enumerate(dataloader):

                train_mask.append(graph["high"].train_mask)                    
                y.append(graph['high'].y)
                w.append(graph['high'].w)
                y_pred.append(model(graph).squeeze())

            # Create tensors
            train_mask = torch.stack(train_mask)
            y_pred = torch.stack(y_pred)
            y = torch.stack(y)
            w = torch.stack(w)
                    
        return y_pred, y, w, train_mask
    
        

