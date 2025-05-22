import numpy as np
from scipy.spatial import transform
from scipy.spatial.distance import cdist
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
from typing import Union
from collections import defaultdict
import copy
import torch


def cut_window(lon_min, lon_max, lat_min, lat_max, lon, lat, *argv):
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
    bool_both = np.logical_and(bool_lon, bool_lat).flatten()
    lon_sel = lon.flatten()[bool_both]
    lat_sel = lat.flatten()[bool_both]
    v = []
    for arg in argv:
        if arg.ndim > 2:
            arg = arg.reshape(arg.shape[0], -1)
            v.append(arg[:, bool_both])
        else:
            arg = arg.flatten()
            v.append(arg[bool_both])
    # print(lon_sel.shape, lat_sel.shape, v[-1].shape)
    return lon_sel, lat_sel, *v


def retain_valid_nodes(pr,mask_land=None):
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
    return valid_nodes

def derive_edge_index_within(lon_radius, lat_radius, lon_senders, lat_senders, lon_receivers, lat_receivers, use_edge_attr=True, radius=None):
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
        
        if radius is not None:
            bool_both = ((lon_receivers - xi[0]) ** 2 + (lat_receivers - xi[1]) ** 2) ** 0.5 < radius
        else:
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

    if use_edge_attr:
        senders = edge_index[0]
        receivers = edge_index[1]
        edge_attr = get_edge_features(lon_senders, lat_senders, lon_receivers, lat_receivers, senders, receivers)
        return edge_index, edge_attr
    else:
        return edge_index


def derive_edge_index_multiscale(lon_senders, lat_senders, lon_receivers, lat_receivers, k, undirected=False, use_edge_attr=True):
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

    lonlat_senders = np.column_stack((lon_senders, lat_senders))
    lonlat_receivers = np.column_stack((lon_receivers,lat_receivers))

    dist = cdist(lonlat_receivers, lonlat_senders, metric='euclidean')
    neighbours = np.argsort(dist, axis=-1)[:, :k]
    # _ , neighbours = dist.topk(k, largest=False, dim=-1)

    for n_n2 in range(lonlat_receivers.shape[0]):
        for n_n1 in neighbours[n_n2,:]:
            if n_n1 == n_n2:
                continue
            # if np.abs(lon_receivers[n_n2] - lon_senders[n_n1]) > 0.01 and np.abs(lat_receivers[n_n2] - lat_senders[n_n1]) > 0.01:
            #     print(np.abs(lon_receivers[n_n2] - lon_senders[n_n1]), np.abs(lat_receivers[n_n2] - lat_senders[n_n1]))
            #     continue
            if [n_n1, n_n2] not in edge_index:
                edge_index.append([n_n1, n_n2])
            # edge_attr.append(dist[n_n2, n_n1])
            if undirected and [n_n2, n_n1] not in edge_index:
                edge_index.append([n_n2, n_n1])

    edge_index = np.array(edge_index).T
    
    if use_edge_attr:
        senders = edge_index[0]
        receivers = edge_index[1]
        edge_attr = get_edge_features(lon_senders, lat_senders, lon_receivers, lat_receivers, senders, receivers)
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



class RemoveNanNodes(BaseTransform):
    r"""Removes nodes with NaN values in `y` from the graph
    (functional name: :obj:`remove_nan_nodes`)."""
    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        # Identify nodes with NaN values
        nan_mask_dict = {
            node_store._key: torch.isnan(node_store.y)
            for node_store in data.node_stores if 'y' in node_store
        }
        
        # Gather all valid nodes (i.e., those without NaNs)
        valid_n_ids_dict = {
            k: torch.where(~v)[0] for k, v in nan_mask_dict.items()
        }
        
        n_map_dict = {}
        for node_store in data.node_stores:
            has_y = True
            if 'y' not in node_store:
                node_store['y'] = torch.zeros(node_store.num_nodes, dtype=torch.float32, device=node_store.x.device)
                has_y = False
            if node_store._key not in valid_n_ids_dict:
                valid_n_ids_dict[node_store._key] = torch.arange(
                    node_store.y.size(0), device=node_store.y.device
                )
        
            idx = valid_n_ids_dict[node_store._key]
            mapping = torch.full((node_store.y.size(0),), -1, dtype=torch.long, device=node_store.y.device)
            mapping[idx] = torch.arange(idx.numel(), device=node_store.y.device)
            n_map_dict[node_store._key] = mapping
            if not has_y:
                del node_store['y']
        
        # Update edge indices
        for edge_store in data.edge_stores:
            if 'edge_index' not in edge_store:
                continue

            if edge_store._key is None:
                src = dst = None
            else:
                src, _, dst = edge_store._key

            row, col = edge_store.edge_index
            valid_mask = (n_map_dict[src][row] != -1) & (n_map_dict[dst][col] != -1)
            edge_store.edge_index = torch.stack([
                n_map_dict[src][row[valid_mask]],
                n_map_dict[dst][col[valid_mask]]
            ], dim=0)
        
        # Update node features
        old_data = copy.copy(data)
        for out, node_store in zip(data.node_stores, old_data.node_stores):
            for key, value in node_store.items():
                if key == 'num_nodes':
                    out.num_nodes = valid_n_ids_dict[node_store._key].numel()
                elif node_store.is_node_attr(key):
                    out[key] = value[valid_n_ids_dict[node_store._key]]
        
        return data
