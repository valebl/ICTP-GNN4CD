import numpy as np
import json
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


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
        e (np.array): elevation for each spatial point (time, nodes)
        mask_land (np.array, optional): a mask for the land points
    Returns:
        The valid points for each input tensor
    '''

    valid_nodes = ~np.isnan(pr).all(axis=0)
    if mask_land is not None:
        valid_nodes = np.logical_and(valid_nodes, ~np.isnan(mask_land))
    return valid_nodes

def derive_edge_index_within(
        lon_radius,
        lat_radius,
        lon_senders,
        lat_senders,
        lon_receivers,
        lat_receivers,
        orog_senders=None,
        orog_receivers=None,
        use_edge_attr=True,
        radius=None,
        dist_scale=None):
    r'''
    Derives edge_indexes within two sets of nodes based on specified lon, lat distances and orog
    Args:
        lon_senders (np.array): longitudes of all first nodes in the edges
        lat_senders (np.array): latitudes of all fisrt nodes in the edges
        lon_receivers (np.array): longitudes of all second nodes in the edges
        lat_receivers (np.array): latitudes of all second nodes in the edges
        orog_receivers (np.array): longitudes of all second nodes in the edges
        orog_receivers (np.array): latitudes of all second nodes in the edges
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

    if use_edge_attr and dist_scale is not None:

        senders = edge_index[0]
        receivers = edge_index[1]
        edge_attr = get_edge_features_lon_lat_orog(
            senders,
            receivers,
            lon_senders,
            lat_senders,
            orog_senders,
            lon_receivers,
            lat_receivers,
            orog_receivers,
            dist_scale
        )
        return edge_index, edge_attr
    else:
        return edge_index, None


def derive_edge_index_multiscale(
        lon_senders,
        lat_senders,
        lon_receivers,
        lat_receivers,
        k,
        undirected=False,
        orog_senders=None,
        orog_receivers=None,
        use_edge_attr=True,
        dist_scale=None):
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
    
    if use_edge_attr and dist_scale is not None:

        senders = edge_index[0]
        receivers = edge_index[1]
        edge_attr = get_edge_features_lon_lat_orog(
            senders,
            receivers,
            lon_senders,
            lat_senders,
            orog_senders,
            lon_receivers,
            lat_receivers,
            orog_receivers,
            dist_scale
        )
        return edge_index, edge_attr
    else:
        return edge_index, None


def compute_dist_scale(lon_senders, lat_senders):
    r'''
    Computes a characteristic spacing of the sender (low-res) grid, to be used
    as a fixed normalization constant for edge distances. Uses the median
    nearest-neighbor distance on the unit sphere.
    
    Transforms lon_*, lat_* to radians.

    Args:
        lon_senders, lat_senders (np.array): sender grid coordinates, in deg
    Returns:
        float: characteristic spacing (unit-sphere chord distance)
    '''

    lon_senders_rad = np.deg2rad(lon_senders)
    lat_senders_rad = np.deg2rad(lat_senders)

    pos = np.column_stack((
        np.cos(lat_senders_rad) * np.cos(lon_senders_rad),
        np.cos(lat_senders_rad) * np.sin(lon_senders_rad),
        np.sin(lat_senders_rad)
    ))
    tree = cKDTree(pos)
    # query 2 neighbours: the point itself (dist=0) and its nearest actual neighbour
    dists, _ = tree.query(pos, k=2)
    nn_dist = dists[:, 1]
    return float(np.median(nn_dist))


def compute_orog_scale(orog_senders, orog_receivers, senders, receivers):
    r'''
    Computes a characteristic elevation-difference scale from the training graph,
    to be reused as a fixed normalization constant at inference.
    '''
    delta_orog = orog_senders[senders] - orog_receivers[receivers]
    return float(np.std(delta_orog))


def save_edge_norm_constants(path, dist_scale, orog_scale=None):
    r'''
    Saves edge normalization constants to disk so the exact same values can be
    reused when building graphs at a different resolution/domain at inference.
    '''
    constants = {"dist_scale": dist_scale}
    if orog_scale is not None:
        constants["orog_scale"] = orog_scale
    with open(path, "w") as f:
        json.dump(constants, f)


def load_edge_norm_constants(path):
    r'''
    Loads previously saved edge normalization constants.
    Returns a dict with keys "dist_scale" and optionally "orog_scale".
    '''
    with open(path, "r") as f:
        return json.load(f)


def get_edge_features_lon_lat_orog(
        senders,
        receivers,
        lon_senders,
        lat_senders,
        orog_senders,
        lon_receivers,
        lat_receivers,
        orog_receivers,
        dist_scale=None,
        orog_scale=None
    ):
    r'''
    Computes geometry-aware edge features as a 3D unit-direction vector plus a
    dist_scale-normalized magnitude. Using the 3D Cartesian embedding of
    (lon, lat) on the unit sphere.

    Args:
        dist_scale (float): characteristic spacing constant to normalize edge
            distances by. Must be a FIXED value shared across train/val/test/
            inference graphs (e.g. computed once via compute_dist_scale on the
            training sender grid, then saved/loaded via
            save_edge_norm_constants / load_edge_norm_constants). Passing a
            freshly-computed value for each graph would reintroduce
            resolution-dependence.
        orog_scale (float, optional): same idea, for elevation differences.
            If None, delta_orog is left unnormalized (raw meters).

    Transforms lon_*, lat_* to radians.
    '''
    if dist_scale is None:
        raise ValueError(
            "dist_scale must be provided explicitly. Compute it once via "
            "compute_dist_scale() on the training graph, save it via "
            "save_edge_norm_constants(), and load+reuse it for every "
            "subsequent graph (including at inference) via "
            "load_edge_norm_constants()."
        )

    def to_unit_sphere(lon, lat):
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return np.column_stack((x, y, z))
    
    lon_senders_rad = np.deg2rad(lon_senders)
    lat_senders_rad = np.deg2rad(lat_senders)
    lon_receivers_rad = np.deg2rad(lon_receivers)
    lat_receivers_rad = np.deg2rad(lat_receivers)

    pos_senders = to_unit_sphere(lon_senders_rad, lat_senders_rad)[senders]
    pos_receivers = to_unit_sphere(lon_receivers_rad, lat_receivers_rad)[receivers]

    edge_vec = pos_senders - pos_receivers          # (E, 3)
    dist = np.linalg.norm(edge_vec, axis=-1, keepdims=True)  # (E, 1)

    eps = 1e-12
    unit_vec = edge_vec / np.clip(dist, eps, None)   # (E, 3), direction only

    dist_norm = dist / dist_scale                    # (E, 1), resolution-invariant magnitude

    features = [unit_vec, dist_norm]

    if orog_senders is not None and orog_receivers is not None:
        delta_orog = orog_senders[senders] - orog_receivers[receivers]
        if orog_scale is not None:
            delta_orog = delta_orog / orog_scale
        features.append(delta_orog[:, None])

    return np.column_stack(features)


