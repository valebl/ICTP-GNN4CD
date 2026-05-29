import numpy as np
from scipy.spatial.distance import cdist


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
        radius=None):
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

    if use_edge_attr:
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
            orog_receivers
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
        use_edge_attr=True):
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
        edge_attr = get_edge_features_lon_lat_orog(
            senders,
            receivers,
            lon_senders,
            lat_senders,
            orog_senders,
            lon_receivers,
            lat_receivers,
            orog_receivers
        )
        return edge_index, edge_attr
    else:
        return edge_index, None

def get_edge_features_lon_lat_orog(
        senders,
        receivers,
        lon_senders,
        lat_senders,
        orog_senders,
        lon_receivers,
        lat_receivers,
        orog_receivers
    ):

    delta_lon = lon_senders[senders] - lon_receivers[receivers]
    delta_lat = lat_senders[senders] - lat_receivers[receivers]
    if orog_senders is not None and orog_receivers is not None:
        delta_orog = orog_senders[senders] - orog_receivers[receivers]
        return np.column_stack((delta_lon, delta_lat, delta_orog))
    else:
        return np.column_stack((delta_lon, delta_lat))


