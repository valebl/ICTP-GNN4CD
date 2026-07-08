def add_base_args(parser):

    #-- paths
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--input_path_target', type=str)
    parser.add_argument('--input_path_predictors', type=str)
    parser.add_argument('--input_path_topo', type=str)
    parser.add_argument('--input_path_mask_sealand', type=str)
    parser.add_argument('--target_file', type=str)
    parser.add_argument('--predictors_file', type=str)
    parser.add_argument('--topo_file', type=str)
    parser.add_argument('--mask_sealand_file', type=str)
    parser.add_argument('--land_use_path', type=str)
    parser.add_argument('--land_use_file', type=str)
    parser.add_argument('--output_file', type=str)

    #-- lat lon grid values
    parser.add_argument('--lon_min', type=float)
    parser.add_argument('--lon_max', type=float)
    parser.add_argument('--lat_min', type=float)
    parser.add_argument('--lat_max', type=float)
    parser.add_argument('--lon_grid_radius_high', type=float)
    parser.add_argument('--lat_grid_radius_high', type=float)
    parser.add_argument('--lon_grid_radius_low', type=float)
    parser.add_argument('--lat_grid_radius_low', type=float)
    parser.add_argument('--k_low2high', type=int)

    #-- other
    parser.add_argument('--mask_path', type=str)
    parser.add_argument('--mask_file', type=str)
    parser.add_argument('--predictors_dataset', type=str)
    parser.add_argument('--target_dataset', type=str)
    parser.add_argument('--target_type', type=str)
    parser.add_argument('--target_multiplier', type=float)
    parser.add_argument('--low2high_norm_constants', type=str)
    parser.add_argument('--high_norm_constants', type=str)

    parser.add_argument("--params", type=str, help="Comma-separated list of variable names")
    parser.add_argument("--levels", type=str, help="Comma-separated list of pressure levels")
    parser.add_argument("--dataset_name", type=str, choices=["CORDEXML", "ERA5", "CMIP6"], help="Name of the dataset loader to use")

    return parser