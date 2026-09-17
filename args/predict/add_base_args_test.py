def add_base_args_test(parser):

    #-- paths
    parser.add_argument('--domain', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--period', type=str)
    parser.add_argument('--train_path', type=str, help='path to logfile directory')
    parser.add_argument('--input_path_P', type=str, help='path to test predictors directory')
    parser.add_argument('--input_path', type=str, help='path to input directory')
    parser.add_argument('--output_path', type=str, help='path to output directory')
    parser.add_argument('--log_path', type=str, help='path to logfile directory')
    parser.add_argument('--log_file', type=str, default='log.txt', help='log file')
    parser.add_argument('--predictors_filename', type=str, help='filename')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--loss_name', type=str, default=None)
    parser.add_argument('--history_length', type=int)
    parser.add_argument('--dataset_loader_name', type=str, default=None)  
    parser.add_argument('--graph_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default="test_predictions.pkl")
    parser.add_argument('--output_file_season', type=str, default="test_seasonal_predictions.pkl")
    parser.add_argument('--seed', type=int, default=80)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--orog_file', type=str, default=None)
    parser.add_argument('--mask_sealand_file', type=str, default=None)
    parser.add_argument('--coords_ij_file', type=str, default=None)
    parser.add_argument('--metadata_file', type=str, default=None) 
    parser.add_argument('--target_type', type=str, default="precipitation")
    parser.add_argument('--use_accelerate',  action='store_true')
    parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')

    parser.add_argument('--make_plots',  action='store_true')
    parser.add_argument('--no-make_plots', dest='make_plots', action='store_false')
    parser.add_argument('--threshold_nll', type=float)

    parser.add_argument("--target_variables", type=str, required=True,
                         help='Comma-separated target variable names, e.g. "tas,tasmax,pr"')
    parser.add_argument("--target_multipliers", type=str, required=True,
                         help='Comma-separated target multipliers, e.g. "1.0,1.0,1.0"')
    parser.add_argument("--params", type=str, help="Comma-separated list of variable names")
    parser.add_argument("--levels", type=str, help="Comma-separated list of pressure levels")

    return parser