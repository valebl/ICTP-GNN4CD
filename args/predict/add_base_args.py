def add_base_args(parser):

    #-- paths
    parser.add_argument('--input_path', type=str, help='path to input directory')
    parser.add_argument('--output_path', type=str, help='path to output directory')
    parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

    parser.add_argument('--epoch', type=int)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_file', type=str, default="G_predictions.pkl")

    parser.add_argument('--graph_file', type=str, default=None)
    parser.add_argument('--low_input_file', type=str, default=None)
    parser.add_argument('--target_file', type=str, default=None)
    parser.add_argument('--orog_file', type=str, default=None)
    parser.add_argument('--mask_sealand_file', type=str, default=None)
    parser.add_argument('--coords_ij_file', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None) 
    parser.add_argument('--loss_name', type=str, default=None)
    parser.add_argument('--history_length', type=int)
    parser.add_argument('--dataset_name', type=str, default=None) 
    parser.add_argument('--target_type', type=str, default="precipitation")
    parser.add_argument('--metadata_file', type=str, help='metadata file')

    #-- start and end training dates
    parser.add_argument('--test_year_start', type=int)
    parser.add_argument('--test_month_start', type=int)
    parser.add_argument('--test_day_start', type=int)
    parser.add_argument('--test_year_end', type=int)
    parser.add_argument('--test_month_end', type=int)
    parser.add_argument('--test_day_end', type=int)
    parser.add_argument("--test_years", type=str, default="")

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int)

    parser.add_argument('--use_accelerate',  action='store_true')
    parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')

    return parser
