def add_base_args(parser):

    #-- paths
    parser.add_argument('--input_path', type=str, help='path to input directory')
    parser.add_argument('--output_path', type=str, help='path to output directory')
    parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

    parser.add_argument('--low_input_file', type=str, default=None)
    parser.add_argument('--orog_file', type=str, default=None)
    parser.add_argument('--mask_sealand_file', type=str, default=None)
    parser.add_argument('--target_file', type=str, default=None)
    parser.add_argument('--graph_file', type=str, default=None) 
    parser.add_argument('--coords_ij_file', type=str, default=None)

    parser.add_argument('--use_accelerate',  action='store_true')
    parser.add_argument('--no-use_accelerate', dest='use_accelerate', action='store_false')
    parser.add_argument('--wandb_project_name', type=str)

    parser.add_argument('--metadata_file', type=str, help='metadata file')

    #-- training hyperparameters
    parser.add_argument('--epochs', type=int, default=15, help='number of total training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (global)')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--lr_scheduler', type=str, default="StepLR")
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (wd)')
    parser.add_argument('--load_checkpoint',  action='store_true')
    parser.add_argument('--no-load_checkpoint', dest='load_checkpoint', action='store_false')


    parser.add_argument('--checkpoint_ctd', type=str, help='checkpoint to load to continue')
    parser.add_argument('--ctd_training',  action='store_true')
    parser.add_argument('--no-ctd_training', dest='ctd_training', action='store_false')
    parser.add_argument('--make_val_plots', action='store_true')
    parser.add_argument('--no-make_val_plots', dest='make_val_plots', action='store_false')
    parser.add_argument('--val_plot_frequency', type=int)
    parser.add_argument('--val_plot_config', type=str)

    parser.add_argument('--loss_name', type=str)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--n_gpu', type=int, default=4)

    parser.add_argument('--model_name', type=str)
    parser.add_argument('--history_length', type=int)

    parser.add_argument('--predictand_transform_mode', type=str)
    parser.add_argument('--predictor_low_transform_mode', type=str)
    parser.add_argument('--predictor_high_transform_mode', type=str)

    parser.add_argument('--dataset_name', type=str, default='graph_dataset')
    parser.add_argument('--collate_name', type=str)
    parser.add_argument('--target_type', type=str)

    #-- start and end training dates
    parser.add_argument('--train_year_start', type=str)
    parser.add_argument('--train_month_start', type=str)
    parser.add_argument('--train_day_start', type=str)
    parser.add_argument('--train_year_end', type=str)
    parser.add_argument('--train_month_end', type=str)
    parser.add_argument('--train_day_end', type=str)
    parser.add_argument('--validation_year', type=str)
    # for random validation years
    parser.add_argument('--first_year', type=str)
    parser.add_argument('--last_year', type=str)
    parser.add_argument('--n_val_years', type=str)
    # for lists of training and validation years
    parser.add_argument('--train_years', nargs='+', type=int, default=[])
    parser.add_argument('--val_years', nargs='+', type=int, default=[])

    parser.add_argument('--WANDB_API_KEY', type=str)
    parser.add_argument('--WANDB_USERNAME', type=str)

    return parser