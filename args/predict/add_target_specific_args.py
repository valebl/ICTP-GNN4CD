def add_target_specific_args(parser, target_type):
    
    if target_type == "precipitation":
        parser.add_argument('--threshold', type=float, help='precipitation threshold')
    
    return parser