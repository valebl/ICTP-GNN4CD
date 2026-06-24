def add_lr_specific_args(parser, lr_scheduler_name):

    if lr_scheduler_name == "StepLR":
        parser.add_argument('--lr_step_size', type=int, default=10, help='scheduler step size (global) for the StepLR scheduler')
        parser.add_argument('--lr_gamma', type=float, help='gamma param for the StepLR scheduler')

    elif lr_scheduler_name == "ReduceLROnPlateau":
        parser.add_argument('--lr_mode', type=str, help='mode param for the ReduceLROnPlateau scheduler', default='min')
        parser.add_argument('--lr_factor', type=float, help='factor param for the ReduceLROnPlateau scheduler', default=0.5)
        parser.add_argument('--lr_patience', type=int, help='patience param for the ReduceLROnPlateau scheduler', default=10)
        
    elif "CosineAnnealingLR" in lr_scheduler_name:
        parser.add_argument('--lr_eta_min', type=float, help='eta_min param for the CosineAnnealingLR scheduler', default=1e-6)
        if lr_scheduler_name == "CosineAnnealingLR_with_warmup":
            parser.add_argument('--lr_warmup_epochs', type=int, help='number of warmup epochs for the CosineAnnealingLR_with_warmup scheduler', default=2)

    return parser
