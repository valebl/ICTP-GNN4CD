from args.train.add_base_args import add_base_args, add_lr_specific_args
from args.train.add_loss_specific_args import add_loss_specific_args
from args.shared.add_model_specific_args import add_model_specific_args

def build_args()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 1. Add base args
    parser = add_base_args(parser)
    args, unknown = parser.parse_known_args()

    # 2. Add lr scheduler - specific args
    parser = add_lr_specific_args(parser, args.lr_scheduler)

    # 3. Add loss - specific args
    parser = add_loss_specific_args(parser, args.loss_name)

    # 4. Add model specific args
    parser = add_model_specific_args(parser, args.model_name)


    args, unknown = parser.parse_known_args()

    return args