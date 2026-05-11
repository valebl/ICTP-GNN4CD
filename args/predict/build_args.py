import argparse

from args.predict.add_base_args import add_base_args
from args.predict.add_target_specific_args import add_target_specific_args
from args.shared.add_model_specific_args import add_model_specific_args

def build_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 1. Add base args
    parser = add_base_args(parser)
    args, unknown = parser.parse_known_args()

    # 2. Add target - specific args
    parser = add_target_specific_args(parser, args.target_type)

    # 3. Add model specific args
    parser = add_model_specific_args(parser, args.model_name)

    args, unknown = parser.parse_known_args()

    return args