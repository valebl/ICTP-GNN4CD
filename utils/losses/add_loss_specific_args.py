from .registry import get_loss

def add_loss_specific_args(parser, loss_name):
    LossClass = get_loss(loss_name)
    if hasattr(LossClass, "add_loss_specific_args"):
        parser = LossClass.add_loss_specific_args(parser)
    return parser