from models.registry import get_model

def add_model_specific_args(parser, model_name):
    ModelClass = get_model(model_name)
    if hasattr(ModelClass, "add_model_specific_args"):
        parser = ModelClass.add_model_specific_args(parser)
    return parser