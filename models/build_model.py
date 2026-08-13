import inspect
from .registry import MODEL_REGISTRY

def build_model(
    x_low_var_dim,
    x_low_lev_dim,
    x_high_dim,
    output_dim,
    args,
    n_target_variables=1):

    ModelClass = MODEL_REGISTRY[args.model_name]

    sig = inspect.signature(ModelClass.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    
    filtered = {
        k: v for k, v in vars(args).items()
        if k in allowed
    }

    explicit_kwargs = dict(
        x_low_var_dim=x_low_var_dim,
        x_low_lev_dim=x_low_lev_dim,
        x_high_dim=x_high_dim,
        output_dim=output_dim,
    )

    # n_target_variables is only accepted by multivariable-aware models
    if "n_target_variables" in allowed:
        explicit_kwargs["n_target_variables"] = n_target_variables

    model = ModelClass(
        **explicit_kwargs,
        **filtered
        )

    return model