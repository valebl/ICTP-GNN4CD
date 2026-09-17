from .registry import get_extractor


def extract_prediction(y_out, loss_name, args=None, **overrides):
    """
    Central place that maps `args` -> extractor kwargs, so every call site
    (validation loop, Predictor, anywhere else) just passes `args` through
    uniformly instead of each one needing to remember which args.xxx maps
    to which extractor kwarg. Every registered extractor must accept a
    **kwargs catch-all (see extract_bernoulli_gamma.py etc.) so passing a
    kwarg an extractor doesn't need (e.g. `threshold` to a plain MSE
    extractor) is silently ignored rather than raising a TypeError.

    `overrides` lets a specific call site override a value if it ever
    genuinely needs to (rare -- most callers should just pass `args`).
    """
    extractor = get_extractor(loss_name)

    kwargs = {}
    if args is not None:
        if hasattr(args, "threshold_nll"):
            kwargs["threshold_nll"] = args.threshold_nll

    kwargs.update(overrides)
    return extractor(y_out, **kwargs)