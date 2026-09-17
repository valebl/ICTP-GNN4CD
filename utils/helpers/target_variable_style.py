"""
Resolves a validation-plot style entry for a given target variable, so
trainer.py can call create_validation_plots(..., target_type=var_name, meta=...)
uniformly for every variable in a multivariable run, even ones your
CORDEXML_plot_params_*.json doesn't (yet) have a dedicated entry for.

Resolution order for a variable name (e.g. "tas", "pr", "hurs"):
  1. meta[var_name]           -- an explicit per-variable config entry, used as-is
  2. meta[<category>]         -- var_name maps to a known physical category
                                  (currently precipitation/temperature) that
                                  the config already has an entry for
  3. a generic/neutral style  -- auto-ranged fallback for anything else
                                  (humidity, pressure, wind components, ...)
"""

# Minimal variable -> plotting-category mapping. Deliberately mirrors (but does
# not import, since preprocess.py is not to be modified) preprocess.py's local
# VAR_TO_CATEGORY dict. Extend here if you add more variables that should
# reuse the precipitation/temperature styling rather than the generic one.
VAR_TO_PLOT_CATEGORY = {
    "pr": "precipitation",
    "tp": "precipitation",
    "tasmax": "temperature",
    "tasmin": "temperature",
    "tas": "temperature",
    "t2m": "temperature",
}

# Neutral fallback style. None values lean on create_validation_plots' own
# auto-ranging (it already does this for binmin/binmax/ylim_pdf/etc. when a
# config value is None) rather than guessing bounds for an unknown variable.
# binwidth can't be None (np.arange needs a numeric step); 1.0 is a coarse,
# safe default -- add a real per-variable config entry if it's too coarse.
GENERIC_PLOT_STYLE = {
    "pdf_unit": "[-]",
    "map_unit": "[-]",
    "vmax": None,
    "vmin": None,
    "vmax_bias": None,
    "vmin_bias": None,
    "s": 30,
    "pdf_title": None,   # filled in with the variable name below
    "map_title": None,   # filled in with the variable name below
    "log_xy": False,
    "ylim_pdf": None,
    "tail_ylim": None,
    "tail_lim": None,
    "cmap": "viridis",
    "cmap_bias": "RdBu_r",
    "tail_zoom": False,
    "binmin": None,
    "binmax": None,
    "binwidth": 1.0,
    "xlim_pdf": None,
    "plot_func_pdf": "step",
    "legend_outside": False,
}


def resolve_plot_meta(var_name, meta):
    """
    Returns a meta dict that is safe to pass straight into
    create_validation_plots(..., target_type=var_name, meta=resolve_plot_meta(var_name, meta)),
    i.e. one guaranteed to have a meta[var_name] style entry. meta["general"]
    (figsize, fontsizes, map extent, ...) is passed through untouched.
    """
    meta_out = dict(meta)  # shallow copy -- only ever adds/overwrites the var_name key

    if var_name in meta_out:
        return meta_out

    category = VAR_TO_PLOT_CATEGORY.get(var_name)
    if category is not None and category in meta_out:
        meta_out[var_name] = meta_out[category]
        return meta_out

    style = dict(GENERIC_PLOT_STYLE)
    style["pdf_title"] = f"PDF of {var_name}"
    style["map_title"] = f"average {var_name}"
    meta_out[var_name] = style
    return meta_out