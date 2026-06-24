import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import wandb

import os
os.environ["CARTOPY_DATA_DIR"] = "/leonardo_work/ICT26_ESP/vblasone/cartopy/"
import cartopy.crs as ccrs

from .plots import get_cmap_dict, plot_maps, plot_pdf

def create_validation_plots(
    y_pred_plot,
    y_plot,
    lon,
    lat,
    target_type,
    meta,
    ):

    #-------------#
    #---- AVG ----#
    #-------------#

    cmap_dict = get_cmap_dict()
    bounds_avg = [0, 1, 1.5, 2, 4, 6, 8, 10, 12] #, 15, 20] #, 25, 30, 35]
    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds_avg, ncolors=256)

    gnn4cd_avg = np.nanmean(y_pred_plot, axis=-1)
    target_avg = np.nanmean(y_plot, axis=-1)

    if meta[target_type]["cmap"] == "cmap_dict['avg']['cmap']":
        cmap = cmap_dict['avg']['cmap']
    else:
        cmap = meta[target_type]["cmap"]

    fig_avg = plot_maps(
        [lon, lon],
        [lat, lat],
        [gnn4cd_avg, target_avg],
        aggr=None,
        s=meta[target_type]["s"],
        legend_title=meta[target_type]["map_unit"],
        cmap=cmap,
        sub_titles=["GNN4CD", "TARGET"],
        x_size=meta["general"]["figsize"][0],
        y_size=meta["general"]["figsize"][1],
        font_size_title=meta["general"]["fontsize_title"],
        font_size=meta["general"]["fontsize"],
        cbar_title_size=meta["general"]["fontsize_cbar_title"],
        pr_max=meta[target_type]["vmax"],
        pr_min=meta[target_type]["vmin"],
        cbar_pad=20,
        suptitle_y=0.87,
        suptitle_x=0.72,
        show_ticks=False,
        plot_func="scatter",
        xlim=meta["general"]["xlim"],
        ylim=meta["general"]["ylim"],
        proj=ccrs.PlateCarree(),
        cbar_ax_lim=[0.93,0.23,0.015,0.55]
    )

    #--------------#
    #---- BIAS ----#
    #--------------#

    bias =  gnn4cd_avg - target_avg

    fig_bias = plot_maps(
        lon,
        lat,
        bias,
        aggr=None,
        s=meta[target_type]["s"],
        legend_title=meta[target_type]["map_unit"],
        cmap=meta[target_type]["cmap_bias"],
        sub_titles=["GNN4CD - TARGET"],
        x_size=meta["general"]["figsize"][0],
        y_size=meta["general"]["figsize"][1],
        font_size_title=25,
        font_size=20,
        cbar_title_size=20,
        pr_max=meta[target_type]["vmax_bias"],
        pr_min=meta[target_type]["vmin_bias"],
        cbar_pad=20,
        suptitle_y=0.87,
        suptitle_x=0.72,
        show_ticks=False,
        plot_func="scatter",
        xlim=meta["general"]["xlim"],
        ylim=meta["general"]["ylim"],
        proj=ccrs.PlateCarree(),
        cbar_ax_lim=[0.93,0.23,0.015,0.55]
    )

    #-------------#
    #---- PDF ----#
    #-------------#

    y_pred_pdf = y_pred_plot.flatten()
    y_pdf = y_plot.flatten()

    plot_meta = meta[target_type]

    # binmin
    binmin = plot_meta.get("binmin")
    if binmin is None:
        binmin = min(np.floor(np.min(y_pred_plot)), np.floor(np.min(y_plot))) - 5
        
    # binmax
    binmax = plot_meta.get("binmax")
    if binmax is None:
        binmax = max(np.ceil(np.max(y_pred_plot)), np.ceil(np.max(y_plot))) + 5

    # binwidth
    binwidth = plot_meta.get("binwidth", 0.5 if target_type == "precipitation" else 1.0)
    if binwidth is None:
        binwidth = 0.5 if target_type == "precipitation" else 1.0

    # bins
    bins = np.arange(binmin, binmax, binwidth).astype(np.float32)

    hist_vals, bins = np.histogram(y_pred_pdf, bins=bins, density=False)
    bins_mid = (bins[:-1] + bins[1:]) / 2
    Ntot = np.nansum(hist_vals)
    hist_vals_target, bins_target = np.histogram(y_pdf, bins=bins, density=False)
    bins_target_mid = (bins_target[:-1] + bins_target[1:]) / 2
    Ntot_target = np.nansum(hist_vals_target)

    if meta[target_type]["xlim_pdf"] is None:
        meta[target_type]["xlim_pdf"] = [0.2, bins.max()+10]

    fig_pdf = plot_pdf(
        bin_list=[bins_target_mid, bins_mid],
        hist_list=[hist_vals_target/Ntot_target, hist_vals/Ntot],
        label_list=["TARGET", "GNN4CD"],
        xlabel=meta[target_type]["pdf_unit"],
        color_list=["black", "darkorange"],
        tail_lim=meta[target_type]["tail_lim"],
        ylim=meta[target_type]["ylim_pdf"],
        title=meta[target_type]["pdf_title"],
        xlim=meta[target_type]["xlim_pdf"],
        plot_func=meta[target_type]["plot_func_pdf"],
        fontsize=20,
        suptitle="",
        tail_ylim=meta[target_type]["tail_ylim"],
        log_xy=meta[target_type]["log_xy"],
        tail_zoom=meta[target_type]["tail_zoom"],
        legend_outside=meta[target_type]["legend_outside"]
    )
    
    return fig_avg, fig_bias, fig_pdf

