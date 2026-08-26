import numpy as np
import matplotlib
import os
os.environ["CARTOPY_DATA_DIR"] = "/leonardo_work/ICT26_ESP/vblasone/cartopy/"
import cartopy.crs as ccrs

from utils.plotting.plots import get_cmap_dict, plot_maps, plot_pdf, plot_maps_grid, plot_pdf_grid

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

    # binmin
    binmin = meta[target_type]["binmin"]
    if binmin is None:
        binmin = min(np.floor(np.min(y_pred_plot)), np.floor(np.min(y_plot))) - 5
        
    # binmax
    binmax = meta[target_type]["binmax"]
    if binmax is None:
        binmax = max(np.ceil(np.max(y_pred_plot)), np.ceil(np.min(y_plot))) + 5

    # bins
    bins = np.arange(binmin,binmax,meta[target_type]["binwidth"]).astype(np.float32)

    hist_vals, bins = np.histogram(y_pred_pdf, bins=bins, density=False)
    bins_mid = (bins[:-1] + bins[1:]) / 2
    Ntot = np.nansum(hist_vals)
    hist_vals_target, bins_target = np.histogram(y_pdf, bins=bins, density=False)
    bins_target_mid = (bins_target[:-1] + bins_target[1:]) / 2
    Ntot_target = np.nansum(hist_vals_target)

    if meta[target_type]["xlim_pdf"] is None:

        meta[target_type]["xlim_pdf"] = [float(bins.min()), float(bins.max())]

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


def create_multivariable_validation_plots(
    y_pred_dict,
    y_dict,
    lon,
    lat,
    var_names,
    meta,
    ):
    cmap_dict = get_cmap_dict()
    general = meta["general"]
    map_plot_func = general.get("plot_func_maps", "scatter")   # single scalar, read once
    x_dim = general.get("x_dim")
    y_dim = general.get("y_dim")

    lon_map, lat_map = {}, {}
    avg_data, bias_data = {}, {}
    cmap_avg, cmap_bias = {}, {}
    pr_min_avg, pr_max_avg = {}, {}
    pr_min_bias, pr_max_bias = {}, {}
    legend_title_map, s_map = {}, {}

    bin_dict, hist_dict = {}, {}
    xlabel_map, xlim_map, ylim_map = {}, {}, {}
    log_xy_map, pdf_plot_func_map = {}, {}          # renamed -- no longer shadows map_plot_func
    tail_zoom_map, tail_lim_map, tail_ylim_map = {}, {}, {}

    for var_name in var_names:
        m = meta[var_name]
        y_pred_plot = y_pred_dict[var_name]
        y_plot = y_dict[var_name]

        #---- AVG / BIAS ----#
        gnn4cd_avg = np.nanmean(y_pred_plot, axis=-1)
        target_avg = np.nanmean(y_plot, axis=-1)
        bias = gnn4cd_avg - target_avg

        lon_map[var_name] = [lon, lon]
        lat_map[var_name] = [lat, lat]
        avg_data[var_name] = [gnn4cd_avg, target_avg]
        bias_data[var_name] = [bias]

        cmap_avg[var_name] = cmap_dict['avg']['cmap'] if m["cmap"] == "cmap_dict['avg']['cmap']" else m["cmap"]
        cmap_bias[var_name] = m["cmap_bias"]
        pr_min_avg[var_name] = m["vmin"]
        pr_max_avg[var_name] = m["vmax"]
        pr_min_bias[var_name] = m["vmin_bias"]
        pr_max_bias[var_name] = m["vmax_bias"]
        legend_title_map[var_name] = m["map_unit"]
        s_map[var_name] = m["s"]

        #---- PDF ----#
        y_pred_pdf = y_pred_plot.flatten()
        y_pdf = y_plot.flatten()

        binmin = m["binmin"]
        if binmin is None:
            binmin = min(np.floor(np.min(y_pred_plot)), np.floor(np.min(y_plot))) - 5

        binmax = m["binmax"]
        if binmax is None:
            binmax = max(np.ceil(np.max(y_pred_plot)), np.ceil(np.min(y_plot))) + 5

        bins = np.arange(binmin, binmax, m["binwidth"]).astype(np.float32)

        hist_vals, bins = np.histogram(y_pred_pdf, bins=bins, density=False)
        bins_mid = (bins[:-1] + bins[1:]) / 2
        Ntot = np.nansum(hist_vals)
        hist_vals_target, bins_target = np.histogram(y_pdf, bins=bins, density=False)
        bins_target_mid = (bins_target[:-1] + bins_target[1:]) / 2
        Ntot_target = np.nansum(hist_vals_target)

        xlim_pdf = m["xlim_pdf"]
        if xlim_pdf is None:
            xlim_pdf = [float(bins.min()), float(bins.max())]

        bin_dict[var_name] = [bins_target_mid, bins_mid]
        hist_dict[var_name] = [hist_vals_target/Ntot_target, hist_vals/Ntot]
        xlabel_map[var_name] = m["pdf_unit"]
        xlim_map[var_name] = xlim_pdf
        ylim_map[var_name] = m["ylim_pdf"]
        log_xy_map[var_name] = m["log_xy"]
        pdf_plot_func_map[var_name] = m.get("plot_func_pdf", general.get("plot_func_pdf", "step"))
        tail_zoom_map[var_name] = m["tail_zoom"]
        tail_lim_map[var_name] = m["tail_lim"]
        tail_ylim_map[var_name] = m["tail_ylim"]

    # -- outside the loop now --
    fig_avg = plot_maps_grid(
        lon_map, lat_map, avg_data, var_names,
        col_labels=["GNN4CD", "TARGET"],
        x_size=6, y_size=6, var_ncols=3,
        font_size_title=general["fontsize_title"], font_size=general["fontsize"],
        plot_func=map_plot_func,
        x_dim=x_dim, y_dim=y_dim,
        cmap_dict=cmap_avg, pr_min_dict=pr_min_avg, pr_max_dict=pr_max_avg,
        legend_title_dict=legend_title_map, s_dict=s_map,
        xlim=general["xlim"], ylim=general["ylim"], show_ticks=False,
        suptitle="Average", suptitle_fontsize=general["fontsize_title"],
    )

    fig_bias = plot_maps_grid(
        lon_map, lat_map, bias_data, var_names,
        col_labels=["GNN4CD - TARGET"],
        x_size=6, y_size=6, var_ncols=3,
        font_size_title=25, font_size=20,
        plot_func=map_plot_func,
        x_dim=x_dim, y_dim=y_dim,
        cmap_dict=cmap_bias, pr_min_dict=pr_min_bias, pr_max_dict=pr_max_bias,
        legend_title_dict=legend_title_map, s_dict=s_map,
        xlim=general["xlim"], ylim=general["ylim"], show_ticks=False,
        suptitle="Bias", suptitle_fontsize=25,
    )

    fig_pdf = plot_pdf_grid(
        bin_dict, hist_dict, var_names,
        color_list=["black", "darkorange"], label_list=["TARGET", "GNN4CD"],
        ncols=3,
        xlabel_dict=xlabel_map, xlim_dict=xlim_map, ylim_dict=ylim_map,
        log_xy_dict=log_xy_map, plot_func_dict=pdf_plot_func_map,
        tail_zoom_dict=tail_zoom_map, tail_lim_dict=tail_lim_map, tail_ylim_dict=tail_ylim_map,
        suptitle="PDF", suptitle_fontsize=24,
    )

    return fig_avg, fig_bias, fig_pdf