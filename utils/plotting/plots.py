import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import os
import sys
import numpy as np
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.tri as tri
from utils.plotting.setup_cartopy import setup_cartopy

# Set-up cartopy before imports
setup_cartopy()
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def draw_rectangle(x_min, x_max, y_min, y_max, color, ax, fill=False, fill_color=None, alpha=0.5):
    y_grid = [y_min, y_min, y_max, y_max, y_min]
    x_grid = [x_min, x_max, x_max, x_min, x_min]
    ax.plot(x_grid, y_grid, color=color)
    if fill:
        if fill_color==None:
            fill_color = color
        ax.fill(x_grid, y_grid, color=fill_color, alpha=alpha)


def plot_maps(
        lon, lat, pr_list, x_size, y_size, font_size_title,
        plot_func = "scatter",
        font_size=80, pr_min=None, pr_max=None, aggr=np.nanmean, title="", extend=None,
        cmap='jet', legend_title="pr", xlim=None, ylim=None, cmap_type=None, sub_titles = ["GNN4CD RC", "GNN4CD R-all", "GRIPHO"],
        cbar_title_size=80, cbar_pad=0, suptitle_y=1, suptitle_x=0.45, s=150, show_ticks=True,
        norm=None, cbar_ticks=None, proj=None, cbar_ax_lim=None):

    plt.rcParams.update({'font.size': int(font_size)})

    if type(pr_list) != list:
        pr_list = [pr_list]
    n_maps = len(pr_list)

    if type(lon) != list:
        lon = [lon for n in range(n_maps)]
    if type(lat) != list:
        lat = [lat for n in range(n_maps)]
    if type(s) != list:
        s = [s for n in range(n_maps)]

    if proj is None:
        proj = ccrs.PlateCarree(central_longitude=0)

    # fig, ax = plt.subplots(nrows=1, ncols=n_maps, figsize=(x_size*n_maps,y_size), subplot_kw={"projection": ccrs.PlateCarree()})
    fig = plt.figure(figsize=(x_size * n_maps, y_size))

    left_margin = 0.02
    right_margin = 0.88
    bottom = 0.10
    height = 0.80
    gap = 0.00   # zero gap = touching maps

    total_width = right_margin - left_margin
    map_width = (total_width - (n_maps - 1) * gap) / n_maps

    ax = []
    for i in range(n_maps):
        left = left_margin + i * (map_width + gap)
        axi = fig.add_axes([left, bottom, map_width, height], projection=proj)

        # disable Cartopy’s aspect ratio enforcement
        axi.set_adjustable("box")
        axi.set_aspect("auto")

        ax.append(axi)

    cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.60])    

    # Define cmaps
    if cmap_type is None:
        pr_min = pr_min if pr_min is not None else np.nanmin([np.nanmin(pr) for pr in pr_list])
        pr_max = pr_max if pr_max is not None else np.nanmax([np.nanmax(pr) for pr in pr_list])

    v_s = []
    for pr in pr_list:
        if aggr is not None:
            v_s.append(aggr(pr, axis=1))
        else:
            v_s.append(pr)

    
    if pr_max is None:
        pr_max = np.max([np.max(v) for v in v_s])

    if pr_min is None:
        pr_min = np.min([np.min(v) for v in v_s])

    for idx in range(n_maps):
        axi = ax[idx]
        if plot_func == "scatter":
            if cmap_type is not None or norm is not None:
                im = axi.scatter(lon[idx], lat[idx], c=v_s[idx], marker="s", s=s[idx], cmap=cmap, norm=norm)
            else:
                im = axi.scatter(lon[idx], lat[idx], c=v_s[idx], marker="s", s=s[idx], cmap=cmap, vmin=pr_min, vmax=pr_max)
        elif plot_func == "pcolormesh":
            if cmap_type is not None or norm is not None:
                im = axi.pcolormesh(lon[idx], lat[idx], v_s[idx], cmap=cmap, norm=norm, shading="auto", transform=ccrs.PlateCarree())
            else:
                im = axi.pcolormesh(lon[idx], lat[idx], v_s[idx], cmap=cmap, vmin=pr_min, vmax=pr_max, shading="auto", transform=ccrs.PlateCarree())
        elif plot_func == "tripcolor":
            triang = tri.Triangulation(lon[idx], lat[idx])
            im = axi.tripcolor(triang, v_s[idx], cmap=cmap, norm=norm, shading="flat", transform=ccrs.PlateCarree())

        # plot_italy(zones, color='black', ax=ax[idx], alpha_fill=0)
        if xlim is not None:
            lon_min = xlim[0]
            lon_max = xlim[1]
        else:
            lon_min = lon[idx].min()
            lon_max = lon[idx].max()
        if ylim is not None:
            lat_min = ylim[0]
            lat_max = ylim[1]
        else:
            lat_min = lat[idx].min()
            lat_max = lat[idx].max()

        axi.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # axi.set_xlim([lon[idx].min()-0.25,lon[idx].max()+0.25])
        # axi.set_ylim([lat[idx].min()-0.25,lat[idx].max()+0.25])
        axi.set_title(sub_titles[idx], fontsize=int(np.ceil(font_size_title*0.9)))

        if not show_ticks:
            axi.xaxis.set_major_locator(ticker.NullLocator())
            axi.yaxis.set_major_locator(ticker.NullLocator())
        axi.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black")
        axi.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")

    # print(fig.get_size_inches())
    # fig_x_size = fig.get_size_inches()[0]
    # fig_y_size = fig.get_size_inches()[1]
    
    # if cbar_ax_lim is None:
    #     width = 1.5/fig_x_size
    #     left = 0.95 #(fig_x_size - width*fig_x_size) / fig_x_size
    #     cbar_ax_lim = [left, 0.15, width, 0.7]
    # cbar_ax = fig.add_axes(cbar_ax_lim)

    if cbar_ticks is not None:
        cbar = fig.colorbar(im, cax=cbar_ax, aspect=25, ticks=cbar_ticks, extend=extend)
    else:
        cbar = fig.colorbar(im, cax=cbar_ax, aspect=25, extend=extend)
    cbar.ax.set_title(legend_title, rotation=0, fontsize=cbar_title_size, pad=cbar_pad)
    _ = fig.suptitle(title, fontsize=font_size_title, x=suptitle_x, y=suptitle_y)
    
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig


def plot_pdf(hist_list, bin_list, color_list=['black', 'darkorange'], label_list=["GRIPHO", "GNN4CD"], suptitle="PDF (I)",
            fig=None, ax=None, show_ticks=True, title="", tail_ylim=[10**-8, 10**-6], tail_lim=50, lg_fontsize=24, fontsize=24,
            ylim=None, xlim=None, xlabel='precipitation [mm/h]', plot_func="scatter",
            tail_zoom=True, show_legend=True, legend_outside=False, log_xy=False):

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    
    if tail_zoom:
        axi_tail = inset_axes(
            ax,
            width="35%", height="35%",          # both dimensions *relative* to parent
            loc='lower left',                   # start in the lower‑left corner…
            bbox_to_anchor=(0.15, 0.1, 1, 1),   # …then shift downward by 33 % of ax height
            bbox_transform=ax.transAxes,        # interpret the anchor in axes coords
            borderpad=0                         # no extra padding
        )

        for i in range(len(bin_list)):
            mask_tail = bin_list[i] >= tail_lim
            axi_tail.scatter(bin_list[i][mask_tail],hist_list[i][mask_tail], color=color_list[i], s=50, label=label_list[i], zorder=2, alpha=0.4)
        if log_xy:
            axi_tail.set_yscale('log')
            axi_tail.set_xscale('log')
        axi_tail.grid(visible=True, which='both', axis='both', color='lightgrey', zorder=0)
        axi_tail.tick_params(axis='both', which='both', labelsize=18)
        axi_tail.xaxis.set_minor_formatter(NullFormatter())
        axi_tail.yaxis.set_minor_formatter(NullFormatter())
        if tail_ylim is not None:
            axi_tail.set_ylim(tail_ylim)

    for i in range(len(bin_list)):
        if plot_func == "scatter":
            ax.scatter(bin_list[i], hist_list[i], color=color_list[i], s=80, label=label_list[i], alpha=0.4, zorder=2)
        else:
            ax.step(bin_list[i], hist_list[i], color=color_list[i], where="mid", linewidth=1, label=label_list[i], zorder=2)
    if show_legend:
        if legend_outside:
            ax.legend(fontsize=lg_fontsize, loc="upper left",
                   bbox_to_anchor=(0.91, 0.88), # x > 1 moves it outside the axes
                   borderaxespad=0,
                   frameon=True,
                   bbox_transform=fig.transFigure # interpret anchor in figure coords
                  )
        else:
            ax.legend(loc='upper right', facecolor='white', framealpha=1, fontsize=lg_fontsize)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if log_xy:
        ax.set_yscale('log')
        ax.set_xscale('log')
    ax.minorticks_on()
    ax.grid(visible=True, which='both', axis='both', color='lightgrey', zorder=0)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('frequency', fontsize=fontsize)

    if not show_ticks:
        ax.yaxis.set_major_formatter(NullFormatter())
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        ax.set_ylabel('', fontsize=28)  

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)

    ax.set_title(title, fontsize=fontsize+4)
    
    plt.suptitle(suptitle, fontsize=fontsize+2)

    return fig

def plot_diurnal_cycles(pr_list, text_list = ['DJF', 'MAM', 'JJA', 'SON'], label_list=['GNN4CD', 'GRIPHO'],
                        color_list=['red', 'black'], linestyle_list=['-',':'], font_size=30, suptitle="Average",
                        unit="[mm/h]", ylim=[0,0.30], time_res=1):

    plt.rcParams.update({'font.size': font_size})
    
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(28,9))
    
    ax_list = [ax[0], ax[1], ax[2], ax[3]]
    
    for s in range(4):
    
        pr_list_s = [pr_i[s] for pr_i in pr_list]
    
        for i, pr in enumerate(pr_list_s):
            ax_list[s].plot(range(time_res,25,time_res), pr, label=label_list[i], linestyle=linestyle_list[i], linewidth=4, color=color_list[i])
        ax_list[s].set_title(text_list[s], fontsize=40)
        ax_list[s].set_xlabel("time [h]", fontsize=35)
        ax_list[s].set_ylim(ylim)
        ax_list[s].set_xlim([time_res-1,25])
        ax_list[s].set_xticks(ticks=range(6,25,6))
        ax_list[s].grid(which='major', color='lightgrey')
    
        if s>0:
            # axi.yaxis.set_major_locator(ticker.NullLocator())
            ax_list[s].yaxis.set_major_formatter(NullFormatter())
            for tick in ax_list[s].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
    
    ax_list[0].set_ylabel(unit, fontsize=35)
    ax_list[0].legend(loc='upper left', prop={'size': font_size})
    
    plt.suptitle(suptitle, y=0.95, fontsize=40)
    plt.tight_layout()
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    return fig

def get_cmap_dict():
    c_list = [
        "#40916D",
        "#6AB996",
        "#A0CF94",
        "#D3E58F",
        "#F9FCB5",
        "#F4E296",
        "#EFBD6C",
        "#E6603F",
        # "#CC3746",
        "#A90F46",
        "#E356A2", # "#D13B8C",
        # "#E068A9",
        # "#C493CB",
    ]

    cmap = LinearSegmentedColormap.from_list("cmap_pr", c_list, N=256)

    cmap_dict = {}

    bounds_avg = [0, 1, 2, 4, 6, 8, 10, 12, 15]
    norm_avg = matplotlib.colors.BoundaryNorm(boundaries=bounds_avg, ncolors=256)
    # cmap_avg = matplotlib.colors.ListedColormap(c_list[:len(bounds_avg)+1], name='precipitation', N=len(c_list[:len(bounds_avg)+1]))
    cmap_dict["avg"] = {"cmap": cmap, "bounds": bounds_avg, "norm": norm_avg}

    bounds_p99 = [0, 1, 2, 4, 6, 8, 10, 15, 20]
    norm_p99 = matplotlib.colors.BoundaryNorm(boundaries=bounds_p99, ncolors=256)
    # cmap_p99 = matplotlib.colors.ListedColormap(c_list[:len(bounds_p99)+1], name='precipitation', N=len(c_list[:len(bounds_p99)+1]))
    cmap_dict["p99"] = {"cmap": cmap, "bounds": bounds_p99, "norm": norm_p99}


    bounds_p999 = [0, 1, 2, 4, 6, 8, 10, 15, 20, 25]
    norm_p999 = matplotlib.colors.BoundaryNorm(boundaries=bounds_p999, ncolors=256)
    # cmap_p999 = matplotlib.colors.ListedColormap(c_list[:len(bounds_p999)+1], name='precipitation', N=len(c_list[:len(bounds_p999)+1]))
    cmap_dict["p999"] = {"cmap": cmap, "bounds": bounds_p999, "norm": norm_p999}

    return cmap_dict