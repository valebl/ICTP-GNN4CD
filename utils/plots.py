import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib
import matplotlib.ticker as ticker


def create_zones(zones_file):
    zones = []
    with open(zones_file) as f:
        lines = f.read()
        for zone in lines.split(';'):
            zones.append(zone)
    for i in range(len(zones)):
        zones[i] = zones[i].split('\n')
        for j in range(len(zones[i])):
            zones[i][j] = zones[i][j].split(',')
        if [''] in zones[i]:
            zones[i].remove([''])
    for i in range(len(zones)):
        for j in range(len(zones[i])):
            if '' in zones[i][j]:
                zones[i][j].remove('')
            if zones[i][j] == []:
                del zones[i][j]
                continue
            for k in range(len(zones[i][j])):
                zones[i][j][k] = float(zones[i][j][k])
    return zones


def plot_italy(zones, ax, color='k', alpha_fill=0.1, linewidth=1, xlim=None, ylim=None):
    j = 0
    for zone in zones:
        x_zone = [zone[i][0] for i in range(len(zone)) if i > 0]
        y_zone = [zone[i][1] for i in range(len(zone)) if i > 0]
        ax.fill(x_zone, y_zone, color, alpha=alpha_fill)
        ax.plot(x_zone, y_zone, color, alpha=1, linewidth=linewidth)
        j += 1
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def draw_rectangle(x_min, x_max, y_min, y_max, color, ax, fill=False, fill_color=None, alpha=0.5):
    y_grid = [y_min, y_min, y_max, y_max, y_min]
    x_grid = [x_min, x_max, x_max, x_min, x_min]
    ax.plot(x_grid, y_grid, color=color)
    if fill:
        if fill_color==None:
            fill_color = color
        ax.fill(x_grid, y_grid, color=fill_color, alpha=alpha)


def extremes_cmap():
    c_lists = [[247, 255, 255],
               [238, 255, 255],
               [230, 255, 255],
               [209, 246, 255],
               [157, 217, 255],
               [105, 187, 255],
               [52, 157, 255],
               [25, 142, 216],
               [17, 137, 147],
               [9, 135, 79],
               [1, 129, 10],
               [12, 146, 12],
               [25, 167, 25],
               [38, 187, 38],
               [58, 203, 48],
               [113, 193, 35],
               [168, 182, 21],
               [233, 171, 8],
               [255, 146, 0],
               [255, 102, 0], 
               [255, 57, 0],
               [255, 13, 0],
               [236, 0, 0],
               [203, 0, 0],
               [164, 0, 0],
               [137, 0, 0]]

    for j, c_list_top in enumerate(c_lists[1:]):
        c_list_bot = c_lists[j]
        c = np.ones((8,4))
        for i in range(3):
            c[:,i] = np.linspace(c_list_bot[i]/255, c_list_top[i]/255, c.shape[0])
        if j == 0:
            cmap = c
        else:
            cmap = np.vstack((cmap, c))
    cmap = matplotlib.colors.ListedColormap(cmap, name='myColorMap', N=cmap.shape[0])
    return cmap


def plot_maps(pos, pr_pred, pr, zones, x_size, y_size, font_size_title=None, font_size=None, pr_min=0, pr_max=2500, aggr=np.nanmean, title="", 
        cmap='jet', legend_title="pr", xlim=None, ylim=None, cbar_y=1, cmap_type=None, pad_cbar_title=80,
        cbar_title_size=None, cbar_pad=0, subtitle_y=1, subtitle_x=0.45, s=150, show_ticks=True, num=16, bounds=None):

    if font_size is not None:
        plt.rcParams.update({'font.size': int(font_size)})
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(x_size*2,y_size))

    lon = pos[:,0]; lat = pos[:,1]
    num_nodes = pos.shape[0]

    # Define cmaps
    if cmap_type is None:
        pr_min = pr_min if pr_min is not None else np.nanmin(np.nanmin(pr_pred), np.nanmin(pr))
        pr_max = pr_max if pr_max is not None else np.nanmax(np.nanmax(pr_pred), np.nanmax(pr))

    if cmap_type == "custom_blue_discrete_avg":
        c_list = ["#F8FBFE",
                  "#E1EBF6",
                  "#CADBED",
                  "#A7C9DE",
                  "#7AADD2",
                  "#5691C1",
                  "#3771B0",
                  "#205297",
                  "#123167"]

        cmap = matplotlib.colors.ListedColormap(c_list, name='cmap_blue', N=len(c_list)) # N=cmap.shape[0])

        # Bounds may be unevenly spaced:
        bounds = np.array([0.0, 0.1, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0])
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=len(c_list))
    elif cmap_type == "custom_jet_discrete_avg":
        bounds = np.array([0.0, 0.1, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0, 12.5, 15.0])
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    elif cmap_type == "custom_jet_discrete_avg_limits":
        bounds = bounds
        # bounds = np.linspace(pr_min, pr_max, num)
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    elif cmap_type == "custom_bwr_discrete_avg":
        bounds = np.array([-100,-50,-25,-10,-5,5,10,50,100,150])
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    v_s = []
    if aggr is not None:
        if pr_pred.shape[1] == num_nodes:
            v_s.append(aggr(pr_pred, axis=0))
            v_s.append(aggr(pr, axis=0))
        else:
            v_s.append(aggr(pr_pred, axis=1))
            v_s.append(aggr(pr, axis=1))
    else:
        v_s.append(pr_pred)
        v_s.append(pr)

    sub_titles = ["GNN4CD", "OBSERVATION"]

    for idx in range(2):
        if cmap_type is not None:
            im = ax[idx].scatter(lon,lat,c=v_s[idx], marker="s", s=s, cmap=cmap, norm=norm)
        else:
            im = ax[idx].scatter(lon,lat,c=v_s[idx], marker="s", s=s, cmap=cmap, vmin=pr_min, vmax=pr_max)
        # im = ax[idx].scatter(lon,lat,c=v_s[idx], marker="s", s=s, vmin=pr_min, vmax=pr_max, cmap=cmap)
        plot_italy(zones, color='black', ax=ax[idx], alpha_fill=0, xlim=xlim, ylim=ylim)
        ax[idx].set_xlim([lon.min()-0.25,lon.max()+0.25])
        ax[idx].set_ylim([lat.min()-0.25,lat.max()+0.25])
        ax[idx].set_title(sub_titles[idx])
        if xlim is not None:
            ax[idx].set_xlim(xlim)
        if ylim is not None:
            ax[idx].set_ylim(ylim)
        if not show_ticks:
            ax[idx].xaxis.set_major_locator(ticker.NullLocator())
            ax[idx].yaxis.set_major_locator(ticker.NullLocator())
    ax[1].yaxis.set_major_locator(ticker.NullLocator())
    
    # cbar_ax = fig.add_axes([0.95,0.1, 0.01, 0.8])
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7]) 
    cbar = fig.colorbar(im, cax=cbar_ax, aspect=25, pad=cbar_pad)
    if cbar_title_size is not None:
        cbar.ax.set_title(legend_title, rotation=0, fontsize=cbar_title_size, pad=pad_cbar_title)
    else:
        cbar.ax.set_title(legend_title, rotation=0, pad=pad_cbar_title)
    if font_size_title is not None:
        _ = fig.suptitle(title, fontsize=font_size_title, x=subtitle_x, y=subtitle_y)
    else:
        _ = fig.suptitle(title, x=subtitle_x, y=subtitle_y)
    
    plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95)

    return fig


def plot_single_map(pos, pr, zones, x_size, y_size, font_size_title, font_size=80, pr_min=0, pr_max=2500, aggr=np.nanmean, title="", 
        cmap='jet', legend_title="pr", xlim=None, ylim=None, cbar_y=1, cmap_type=None,
        cbar_title_size=80, cbar_pad=0, subtitle_y=0.98, subtitle_x=0.45, s=150, show_ticks=True, num=16, bounds=None):

    plt.rcParams.update({'font.size': int(font_size)})
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(x_size,y_size))

    lon = pos[:,0]; lat = pos[:,1]

    # Define cmaps
    if cmap_type is None:
        pr_min = pr_min if pr_min is not None else np.nanmin(np.nanmin(pr), np.nanmin(pr))
        pr_max = pr_max if pr_max is not None else np.nanmax(np.nanmax(pr), np.nanmax(pr))

    if cmap_type == "custom_blue_discrete_avg":
        c_list = ["#F8FBFE",
                  "#E1EBF6",
                  "#CADBED",
                  "#A7C9DE",
                  "#7AADD2",
                  "#5691C1",
                  "#3771B0",
                  "#205297",
                  "#123167"]

        cmap = matplotlib.colors.ListedColormap(c_list, name='cmap_blue', N=len(c_list)) # N=cmap.shape[0])

        # Bounds may be unevenly spaced:
        bounds = np.array([0.0, 0.1, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0])
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=len(c_list))
    elif cmap_type == "custom_jet_discrete_avg":
        bounds = np.array([0.0, 0.1, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0, 12.5, 15.0])
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    elif cmap_type == "custom_jet_discrete_avg_limits":
        bounds = bounds
        # bounds = np.linspace(pr_min, pr_max, num)
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    elif cmap_type == "custom_bwr_discrete_avg":
        bounds = np.array([-100,-50,-25,-10,-5,5,10,50,100,150])
        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    if aggr is not None:
        v_s = aggr(pr, axis=1)
    else:
        v_s = pr

    # sub_titles = ["DL-MODEL", "OBSERVATION"]

    if cmap_type is not None:
        im = ax.scatter(lon,lat,c=v_s, marker="s", s=s, cmap=cmap, norm=norm)
    else:
        im = ax.scatter(lon,lat,c=v_s, marker="s", s=s, cmap=cmap, vmin=pr_min, vmax=pr_max)
    # im = ax.scatter(lon,lat,c=v_s[idx], marker="s", s=s, vmin=pr_min, vmax=pr_max, cmap=cmap)
    plot_italy(zones, color='black', ax=ax, alpha_fill=0)
    ax.set_xlim([lon.min()-0.25,lon.max()+0.25])
    ax.set_ylim([lat.min()-0.25,lat.max()+0.25])
    ax.set_title("GNN4CD - OBSERVATION")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    #     if not show_ticks:
    #         ax[idx].xaxis.set_major_locator(ticker.NullLocator())
    #         ax[idx].yaxis.set_major_locator(ticker.NullLocator())
    # ax[1].yaxis.set_major_locator(ticker.NullLocator())
    
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7]) # (left, bottom, width, height) in fractions of figure width and height
    cbar = fig.colorbar(im, cax=cbar_ax, aspect=25, pad=cbar_pad)
    cbar.ax.set_title(legend_title, rotation=0, fontsize=cbar_title_size, pad=80)
    _ = fig.suptitle(title, fontsize=font_size_title, x=subtitle_x, y=subtitle_y)
    
    plt.subplots_adjust(wspace=0, hspace=0)


def plot_seasonal_maps(pos, pr, pr_min, pr_max, zones, aggr=np.mean, title=""):
    
    plt.rcParams.update({'font.size': 50})
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(49,49))
    ax_list = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]
    text_list = ['DJF', 'MAM', 'JJA', 'SON']

    lon = pos[:,0]
    lat = pos[:,1]

    for s in range(4):
        axi = ax_list[s]
        v = pr[s]
        v_s = aggr(v, axis=1)
        im = axi.scatter(lon,lat,c=v_s, marker="s", s=10, vmin = pr_min, vmax = pr_max, cmap='jet')
        axi.text(0.95, 0.9, text_list[s], transform=axi.transAxes,  horizontalalignment='right',fontsize=50, family='sans-serif')

        plot_italy(zones, color='black', ax=axi, alpha_fill=0)
        axi.set_xlim([6.75, 18.50])
        axi.set_ylim([36.50, 47.00])

    cbar = fig.colorbar(im, ax=ax, aspect=25, pad=0.025)
    cbar.ax.set_title("[mm/h]", rotation=0, fontsize=50, pad=80)

    _ = fig.suptitle(title, fontsize=70, x=0.45, y=.93)
    # plt.savefig(f'results/cumulative_2015_seasons_predictions.png', dpi=800, bbox_inches='tight', pad_inches=0.0)


def plot_time_series(pos, pr, pr_pred, point, season, aggr=np.mean, title=""):

    lon = pos[:,0]
    lat = pos[:,1]
    
    plt.rcParams.update({'font.size': 50})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(40,30))
    text_list = ['DJF', 'MAM', 'JJA', 'SON']

    n = len(pr[season][point,:])
    ax[0].plot(range(n), pr[season][point,:], label='observations')
    ax[0].plot(range(n), pr_pred[season][point,:], label='predictions')
    ax[0].set_title(text_list[season] + f" in position ({lon[point].item():.2f}, {lat[point].item():.2f})")
    ax[0].set_ylabel("pr [mm]")
    ax[0].set_xlabel("time")
    ax[1].plot(range(n), np.sqrt((pr[season][point,:] - pr_pred[season][point,:])**2) * np.sign(pr[season][point,:] - pr_pred[season][point,:]), label='RMSE',color='gold')
    ax[1].set_ylabel("RMSE [mm]")
    ax[1].set_xlabel("time")
    plt.legend()
    plt.show()

    # _ = fig.suptitle(title, fontsize=70, x=0.45, y=0.91)
    # plt.savefig(f'results/cumulative_2015_seasons_predictions.png', dpi=800, bbox_inches='tight', pad_inches=0.0)


def plot_mean_time_series(pos, pr, pr_pred, points, aggr=np.nanmean, title="Precipitation time series",color="paleturquoise"):
    
    plt.rcParams.update({'font.size': 40})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(60,25))

    pr_mean = aggr(pr[points,:], axis=0)
    pr_pred_mean = aggr(pr_pred[points,:], axis=0)

    ticks = list(np.arange(0,31*24*12,31*24))
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug","Sep","Oct","Nov","Dec"]

    rmse = np.sqrt((pr_pred_mean - pr_mean)**2) * np.sign(pr_pred_mean - pr_mean)
    rmse_perc = np.sqrt((pr_pred_mean - pr_mean)**2) * np.sign(pr_pred_mean - pr_mean) / np.abs(pr_mean)

    n = pr_mean.shape[0]
    ax[0].fill_between(range(n), pr_mean, label='OBSERVATIONS', color=color)
    ax[0].plot(range(n), pr_pred_mean, label='GNN4CD', linestyle=':', color='indigo')
    ax[0].set_title(title)
    ax[0].set_ylabel("pr [mm/hr]")
    ax[0].set_xlabel("time")
    ax[0].legend(loc="upper right")
    ylim = ax[0].get_ylim()
    ax[0].set_xticks(ticks=ticks, labels=labels)
    ax[1].plot(range(n), rmse, label='RMSE',color='gold')
    ax[1].set_ylabel("RMSE [mm]")
    ax[1].set_xlabel("time")
    ax[1].legend(loc="upper right")
    ax[1].set_ylim(-ylim[1], ylim[1])
    ax[1].set_xticks(ticks=ticks, labels=labels)

    return rmse, rmse_perc

def plot_pdf(pr_pred, y, fontsize=18):
    plt.rcParams.update({'font.size': fontsize})

    pr_pred = pr_pred.flatten()
    y = y.flatten()
    
    hist_pr, bin_edges_pr = np.histogram(pr_pred, bins=np.arange(0,200,0.5).astype(np.float32), density=False)
    hist_y, bin_edges_y = np.histogram(y, bins=np.arange(0,200,0.5).astype(np.float32), density=False)
    
    Ntot_pr = hist_pr.sum()
    Ntot_y = hist_y.sum()
    
    bin_edges_pr_centre = (bin_edges_pr[:-1] + bin_edges_pr[1:]) / 2
    bin_edges_y_centre = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
        
    p99_pred = np.nanpercentile(pr_pred, q=99)
    p999_pred = np.nanpercentile(pr_pred, q=99.9)

    p99 = np.nanpercentile(y, q=99)
    p999 = np.nanpercentile(y, q=99.9)
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(bin_edges_y_centre, hist_y/Ntot_y, color='darkturquoise', s=80, label="GRIPHO", alpha=0.4, zorder=2)
    ax.scatter(bin_edges_pr_centre, hist_pr/Ntot_pr, color='indigo', s=80, label="GNN4CD R", alpha=0.4, zorder=2)
    ax.plot([p99, p99], [0,50], '-', color='gold', label='GRIPHO - p99', zorder=1)
    ax.plot([p999, p999], [0,50], '-r', label='GRIPHO - p99.9', zorder=1)
    ax.plot([p99_pred, p99_pred], [0,50], ':', color='gold', label='GNN4CD R - p99', zorder=1)
    ax.plot([p999_pred, p999_pred], [0,50], ':r', label='GNN4CD R - p99.9', zorder=1)
    l = plt.legend(loc='upper right', facecolor='white', framealpha=1, fontsize=16)
    ax.set_ylim([10**(-8),5])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.minorticks_on()
    ax.grid(visible=True, which='both', axis='both', color='lightgrey', zorder=0)
    ax.set_xlabel('precipitation [mm/h]', fontsize=fontsize)
    ax.set_ylabel('frequency', fontsize=fontsize)

    return fig

def plot_diurnal_cycles(y_pred, y, aggr=np.nanmean, fontsize=25, figsize=(16,18),
                        xlim=[0,24], ylim=[0.5, 3.0]):

    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    # create seasons
    jf_start = 0 # january-february
    if y_pred.shape[1] == 8760:
        jf_end = (31 + 28) * 24
    elif y_pred.shape[1] == 8784:
        jf_end = (31 + 29) * 24
    else:
        print("\nInvalid number of hours in a year: {y_pred.shape[1]}")
        return fig

    mam_start = jf_end
    mam_end = mam_start + (31 + 30 + 31) * 24
    jja_start = mam_end
    jja_end = jja_start + (31 + 31 + 30) * 24
    son_start = jja_end
    son_end = son_start + (30 + 31 + 30) * 24
    d_start = son_end
    d_end = son_end + 31 * 24 # december

    idxs_seasons = [np.concatenate((np.arange(d_start, d_end), np.arange(jf_start, jf_end)), axis=0),
                    np.arange(mam_start, mam_end),
                    np.arange(jja_start, jja_end),
                    np.arange(son_start, son_end)]
    
    text_list = ['DJF', 'MAM', 'JJA', 'SON']

    for s in range(4):
        idxs = idxs_seasons[s]
        y_pred_s = y_pred[:,idxs]
        y_s = y[:,idxs]
        y_pred_daily_cycles = np.zeros(24)
        y_daily_cycles = np.zeros(24)
        axi = ax[s//2, s%2]
        for i in range(0,24):
            y_pred_daily_cycles[i] = aggr(y_pred_s[:,i::24][y_pred_s[:,i::24]>0])
            y_daily_cycles[i] = aggr(y_s[:,i::24][y_s[:,i::24]>0])
        n = 25
        axi.plot(range(1,n), y_pred_daily_cycles, label='GNN4CD', linestyle='-', linewidth=2, color='red')
        axi.plot(range(1,n), y_daily_cycles, label='GRIPHO', linestyle=':', linewidth=2, color='black')
        axi.set_ylabel("pr [mm/h]", fontsize=40)
        axi.set_xlabel("time [h]", fontsize=40)
        axi.set_xlim(xlim)
        axi.set_ylim(ylim)
        axi.set_xticks(ticks=range(0,n,6))
        axi.set_title(text_list[s], fontsize=45)
        axi.grid(which='major', color='lightgrey')
    plt.legend(loc='upper left', prop={'size': 30})
    plt.tight_layout()

    return fig