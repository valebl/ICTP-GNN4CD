import matplotlib.pyplot as plt
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
    xy_zones = []
    for zone in zones:
        xy_zones.append([[zone[i][0] for i in range(len(zone))], [zone[i][1] for i in range(len(zone))]])
    return xy_zones


def plot_italy(zones, ax, color='k', color_fill=None, alpha_fill=0.1, linewidth=1, xlim=None, ylim=None):
    for zone in zones:
        x_zone, y_zone = zone[0], zone[1]
        if color_fill is not None:
            ax.fill(x_zone, y_zone, color, alpha=alpha_fill)
        ax.plot(x_zone, y_zone, color, alpha=1, linewidth=linewidth)
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

def get_discrete_cmap(end=None, c_list=None):
    
    if c_list is None:
        c_list = ["#6AB996",
                "#A0CF94",
                "#D3E58F",
                "#F9FCB5",
                "#F4E296",
                "#EFBD6C",
                "#E6603F",
                "#CC3746",
                "#A90F46",
                "#BC2577",
                "#E068A9",
                "#C493CB"]

    if end is None or type=="avg" or type=="sum":
        return matplotlib.colors.ListedColormap(c_list, name='cmap_custom', N=len(c_list)) # N=cmap.shape[0])
    elif end == -1 or abs(end) < len(c_list):
        return matplotlib.colors.ListedColormap(c_list[:end], name='cmap_custom', N=len(c_list[:end])) # N=cmap.shape[0])
    else:
        raise Exception(f"end is {end} but should be < {len(c_list)}")
        

def plot_maps(lon, lat, pr_list, x_size=38, y_size=44, font_size_title=200, zones=None,
              zones_file='/leonardo_work/ICT25_ESP/vblasone/ICTP-GNN4CD/utils/Italia.txt',
              font_size=160, vmin=0, vmax=2500, aggr=np.nanmean, title="", cmap='jet',
              legend_title="pr", xlim=[6.75, 18.50], ylim=[36.50, 47.00], cmap_type=None,
              sub_titles = ["GNN4CD RC", "GNN4CD R-all", "GRIPHO"], cbar_title_size=180, cbar_pad=150,
              subtitle_y=1, subtitle_x=0.45, s=150, show_ticks=False, norm=None, save_path_file=None):

    if zones is None:
        zones = create_zones(zones_file=zones_file)

    plt.rcParams.update({'font.size': int(font_size)})

    n_maps = len(pr_list)
    
    fig, ax = plt.subplots(nrows=1, ncols=n_maps, figsize=(x_size*n_maps,y_size))

    # Define cmaps
    if cmap_type is None:
        vmin = vmin if vmin is not None else np.nanmin([np.nanmin(pr) for pr in pr_list])
        vmax = vmax if vmax is not None else np.nanmax([np.nanmax(pr) for pr in pr_list])

    v_s = []
    for pr in pr_list:
        if aggr is not None:
            v_s.append(aggr(pr, axis=1))
        else:
            v_s.append(pr)

    for idx in range(n_maps):
        if cmap_type is not None or norm is not None:
            im = ax[idx].scatter(lon,lat,c=v_s[idx], marker="s", s=s, cmap=cmap, norm=norm)
        else:
            im = ax[idx].scatter(lon,lat,c=v_s[idx], marker="s", s=s, cmap=cmap, vmin=vmin, vmax=vmax)
        plot_italy(zones, color='black', ax=ax[idx], alpha_fill=0)
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

    # print(fig.get_size_inches())
    fig_x_size = fig.get_size_inches()[0]
    
    width = 1.5/fig_x_size
    left = 0.95 #(fig_x_size - width*fig_x_size) / fig_x_size
    cbar_ax_lim = [left, 0.15, width, 0.7]
    cbar_ax = fig.add_axes(cbar_ax_lim)
    
    cbar = fig.colorbar(im, cax=cbar_ax, aspect=25)
    cbar.ax.set_title(legend_title, rotation=0, fontsize=cbar_title_size, pad=cbar_pad)
    _ = fig.suptitle(title, fontsize=font_size_title, x=subtitle_x, y=subtitle_y)
    
    plt.subplots_adjust(wspace=0, hspace=0)

    if save_path_file is not None:
        plt.savefig(save_path_file, bbox_inches='tight', pad_inches=0.0)


def plot_single_map(pos, pr, zones, save_path, save_file_name, 
        x_size, y_size, font_size_title, font_size=80, pr_min=0, pr_max=2500, aggr=np.nanmean, title="", 
        cmap='jet', legend_title="pr", xlim=None, ylim=None, cbar_y=1, cmap_type=None, pad_cbar_title=80,
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
    cbar.ax.set_title(legend_title, rotation=0, fontsize=cbar_title_size, pad=pad_cbar_title)
    _ = fig.suptitle(title, fontsize=font_size_title, x=subtitle_x, y=subtitle_y)
    
    plt.subplots_adjust(wspace=0, hspace=0)


def plot_diurnal_cycles(pr_list, text_list = ['DJF', 'MAM', 'JJA', 'SON'], label_list=['GNN4CD', 'GRIPHO'],
                        color_list=['red', 'black'], linestyle_list=['-',':'], font_size=30, suptitle="Average",
                        unit="[mm/h]", ylim=[0,0.30], save_path_file=None):

    plt.rcParams.update({'font.size': font_size})
    
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(28,9))
    
    ax_list = [ax[0], ax[1], ax[2], ax[3]]
    
    for s in range(4):
    
        pr_list_s = [pr_i[s] for pr_i in pr_list]
    
        n = 25
        for i, pr in enumerate(pr_list_s):
            ax_list[s].plot(range(1,n), pr, label=label_list[i], linestyle=linestyle_list[i], linewidth=4, color=color_list[i])
        ax_list[s].set_title(text_list[s], fontsize=40)
        ax_list[s].set_xlabel("time [h]", fontsize=35)
        ax_list[s].set_ylim(ylim)
        # ax_list[s].set_xlim([0,24])
        ax_list[s].set_xticks(ticks=range(6,n,6))
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
    
    if save_path_file is not None:
        plt.savefig(save_path_file, bbox_inches='tight', pad_inches=0.0)

def plot_pdf(bin_list, hist_list, color_list=['turquoise', 'darkorange'], label_list=["GRIPHO", "GNN4CD"], suptitle="PDF (I)",
            fig=None, ax=None, show_ticks=True, title="", save_path_file=None):

    plt.rcParams.update({'font.size': 26})

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    
    axi_tail = inset_axes(
        ax,
        width="35%", height="35%",       # both dimensions *relative* to parent
        loc='lower left',                # start in the lower‑left corner…
        bbox_to_anchor=(0.15, 0.1, 1, 1), # …then shift downward by 33 % of ax height
        bbox_transform=ax.transAxes,     # interpret the anchor in axes coords
        borderpad=0                      # no extra padding
    )

    for i in range(len(bin_list)):
        mask_tail = bin_list[i] >= 50 #results_RC["bin_edges_y_centre"] >= 50
        axi_tail.scatter(bin_list[i][mask_tail],hist_list[i][mask_tail], color=color_list[i], s=50, label=label_list[i], zorder=2, alpha=0.4)
    axi_tail.set_yscale('log')
    axi_tail.set_xscale('log')
    axi_tail.grid(visible=True, which='both', axis='both', color='lightgrey', zorder=0)
    axi_tail.tick_params(axis='both', which='both', labelsize=18)
    axi_tail.xaxis.set_minor_formatter(NullFormatter())
    axi_tail.yaxis.set_minor_formatter(NullFormatter())

    for i in range(len(bin_list)):
        ax.scatter(bin_list[i], hist_list[i], color=color_list[i], s=80, label=label_list[i], alpha=0.4, zorder=2)
    l = ax.legend(loc='upper right', facecolor='white', framealpha=1, fontsize=24)
    ax.set_ylim([10**(-10),5])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.minorticks_on()
    ax.grid(visible=True, which='both', axis='both', color='lightgrey', zorder=0)
    ax.set_xlabel('precipitation [mm/h]', fontsize=28)
    ax.set_ylabel('frequency', fontsize=28)

    if not show_ticks:
        ax.yaxis.set_major_formatter(NullFormatter())
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        ax.set_ylabel('', fontsize=28)  

    ax.set_title(title)
    
    plt.suptitle(suptitle)

    if save_path_file is not None:
        plt.savefig(save_path_file, bbox_inches='tight', pad_inches=0.0)
