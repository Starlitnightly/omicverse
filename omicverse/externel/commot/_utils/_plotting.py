import pandas as pd
import numpy as np
import plotly
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.stats import norm
from networkx.drawing.nx_agraph import to_agraph

def get_cmap_qualitative(cmap_name):
    if cmap_name == "Plotly":
        cmap = plotly.colors.qualitative.Plotly
    elif cmap_name == "Alphabet":
        cmap = plotly.colors.qualitative.Alphabet
    elif cmap_name == "Light24":
        cmap = plotly.colors.qualitative.Light24
    elif cmap_name == "Dark24":
        cmap = plotly.colors.qualitative.Dark24
    # Safe and Vivid are strings of form "rbg(...)"
    # Handle this later.
    elif cmap_name == "Safe":
        cmap = plotly.colors.qualitative.Safe
    elif cmap_name == "Vivid":
        cmap = plotly.colors.qualitative.Vivid
    return cmap

def linear_clamp_value(x, lower_bound, upper_bound, out_min, out_max):
    if x <= lower_bound:
        y = out_min
    elif x >= upper_bound:
        y = out_max
    else:
        y = out_min + (x - lower_bound)/(upper_bound-lower_bound) * (out_max-out_min)
    return y


def plot_cluster_signaling_chord(S, p_values, label_name=None, colormap='Plotly', quantile_cutoff=None, p_value_cutoff=None, cutoff=0, separate=False, filename="chord_plot.pdf", diagonal_off=False):
    cmap = get_cmap_qualitative(colormap)
    data_all = np.empty([0,3], str)
    if not quantile_cutoff is None:
        cutoff = np.quantile(S.reshape(-1), quantile_cutoff)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if diagonal_off and i==j: continue
            if S[i,j] >= cutoff or p_values[i,j] <= p_value_cutoff:
                if not label_name is None:
                    if separate:
                        data = np.array([['S'+label_name[i], 'T'+label_name[j], str(S[i,j])]], str)
                    else:
                        data = np.array([[label_name[i], label_name[j], str(S[i,j])]], str)
                else:
                    if separate:
                        data = np.array([['S'+str(i), 'T'+str(j), str(S[i,j])]], str)
                    else:
                        data = np.array([[str(i), str(j), str(S[i,j])]], str)
                data_all = np.concatenate((data_all, data), axis=0)
    df = pd.DataFrame(data=data_all, columns=['from','to','value'])
    df.to_csv("./plot_chord_data.csv", index=False)

    data_cmap = np.empty([0, 2], str)
    for i in range(S.shape[0]):
        if not label_name is None:
            if separate:
                data_tmp = np.array([['S'+label_name[i], cmap[i % len(cmap)]],['T'+label_name[i], cmap[i % len(cmap)]]], str)
            else:
                data_tmp = np.array([[label_name[i], cmap[i % len(cmap)]]], str)
        else:
            if separate:
                data_tmp = np.array([['S'+str(i), cmap[i % len(cmap)]],['T'+str(i), cmap[i % len(cmap)]]], str)
            else:
                data_tmp = np.array([[str(i), cmap[i % len(cmap)]]], str)
        data_cmap = np.concatenate((data_cmap, data_tmp), axis=0)
    df = pd.DataFrame(data=data_cmap)
    df.to_csv("./plot_chord_cmap.csv", index=False, header=False)

    os.system("Rscript plot_chord.R")
    os.system("mv Rplots.pdf "+filename)
    os.system("pdfcrop %s %s" % (filename, filename))
    filename_png = filename[:-4]
    filename_png = filename_png + ".png"
    os.system("convert -density 800 %s -quality 100 %s" % (filename, filename_png))

def plot_cluster_signaling_network(S,
    labels = None,
    node_size = 0.2,
    node_colormap = "Plotly",
    node_cluster_colormap = None,
    node_pos = None,
    edge_width_lb_quantile = 0.05,
    edge_width_ub_quantile = 0.95,
    edge_width_min = 1,
    edge_width_max = 4,
    edge_color = None, # expect to range from 0 to 1
    edge_colormap = None,
    background_pos = None,
    background_ndcolor = "lavender",
    background_ndsize = 1,
    filename = "network_plot.pdf",
):
    if labels is None:
        labels = [str(i) for i in range(S.shape[0])]
    node_cmap = get_cmap_qualitative(node_colormap)
    G = nx.MultiDiGraph()

    edge_width_lb = np.quantile(S.reshape(-1), edge_width_lb_quantile)
    edge_width_ub = np.quantile(S.reshape(-1), edge_width_ub_quantile)


    # Draw the background geometry
    if not background_pos is None:
        for i in range(background_pos.shape[0]):
            G.add_node("cell_"+str(i), shape='point', color=background_ndcolor, fillcolor=background_ndcolor, width=background_ndsize)
            G.nodes["cell_"+str(i)]["pos"] = "%f,%f!" %(background_pos[i,0],background_pos[i,1])

    # Draw the nodes (cluster)
    for i in range(len(labels)):
        if node_cluster_colormap is None:
            G.add_node(labels[i], shape="point", fillcolor=node_cmap[i], color=node_cmap[i])
        elif not node_cluster_colormap is None:
            G.add_node(labels[i], shape="point", fillcolor=node_cluster_colormap[labels[i]], color=node_cmap[i])
        if not node_pos is None:
            G.nodes[labels[i]]["pos"] = "%f,%f!" % (node_pos[i,0],node_pos[i,1])
        G.nodes[labels[i]]["width"] = str(node_size)

    # Draw the edges
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i,j] > 0:
                G.add_edge(labels[i], labels[j], splines="curved")
                G[labels[i]][labels[j]][0]["penwidth"] = str(linear_clamp_value(S[i,j],edge_width_lb,edge_width_ub,edge_width_min,edge_width_max))
                if edge_color == "node":
                    G[labels[i]][labels[j]][0]['color'] = node_cmap[i]
                elif isinstance(edge_color, np.ndarray):
                    G[labels[i]][labels[j]][0]['color'] = mpl.colors.to_hex( edge_colormap(edge_color[i,j]) )
                else:
                    G[labels[i]][labels[j]][0]['color'] = edge_color
    
    # Draw the network
    A = to_agraph(G)
    if node_pos is None:
        A.layout("dot")
    else:
        A.layout()
    A.draw(filename)

# Recommend plotting at most three LR species
def plot_cluster_signaling_network_multipair(S_list,
    p_values_list,
    labels = None,
    quantile_cutoff = None,
    p_value_cutoff = 0.05,
    node_size = 0.2,
    node_colormap = "Plotly",
    node_pos = None,
    edge_width_lb_quantile = 0.05,
    edge_width_ub_quantile = 0.95,
    edge_width_min = 1,
    edge_width_max = 4,
    edge_colors = ["blue","red","green","grey","yellow"],
    background_pos = None,
    background_ndcolor = "darkseagreen1",
    background_ndsize = 1,
    filename = "network_plot.pdf"
):
    tmp_all_S = []
    for S in S_list:
        tmp_all_S.extend( list( S.reshape(-1) ) )
    tmp_all_S = np.array(tmp_all_S)
    if labels is None:
        labels = [str(i) for i in range(S.shape[0])]
    node_cmap = get_cmap_qualitative(node_colormap)
    G = nx.MultiDiGraph()
    if not quantile_cutoff is None:
        cutoff = np.quantile(tmp_all_S.reshape(-1), quantile_cutoff)
    else:
        cutoff = 0.0
    edge_width_lb = np.quantile(tmp_all_S.reshape(-1), edge_width_lb_quantile)
    edge_width_ub = np.quantile(tmp_all_S.reshape(-1), edge_width_ub_quantile)

    # Draw the background geometry
    if not background_pos is None:
        for i in range(background_pos.shape[0]):
            G.add_node("cell_"+str(i), shape='point', color=background_ndcolor, fillcolor=background_ndcolor, width=background_ndsize)
            G.nodes["cell_"+str(i)]["pos"] = "%f,%f!" %(background_pos[i,0],background_pos[i,1])

    # Draw the nodes (cluster)
    for i in range(len(labels)):
        G.add_node(labels[i], shape="point", fillcolor=node_cmap[i], color=node_cmap[i])
        if not node_pos is None:
            G.nodes[labels[i]]["pos"] = "%f,%f!" % (node_pos[i,0],node_pos[i,1])
        G.nodes[labels[i]]["width"] = str(node_size)

    # Draw the edges
    for k in range(len(S_list)):
        S = S_list[k]
        p_values = p_values_list[k]
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                if S[i,j] >= cutoff or p_values[i,j] <= p_value_cutoff:
                    penwidth = str(linear_clamp_value(S[i,j],edge_width_lb,edge_width_ub,edge_width_min,edge_width_max))
                    color = edge_colors[k]
                    G.add_edge(labels[i], labels[j], splines="curved", penwidth=penwidth, color=color)
    
    # Draw the network
    A = to_agraph(G)
    if node_pos is None:
        A.layout("dot")
    else:
        A.layout()
    A.draw(filename)

def plot_cell_signaling(X,
    V,
    signal_sum,
    cmap="coolwarm",
    cluster_cmap = None,
    arrow_color="tab:blue",
    plot_method="cell",
    background='summary',
    clustering=None,
    background_legend=False,
    adata=None,
    summary='sender',
    ndsize = 1,
    scale = 1.0,
    grid_density = 1,
    grid_knn = None,
    grid_scale = 1.0,
    grid_thresh = 1.0,
    grid_width = 0.005,
    stream_density = 1.0,
    stream_linewidth = 1,
    stream_cutoff_perc = 5,
    filename=None,
    ax = None,
    fig = None
):
    ndcolor = signal_sum
    ncell = X.shape[0]
    
    V_cell = V.copy()
    V_cell_sum = np.sum(V_cell, axis=1)
    V_cell[np.where(V_cell_sum==0)[0],:] = np.nan
    if summary == "sender":
        X_vec = X
    elif summary == "receiver":
        X_vec = X - V / scale

    if plot_method == "grid" or plot_method == "stream":
        # Get a rectangular grid
        xl, xr = np.min(X[:,0]), np.max(X[:,0])
        epsilon = 0.02*(xr-xl); xl -= epsilon; xr += epsilon
        yl, yr = np.min(X[:,1]), np.max(X[:,1])
        epsilon = 0.02*(yr-yl); yl -= epsilon; yr += epsilon
        ngrid_x = int(50 * grid_density)
        gridsize = (xr-xl) / float(ngrid_x)
        ngrid_y = int((yr-yl)/gridsize)
        meshgrid = np.meshgrid(np.linspace(xl,xr,ngrid_x), np.linspace(yl,yr,ngrid_y))
        grid_pts = np.concatenate((meshgrid[0].reshape(-1,1), meshgrid[1].reshape(-1,1)), axis=1)
    
        if grid_knn is None:
            grid_knn = int( X.shape[0] / 50 )
        nn_mdl = NearestNeighbors()
        nn_mdl.fit(X)
        dis, nbs = nn_mdl.kneighbors(grid_pts, n_neighbors=grid_knn)
        w = norm.pdf(x=dis, scale=gridsize * grid_scale)
        w_sum = w.sum(axis=1)

        V_grid = (V[nbs] * w[:,:,None]).sum(axis=1)
        V_grid /= np.maximum(1, w_sum)[:,None]

        if plot_method == "grid":
            grid_thresh *= np.percentile(w_sum, 99) / 100
            grid_pts, V_grid = grid_pts[w_sum > grid_thresh], V_grid[w_sum > grid_thresh]
        elif plot_method == "stream":
            x_grid = np.linspace(xl, xr, ngrid_x)
            y_grid = np.linspace(yl, yr, ngrid_y)
            V_grid = V_grid.T.reshape(2, ngrid_y, ngrid_x)
            vlen = np.sqrt((V_grid ** 2).sum(0))
            grid_thresh = 10 ** (grid_thresh - 6)
            grid_thresh = np.clip(grid_thresh, None, np.max(vlen) * 0.9)
            cutoff = vlen.reshape(V_grid[0].shape) < grid_thresh
            length = np.sum(np.mean(np.abs(V[nbs]),axis=1),axis=1).T
            length = length.reshape(ngrid_y, ngrid_x)
            cutoff |= length < np.percentile(length, stream_cutoff_perc)

            V_grid[0][cutoff] = np.nan

    if cmap == 'Plotly':
        cmap = plotly.colors.qualitative.Plotly
    elif cmap == 'Light24':
        cmap = plotly.colors.qualitative.Light24
    elif cmap == 'Dark24':
        cmap = plotly.colors.qualitative.Dark24
    elif cmap == 'Alphabet':
        cmap = plotly.colors.qualitative.Alphabet

    idx = np.argsort(ndcolor)
    if background == 'summary' or background == 'cluster':
        if not ndsize==0:
            if background == 'summary':
                ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=ndcolor[idx], cmap=cmap, linewidth=0)
            elif background == 'cluster':
                adata.obs[clustering]=adata.obs[clustering].astype('category')
                labels = np.array( adata.obs[clustering], str )
                unique_labels = adata.obs[clustering].cat.categories
                for i_label in range(len(unique_labels)):
                    idx = np.where(labels == unique_labels[i_label])[0]
                    if cluster_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=cmap[i_label], linewidth=0, label=unique_labels[i_label])
                    elif not cluster_cmap is None:
                        ax.scatter(X[idx,0], X[idx,1], s=ndsize, c=cluster_cmap[unique_labels[i_label]], linewidth=0, label=unique_labels[i_label])
                if background_legend:
                    ax.legend(markerscale=2.0, loc=[1.0,0.0])
        if plot_method == "cell":
            ax.quiver(X_vec[:,0], X_vec[:,1], V_cell[:,0], V_cell[:,1], scale=scale, scale_units='x', color=arrow_color)
        elif plot_method == "grid":
            ax.quiver(grid_pts[:,0], grid_pts[:,1], V_grid[:,0], V_grid[:,1], scale=scale, scale_units='x', width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid ** 2).sum(0))
            stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
            ax.streamplot(x_grid, y_grid, V_grid[0], V_grid[1], color=arrow_color, density=stream_density, linewidth=stream_linewidth)
    elif background == 'image':
        spatial_mapping = adata.uns.get("spatial", {})
        library_id = list(spatial_mapping.keys())[0]
        spatial_data = spatial_mapping[library_id]
        img = spatial_data['images']['hires']
        sf = spatial_data['scalefactors']['tissue_hires_scalef']
        ax.imshow(img, origin='lower')
        if plot_method == "cell":
            ax.quiver(X_vec[:,0]*sf, X_vec[:,1]*sf, V_cell[:,0]*sf, V_cell[:,1]*sf, scale=scale, scale_units='x', color=arrow_color)
        elif plot_method == "grid":
            ax.quiver(grid_pts[:,0]*sf, grid_pts[:,1]*sf, V_grid[:,0]*sf, V_grid[:,1]*sf, scale=scale, scale_units='x', width=grid_width, color=arrow_color)
        elif plot_method == "stream":
            lengths = np.sqrt((V_grid ** 2).sum(0))
            stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
            ax.streamplot(x_grid*sf, y_grid*sf, V_grid[0]*sf, V_grid[1]*sf, color=arrow_color, density=stream_density, linewidth=stream_linewidth)

    ax.axis("equal")
    ax.axis("off")
    if not filename is None:
        plt.savefig(filename, dpi=500, bbox_inches = 'tight', transparent=True)


def plot_cell_signaling_compare(X,
    V,
    ax,
    arrow_color="tab:blue",
    plot_method="cell",
    summary='sender',
    scale = 1.0,
    grid_density = 1,
    grid_knn = None,
    grid_scale = 1.0,
    grid_thresh = 1.0,
):
    V_cell = V.copy()
    V_cell_sum = np.sum(V_cell, axis=1)
    V_cell[np.where(V_cell_sum==0)[0],:] = np.nan
    if summary == "sender":
        X_vec = X
    elif summary == "receiver":
        X_vec = X - V / scale

    if plot_method == "grid":
        # Get a rectangular grid
        xl, xr = np.min(X[:,0]), np.max(X[:,0])
        epsilon = 0.02*(xr-xl); xl -= epsilon; xr += epsilon
        yl, yr = np.min(X[:,1]), np.max(X[:,1])
        epsilon = 0.02*(yr-yl); yl -= epsilon; yr += epsilon
        ngrid_x = int(50 * grid_density)
        gridsize = (xr-xl) / float(ngrid_x)
        ngrid_y = int((yr-yl)/gridsize)
        meshgrid = np.meshgrid(np.linspace(xl,xr,ngrid_x), np.linspace(yl,yr,ngrid_y))
        grid_pts = np.concatenate((meshgrid[0].reshape(-1,1), meshgrid[1].reshape(-1,1)), axis=1)
    
        if grid_knn is None:
            grid_knn = int( X.shape[0] / 50 )
        nn_mdl = NearestNeighbors()
        nn_mdl.fit(X)
        dis, nbs = nn_mdl.kneighbors(grid_pts, n_neighbors=grid_knn)
        w = norm.pdf(x=dis, scale=gridsize * grid_scale)
        w_sum = w.sum(axis=1)

        V_grid = (V[nbs] * w[:,:,None]).sum(axis=1)
        V_grid /= np.maximum(1, w_sum)[:,None]

        if plot_method == "grid":
            grid_thresh *= np.percentile(w_sum, 99) / 100
            grid_pts, V_grid = grid_pts[w_sum > grid_thresh], V_grid[w_sum > grid_thresh]

    if plot_method == "cell":
        ax.quiver(X_vec[:,0], X_vec[:,1], V_cell[:,0], V_cell[:,1], scale=scale, scale_units='x', color=arrow_color, alpha=1.0)
    elif plot_method == "grid":
        ax.quiver(grid_pts[:,0], grid_pts[:,1], V_grid[:,0], V_grid[:,1], scale=scale, scale_units='x', color=arrow_color, alpha=1.0)
    return ax