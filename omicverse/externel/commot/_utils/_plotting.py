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

def get_cmap_qualitative(cmap_name, n_colors=None):
    """
    Get qualitative colormap with support for large numbers of colors.
    
    Parameters
    ----------
    cmap_name : str
        Name of the colormap. Options include:
        - Standard plotly: "Plotly", "Alphabet", "Light24", "Dark24", "Safe", "Vivid"
        - Extended omicverse: "omicverse_28", "omicverse_56", "omicverse_112"  
        - Traditional Chinese: "forbidden_city"
        - Themed: "vibrant", "earth", "pastel", "ditto"
    n_colors : int, optional
        Number of colors needed. If provided, will automatically select 
        appropriate palette size.
    
    Returns
    -------
    cmap : list
        List of color strings
    """
    # Handle omicverse extended palettes directly
    if cmap_name in ["omicverse_28", "omicverse_56", "omicverse_112", "forbidden_city", 
                     "vibrant", "earth", "pastel", "ditto"]:
        try:
            from ...pl._palette import (palette_28, palette_56, palette_112, 
                                       vibrant_palette, earth_palette, pastel_palette, 
                                       ditto_color, ForbiddenCity)
            
            if cmap_name == "omicverse_28":
                cmap = palette_28
            elif cmap_name == "omicverse_56":
                cmap = palette_56  
            elif cmap_name == "omicverse_112":
                cmap = palette_112
            elif cmap_name == "vibrant":
                cmap = vibrant_palette
            elif cmap_name == "earth":
                cmap = earth_palette
            elif cmap_name == "pastel":
                cmap = pastel_palette
            elif cmap_name == "ditto":
                cmap = ditto_color
            elif cmap_name == "forbidden_city":
                fc = ForbiddenCity()
                # Use a diverse selection of traditional Chinese colors
                if n_colors is not None:
                    # Select colors from different color families for diversity
                    color_families = [
                        fc.green[:15], fc.red[:15], fc.blue[:15], fc.yellow[:15], 
                        fc.purple[:15], fc.brown[:15], fc.pink[:15], fc.grey[:10]
                    ]
                    cmap = []
                    family_idx = 0
                    color_idx = 0
                    while len(cmap) < n_colors and family_idx < len(color_families):
                        if color_idx < len(color_families[family_idx]):
                            cmap.append(color_families[family_idx][color_idx])
                            family_idx = (family_idx + 1) % len(color_families)
                            if family_idx == 0:
                                color_idx += 1
                        else:
                            family_idx += 1
                else:
                    cmap = fc.green[:10] + fc.red[:10] + fc.blue[:10] + fc.yellow[:10]
        except ImportError:
            # Fallback to plotly if import fails
            cmap = plotly.colors.qualitative.Plotly
    else:
        # Standard plotly palettes
        if cmap_name == "Plotly":
            cmap = plotly.colors.qualitative.Plotly
        elif cmap_name == "Alphabet":
            cmap = plotly.colors.qualitative.Alphabet
        elif cmap_name == "Light24":
            cmap = plotly.colors.qualitative.Light24
        elif cmap_name == "Dark24":
            cmap = plotly.colors.qualitative.Dark24
        elif cmap_name == "Safe":
            cmap = plotly.colors.qualitative.Safe
        elif cmap_name == "Vivid":
            cmap = plotly.colors.qualitative.Vivid
        else:
            cmap = plotly.colors.qualitative.Plotly
    
    # If n_colors is specified and exceeds current palette size, 
    # use extended palettes from omicverse
    if n_colors is not None and n_colors > len(cmap) and cmap_name not in ["forbidden_city"]:
        try:
            from ...pl._palette import palette_28, palette_56, palette_112
            
            if n_colors <= 28:
                cmap = palette_28[:n_colors]
            elif n_colors <= 56:
                cmap = palette_56[:n_colors]
            elif n_colors <= 112:
                cmap = palette_112[:n_colors]
            else:
                # For very large numbers, use palette_112 and cycle through
                base_palette = palette_112
                cycles_needed = (n_colors + len(base_palette) - 1) // len(base_palette)
                extended_palette = []
                for cycle in range(cycles_needed):
                    for color in base_palette:
                        if len(extended_palette) >= n_colors:
                            break
                        extended_palette.append(color)
                    if len(extended_palette) >= n_colors:
                        break
                cmap = extended_palette[:n_colors]
        except ImportError:
            # Fallback to original behavior if import fails
            pass
    
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
    cmap = get_cmap_qualitative(colormap, n_colors=S.shape[0])
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
    """
    Plot cluster signaling network using matplotlib and NetworkX.
    
    Parameters
    ----------
    S : array-like
        Signaling matrix between clusters
    labels : list, optional
        Cluster labels
    node_size : float
        Size of nodes
    node_colormap : str
        Colormap for nodes
    node_cluster_colormap : dict, optional
        Custom color mapping for clusters
    node_pos : array-like, optional
        Positions for nodes
    edge_width_lb_quantile : float
        Lower bound quantile for edge width
    edge_width_ub_quantile : float
        Upper bound quantile for edge width
    edge_width_min : float
        Minimum edge width
    edge_width_max : float
        Maximum edge width
    edge_color : str or array-like
        Edge color specification
    edge_colormap : matplotlib colormap
        Colormap for edges when using numerical edge colors
    background_pos : array-like, optional
        Background positions to plot
    background_ndcolor : str
        Background color
    background_ndsize : float
        Background node size
    filename : str
        Output filename
    """
    
    if labels is None:
        labels = [str(i) for i in range(S.shape[0])]
    node_cmap = get_cmap_qualitative(node_colormap, n_colors=len(labels))
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, label in enumerate(labels):
        G.add_node(label)
    
    # Add edges with weights
    edge_weights = []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i,j] > 0:
                G.add_edge(labels[i], labels[j], weight=S[i,j])
                edge_weights.append(S[i,j])
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate layout
    if node_pos is not None:
        pos = {labels[i]: node_pos[i,:2] for i in range(len(labels))}
    else:
        # Use spring layout as default
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw background positions if provided
    if background_pos is not None:
        ax.scatter(background_pos[:,0], background_pos[:,1], 
                   c=background_ndcolor, s=background_ndsize*50, alpha=0.3, zorder=1)
    
    # Calculate edge widths
    if edge_weights:
        edge_width_lb = np.quantile(edge_weights, edge_width_lb_quantile)
        edge_width_ub = np.quantile(edge_weights, edge_width_ub_quantile)
        normalized_weights = []
        for edge in G.edges(data=True):
            weight = edge[2]['weight']
            norm_weight = linear_clamp_value(weight, edge_width_lb, edge_width_ub, edge_width_min, edge_width_max)
            normalized_weights.append(norm_weight)
    else:
        normalized_weights = [1] * len(G.edges())
    
    # Calculate edge colors
    if isinstance(edge_color, str) and edge_color == "node":
        edge_colors = []
        for edge in G.edges():
            source_idx = labels.index(edge[0])
            edge_colors.append(node_cmap[source_idx % len(node_cmap)])
    elif isinstance(edge_color, np.ndarray):
        edge_colors = []
        for edge in G.edges():
            i, j = labels.index(edge[0]), labels.index(edge[1])
            color_val = edge_color[i, j] if edge_color.ndim == 2 else edge_color[i]
            if edge_colormap is not None:
                edge_colors.append(mpl.colors.to_hex(edge_colormap(color_val)))
            else:
                edge_colors.append(plt.cm.Greys(color_val))
    else:
        edge_colors = edge_color if edge_color else 'gray'
    
    # Draw edges
    if len(G.edges()) > 0:
        nx.draw_networkx_edges(G, pos, width=normalized_weights, 
                              edge_color=edge_colors, alpha=0.7, arrows=True, 
                              arrowsize=20, arrowstyle='->', 
                              connectionstyle='arc3,rad=0.1', ax=ax)
    
    # Calculate node colors
    node_colors = []
    for i, label in enumerate(labels):
        if node_cluster_colormap is None:
            node_colors.append(node_cmap[i % len(node_cmap)])
        else:
            node_colors.append(node_cluster_colormap[label])
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_size*8000, alpha=0.9, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', 
                           font_color='white', ax=ax)
    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    
    # Save the plot
    if filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    elif filename.endswith('.png'):
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    else:
        plt.savefig(filename + '.png', format='png', bbox_inches='tight', dpi=300)
    
    plt.close()

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
    """
    Plot cluster signaling network for multiple ligand-receptor pairs using matplotlib and NetworkX.
    
    Parameters
    ----------
    S_list : list
        List of signaling matrices
    p_values_list : list
        List of p-value matrices
    labels : list, optional
        Cluster labels
    quantile_cutoff : float, optional
        Quantile cutoff for significance
    p_value_cutoff : float
        P-value cutoff
    node_size : float
        Size of nodes
    node_colormap : str
        Colormap for nodes
    node_pos : array-like, optional
        Positions for nodes
    edge_width_lb_quantile : float
        Lower bound quantile for edge width
    edge_width_ub_quantile : float
        Upper bound quantile for edge width
    edge_width_min : float
        Minimum edge width
    edge_width_max : float
        Maximum edge width
    edge_colors : list
        Colors for different LR pairs
    background_pos : array-like, optional
        Background positions to plot
    background_ndcolor : str
        Background color
    background_ndsize : float
        Background node size
    filename : str
        Output filename
    """
    
    tmp_all_S = []
    for S in S_list:
        tmp_all_S.extend( list( S.reshape(-1) ) )
    tmp_all_S = np.array(tmp_all_S)
    if labels is None:
        labels = [str(i) for i in range(S_list[0].shape[0])]
    node_cmap = get_cmap_qualitative(node_colormap, n_colors=len(labels))
    
    # Create NetworkX graph
    G = nx.MultiDiGraph()
    
    # Add nodes
    for i, label in enumerate(labels):
        G.add_node(label)
    
    # Process each signaling matrix
    all_edge_weights = []
    all_edges_info = []  # Store edge information for each S matrix
    
    for s_idx, (S, p_values) in enumerate(zip(S_list, p_values_list)):
        edges_for_this_s = []
        
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                include_edge = False
                
                if quantile_cutoff is not None:
                    cutoff = np.quantile(tmp_all_S, quantile_cutoff)
                    if S[i,j] >= cutoff:
                        include_edge = True
                
                if p_values[i,j] <= p_value_cutoff:
                    include_edge = True
                
                if include_edge:
                    edge_key = f"{s_idx}_{i}_{j}"  # Unique key for this edge
                    G.add_edge(labels[i], labels[j], key=edge_key, weight=S[i,j], s_idx=s_idx)
                    all_edge_weights.append(S[i,j])
                    edges_for_this_s.append((labels[i], labels[j], edge_key, S[i,j]))
        
        all_edges_info.append(edges_for_this_s)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate layout
    if node_pos is not None:
        pos = {labels[i]: node_pos[i,:2] for i in range(len(labels))}
    else:
        # Use spring layout as default
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw background positions if provided
    if background_pos is not None:
        ax.scatter(background_pos[:,0], background_pos[:,1], 
                   c=background_ndcolor, s=background_ndsize*50, alpha=0.3, zorder=1)
    
    # Calculate edge width bounds
    if all_edge_weights:
        edge_width_lb = np.quantile(all_edge_weights, edge_width_lb_quantile)
        edge_width_ub = np.quantile(all_edge_weights, edge_width_ub_quantile)
    else:
        edge_width_lb, edge_width_ub = 0, 1
    
    # Draw edges for each signaling matrix with different colors
    for s_idx, edges_info in enumerate(all_edges_info):
        if not edges_info:
            continue
            
        # Prepare edges for this signaling matrix
        edges_to_draw = [(edge[0], edge[1]) for edge in edges_info]
        edge_weights = [edge[3] for edge in edges_info]
        
        # Normalize edge weights
        normalized_weights = [
            linear_clamp_value(w, edge_width_lb, edge_width_ub, edge_width_min, edge_width_max)
            for w in edge_weights
        ]
        
        # Get edge color for this signaling matrix
        edge_color = edge_colors[s_idx % len(edge_colors)]
        
        # Draw edges with slight offset to avoid overlap
        offset = s_idx * 0.05  # Small offset for visual separation
        pos_offset = {node: (x + offset, y + offset) for node, (x, y) in pos.items()}
        
        if edges_to_draw:
            nx.draw_networkx_edges(G, pos_offset, edgelist=edges_to_draw,
                                  width=normalized_weights, edge_color=edge_color, 
                                  alpha=0.7, arrows=True, arrowsize=15, 
                                  arrowstyle='->', connectionstyle=f'arc3,rad={0.1 + s_idx*0.05}', 
                                  ax=ax, label=f'Signal {s_idx+1}')
    
    # Calculate node colors
    node_colors = [node_cmap[i % len(node_cmap)] for i in range(len(labels))]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_size*8000, alpha=0.9, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', 
                           font_color='white', ax=ax)
    
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    
    # Add legend for edge colors if multiple signaling matrices
    if len(S_list) > 1:
        legend_elements = [plt.Line2D([0], [0], color=edge_colors[i % len(edge_colors)], 
                                     lw=2, label=f'Signal {i+1}') 
                          for i in range(len(S_list))]
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Save the plot
    if filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
    elif filename.endswith('.png'):
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
    else:
        plt.savefig(filename + '.png', format='png', bbox_inches='tight', dpi=300)
    
    plt.close()

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