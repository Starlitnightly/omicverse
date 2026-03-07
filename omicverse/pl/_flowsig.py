import numpy as np

import random
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from ._palette import palette_112
from .._registry import register_function

def curved_line(x0, y0, x1, y1, eps=0.8, pointn=30):
    r"""
    Generate points for a curved line between two coordinates using Bezier curves.
    
    Parameters
    ----------
    x0 : float
        Start x-coordinate.
    y0 : float
        Start y-coordinate.
    x1 : float
        End x-coordinate.
    y1 : float
        End y-coordinate.
    eps : float
        Curvature control factor.
    pointn : int
        Number of sampled points along the curve.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        Sampled x/y coordinate arrays.
    """
    import bezier
    x2 = (x0 + x1) / 2.0 + 0.1 ** (eps + abs(x0 - x1)) * (-1) ** (random.randint(1, 4))
    y2 = (y0 + y1) / 2.0 + 0.1 ** (eps + abs(y0 - y1)) * (-1) ** (random.randint(1, 4))
    nodes = np.asfortranarray([
        [x0, x2, x1],
        [y0, y2, y1]
    ])
    curve = bezier.Curve(nodes, degree=2)
    s_vals = np.linspace(0.0, 1.0, pointn)
    data = curve.evaluate_multi(s_vals)
    x = data[0]
    y = data[1]
    return x, y

def curved_graph(_graph, pos=None, eps=0.2, pointn=30, 
                 linewidth=2, alpha=0.3, color_dict=None):
    r"""
    Draw a network graph with curved edges and arrows.
    
    Parameters
    ----------
    _graph : nx.Graph
        Graph whose edges will be rendered as curved links.
    pos : dict or None
        Node-position dictionary mapping node IDs to xy coordinates.
    eps : float
        Curvature control factor.
    pointn : int
        Number of points sampled on each edge curve.
    linewidth : float
        Width of rendered edges.
    alpha : float
        Edge transparency.
    color_dict : dict or None
        Optional node-to-color mapping used for edge coloring.

    Returns
    -------
    None
        Draws curved edges on current matplotlib axes.
    """
    ax = plt.gca()
    for u, v in _graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        x, y = curved_line(x0, y0, x1, y1, eps=eps, pointn=pointn)
        
        # 使用颜色字典为边着色
        color = color_dict[u] if color_dict and u in color_dict else 'k'
        
        # 绘制曲线部分
        segments = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([segments[:-1], segments[1:]], axis=1)
        lc = LineCollection(segments, linewidths=linewidth, alpha=alpha, color=color)
        ax.add_collection(lc)
        
        # 在曲线的最后添加箭头
        arrow = FancyArrowPatch((x[-2], y[-2]), (x[-1], y[-1]),
                                connectionstyle="arc3,rad=0.2",
                                arrowstyle='-|>', mutation_scale=10,
                                color=color, linewidth=linewidth, alpha=1)
        ax.add_patch(arrow)
        
    ax.autoscale_view()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import networkx as nx
import pandas as pd
import anndata
def plot_curve_network(G: nx.Graph, G_type_dict: dict, G_color_dict: dict, pos_type: str = 'spring', pos_dim: int = 2,
                       figsize: tuple = (4, 4), pos_scale: int = 10, pos_k=None, pos_alpha: float = 0.4,
                       node_size: int = 50, node_alpha: float = 0.6, node_linewidths: int = 1,
                       plot_node=None, plot_node_num: int = 20, node_shape=None,
                       label_verticalalignment: str = 'center_baseline', label_fontsize: int = 12,
                       label_fontfamily: str = 'Arial', label_fontweight: str = 'bold', label_bbox=None,
                       legend_bbox: tuple = (0.7, 0.05), legend_ncol: int = 3, legend_fontsize: int = 12,
                       legend_fontweight: str = 'bold', curve_awarg=None,ylim=(-0.5,0.5),xlim=(-3,3),ax=None):
    r"""
    Create a network plot with curved edges and customizable node styling.
    
    Parameters
    ----------
    G : nx.Graph
        Graph object to visualize.
    G_type_dict : dict
        Mapping from node IDs to node-type categories.
    G_color_dict : dict
        Mapping from node IDs to plotting colors.
    pos_type : str or dict
        Layout strategy (``'spring'``, ``'kamada_kawai'``) or explicit position dict.
    pos_dim : int
        Dimensionality used by layout algorithm.
    figsize : tuple
        Figure size for plotting canvas.
    pos_scale : int or float
        Scale parameter for layout computation.
    pos_k : float or None
        Optimal pairwise distance parameter in spring layout.
    pos_alpha : float
        Reserved layout-related alpha parameter.
    node_size : int
        Base node size multiplier.
    node_alpha : float
        Node transparency.
    node_linewidths : int or float
        Border width for nodes.
    plot_node : list or None
        Explicit list of nodes to label.
    plot_node_num : int
        Number of top-degree nodes labeled when ``plot_node`` is None.
    node_shape : dict or None
        Optional mapping from node category to marker shape.
    label_verticalalignment : str
        Vertical alignment of text labels.
    label_fontsize : int
        Label font size.
    label_fontfamily : str
        Label font family.
    label_fontweight : str
        Label font weight.
    label_bbox : dict or None
        Optional matplotlib bbox style for labels.
    legend_bbox : tuple
        Legend anchor position.
    legend_ncol : int
        Number of legend columns.
    legend_fontsize : int
        Legend font size.
    legend_fontweight : str
        Legend font weight.
    curve_awarg : dict or None
        Optional arguments for curved-edge drawing helper.
    ylim : tuple
        Y-axis limits.
    xlim : tuple
        X-axis limits.
    ax : matplotlib.axes.Axes or None
        Existing axes for plotting.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes containing network visualization.
    """
    from adjustText import adjust_text
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine node positions
    if pos_type == 'spring':
        pos = nx.spring_layout(G, scale=pos_scale, k=pos_k)
    elif pos_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, dim=pos_dim, scale=pos_scale)
    else:
        pos = pos_type

    degree_dict = dict(G.degree(G.nodes()))

    # Update color and type dictionaries
    G_color_dict = {node: G_color_dict[node] for node in G.nodes}
    G_type_dict = {node: G_type_dict[node] for node in G.nodes}

    # Draw nodes with different shapes if specified
    if node_shape is not None:
        for node_type, shape in node_shape.items():
            node_list = [node for node in G.nodes if G_type_dict[node] == node_type]
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=node_list,
                node_size=100,
                node_color=[G_color_dict[v] for v in node_list],
                alpha=node_alpha,
                linewidths=node_linewidths,
                node_shape=shape  # Set node shape
            )
            resultant_pattern_graph_edge_colours = [G_color_dict[edge[0]] for edge in G.edges()]
            edge_widths = [0.2*G[u][v]['weight'] for u,v in G.edges()]
            nx.draw_networkx_edges(G, pos, 
                                   edge_color=resultant_pattern_graph_edge_colours, 
                                   width=edge_widths, alpha=0.75, connectionstyle="arc3,rad=0.2")
    else:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(G_color_dict.keys()),
            node_size=[degree_dict[v] * node_size for v in G],
            node_color=list(G_color_dict.values()),
            alpha=node_alpha,
            linewidths=node_linewidths,
        )
        resultant_pattern_graph_edge_colours = [G_color_dict[edge[0]] for edge in G.edges()]
        edge_widths = [0.2*G[u][v]['weight'] for u,v in G.edges()]
        nx.draw_networkx_edges(G, pos, 
                                   edge_color=resultant_pattern_graph_edge_colours, 
                                   width=edge_widths, alpha=0.75, connectionstyle="arc3,rad=0.2")

    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])

    # Determine hub genes
    if plot_node is not None:
        hub_gene = plot_node
    else:
        hub_gene = [i[0] for i in sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:plot_node_num]]

    # Draw curved edges if specified
    #if curve_awarg is not None:
    #    curved_graph(G, pos, color_dict=G_color_dict, **curve_awarg)
    #else:
    #    curved_graph(G, pos, color_dict=G_color_dict)

    pos1 = {i: pos[i] for i in hub_gene}

    # Add labels to hub genes
    texts = [ax.text(pos1[i][0], pos1[i][1], i, fontdict={'size': label_fontsize, 'weight': label_fontweight, 'color': 'black'})
             for i in hub_gene if 'ENSG' not in i]
    print(texts)
    
    import adjustText
    if adjustText.__version__<='0.8':
        adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)
    else:
        adjust_text(texts,only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                    arrowprops=dict(arrowstyle='->', color='red'))

    ax.axis("off")
    
    
    # Prepare legend
    t = pd.DataFrame(index=G_type_dict.keys())
    t['gene_type_dict'] = G_type_dict
    t['gene_color_dict'] = G_color_dict
    type_color_dict = {i: t.loc[t['gene_type_dict'] == i, 'gene_color_dict'].values[0] for i in t['gene_type_dict'].value_counts().index}

    patches = [mpatches.Patch(color=type_color_dict[i], label="{:s}".format(i)) for i in type_color_dict.keys()]
    plt.tight_layout()
    plt.margins(x=0.3)
    plt.margins(y=0)
    #plt.legend(handles=patches, bbox_to_anchor=legend_bbox, ncol=legend_ncol, fontsize=legend_fontsize)
    #leg = plt.gca().get_legend()
    #ltext = leg.get_texts()
    #plt.setp(ltext, fontsize=legend_fontsize, fontweight=legend_fontweight)

    return fig, ax


@register_function(
    aliases=['FlowSig 网络图', 'plot_flowsig_network', 'intercellular flow network plot'],
    category="pl",
    description="Visualize inferred intercellular signal-flow graphs, highlighting pathway-level directional communication structure.",
    prerequisites={'optional_functions': ['pathway_enrichment']},
    requires={'uns': ['flow network results']},
    produces={},
    auto_fix='none',
    examples=['ov.pl.plot_flowsig_network(flow_network, gem_plot=True)'],
    related=['single.pathway_enrichment', 'pl.plot_spatial']
)
def plot_flowsig_network(flow_network,
                         gem_plot,
                         figsize=(8,4),
                        curve_awarg={'eps':2},
                        node_shape={'GEM':'^','Sender':'o','Receptor':'o'},
                        **kwargs):
    r"""
    Create a flowsig network visualization showing GEM modules and gene flows.
    
    Parameters
    ----------
    flow_network : nx.Graph
        FlowSig graph containing GEM, sender, and receptor nodes.
    gem_plot : list
        GEM module subset to include in rendered subgraph.
    figsize : tuple
        Figure size for network plot.
    curve_awarg : dict
        Optional keyword arguments for curved-edge drawing.
    node_shape : dict
        Mapping from node class to marker shape.
    **kwargs
        Additional arguments passed to ``plot_curve_network``.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes of the FlowSig network plot.
    """
    
    gem_li=[i for i in flow_network.nodes if 'GEM' in i]
    receptor_li=[i for i,j in flow_network.edges if ('GEM' in j)and('GEM' not in i)]
    sender_li=[j for i,j in flow_network.edges if ('GEM' in i)and('GEM' not in j)]
    for g in gem_li:
        flow_network.nodes[g]['type']='module'
    for g in receptor_li:
        flow_network.nodes[g]['type']='inflow'
    for g in sender_li:
        flow_network.nodes[g]['type']='outflow'
    #gene_li=[i for i in flow_network.nodes if 'GEM' not in i]
    G_type_dict=dict(zip(gem_li+receptor_li+sender_li,
            ['GEM' for i in range(len(gem_li))]+\
                        ['Receptor' for i in range(len(receptor_li))]+\
                        ['Sender' for i in range(len(sender_li))]))
    
    gem_li=[i for i in flow_network.nodes if 'GEM' in i]
    receptor_li=[i for i,j in flow_network.edges if ('GEM' in j)and('GEM' not in i)]
    sender_li=[j for i,j in flow_network.edges if ('GEM' in i)and('GEM' not in j)]
    #gene_li=[i for i in flow_network.nodes if 'GEM' not in i]
    if len(gem_li)<28:
        sc_color=['#7CBB5F','#368650','#A499CC','#5E4D9A','#78C2ED','#866017', '#9F987F','#E0DFED',
            '#EF7B77', '#279AD7','#F0EEF0', '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', '#FCBC10',
            '#EAEFC5', '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']

        G_color_dict=dict(zip(gem_li+receptor_li+sender_li,
                [sc_color[i] for i in range(len(gem_li))]+\
                            ['#c2c2c2'  for i in range(len(receptor_li))]+\
                            ['#a51616' for i in range(len(sender_li))]))
    else:
        sc_color=palette_112
        G_color_dict=dict(zip(gem_li+receptor_li+sender_li,
                [sc_color[i] for i in range(len(gem_li))]+\
                            ['#c2c2c2'  for i in range(len(receptor_li))]+\
                            ['#a51616' for i in range(len(sender_li))]))
    
    #G = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
    import networkx as nx
    if len(sender_li)==0 and len(receptor_li)>0:
        layers = { "layer3": list(set(receptor_li)),
                  "layer2": list(set(gem_li)),}
    elif len(sender_li)>0 and len(receptor_li)==0:
        layers = {"layer1": list(set(sender_li)), 
                   "layer3": list(set(gem_li))}
    else:
        layers = {"layer1": list(set(sender_li)), 
                   "layer3": list(set(receptor_li)),
                   "layer2": list(set(gem_li)),}
    
    # Assign layer information to node attributes
    for layer_name, node_list in layers.items():
        for node in node_list:
            flow_network.nodes[node]['layer'] = layer_name
    
    pos = nx.multipartite_layout(flow_network, subset_key='layer', align='horizontal', scale=2.0)

    #fig, ax = plt.subplots(figsize=(8,8)) 
    sub_G=flow_network.subgraph(sender_li+receptor_li+gem_plot).copy()
    # 获取所有节点和边
    all_nodes = set(sub_G.nodes())
    all_edges = sub_G.edges()
    # 找到没有连接边的节点
    nodes_with_edges = set()
    for edge in all_edges:
        nodes_with_edges.update(edge)
        
    nodes_without_edges = all_nodes - nodes_with_edges

    # 从图中删除这些节点
    sub_G.remove_nodes_from(nodes_without_edges)

    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch
    import networkx as nx
    fig,ax=plot_curve_network(sub_G,
                        G_type_dict=G_type_dict,
                        G_color_dict=G_color_dict,
                        pos_type=pos,
                        figsize=figsize,
                        curve_awarg=curve_awarg,
                        node_shape=node_shape,
                        **kwargs)
                        
    plt.box(False)
    return fig,ax
