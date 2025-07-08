


def add_palue(ax,line_x1,line_x2,line_y,
              text_y,text,fontsize=11,fontcolor='#000000',
             horizontalalignment='center',):
    r"""
    Add p-value annotation with connecting line to a matplotlib plot.
    
    Arguments:
        ax: matplotlib.axes.Axes object to add annotation to
        line_x1: Starting x-coordinate of connecting line
        line_x2: Ending x-coordinate of connecting line
        line_y: Y-coordinate of connecting line
        text_y: Y-offset for text placement above line
        text: Text to display (typically p-value)
        fontsize: Font size for text (11)
        fontcolor: Color for line and text ('#000000')
        horizontalalignment: Text horizontal alignment ('center')
        
    Returns:
        None: Adds annotation directly to the axes
    """
    ax.plot((line_x1,line_x2),(line_y,line_y),c=fontcolor)
    ax.text((line_x1+line_x2)/2,line_y+text_y,text,fontsize=fontsize,
            horizontalalignment=horizontalalignment,)
    

def create_transparent_gradient_colormap(color1, color2,N=100):
    r"""
    Create a gradient colormap from transparent color1 to opaque color2.

    Arguments:
        color1: Starting color (transparent)
        color2: Ending color (opaque)
        N: Number of color steps (default: 100)

    Returns:
        matplotlib.colors.LinearSegmentedColormap: Gradient colormap
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    
    rgb1 = to_rgb(color1)
    rgb2 = to_rgb(color2)
    
    colors = [
        rgb1 + (0.0,),  # 透明的起始颜色
        rgb2 + (1.0,)   # 不透明的结束颜色
    ]
    
    cmap = LinearSegmentedColormap.from_list('custom_transparent_gradient', colors, N=N)
    return cmap

def create_custom_colormap(cell_color):
    r"""
    Create a custom colormap based on cell type color.
    
    Arguments:
        cell_color: str
            Base color for the cell type
    
    Returns:
        cmap: matplotlib.colors.LinearSegmentedColormap
            Custom colormap
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors
    
    # Convert color to RGB if it's a hex string
    if isinstance(cell_color, str):
        base_rgb = mcolors.to_rgb(cell_color)
    else:
        base_rgb = cell_color[:3] if len(cell_color) >= 3 else cell_color
    
    # Create gradient from light to dark
    colors = [(1.0, 1.0, 1.0, 1), base_rgb + (1.0,)]  # White transparent to full color
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return cmap