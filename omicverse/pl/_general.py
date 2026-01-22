from ..utils.registry import register_function


@register_function(
    aliases=["添加P值", "add_palue", "add_pvalue", "p值标注", "统计标注"],
    category="pl",
    description="Add p-value annotation with connecting line to matplotlib plot",
    examples=[
        "# Basic p-value annotation",
        "fig, ax = ov.pl.boxplot(data, hue='group', x_value='condition', y_value='value')",
        "ov.pl.add_palue(ax, line_x1=0, line_x2=1, line_y=50, text_y=2, text='p<0.001')",
        "# Multiple comparisons",
        "ov.pl.add_palue(ax, line_x1=-0.5, line_x2=0.5, line_y=40, text_y=1,",
        "                text='$p={}$'.format(0.001), fontsize=12)",
        "# Custom styling",
        "ov.pl.add_palue(ax, line_x1=1, line_x2=2, line_y=60, text_y=3,",
        "                text='***', fontcolor='red', fontsize=14)"
    ],
    related=["pl.boxplot", "pl.violin", "pl.bardotplot"]
)
def add_palue(ax,line_x1,line_x2,line_y,
              text_y,text,fontsize=11,fontcolor='#000000',
             horizontalalignment='center',):
    r"""
    Add p-value annotation with connecting line to a matplotlib plot.
    
    Args:
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

    Args:
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
    
    Args:
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