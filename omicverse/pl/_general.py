


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