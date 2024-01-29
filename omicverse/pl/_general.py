


def add_palue(ax,line_x1,line_x2,line_y,
              text_y,text,fontsize=11,fontcolor='#000000',
             horizontalalignment='center',):
    ax.plot((line_x1,line_x2),(line_y,line_y),c=fontcolor)
    ax.text((line_x1+line_x2)/2,line_y+text_y,text,fontsize=fontsize,
            horizontalalignment=horizontalalignment,)