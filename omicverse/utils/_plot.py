import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import scanpy as sc
import networkx as nx
import pandas as pd

sc_color=['#7CBB5F','#368650','#A499CC','#5E4D9A','#78C2ED','#866017', '#9F987F','#E0DFED',
 '#EF7B77', '#279AD7','#F0EEF0', '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', '#FCBC10',
 '#EAEFC5', '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']
sc_color_cmap = LinearSegmentedColormap.from_list('Custom', sc_color, len(sc_color))

def pyomic_plot_set(verbosity=3,dpi=80,facecolor='white'):
    sc.settings.verbosity = verbosity             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.settings.set_figure_params(dpi=dpi, facecolor=facecolor)
    import warnings
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)

def pyomic_palette()->list:
    """
    Returns a dictionary of colors for various plots used in pyomic package.

    Returns:
        sc_color: List containing the hex codes as values.
    """ 
    return sc_color

def plot_text_set(text,text_knock=2,text_maxsize=20):
    """
    Formats the text to fit in a plot by adding line breaks.

    Parameters
    ----------
    - text : `str`
        Text to format.
    - text_knock : `int`, optional
        Number of words to skip between two line breaks, by default 2.
    - text_maxsize : `int`, optional
        Maximum length of the text before formatting, by default 20.

    Returns
    -------
    - text: `str`
        Formatted text.
    """
    #print(text)
    text_len=len(text)
    if text_len>text_maxsize:
        ty=text.split(' ')
        ty_len=len(ty)
        if ty_len%2==1:
            ty_mid=(ty_len//text_knock)+1
        else:
            ty_mid=(ty_len//text_knock)
        #print(ty_mid)

        if ty_mid==0:
            ty_mid=1

        res=''
        ty_len_max=np.max([i%ty_mid for i in range(ty_len)])
        if ty_len_max==0:
            ty_len_max=1
        for i in range(ty_len):
            #print(ty_mid,i%ty_mid,i,ty_len_max)
            if (i%ty_mid)!=ty_len_max:
                res+=ty[i]+' '
            else:
                res+='\n'+ty[i]+' '
        return res
    else:
        return text
    
def ticks_range(x,width):
    """
    Returns a list of ticks for a plot.
    
    Parameters
    ----------
    - x : `int`
        Number of ticks.
    - width : `float`
        Width of the plot.

    Returns
    -------
    - ticks: `list`
        List of ticks.
    """
    nticks=[]
    pticks=[]
    start=-(x//2)
    end=(x//2)
    for i in range(x//2):
        nticks.append(start+width)
        start+=width
        pticks.append(end-width)
        end-=width
    if x%2==0:
        ticks=nticks+pticks
    elif x%2==1:
        ticks=nticks+[0]+pticks
    return ticks

def plot_boxplot(data,hue,x_value,y_value,width=0.6,title='',
                 figsize=(6,3),palette=None,fontsize=10,
                 legend_bbox=(1, 0.55),legend_ncol=1,):
    """
    Plots a boxplot with jittered points.

    Parameters
    ----------
    - data : `pandas.DataFrame`
        Dataframe containing the data to plot.
    - hue : `str`
        Column name of the dataframe containing the hue data.
    - x_value : `str`
        Column name of the dataframe containing the x-axis data.
    - y_value : `str`
        Column name of the dataframe containing the y-axis data.
    - width : `float`, optional
        Width of the boxplot, by default 0.6.
    - title : `str`, optional
        Title of the plot, by default ''.
    - figsize : `tuple`, optional
        Size of the figure, by default (6,3).
    - palette : `list`, optional
        List of colors to use for the plot, by default None.
    - fontsize : `int`, optional
        Font size of the plot, by default 10.
    - legend_bbox : `tuple`, optional
        Bounding box of the legend, by default (1, 0.55).
    - legend_ncol : `int`, optional
        Number of columns in the legend, by default 1.

    Returns
    -------
    - fig: `matplotlib.figure.Figure`
        Figure object.
    - ax: `matplotlib.axes._subplots.AxesSubplot`
        Axes object.
    """

    #获取需要分割的数据
    hue=hue
    hue_datas=list(set(data[hue]))

    #获取箱线图的横坐标
    x=x_value
    ticks=list(set(data[x]))

    #在这个数据中，我们有6个不同的癌症，每个癌症都有2个基因（2个箱子）
    #所以我们需要得到每一个基因的6个箱线图位置，6个散点图的抖动
    plot_data1={}#字典里的每一个元素就是每一个基因的所有值
    plot_data_random1={}#字典里的每一个元素就是每一个基因的随机20个值
    plot_data_xs1={}#字典里的每一个元素就是每一个基因的20个抖动值


    #箱子的参数
    #width=0.6
    y=y_value
    for hue_data,num in zip(hue_datas,ticks_range(len(hue_datas),width)):
        data_a=[]
        data_a_random=[]
        data_a_xs=[]
        for i,k in zip(ticks,range(len(ticks))):
            test_data=data.loc[((data[x]==i)&(data[hue]==hue_data)),y].tolist()
            data_a.append(test_data)
            if len(test_data)<20:
                data_size=len(test_data)
            else:
                data_size=20
            random_data=random.sample(test_data,data_size)
            data_a_random.append(random_data)
            data_a_xs.append(np.random.normal(k*len(hue_datas)+num, 0.04, len(random_data)))
        #data_a=np.array(data_a)
        data_a_random=np.array(data_a_random)
        plot_data1[hue_data]=data_a 
        plot_data_random1[hue_data]=data_a_random
        plot_data_xs1[hue_data]=data_a_xs

    fig, ax = plt.subplots(figsize=figsize)
    #色卡
    if palette==None:
        palette=pyomic_palette()
    #palette=["#a64d79","#674ea7"]
    #绘制箱线图
    for hue_data,hue_color,num in zip(hue_datas,palette,ticks_range(len(hue_datas),width)):
        b1=ax.boxplot(plot_data1[hue_data], 
                    positions=np.array(range(len(ticks)))*len(hue_datas)+num, 
                    sym='', 
                    widths=width,)
        plt.setp(b1['boxes'], color=hue_color)
        plt.setp(b1['whiskers'], color=hue_color)
        plt.setp(b1['caps'], color=hue_color)
        plt.setp(b1['medians'], color=hue_color)

        clevels = np.linspace(0., 1., len(plot_data_random1[hue_data]))
        for x, val, clevel in zip(plot_data_xs1[hue_data], plot_data_random1[hue_data], clevels):
            plt.scatter(x, val,c=hue_color,alpha=0.4)

    #坐标轴字体
    #fontsize=10
    #修改横坐标
    ax.set_xticks(range(0, len(ticks) * len(hue_datas), len(hue_datas)), ticks,fontsize=fontsize)
    #修改纵坐标
    yticks=ax.get_yticks()
    ax.set_yticks(yticks[yticks>=0],yticks[yticks>=0],fontsize=fontsize)

    labels = hue_datas  #legend标签列表，上面的color即是颜色列表
    color = palette
    #用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
    patches = [ mpatches.Patch(color=color[i], label="{:s}".format(labels[i]) ) for i in range(len(hue_datas)) ] 
    ax.legend(handles=patches,bbox_to_anchor=legend_bbox, ncol=legend_ncol,fontsize=fontsize)

    #设置标题
    ax.set_title(title,fontsize=fontsize+2)
    #设置spines可视化情况
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    return fig,ax

def plot_network(G,G_type_dict,G_color_dict,pos_type='spring',pos_dim=2,
                figsize=(4,4),pos_scale=10,pos_k=None,pos_alpha=0.4,
                node_size=50,node_alpha=0.6,node_linewidths=1,
                plot_node=None,plot_node_num=20,
                label_verticalalignment='center_baseline',label_fontsize=12,
                label_fontfamily='Arial',label_fontweight='bold',label_bbox=None,
                legend_bbox=(0.7, 0.05),legend_ncol=3,legend_fontsize=12,
                legend_fontweight='bold'):
    
    fig, ax = plt.subplots(figsize=figsize)
    if pos_type=='spring':
        pos = nx.spring_layout(G, scale=pos_scale, k=pos_k)
    elif pos_type=='kamada_kawai':
        pos=nx.kamada_kawai_layout(G,dim=pos_dim,scale=pos_scale)
    degree_dict = dict(G.degree(G.nodes()))
    
    nx.draw_networkx_edges(G, pos,nodelist=list(G_color_dict.keys()), alpha=pos_alpha)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(G_color_dict.keys()),
        node_size=[degree_dict[v]*node_size for v in G],
        node_color=list(G_color_dict.values()),
        alpha=node_alpha,
        linewidths=node_linewidths,
    )
    if plot_node!=None:
        hub_gene=plot_node
    else:
        hub_gene=[i[0] for i in sorted(degree_dict.items(),key=lambda x: x[1],reverse=True)[:plot_node_num]]
    
    pos1=dict()
    for i in pos.keys():
        pos1[i]=np.array([-1000,-1000])
    for i in hub_gene:
        pos1[i]=pos[i]
    #label_options = {"ec": "white", "fc": "white", "alpha": 0.6}
    nx.draw_networkx_labels(
        G,pos1,verticalalignment=label_verticalalignment,
        font_size=label_fontsize,font_family=label_fontfamily,
        font_weight=label_fontweight,bbox=label_bbox,
    )

    ax.axis("off")
    
    t=pd.DataFrame(index=G_type_dict.keys())
    t['gene_type_dict']=G_type_dict
    t['gene_color_dict']=G_color_dict
    type_color_dict={}
    for i in t['gene_type_dict'].value_counts().index:
        type_color_dict[i]=t.loc[t['gene_type_dict']==i,'gene_color_dict'].values[0]
    
    patches = [ mpatches.Patch(color=type_color_dict[i], label="{:s}".format(i) ) for i in type_color_dict.keys() ] 

    plt.legend(handles=patches,bbox_to_anchor=legend_bbox, ncol=legend_ncol,fontsize=legend_fontsize)
    leg = plt.gca().get_legend() #或leg=ax.get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=legend_fontsize,fontweight=legend_fontweight)
    
    return fig,ax