import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import scanpy as sc
import networkx as nx
import pandas as pd
import anndata

sc_color=['#7CBB5F','#368650','#A499CC','#5E4D9A','#78C2ED','#866017', '#9F987F','#E0DFED',
 '#EF7B77', '#279AD7','#F0EEF0', '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', '#FCBC10',
 '#EAEFC5', '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']

red_color=['#F0C3C3','#E07370','#CB3E35','#A22E2A','#5A1713','#D3396D','#DBC3DC','#85539B','#5C2B80','#5C4694']
green_color=['#91C79D','#8FC155','#56AB56','#2D5C33','#BBCD91','#6E944A','#A5C953','#3B4A25','#010000']
orange_color=['#EFBD49','#D48F3E','#AC8A3E','#7D7237','#745228','#E1C085','#CEBC49','#EBE3A1','#6C6331','#8C9A48','#D7DE61']
blue_color=['#347862','#6BBBA0','#81C0DD','#3E8CB1','#88C8D2','#52B3AD','#265B58','#B2B0D4','#5860A7','#312C6C']

sc_color_cmap = LinearSegmentedColormap.from_list('Custom', sc_color, len(sc_color))

def pyomic_plot_set(verbosity=3,dpi=80,facecolor='white'):
    sc.settings.verbosity = verbosity             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.settings.set_figure_params(dpi=dpi, facecolor=facecolor)
    import warnings
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)

def ov_plot_set(verbosity=3,dpi=80,facecolor='white'):
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

def palette()->list:
    """
    Returns a dictionary of colors for various plots used in pyomic package.

    Returns:
        sc_color: List containing the hex codes as values.
    """ 
    return sc_color

def red_palette()->list:
    """
    Returns a dictionary of colors for various plots used in pyomic package.

    Returns:
        sc_color: List containing the hex codes as values.
    """ 
    return red_color

def green_palette()->list:
    """
    Returns a dictionary of colors for various plots used in pyomic package.

    Returns:
        sc_color: List containing the hex codes as values.
    """ 
    return green_color

def orange_palette()->list:
    """
    Returns a dictionary of colors for various plots used in pyomic package.

    Returns:
        sc_color: List containing the hex codes as values.
    """ 
    return orange_color

def blue_palette()->list:
    """
    Returns a dictionary of colors for various plots used in pyomic package.

    Returns:
        sc_color: List containing the hex codes as values.
    """ 
    return blue_color

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

def plot_network(G:nx.Graph,G_type_dict:dict,G_color_dict:dict,pos_type:str='spring',pos_dim:int=2,
                figsize:tuple=(4,4),pos_scale:int=10,pos_k=None,pos_alpha:float=0.4,
                node_size:int=50,node_alpha:float=0.6,node_linewidths:int=1,
                plot_node=None,plot_node_num:int=20,
                label_verticalalignment:str='center_baseline',label_fontsize:int=12,
                label_fontfamily:str='Arial',label_fontweight:str='bold',label_bbox=None,
                legend_bbox:tuple=(0.7, 0.05),legend_ncol:int=3,legend_fontsize:int=12,
                legend_fontweight:str='bold'):
    """
    Plot network graph.

    Arguments:
        G: networkx graph
        G_type_dict: dict, node type dict
        G_color_dict: dict, node color dict
        pos_type: str, node position type, 'spring' or 'kamada_kawai'
        pos_dim: int, node position dimension, 2 or 3
        figsize: tuple, figure size
        pos_scale: int, node position scale
        pos_k: float, node position k
        pos_alpha: float, node position alpha
        node_size: int, node size
        node_alpha: float, node alpha
        node_linewidths: float, node linewidths
        plot_node: list, plot node list
        plot_node_num: int, plot node number
        label_verticalalignment: str, label verticalalignment
        label_fontsize: int, label fontsize
        label_fontfamily: str, label fontfamily
        label_fontweight: str, label fontweight
        label_bbox: tuple, label bbox
        legend_bbox: tuple, legend bbox
        legend_ncol: int, legend ncol
        legend_fontsize: int, legend fontsize
        legend_fontweight: str, legend fontweight

    
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    if pos_type=='spring':
        pos = nx.spring_layout(G, scale=pos_scale, k=pos_k)
    elif pos_type=='kamada_kawai':
        pos=nx.kamada_kawai_layout(G,dim=pos_dim,scale=pos_scale)
    degree_dict = dict(G.degree(G.nodes()))
    
    G_color_dict=dict(zip(G.nodes,[G_color_dict[i] for i in G.nodes]))
    G_type_dict=dict(zip(G.nodes,[G_type_dict[i] for i in G.nodes]))

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

def plot_cellproportion(adata:anndata.AnnData,celltype_clusters:str,visual_clusters:str,
                       visual_li=None,visual_name:str='',figsize:tuple=(4,6),
                       ticks_fontsize:int=12,labels_fontsize:int=12,
                       legend:bool=False):
    """
    Plot cell proportion of each cell type in each visual cluster.

    Arguments:
        adata: AnnData object.
        celltype_clusters: Cell type clusters.
        visual_clusters: Visual clusters.
        visual_li: Visual cluster list.
        visual_name: Visual cluster name.
        figsize: Figure size.
        ticks_fontsize: Ticks fontsize.
        labels_fontsize: Labels fontsize.
        legend: Whether to show legend.
    
    
    """

    b=pd.DataFrame(columns=['cell_type','value','Week'])
    
    if visual_li==None:
        adata.obs[visual_clusters]=adata.obs[visual_clusters].astype('category')
        visual_li=adata.obs[visual_clusters].cat.categories
    
    for i in visual_li:
        b1=pd.DataFrame()
        test=adata.obs.loc[adata.obs[visual_clusters]==i,celltype_clusters].value_counts()
        b1['cell_type']=test.index
        b1['value']=test.values/test.sum()
        b1['Week']=i.replace('Retinoblastoma_','')
        b=pd.concat([b,b1])
    
    plt_data2=adata.obs[celltype_clusters].value_counts()
    plot_data2_color_dict=dict(zip(adata.obs[celltype_clusters].cat.categories,adata.uns['{}_colors'.format(celltype_clusters)]))
    plt_data3=adata.obs[visual_clusters].value_counts()
    plot_data3_color_dict=dict(zip([i.replace('Retinoblastoma_','') for i in adata.obs[visual_clusters].cat.categories],adata.uns['{}_colors'.format(visual_clusters)]))
    b['cell_type_color'] = b['cell_type'].map(plot_data2_color_dict)
    b['stage_color']=b['Week'].map(plot_data3_color_dict)
    
    fig, ax = plt.subplots(figsize=figsize)
    #用ax控制图片
    #sns.set_theme(style="whitegrid")
    #sns.set_theme(style="ticks")
    n=0
    all_celltype=adata.obs[celltype_clusters].cat.categories
    for i in all_celltype:
        if n==0:
            test1=b[b['cell_type']==i]
            ax.bar(x=test1['Week'],height=test1['value'],width=0.8,color=list(set(test1['cell_type_color']))[0], label=i)
            bottoms=test1['value'].values
        else:
            test2=b[b['cell_type']==i]
            ax.bar(x=test2['Week'],height=test2['value'],bottom=bottoms,width=0.8,color=list(set(test2['cell_type_color']))[0], label=i)
            test1=test2
            bottoms+=test1['value'].values
        n+=1
    if legend!=False:
        plt.legend(bbox_to_anchor=(1.05, -0.05), loc=3, borderaxespad=0,fontsize=10)
    
    plt.grid(False)
    
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # 设置左边和下边的坐标刻度为透明色
    #ax.yaxis.tick_left()
    #ax.xaxis.tick_bottom()
    #ax.xaxis.set_tick_params(color='none')
    #ax.yaxis.set_tick_params(color='none')

    # 设置左边和下边的坐标轴线为独立的线段
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    plt.xticks(fontsize=ticks_fontsize,rotation=90)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel(visual_name,fontsize=labels_fontsize)
    plt.ylabel('Cells per Stage',fontsize=labels_fontsize)
    fig.tight_layout()
    return fig,ax

def plot_embedding_celltype(adata:anndata.AnnData,figsize:tuple=(6,4),basis:str='umap',
                            celltype_key:str='major_celltype',title:str=None,
                            celltype_range:tuple=(2,9),
                            embedding_range:tuple=(3,10),
                            xlim:int=-1000)->tuple:
    """
    Plot embedding with celltype color by omicverse

    Arguments:
        adata: AnnData object  
        figsize: figure size
        basis: embedding method
        celltype_key: celltype key in adata.obs
        title: figure title
        celltype_range: celltype range to plot
        embedding_range: embedding range to plot
        xlim: x axis limit

    Returns:
        fig : figure and axis
        ax: axis
    
    """


    cell_num_pd=pd.DataFrame(adata.obs[celltype_key].value_counts())
    if '{}_colors'.format(celltype_key) in adata.uns.keys():
        cell_color_dict=dict(zip(adata.obs[celltype_key].cat.categories.tolist(),
                        adata.uns['{}_colors'.format(celltype_key)]))
    else:
        if len(adata.obs[celltype_key].cat.categories)>28:
            cell_color_dict=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.default_102))
        else:
            cell_color_dict=dict(zip(adata.obs[celltype_key].cat.categories,sc.pl.palettes.zeileis_28))

    if figsize==None:
        if len(adata.obs[celltype_key].cat.categories)<10:
            fig = plt.figure(figsize=(6,4))
        else:
            print('The number of cell types is too large, please set the figsize parameter')
            return
    else:
        fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(10, 10)
    ax1 = fig.add_subplot(grid[:, embedding_range[0]:embedding_range[1]])       # 占据第一行的所有列
    ax2 = fig.add_subplot(grid[celltype_range[0]:celltype_range[1], :2]) 
    # 定义子图的大小和位置
         # 占据第二行的前两列
    #ax3 = fig.add_subplot(grid[1:, 2])      # 占据第二行及以后的最后一列
    #ax4 = fig.add_subplot(grid[2, 0])       # 占据最后一行的第一列
    #ax5 = fig.add_subplot(grid[2, 1])       # 占据最后一行的第二列

    sc.pl.embedding(
        adata,
        basis=basis,
        color=[celltype_key],
        title='',
        frameon=False,
        #wspace=0.65,
        ncols=3,
        ax=ax1,
        legend_loc=False,
        show=False
    )

    for idx,cell in zip(range(cell_num_pd.shape[0]),
                        adata.obs[celltype_key].cat.categories):
        ax2.scatter(100,
                cell,c=cell_color_dict[cell],s=50)
        ax2.plot((100,cell_num_pd.loc[cell,celltype_key]),(idx,idx),
                c=cell_color_dict[cell],lw=4)
        ax2.text(100,idx+0.2,
                cell+'('+str("{:,}".format(cell_num_pd.loc[cell,celltype_key]))+')',fontsize=11)
    ax2.set_xlim(xlim,cell_num_pd.iloc[1].values[0]) 
    ax2.text(xlim,idx+1,title,fontsize=12)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.axis('off')

    return fig,[ax1,ax2]

def gen_mpl_labels(
    adata, groupby, exclude=(), 
    basis='X_umap',ax=None, adjust_kwargs=None, text_kwargs=None
):
    """ 
    Get locations of cluster median . Borrowed from scanpy github forum.
    """
    if adjust_kwargs is None:
        adjust_kwargs = {"text_from_points": False}
    if text_kwargs is None:
        text_kwargs = {}

    medians = {}

    for g, g_idx in adata.obs.groupby(groupby).groups.items():
        if g in exclude:
            continue
        medians[g] = np.median(adata[g_idx].obsm[basis], axis=0)

    if ax is None:
        texts = [
            plt.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()
        ]
    else:
        texts = [ax.text(x=x, y=y, s=k, **text_kwargs) for k, (x, y) in medians.items()]
    from adjustText import adjust_text
    adjust_text(texts, **adjust_kwargs)

def plot_embedding(adata:anndata.AnnData,basis:str,color:str,color_dict=None,
                   figsize:tuple=(4,4),**kwargs):
    
    """
    Plot embedding with celltype color by omicverse

    Arguments:
        adata: AnnData object
        basis: embedding method
        color: celltype key in adata.obs
        figsize: figure size
        kwargs: other parameters for sc.pl.embedding

    Returns:
        fig : figure
        ax: axes
    
    """
    if type(color)!=str:
        print("Only one color could be input, don't input list")
        return
    fig,ax=plt.subplots(1,1,figsize=figsize)
    adata.obs[color]=adata.obs[color].astype('category')

    if '{}_colors'.format(color) in adata.uns.keys():
        print('{}_colors'.format(color))
        type_color_all=dict(zip(adata.obs[color].cat.categories,adata.uns['{}_colors'.format(color)]))
    else:
        if len(adata.obs[color].cat.categories)>28:
            type_color_all=dict(zip(adata.obs[color].cat.categories,sc.pl.palettes.default_102))
        else:
            type_color_all=dict(zip(adata.obs[color].cat.categories,sc.pl.palettes.zeileis_28))
    if color_dict is not None:
        for color_key in color_dict.keys():
            type_color_all[color_key]=color_dict[color_key]
    
    adata.uns['{}_colors'.format(color)]=np.array([i for i in type_color_all.values()])
    sc.pl.embedding(adata,basis=basis,
                    color=color,ax=ax,**kwargs)
    return fig,ax

from sklearn.preprocessing import MinMaxScaler

def normalize_to_minus_one_to_one(arr):
    # 将数组reshape为二维数组，因为MinMaxScaler接受二维数据
    arr = arr.reshape(-1, 1)
    
    # 创建MinMaxScaler对象，并设定归一化的范围为[-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # 对数据进行归一化处理
    normalized_arr = scaler.fit_transform(arr)
    
    # 将结果转换回一维数组
    normalized_arr = normalized_arr.flatten()
    
    return normalized_arr

def stacking_vol(data_dict:dict,color_dict:dict,
                 pval_threshold:float=0.01,
                 log2fc_threshold:int=2,
                 figsize:tuple=(8,4),
                 sig_color:str='#a51616',
                 normal_color:str='#c7c7c7',
                 plot_genes_num:int=10,
                 plot_genes_fontsize:int=8,
                plot_genes_weight:str='bold')->tuple:
    """
    Plot the stacking volcano plot for multiple omics

    Arguments:
        data_dict: dict, in each key, there is a dataframe with columns of ['logfoldchanges','pvals_adj','names']
        color_dict: dict, in each key, there is a color for each omic
        pval_threshold: float, pvalue threshold for significant genes
        log2fc_threshold: float, log2fc threshold for significant genes
        figsize: tuple, figure size
        sig_color: str, color for significant genes
        normal_color: str, color for non-significant genes
        plot_genes_num: int, number of genes to plot
        plot_genes_fontsize: int, fontsize for gene names
        plot_genes_weight: str, weight for gene names
    
    Returns:
        fig: figure
        axes: the dict of axes
    
    """
    
    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(len(data_dict.keys())*2, len(data_dict.keys())*2)
    axes={}
    j_before=0
    y_min,y_max=0,0
    for i in data_dict.keys():
        y_min=min(y_min,data_dict[i]['logfoldchanges'].min())
        y_max=max(y_max,data_dict[i]['logfoldchanges'].max())

    for i,j in zip(data_dict.keys(),
               range(2,len(data_dict.keys())*2+2,2)):
        print(j_before,j)
        axes[i]=fig.add_subplot(grid[:, j_before:j])
        j_before+=2
    
        x=np.random.normal(0, 1, data_dict[i].shape[0])
        x=normalize_to_minus_one_to_one(x)

        plot_data=pd.DataFrame()
        plot_data['logfoldchanges']=data_dict[i]['logfoldchanges']
        plot_data['pvals_adj']=data_dict[i]['pvals_adj']
        plot_data['abslogfoldchanges']=abs(data_dict[i]['logfoldchanges'])
        plot_data['sig']='normal'
        plot_data.loc[(plot_data['pvals_adj']<pval_threshold)&(plot_data['abslogfoldchanges']>log2fc_threshold),'sig']='sig'
        plot_data['x']=x
        plot_data.index=data_dict[i]['names']


        axes[i].scatter(plot_data.loc[plot_data['sig']!='sig','x'],
                   plot_data.loc[plot_data['sig']!='sig','logfoldchanges'],
                   color=normal_color,alpha=0.5)

        axes[i].scatter(plot_data.loc[plot_data['sig']=='sig','x'],
                   plot_data.loc[plot_data['sig']=='sig','logfoldchanges'],
                   color=sig_color,alpha=0.8)

        axes[i].axhspan(0-log2fc_threshold/2, log2fc_threshold/2, 
                        facecolor=color_dict[i], alpha=1)

        
        axes[i].set_ylim(y_min,y_max)

        plt.grid(False)
        plt.yticks(fontsize=12)

        axes[i].spines['top'].set_visible(False)
        if j_before!=2:
            axes[i].spines['left'].set_visible(False)
            axes[i].axis('off')
        axes[i].spines['bottom'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_ylim(-15,15)
        axes[i].set_xticks([])
        
        hub_gene=plot_data.loc[plot_data['sig']=='sig'].sort_values('abslogfoldchanges',
                                                                    ascending=False).index[:plot_genes_num]
        from adjustText import adjust_text
        texts=[axes[i].text(plot_data.loc[gene,'x'], 
                            plot_data.loc[gene,'logfoldchanges'],
                            gene,
                            fontdict={'size':plot_genes_fontsize,
                                    'weight':plot_genes_weight,
                                     'color':'black'}) 
               for gene in hub_gene]
        adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)
            
    return fig,axes