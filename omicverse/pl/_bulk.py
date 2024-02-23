import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib

def volcano(result,pval_name='qvalue',fc_name='log2FC',pval_max=None,FC_max=None,
            figsize:tuple=(4,4),title:str='',titlefont:dict={'weight':'normal','size':14,},
                     up_color:str='#e25d5d',down_color:str='#7388c1',normal_color:str='#d7d7d7',
                     up_fontcolor:str='#e25d5d',down_fontcolor:str='#7388c1',normal_fontcolor:str='#d7d7d7',
                     legend_bbox:tuple=(0.8, -0.2),legend_ncol:int=2,legend_fontsize:int=12,
                     plot_genes:list=None,plot_genes_num:int=10,plot_genes_fontsize:int=10,
                     ticks_fontsize:int=12,pval_threshold:float=0.05,fc_max:float=1.5,fc_min:float=-1.5,
                     ax = None,):
    result=result.copy()
    result['-log(qvalue)']=-np.log10(result[pval_name])
    result['log2FC']= result[fc_name].copy()
    if pval_max!=None:
        result.loc[result['-log(qvalue)']>pval_max,'-log(qvalue)']=pval_max
    if FC_max!=None:
        result.loc[result['log2FC']>FC_max,'log2FC']=FC_max
        result.loc[result['log2FC']<-FC_max,'log2FC']=0-FC_max
    
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x=result[result['sig']=='normal']['log2FC'],
            y=result[result['sig']=='normal']['-log(qvalue)'],
            color=normal_color,#颜色
            alpha=.5,#透明度
            )
    #接着绘制上调基因
    ax.scatter(x=result[result['sig']=='up']['log2FC'],
            y=result[result['sig']=='up']['-log(qvalue)'],
            color=up_color,#选择色卡第15个颜色
            alpha=.5,#透明度
            )
    #绘制下调基因
    ax.scatter(x=result[result['sig']=='down']['log2FC'],
            y=result[result['sig']=='down']['-log(qvalue)'],
            color=down_color,#颜色
            alpha=.5,#透明度
            )

    ax.plot([result['log2FC'].min(),result['log2FC'].max()],#辅助线的x值起点与终点
            [-np.log10(pval_threshold),-np.log10(pval_threshold)],#辅助线的y值起点与终点
            linewidth=2,#辅助线的宽度
            linestyle="--",#辅助线类型：虚线
            color='black'#辅助线的颜色
    )
    ax.plot([fc_max,fc_max],
            [result['-log(qvalue)'].min(),result['-log(qvalue)'].max()],
            linewidth=2, 
            linestyle="--",
            color='black')
    ax.plot([fc_min,fc_min],
            [result['-log(qvalue)'].min(),result['-log(qvalue)'].max()],
            linewidth=2, 
            linestyle="--",
            color='black')
    #设置横标签与纵标签
    ax.set_ylabel(r'$-log_{10}(qvalue)$',titlefont)                                    
    ax.set_xlabel(r'$log_{2}FC$',titlefont)
    #设置标题
    ax.set_title(title,titlefont)

    #绘制图注
    #legend标签列表，上面的color即是颜色列表
    labels = ['up:{0}'.format(len(result[result['sig']=='up'])),
            'down:{0}'.format(len(result[result['sig']=='down']))]  
    #用label和color列表生成mpatches.Patch对象，它将作为句柄来生成legend
    color = [up_color,down_color]
    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i]) ) for i in range(len(color))] 

    ax.legend(handles=patches,
        bbox_to_anchor=legend_bbox, 
        ncol=legend_ncol,
        fontsize=legend_fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    from adjustText import adjust_text
    import adjustText

    if plot_genes is not None:
        hub_gene=plot_genes
    else:
        up_result=result.loc[result['sig']=='up']
        down_result=result.loc[result['sig']=='down']
        hub_gene=up_result.sort_values(pval_name).index[:plot_genes_num//2].tolist()+down_result.sort_values(pval_name).index[:plot_genes_num//2].tolist()

    color_dict={
    'up':up_fontcolor,
        'down':down_fontcolor,
        'normal':normal_fontcolor
    }

    texts=[ax.text(result.loc[i,'log2FC'], 
        result.loc[i,'-log(qvalue)'],
        i,
        fontdict={'size':plot_genes_fontsize,'weight':'bold','color':color_dict[result.loc[i,'sig']]}
        ) for i in hub_gene]

    if adjustText.__version__<='0.8':
        adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)
    else:
        adjust_text(texts,only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                    arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xticks([round(i,2) for i in ax.get_xticks()[1:-1]],#获取x坐标轴内容
        [round(i,2) for i in ax.get_xticks()[1:-1]],#更新x坐标轴内容
        fontsize=ticks_fontsize,
        fontweight='normal'
        )

    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    return ax

def venn(sets={}, out='./', palette='bgrc',
             ax=False, ext='png', dpi=300, fontsize=3.5):
    
    from ..utils import venny4py
    venny4py(sets=sets,out=out,ce=palette,asax=ax,ext=ext,
             dpi=dpi,size=fontsize)
    return ax

def boxplot(data,hue,x_value,y_value,width=0.6,title='',
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
    from ..utils import ticks_range
    import random
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
        from ._palette import sc_color
        palette=sc_color
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
    ax.set_title(title,fontsize=fontsize+1)
    #设置spines可视化情况
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    
    return fig,ax