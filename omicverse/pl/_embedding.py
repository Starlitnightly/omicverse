import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

def embedding_atlas(adata,basis,color,
                    title=None,figsize=(4,4),ax=None,cmap='RdBu',
                    legend_loc = 'right margin',frameon='small',
                    fontsize=12):
    r"""
    Create high-resolution embedding plots using Datashader for large datasets.
    
    Uses Datashader to render embeddings at high resolution, suitable for datasets
    with millions of cells where standard scatter plots become ineffective.
    
    Args:
        adata: Annotated data object with embedding coordinates
        basis: Key in adata.obsm containing embedding coordinates (e.g., 'X_umap')
        color: Gene name or obs column to color cells by
        title: Plot title (None, uses color name)
        figsize: Figure dimensions as (width, height) ((4,4))
        ax: Existing matplotlib axes object (None)
        cmap: Colormap for continuous values ('RdBu')
        legend_loc: Legend position ('right margin')
        frameon: Frame style - False, 'small', or True ('small')
        fontsize: Font size for labels and title (12)
        
    Returns:
        ax: matplotlib.axes.Axes object with rendered embedding
    """
    import scanpy as sc
    import pandas as pd
    import datashader as ds
    import datashader.transfer_functions as tf
    from scipy.sparse import issparse
    from bokeh.palettes import RdBu9
    import bokeh
    # 创建一个 Canvas 对象
    cvs = ds.Canvas(plot_width=800, plot_height=800)
    
    
    embedding = adata.obsm[basis]
    # 如果你有一个感兴趣的分类标签，比如细胞类型
    
    # 将数据转换为 DataFrame
    df = pd.DataFrame(embedding, columns=['x', 'y'])

    if color in adata.obs.columns:
        labels = adata.obs[color].tolist()  # 假设'cell_type'是一个列名
    elif color in adata.var_names:
        X=adata[:,color].X
        if issparse(X):
            labels=X.toarray().reshape(-1)
        else:
            labels=X.reshape(-1)
    elif (not adata.raw is None) and (color in adata.raw.var_names):
        X=adata.raw[:,color].X
        if issparse(X):
            labels=X.toarray().reshape(-1)
        else:
            labels=X.reshape(-1)

    
    df['label'] = labels
    #return labels
    #print(labels[0],type(labels[0]))
    if type(labels[0]) is str:
        df['label']=df['label'].astype('category')
        # 聚合数据
        agg = cvs.points(df, 'x', 'y',ds.count_cat('label'),
                        )
        legend_tag=True
        color_key = dict(zip(adata.obs[color].cat.categories,
                        adata.uns[f'{color}_colors']))
        
        
        # 使用色彩映射
        img = tf.shade(tf.spread(agg,px=0),color_key=[color_key[i] for i in df['label'].cat.categories], 
                       how='eq_hist')
    elif (type(labels[0]) is int) or (type(labels[0]) is float) or (type(labels[0]) is np.float32)\
    or (type(labels[0]) is np.float64) or (type(labels[0]) is np.int64):
        # 聚合数据
        agg = cvs.points(df, 'x', 'y',ds.mean('label'),
                        )
        legend_tag=False
        if cmap in bokeh.palettes.all_palettes.keys():
            num=list(bokeh.palettes.all_palettes[cmap].keys())[-1]
            img = tf.shade(agg,cmap=bokeh.palettes.all_palettes[cmap][num], 
                           )
        else:
            img = tf.shade(agg,cmap=cmap, 
                           )
    else:
        print('Unrecognized label type')
        return None
    
    
    
        # 假设 img 是 Datashader 渲染的图像
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig=ax.figure
    
    # 假设 img 是一个 NumPy 数组或类似的对象，这里使用 img 的占位符
    # img = np.random.rand(100, 100)  # 示例数据
    ax.imshow(img.to_pil(), aspect='auto')
    
    
    # 自定义格式化函数以显示坐标
    def format_coord(x, y):
        return f"x={x:.2f}, y={y:.2f}"
    
    ax.format_coord = format_coord

    if legend_tag==True:
        # 手动创建图例
        unique_labels = adata.obs[color].cat.categories
        
        # 创建图例项
        for label in unique_labels:
            ax.scatter([], [], c=color_key[label], label=label)
        
        if legend_loc == "right margin":
            ax.legend(
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                ncol=(1 if len(unique_labels) <= 14 else 2 if len(unique_labels) <= 30 else 3),
                fontsize=fontsize-1,
            )
    if frameon==False:
        ax.axis('off')
    elif frameon=='small':
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_bounds(0,150)
        ax.spines['left'].set_bounds(650,800)
        ax.set_xlabel(f'{basis}1',loc='left',fontsize=fontsize)
        ax.set_ylabel(f'{basis}2',loc='bottom',fontsize=fontsize)

    else:
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.set_xlabel(f'{basis}1',loc='center',fontsize=fontsize)
        ax.set_ylabel(f'{basis}2',loc='center',fontsize=fontsize)

    
    # 调整坐标轴线的粗细
    line_width = 1.2  # 设置线宽
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)
    
    
    if title is None:
        title=color
    ax.set_title(title,fontsize=fontsize+1)

    return ax


