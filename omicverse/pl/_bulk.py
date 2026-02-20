import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
from .._registry import register_function

@register_function(
    aliases=["火山图", "volcano", "volcano_plot", "差异基因可视化", "火山图绘制"],
    category="pl",
    description="Create volcano plot for differential expression analysis visualization",
    examples=[
        "# Basic volcano plot",
        "ov.pl.volcano(deg_results, pval_name='qvalue', fc_name='log2FoldChange')",
        "# Customized volcano plot",
        "ov.pl.volcano(deg_results, pval_threshold=0.05, fc_max=2, fc_min=-2,",
        "             title='DEGs Analysis', figsize=(6,6))",
        "# With gene labels",
        "ov.pl.volcano(deg_results, plot_genes=['GENE1', 'GENE2'], plot_genes_num=15)"
    ],
    related=["bulk.pyDEG", "bulk.pyDEG.deg_analysis"]
)
def volcano(result,pval_name='qvalue',fc_name='log2FC',pval_max=None,FC_max=None,
            figsize:tuple=(4,4),title:str='',titlefont:dict={'weight':'normal','size':14,},
                     up_color:str='#e25d5d',down_color:str='#7388c1',normal_color:str='#d7d7d7',
                     up_fontcolor:str='#e25d5d',down_fontcolor:str='#7388c1',normal_fontcolor:str='#d7d7d7',
                     legend_bbox:tuple=(0.8, -0.2),legend_ncol:int=2,legend_fontsize:int=12,
                     plot_genes:list=None,plot_genes_num:int=10,plot_genes_fontsize:int=10,
                     ticks_fontsize:int=12,pval_threshold:float=0.05,fc_max:float=1.5,fc_min:float=-1.5,
                     ax = None,):
    r"""
    Create a volcano plot for differential expression analysis.
    
    Args:
        result: pandas.DataFrame
            Dataframe containing differential expression results with 'sig' column
        pval_name: str, optional (default='qvalue')
            Column name for p-values/q-values
        fc_name: str, optional (default='log2FC')
            Column name for fold change values  
        pval_max: float, optional (default=None)
            Maximum p-value for y-axis scaling
        FC_max: float, optional (default=None)
            Maximum fold change for x-axis scaling
        figsize: tuple, optional (default=(4,4))
            Figure size (width, height)
        title: str, optional (default='')
            Plot title
        titlefont: dict, optional (default={'weight':'normal','size':14})
            Font settings for title and axis labels
        up_color: str, optional (default='#e25d5d')
            Color for upregulated genes
        down_color: str, optional (default='#7388c1')
            Color for downregulated genes
        normal_color: str, optional (default='#d7d7d7')
            Color for non-significant genes
        up_fontcolor: str, optional (default='#e25d5d')
            Font color for upregulated gene labels
        down_fontcolor: str, optional (default='#7388c1')
            Font color for downregulated gene labels
        normal_fontcolor: str, optional (default='#d7d7d7')
            Font color for normal gene labels
        legend_bbox: tuple, optional (default=(0.8, -0.2))
            Legend bounding box position
        legend_ncol: int, optional (default=2)
            Number of legend columns
        legend_fontsize: int, optional (default=12)
            Legend font size
        plot_genes: list, optional (default=None)
            Specific genes to label on plot
        plot_genes_num: int, optional (default=10)
            Number of top genes to label automatically
        plot_genes_fontsize: int, optional (default=10)
            Font size for gene labels
        ticks_fontsize: int, optional (default=12)
            Font size for axis ticks
        pval_threshold: float, optional (default=0.05)
            P-value threshold for significance
        fc_max: float, optional (default=1.5)
            Upper fold change threshold
        fc_min: float, optional (default=-1.5)
            Lower fold change threshold
        ax: matplotlib.axes, optional (default=None)
            Existing axes to plot on
        
    Returns:
        ax: matplotlib.axes
            The plot axes object
    """
    
    # Color codes for terminal output
    class Colors:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    # Analyze the input data
    print(f"{Colors.HEADER}{Colors.BOLD}🌋 Volcano Plot Analysis:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Total genes: {Colors.BOLD}{len(result)}{Colors.ENDC}")
    
    # Check required columns
    required_cols = [pval_name, fc_name, 'sig']
    missing_cols = [col for col in required_cols if col not in result.columns]
    if missing_cols:
        print(f"   {Colors.FAIL}❌ Missing required columns: {Colors.BOLD}{missing_cols}{Colors.ENDC}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate gene counts by significance
    sig_counts = result['sig'].value_counts()
    total_sig = sig_counts.get('up', 0) + sig_counts.get('down', 0)
    
    print(f"   {Colors.GREEN}↗️  Upregulated genes: {Colors.BOLD}{sig_counts.get('up', 0)}{Colors.ENDC}")
    print(f"   {Colors.BLUE}↘️  Downregulated genes: {Colors.BOLD}{sig_counts.get('down', 0)}{Colors.ENDC}")
    print(f"   {Colors.CYAN}➡️  Non-significant genes: {Colors.BOLD}{sig_counts.get('normal', 0)}{Colors.ENDC}")
    print(f"   {Colors.WARNING}🎯 Total significant genes: {Colors.BOLD}{total_sig}{Colors.ENDC}")
    
    # Data range information
    fc_range = result[fc_name].max() - result[fc_name].min()
    pval_range = result[pval_name].max() - result[pval_name].min()
    print(f"   {Colors.BLUE}{fc_name} range: {Colors.BOLD}{result[fc_name].min():.2f} to {result[fc_name].max():.2f}{Colors.ENDC}")
    print(f"   {Colors.BLUE}{pval_name} range: {Colors.BOLD}{result[pval_name].min():.2e} to {result[pval_name].max():.2e}{Colors.ENDC}")
    
    # Display current function parameters
    print(f"\n{Colors.HEADER}{Colors.BOLD}⚙️  Current Function Parameters:{Colors.ENDC}")
    print(f"   {Colors.BLUE}Data columns: pval_name='{pval_name}', fc_name='{fc_name}'{Colors.ENDC}")
    print(f"   {Colors.BLUE}Thresholds: pval_threshold={Colors.BOLD}{pval_threshold}{Colors.ENDC}{Colors.BLUE}, fc_max={Colors.BOLD}{fc_max}{Colors.ENDC}{Colors.BLUE}, fc_min={Colors.BOLD}{fc_min}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Plot size: figsize={Colors.BOLD}{figsize}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Gene labels: plot_genes_num={Colors.BOLD}{plot_genes_num}{Colors.ENDC}{Colors.BLUE}, plot_genes_fontsize={Colors.BOLD}{plot_genes_fontsize}{Colors.ENDC}")
    if plot_genes is not None:
        print(f"   {Colors.BLUE}Custom genes: {Colors.BOLD}{len(plot_genes)} specified{Colors.ENDC}")
    else:
        print(f"   {Colors.BLUE}Custom genes: {Colors.BOLD}None{Colors.ENDC}{Colors.BLUE} (auto-select top genes){Colors.ENDC}")
    
    # Parameter optimization suggestions
    print(f"\n{Colors.HEADER}{Colors.BOLD}💡 Parameter Optimization Suggestions:{Colors.ENDC}")
    suggestions = []
    
    # Check if there are enough significant genes
    if total_sig < 10:
        suggestions.append(f"   {Colors.WARNING}▶ Few significant genes detected ({total_sig}):{Colors.ENDC}")
        suggestions.append(f"     {Colors.CYAN}Current: pval_threshold={Colors.BOLD}{pval_threshold}{Colors.ENDC}")
        suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}pval_threshold=0.1{Colors.ENDC} or {Colors.BOLD}pval_threshold=0.2{Colors.ENDC}")
    
    # Check fold change thresholds
    if fc_range > 10 and (fc_max <= 2 or abs(fc_min) <= 2):
        new_fc_max = min(round(result[fc_name].quantile(0.95), 1), 4.0)
        new_fc_min = max(round(result[fc_name].quantile(0.05), 1), -4.0)
        suggestions.append(f"   {Colors.WARNING}▶ Wide fold change range detected:{Colors.ENDC}")
        suggestions.append(f"     {Colors.CYAN}Current: fc_max={Colors.BOLD}{fc_max}{Colors.ENDC}{Colors.CYAN}, fc_min={Colors.BOLD}{fc_min}{Colors.ENDC}")
        suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}fc_max={new_fc_max}, fc_min={new_fc_min}{Colors.ENDC}")
    
    # Check plot size based on gene label settings
    if plot_genes_num > 20 and figsize[0] < 6:
        suggestions.append(f"   {Colors.WARNING}▶ Many gene labels with small plot:{Colors.ENDC}")
        suggestions.append(f"     {Colors.CYAN}Current: plot_genes_num={Colors.BOLD}{plot_genes_num}{Colors.ENDC}{Colors.CYAN}, figsize={Colors.BOLD}{figsize}{Colors.ENDC}")
        suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}figsize=(6, 6){Colors.ENDC} or {Colors.BOLD}plot_genes_num=15{Colors.ENDC}")
    
    # Check if gene labels might be too small
    if plot_genes_fontsize < 8 and plot_genes_num > 15:
        suggestions.append(f"   {Colors.WARNING}▶ Small font with many labels:{Colors.ENDC}")
        suggestions.append(f"     {Colors.CYAN}Current: plot_genes_fontsize={Colors.BOLD}{plot_genes_fontsize}{Colors.ENDC}{Colors.CYAN}, plot_genes_num={Colors.BOLD}{plot_genes_num}{Colors.ENDC}")
        suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}plot_genes_fontsize=10{Colors.ENDC} or {Colors.BOLD}plot_genes_num=10{Colors.ENDC}")
    
    # Check figure aspect ratio
    if abs(figsize[0] - figsize[1]) > 2:
        suggestions.append(f"   {Colors.BLUE}▶ Unbalanced figure dimensions:{Colors.ENDC}")
        suggestions.append(f"     {Colors.CYAN}Current: figsize={Colors.BOLD}{figsize}{Colors.ENDC}")
        optimal_size = max(figsize)
        suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}figsize=({optimal_size}, {optimal_size}){Colors.ENDC}")
    
    if suggestions:
        for suggestion in suggestions:
            print(suggestion)
        print(f"\n   {Colors.BOLD}📋 Copy-paste ready function call:{Colors.ENDC}")
        
        # Generate optimized function call
        optimized_params = ["result"]
        
        # Add data column parameters if different from defaults
        if pval_name != 'qvalue':
            optimized_params.append(f"pval_name='{pval_name}'")
        if fc_name != 'log2FC':
            optimized_params.append(f"fc_name='{fc_name}'")
            
        # Add optimized parameters based on suggestions
        if total_sig < 10:
            optimized_params.append("pval_threshold=0.1")
        
        if fc_range > 10 and (fc_max <= 2 or abs(fc_min) <= 2):
            new_fc_max = min(round(result[fc_name].quantile(0.95), 1), 4.0)
            new_fc_min = max(round(result[fc_name].quantile(0.05), 1), -4.0)
            optimized_params.append(f"fc_max={new_fc_max}")
            optimized_params.append(f"fc_min={new_fc_min}")
        
        if plot_genes_num > 20 and figsize[0] < 6:
            optimized_params.append("figsize=(6, 6)")
        elif abs(figsize[0] - figsize[1]) > 2:
            optimal_size = max(figsize)
            optimized_params.append(f"figsize=({optimal_size}, {optimal_size})")
        
        if plot_genes_fontsize < 8 and plot_genes_num > 15:
            optimized_params.append("plot_genes_fontsize=10")
        
        if plot_genes is not None:
            optimized_params.append(f"plot_genes={plot_genes}")
            
        optimized_call = f"   {Colors.GREEN}ov.pl.volcano({', '.join(optimized_params)}){Colors.ENDC}"
        print(optimized_call)
    else:
        print(f"   {Colors.GREEN}✅ Current parameters are optimal for your data!{Colors.ENDC}")
    
    print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")

    # Original volcano plot code starts here
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


    from adjustText import adjust_text
    import adjustText

    if plot_genes is not None:
        hub_gene=plot_genes
    elif (plot_genes is None) and (plot_genes_num is None):
        return ax
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

    
    return ax

@register_function(
    aliases=["韦恩图", "venn", "venn_diagram", "集合可视化", "韦恩图绘制"],
    category="pl",
    description="Create Venn diagram to visualize set overlaps and intersections",
    examples=[
        "# Basic Venn diagram",
        "sets = {'Set1': {1,2,3}, 'Set2': {2,3,4}, 'Set3': {3,4,5}}",
        "ov.pl.venn(sets=sets)",
        "# Customized Venn diagram",
        "ov.pl.venn(sets=sets, palette=ov.pl.red_color, fontsize=8)",
        "# Gene set comparison",
        "gene_sets = {'UP_genes': up_genes, 'DOWN_genes': down_genes}",
        "ov.pl.venn(sets=gene_sets, figsize=(6,6))"
    ],
    related=["bulk.pyDEG.deg_analysis"]
)
def venn(sets={}, out='./', palette='bgrc',
             ax=False, ext='png', dpi=300, fontsize=3.5,
             bbox_to_anchor=(.5, .99),nc=2,cs=4):
    r"""
    Create a Venn diagram to visualize set overlaps.
    
    Args:
        sets: Dictionary with set names as keys and sets as values ({})
        out: Output directory path ('./')
        palette: Color palette for sets ('bgrc')
        ax: Matplotlib axes object or False to create new (False)
        ext: File extension for output ('png')
        dpi: Resolution for output image (300)
        fontsize: Font size for text (3.5)
        bbox_to_anchor: Legend position ((.5, .99))
        nc: Number of legend columns (2)
        cs: Font size for legend (4)
        
    Returns:
        ax: matplotlib.axes.Axes object
    """
    
    from ..utils import venny4py
    venny4py(sets=sets,out=out,ce=palette,asax=ax,ext=ext,
             dpi=dpi,size=fontsize,bbox_to_anchor=bbox_to_anchor,
             nc=nc,cs=cs,
             )
    return ax

@register_function(
    aliases=["箱线图", "boxplot", "box_plot", "盒图", "箱形图"],
    category="pl",
    description="Create boxplot with jittered points for data distribution visualization",
    examples=[
        "# Basic boxplot",
        "import seaborn as sns",
        "data = sns.load_dataset('tips')",
        "ov.pl.boxplot(data, hue='sex', x_value='day', y_value='total_bill')",
        "# Customized boxplot",
        "ov.pl.boxplot(data, hue='time', x_value='day', y_value='tip',",
        "             palette=ov.pl.blue_color, figsize=(8,4))",
        "# With statistical annotation",
        "fig, ax = ov.pl.boxplot(data, hue='smoker', x_value='time', y_value='total_bill')",
        "ov.pl.add_palue(ax, line_x1=0, line_x2=1, line_y=40, text='p<0.001')"
    ],
    related=["pl.add_palue"]
)
def boxplot(data,hue,x_value,y_value,width=0.3,title='',
                 figsize=(6,3),palette=None,fontsize=10,
                 legend_bbox=(1, 0.55),legend_ncol=1,hue_order=None):
    r"""
    Create a boxplot with jittered points to visualize data distribution across categories.
    
    Args:
        data: DataFrame containing the data to plot
        hue: Column name for grouping variable (color coding)
        x_value: Column name for x-axis categories
        y_value: Column name for y-axis values
        width: Width of each boxplot (0.3)
        title: Plot title ('')
        figsize: Figure dimensions as (width, height) ((6,3))
        palette: Color palette for groups (None, uses default)
        fontsize: Font size for labels and ticks (10)
        legend_bbox: Legend position as (x, y) ((1, 0.55))
        legend_ncol: Number of legend columns (1)
        hue_order: Custom order for hue categories (None, uses alphabetical)
        
    Returns:
        fig: matplotlib.figure.Figure object
        ax: matplotlib.axes.Axes object
    """

    # Color codes for terminal output
    class Colors:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    # Print data information for user guidance
    print(f"{Colors.HEADER}{Colors.BOLD}📊 Boxplot Data Analysis:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Total samples: {Colors.BOLD}{len(data)}{Colors.ENDC}")
    print(f"   {Colors.BLUE}X-axis variable ('{x_value}'): {Colors.BOLD}{sorted(data[x_value].unique())}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Hue variable ('{hue}'): {Colors.BOLD}{sorted(data[hue].unique())}{Colors.ENDC}")
    print(f"   {Colors.BLUE}Y-axis variable: '{y_value}' (range: {Colors.BOLD}{data[y_value].min():.2f} - {data[y_value].max():.2f}{Colors.ENDC})")
    
    # Check for missing data
    missing_data = data[[hue, x_value, y_value]].isnull().sum().sum()
    if missing_data > 0:
        print(f"   {Colors.WARNING}⚠️  Warning: Found {Colors.BOLD}{missing_data}{Colors.ENDC}{Colors.WARNING} missing values in key columns{Colors.ENDC}")
    
    # Display current function parameters
    print(f"\n{Colors.HEADER}{Colors.BOLD}⚙️  Current Function Parameters:{Colors.ENDC}")
    print(f"   {Colors.BLUE}hue='{hue}', x_value='{x_value}', y_value='{y_value}'{Colors.ENDC}")
    print(f"   {Colors.BLUE}width={Colors.BOLD}{width}{Colors.ENDC}{Colors.BLUE}, figsize={Colors.BOLD}{figsize}{Colors.ENDC}{Colors.BLUE}, fontsize={Colors.BOLD}{fontsize}{Colors.ENDC}")
    if hue_order is not None:
        print(f"   {Colors.BLUE}hue_order={Colors.BOLD}{hue_order}{Colors.ENDC}")
    else:
        print(f"   {Colors.BLUE}hue_order={Colors.BOLD}None{Colors.ENDC}{Colors.BLUE} (using alphabetical order){Colors.ENDC}")
    
    def calculate_box_positions(n_hues, spacing=0.8):
        """
        Calculate evenly distributed positions for boxes within the range [-0.5, 0.5].
        
        Args:
            n_hues: int
                Number of hue categories
            spacing: float
                Fraction of the total range to use for spacing boxes (0.8 means use 80% of [-0.5, 0.5])
            
        Returns:
            positions: list
                List of positions for each hue
        """
        if n_hues == 1:
            return [0.0]
        
        # Calculate the range to use for positioning
        total_range = spacing  # Use 80% of the [-0.5, 0.5] range by default
        half_range = total_range / 2
        
        # Calculate positions evenly distributed within the range
        if n_hues > 1:
            step = total_range / (n_hues - 1)
            positions = [-half_range + i * step for i in range(n_hues)]
        else:
            positions = [0.0]
            
        return positions

    #获取需要分割的数据
    if hue_order is not None:
        hue_datas = hue_order
        # Check if all hue values in data are in hue_order
        data_hue_values = set(data[hue].unique())
        hue_order_set = set(hue_order)
        if not data_hue_values.issubset(hue_order_set):
            missing_values = data_hue_values - hue_order_set
            raise ValueError(f"The following hue values are in data but not in hue_order: {missing_values}")
        print(f"   {Colors.GREEN}📋 Using custom hue order: {Colors.BOLD}{hue_order}{Colors.ENDC}")
    else:
        hue_datas = sorted(list(set(data[hue])))
        print(f"   {Colors.GREEN}📋 Using alphabetical hue order: {Colors.BOLD}{hue_datas}{Colors.ENDC}")

    #获取箱线图的横坐标
    x=x_value
    ticks=sorted(list(set(data[x])))

    # Calculate box positions
    box_positions = calculate_box_positions(len(hue_datas))
    print(f"\n{Colors.HEADER}{Colors.BOLD}🎯 Box Positioning:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Number of hue groups: {Colors.BOLD}{len(hue_datas)}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Box positions: {Colors.BOLD}{[round(pos, 3) for pos in box_positions]}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Box width: {Colors.BOLD}{width}{Colors.ENDC}")
    
    # Calculate sample sizes for each combination
    print(f"\n{Colors.HEADER}{Colors.BOLD}📈 Sample sizes per group:{Colors.ENDC}")
    for hue_cat in hue_datas:
        for x_cat in ticks:
            count = len(data[(data[hue] == hue_cat) & (data[x] == x_cat)])
            if count < 5:
                color = Colors.WARNING
            elif count < 10:
                color = Colors.BLUE
            else:
                color = Colors.GREEN
            print(f"   {color}{hue_cat} × {x_cat}: {Colors.BOLD}{count}{Colors.ENDC}{color} samples{Colors.ENDC}")
    
    # Provide parameter suggestions with current vs suggested comparison
    print(f"\n{Colors.HEADER}{Colors.BOLD}💡 Parameter Optimization Suggestions:{Colors.ENDC}")
    suggestions = []
    
    if len(hue_datas) > 4:
        suggested_width = round(max(0.1, 0.8 / len(hue_datas)), 1)
        suggested_figsize_width = max(8, len(ticks) * 2)
        suggestions.append(f"   {Colors.WARNING}▶ Many hue groups detected:{Colors.ENDC}")
        suggestions.append(f"     {Colors.CYAN}Current: {Colors.BOLD}width={width}{Colors.ENDC}{Colors.CYAN}, figsize={figsize}{Colors.ENDC}")
        suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}width={suggested_width}, figsize=({suggested_figsize_width}, {figsize[1]}){Colors.ENDC}")
    
    if len(ticks) > 5:
        suggested_figsize_width = max(10, len(ticks) * 1.5)
        suggestions.append(f"   {Colors.WARNING}▶ Many x-categories detected:{Colors.ENDC}")
        suggestions.append(f"     {Colors.CYAN}Current: {Colors.BOLD}figsize={figsize}{Colors.ENDC}")
        suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}figsize=({suggested_figsize_width}, {figsize[1]}){Colors.ENDC}")
    
    max_samples = max([len(data[(data[hue] == h) & (data[x] == x_cat)]) for h in hue_datas for x_cat in ticks])
    if max_samples < 5:
        suggested_width = round(max(0.1, width * 0.7), 1)
        suggestions.append(f"   {Colors.WARNING}▶ Small sample sizes detected:{Colors.ENDC}")
        suggestions.append(f"     {Colors.CYAN}Current: {Colors.BOLD}width={width}{Colors.ENDC}")
        suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}width={suggested_width}{Colors.ENDC}")
    
    # Check if current width might cause overlap
    if len(hue_datas) > 3 and width > 0.25:
        suggested_width = round(0.8 / len(hue_datas), 1)
        suggestions.append(f"   {Colors.FAIL}▶ Box overlap risk detected:{Colors.ENDC}")
        suggestions.append(f"     {Colors.CYAN}Current: {Colors.BOLD}width={width}{Colors.ENDC}{Colors.CYAN} (too wide for {len(hue_datas)} groups){Colors.ENDC}")
        suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}width={suggested_width}{Colors.ENDC}")
    
    # Figure size optimization based on both dimensions
    if len(ticks) > 3 and len(hue_datas) > 3:
        suggested_width = max(8, len(ticks) * 1.5)
        suggested_height = max(4, figsize[1])
        suggestions.append(f"   {Colors.BLUE}▶ Complex plot optimization:{Colors.ENDC}")
        suggestions.append(f"     {Colors.CYAN}Current: {Colors.BOLD}figsize={figsize}{Colors.ENDC}")
        suggestions.append(f"     {Colors.GREEN}Suggested: {Colors.BOLD}figsize=({suggested_width}, {suggested_height}){Colors.ENDC}")
    
    if suggestions:
        for suggestion in suggestions:
            print(suggestion)
        print(f"\n   {Colors.BOLD}📋 Copy-paste ready function call:{Colors.ENDC}")
        # Generate optimized function call
        optimized_params = []
        optimized_params.append(f"data, hue='{hue}', x_value='{x_value}', y_value='{y_value}'")
        
        # Add optimized parameters
        if len(hue_datas) > 4 or (len(hue_datas) > 3 and width > 0.25) or max_samples < 5:
            if len(hue_datas) > 4:
                opt_width = round(max(0.1, 0.8 / len(hue_datas)), 1)
            elif max_samples < 5:
                opt_width = round(max(0.1, width * 0.7), 1)
            else:
                opt_width = round(0.8 / len(hue_datas), 1)
            optimized_params.append(f"width={opt_width}")
        
        if len(ticks) > 5 or len(hue_datas) > 4 or (len(ticks) > 3 and len(hue_datas) > 3):
            if len(hue_datas) > 4:
                opt_fig_w = max(8, len(ticks) * 2)
            elif len(ticks) > 5:
                opt_fig_w = max(10, len(ticks) * 1.5)
            else:
                opt_fig_w = max(8, len(ticks) * 1.5)
            opt_fig_h = max(4, figsize[1])
            optimized_params.append(f"figsize=({opt_fig_w}, {opt_fig_h})")
        
        if hue_order is not None:
            optimized_params.append(f"hue_order={hue_order}")
            
        optimized_call = f"   {Colors.GREEN}ov.pl.boxplot({', '.join(optimized_params)}){Colors.ENDC}"
        print(optimized_call)
    else:
        print(f"   {Colors.GREEN}✅ Current parameters are optimal for your data!{Colors.ENDC}")
    
    print(f"{Colors.CYAN}{'─' * 60}{Colors.ENDC}")

    #在这个数据中，我们有6个不同的癌症，每个癌症都有2个基因（2个箱子）
    #所以我们需要得到每一个基因的6个箱线图位置，6个散点图的抖动
    plot_data1={}#字典里的每一个元素就是每一个基因的所有值
    plot_data_random1={}#字典里的每一个元素就是每一个基因的随机20个值
    plot_data_xs1={}#字典里的每一个元素就是每一个基因的20个抖动值


    #箱子的参数
    #width=0.6
    y=y_value
    import random
    for hue_data,num in zip(hue_datas,box_positions):
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
            if len(test_data) > 0:
                random_data=random.sample(test_data,data_size)
            else:
                random_data=[]
            data_a_random.append(random_data)
            data_a_xs.append(np.random.normal(k+num, 0.04, len(random_data)))
        #data_a=np.array(data_a)
        data_a_random=np.array(data_a_random,dtype=object)
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
    for hue_data,hue_color,num in zip(hue_datas,palette,box_positions):
        b1=ax.boxplot(plot_data1[hue_data], 
                    positions=np.array(range(len(ticks)))+num, 
                    sym='', 
                    widths=width,)
        plt.setp(b1['boxes'], color=hue_color)
        plt.setp(b1['whiskers'], color=hue_color)
        plt.setp(b1['caps'], color=hue_color)
        plt.setp(b1['medians'], color=hue_color)

        clevels = np.linspace(0., 1., len(plot_data_random1[hue_data]))
        for x, val, clevel in zip(plot_data_xs1[hue_data], plot_data_random1[hue_data], clevels):
            if len(val) > 0:  # Only plot if there's data
                plt.scatter(x, val,c=hue_color,alpha=0.4)

    #坐标轴字体
    #fontsize=10
    #修改横坐标
    ax.set_xticks(range(len(ticks)), ticks,fontsize=fontsize)
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


def plot_grouped_fractions(res, obs, group_key, 
                           color_dict=None,agg='mean', normalize=True,
                           figsize=(4, 4),
                          ):
    """
    res: 行为 sample、列为 cell type 的预测比例表（你现在的 res）
    obs: bulk_ad.obs
    group_key: 例如 'severity' / 'disease_state' / 'gender' / 任意 obs 列
    agg: 'mean' / 'median' / 'sum'
    normalize: 是否把每组的细胞类型比例归一化到和为1
    """
    # 1) 对齐索引（样本名）
    common = res.index.intersection(obs.index)
    df = res.loc[common].copy()
    groups = obs.loc[common, group_key].copy()

    # 可选：去掉前缀（例如 "severity: Severe" -> "Severe"）
    if groups.dtype == 'object':
        groups = groups.astype(str).str.split(':').str[-1].str.strip()

    # 2) 聚合
    df[group_key] = groups.values
    if   agg == 'median': g = df.groupby(group_key).median(numeric_only=True)
    elif agg == 'sum':    g = df.groupby(group_key).sum(numeric_only=True)
    else:                 g = df.groupby(group_key).mean(numeric_only=True)

    # 3) 归一化到每组合计为1（按需）
    if normalize:
        g = g.div(g.sum(axis=1), axis=0).fillna(0)

    # 4) 列顺序与颜色一致
    ct_order=list(color_dict.keys())
    g = g.reindex(columns=ct_order, fill_value=0)
    colors = [color_dict[c] for c in g.columns]

    # 5) 画图
    ax = g.plot(kind='bar', stacked=True, figsize=figsize, color=colors)
    ax.set_xlabel(group_key)
    ax.set_ylabel('Cell Fraction')
    ax.set_title(f'Cell fractions grouped by {group_key} ({agg})')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Cell type', ncol=1)
    #plt.tight_layout()
    return ax