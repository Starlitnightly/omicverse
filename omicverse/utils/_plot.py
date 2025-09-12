import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib import rcParams
import random
import scanpy as sc
import networkx as nx
import pandas as pd
import anndata
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import seaborn as sns
from datetime import datetime,timedelta
import tomli
import os
from typing import Union


from datetime import datetime, timedelta
import warnings
import platform
import os
from .registry import register_function
try:
    import torch  # Optional, used for GPU information
except ImportError:  # pragma: no cover - optional dependency
    torch = None

# Global variable to control vector-friendly rasterization
_vector_friendly = True

sc_color=[
 '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', 
 '#FCBC10', '#EF7B77', '#279AD7','#F0EEF0',
 '#EAEFC5', '#7CBB5F','#368650','#A499CC','#5E4D9A',
 '#78C2ED','#866017', '#9F987F','#E0DFED',
 '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48',
 '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']

red_color=['#F0C3C3','#E07370','#CB3E35','#A22E2A','#5A1713',
           '#D3396D','#8B0000', '#A52A2A', '#CD5C5C', '#DC143C' ]

green_color=['#91C79D','#8FC155','#56AB56','#2D5C33','#BBCD91',
             '#6E944A','#A5C953','#3B4A25','#010000']

orange_color=['#EFBD49','#D48F3E','#AC8A3E','#7D7237','#745228',
              '#E1C085','#CEBC49','#EBE3A1','#6C6331','#8C9A48','#D7DE61']

blue_color=['#1F577B', '#279AD7', '#78C2ED', '#01A0A7', '#75C8CC', '#9DC3C3',
            '#3E8CB1', '#52B3AD', '#265B58', '#5860A7', '#312C6C', '#4CC9F0']

purple_color=['#823d86','#825b94','#bb98c6','#c69bc6','#a69ac9',
              '#c5a6cc','#caadc4','#d1c3d4']

#more beautiful colors
# 28-color palettes with distinct neighboring colors
palette_28 = sc_color[:28]
# 56-color palette with clear transitions

# 112-color palette with distinct transitions
cet_g_bw = [
 '#d60000', '#8c3bff', '#018700', '#00acc6', '#97ff00', '#ff7ed1', '#6b004f', '#ffa52f', '#00009c', '#857067',
 '#004942', '#4f2a00', '#00fdcf', '#bcb6ff', '#95b379', '#bf03b8', '#2466a1', '#280041', '#dbb3af', '#fdf490',
 '#4f445b', '#a37c00', '#ff7066', '#3f806e', '#82000c', '#a37bb3', '#344d00', '#9ae4ff', '#eb0077', '#2d000a',
 '#5d90ff', '#00c61f', '#5701aa', '#001d00', '#9a4600', '#959ea5', '#9a425b', '#001f31', '#c8c300', '#ffcfff',
 '#00bd9a', '#3615ff', '#2d2424', '#df57ff', '#bde6bf', '#7e4497', '#524f3b', '#d86600', '#647438', '#c17287',
 '#6e7489', '#809c03', '#bd8a64', '#623338', '#cacdda', '#6beb82', '#213f69', '#a17eff', '#fd03ca', '#75bcfd',
 '#d8c382', '#cda3cd', '#6d4f00', '#006974', '#469e5d', '#93c6bf', '#f9ff00', '#bf5444', '#00643b', '#5b4fa8',
 '#521f64', '#4f5eff', '#7e8e77', '#b808f9', '#8a91c3', '#b30034', '#87607e', '#9e0075', '#ffddc3', '#500800',
 '#1a0800', '#4b89b5', '#00dfdf', '#c8fff9', '#2f3415', '#ff2646', '#ff97aa', '#03001a', '#c860b1', '#c3a136',
 '#7c4f3a', '#f99e77', '#566464', '#d193ff', '#2d1f69', '#411a34', '#af9397', '#629e99', '#bcdd7b', '#ff5d93',
 '#0f2823', '#b8bdac', '#743b64', '#0f000c', '#7e6ebc', '#9e6b3b', '#ff4600', '#7e0087', '#ffcd3d', '#2f3b42',
 '#fda5ff', '#89013d', '#752b01', '#0a8995', '#050052', '#8ed631', '#52c372', '#465970', '#570121', '#a52101',
 '#90934b', '#00421d', '#8000d1', '#2f263f', '#bf3883', '#f4ffd4', '#00d3ff', '#6900f7', '#9cbad1', '#79d8aa',
 '#69565d', '#006905', '#36369c', '#018246', '#441d18', '#07a5ef', '#ff802f', '#a754b8', '#675982', '#72ffff',
 '#d88701', '#bad3ff', '#8e362f', '#a7a080', '#007ce2', '#8e7e8e', '#994487', '#00f034', '#aeaac8', '#a06062',
 '#4b3a77', '#6b8282', '#f0dde6', '#ffbad3', '#38a523', '#b3ffa8', '#0c1107', '#d6526e', '#959efd', '#7c7e00',
 '#759eb8', '#db877e', '#111318', '#d482d4', '#9e00bf', '#dbefff', '#8eaa9a', '#706442', '#493b3d', '#084d5e',
 '#9cb844', '#d8ddd4', '#caff6b', '#b364eb', '#465d33', '#009e7c', '#c14100', '#4fbcba', '#d88ab1', '#5b72b5',
 '#4b4101', '#95825d', '#49748a', '#ff72ff', '#82691c', '#dbcfff', '#7e6bfd', '#627560', '#ffc191', '#595d00',
 '#e408e6', '#b8b1b6', '#d32d41', '#314236', '#d8a362', '#5b8a33', '#2f1f00', '#97e6d6', '#2a6256', '#cd724d',
 '#5d3d28', '#0059d8', '#ac93d6', '#6b1d93', '#b3015d', '#410046', '#9cffcf', '#e4489c', '#e2e246', '#dbe2a5',
 '#002859', '#aa5b82', '#0000db', '#4b4d50', '#dabfd4', '#004d99', '#87649e', '#691d1c', '#8e52c4', '#b8dadf',
 '#ddb3fd', '#7b4854', '#4b7200', '#440077', '#b15e00', '#91d185', '#54334b', '#69af85', '#aa93af', '#e65442',
 '#8e8c89', '#70ac50', '#aa7c74', '#00343b', '#240f13', '#e6af00', '#79ccdb', '#18133a', '#9c5238', '#ba7b31',
 '#b6ca93', '#310800', '#a39505', '#00daba', '#74a0dd', '#623b72', '#ffda8e', '#77b800', '#3f2f1c', '#578759',
 '#2d0021', '#f4a1d4', '#da00aa', '#752849', '#bce400', '#c3c15d'
]

palette_112 = cet_g_bw[:112]
palette_56 = cet_g_bw[:56]

# Vibrant palette with clear distinctions (24 colors)
vibrant_palette = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', 
    '#FF8000', '#FF0080', '#80FF00', '#00FF80', '#8000FF', '#0080FF',
    '#FF3333', '#33FF33', '#3333FF', '#FFFF33', '#FF33FF', '#33FFFF',
    '#FF9933', '#FF3399', '#99FF33', '#33FF99', '#9933FF', '#3399FF'
]

# Earth tones palette (24 colors)
earth_palette = [
    '#8B4513', '#DAA520', '#556B2F', '#2F4F4F', '#8B008B', '#4682B4',
    '#CD853F', '#BDB76B', '#6B8E23', '#4F666A', '#800080', '#4169E1',
    '#D2691E', '#F0E68C', '#9ACD32', '#5F9EA0', '#9932CC', '#1E90FF',
    '#A0522D', '#EEE8AA', '#698B22', '#008B8B', '#9400D3', '#00BFFF'
]

# Pastel palette (24 colors)
pastel_palette = [
    '#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFDFBA', '#E0BBE4',
    '#957DAD', '#D291BC', '#FEC8D8', '#FFDFD3', '#B5EAD7', '#C7CEEA',
    '#FFB997', '#F5B0CB', '#D4F0F0', '#FFF5BA', '#A8E6CF', '#DBB4D8',
    '#FFD3B5', '#D4F0F0', '#B5EAD7', '#E2F0CB', '#C7CEEA', '#FFDAC1'
]






sc_color_cmap = LinearSegmentedColormap.from_list('Custom', sc_color, len(sc_color))



spring_festival = { 
    2022: datetime(2022, 2, 1), 
    2023: datetime(2023, 1, 22), 
    2024: datetime(2024, 2, 10), # ... 
    2025: datetime(2025, 1, 29),
    2026: datetime(2026, 2, 17),
    2027: datetime(2027, 2, 6),
}

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

name = "omicverse"
try:
    __version__ = version(name)
except Exception:
    __version__ = "unknown"

_has_printed_logo = False  # Flag to ensure logo prints only once
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# emoji map for status reporting
EMOJI = {
    "start":        "🔬",  # experiment start
    "deps":         "🔗",  # dependency check
    "settings":     "⚙️",  # configure settings
    "warnings":     "🚫",  # suppress warnings
    "gpu":          "🧬",  # GPU check
    "logo":         "🌟",  # print logo
    "done":         "✅",  # done
}


@register_function(
    aliases=["绘图设置", "plot_set", "ov_plot_set", "plotset", "设置绘图"],
    category="utils",
    description="Configure plotting settings for OmicVerse including matplotlib, scanpy, and GPU detection",
    examples=[
        "ov.utils.ov_plot_set()",
        "ov.utils.plot_set(dpi=100, figsize=6)",
        "ov.utils.plot_set(scanpy=False, fontsize=12)",
        "ov.utils.plot_set(vector_friendly=True)"
    ],
    related=["pl.embedding", "pl.volcano", "utils.palette"]
)
def plot_set(verbosity: int = 3, dpi: int = 80, 
             facecolor: str = 'white', 
             font_path: str = None,
             ipython_format: str  = "retina",
             dpi_save: int = 300,
             transparent: bool = None,
             scanpy: bool = True,
             fontsize: int = 14,
             color_map: Union[str, None] = None,
             figsize: Union[int, None] = None,
             vector_friendly: bool = True,
             ):
    r"""Configure plotting settings for OmicVerse.
    
    Arguments:
        verbosity: Scanpy verbosity level. Default: 3.
        dpi: Resolution for matplotlib figures. Default: 80.
        facecolor: Background color for figures. Default: 'white'.
        font_path: Path to font for custom fonts. Default: None.
        ipython_format: IPython display format. Default: 'retina'.
        dpi_save: Resolution for saved figures. Default: 300.
        transparent: Whether to use transparent background. Default: None.
        scanpy: Whether to apply scanpy settings. Default: True.
        fontsize: Default font size for plots. Default: 14.
        color_map: Default color map for plots. Default: None.
        figsize: Default figure size. Default: None.
        vector_friendly: Control rasterization for vector-friendly plots. Default: True.
        
    Returns:
        None: The function configures global plotting settings and displays initialization information.
    """
    global _has_printed_logo

    print(f"{EMOJI['start']} Starting plot initialization...")

    # 1) dependency check
    #print(f"{EMOJI['deps']} Checking dependencies...")
    #check_dependencies()
    # print(f"{EMOJI['done']} Dependencies OK")

    # 2) scanpy verbosity & figure params
    #print(f"{EMOJI['settings']} Applying plotting settings (verbosity={verbosity}, dpi={dpi})")
    sc.settings.verbosity = verbosity
    import builtins
    is_ipython = getattr(builtins, "__IPYTHON__", False)
    if is_ipython:
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        ipython_format = [ipython_format]
        set_matplotlib_formats(*ipython_format)
    
    from matplotlib import rcParams
    if dpi is not None:
        rcParams["figure.dpi"] = dpi
    if dpi_save is not None:
        rcParams["savefig.dpi"] = dpi_save
    if transparent is not None:
        rcParams["savefig.transparent"] = transparent
    if facecolor is not None:
        rcParams["figure.facecolor"] = facecolor
        rcParams["axes.facecolor"] = facecolor
    if scanpy:
        set_rcParams_scanpy(fontsize=fontsize, color_map=color_map)
    if figsize is not None:
        rcParams["figure.figsize"] = figsize
    
    # Set global vector_friendly setting
    global _vector_friendly
    _vector_friendly = vector_friendly
    #print(f"{EMOJI['done']} Settings applied")

    # 3) Custom font setup
    if font_path is not None:
        # Check if user wants Arial font (auto-download)
        if font_path.lower() in ['arial', 'arial.ttf'] and not font_path.endswith('.ttf'):
            try:
                # Create a persistent cache location for the Arial font
                import tempfile
                import requests

                cache_dir = tempfile.gettempdir()
                cached_arial_path = os.path.join(cache_dir, 'omicverse_arial.ttf')
                
                # Check if Arial font is already cached
                if os.path.exists(cached_arial_path):
                    print(f"Using already downloaded Arial font from: {cached_arial_path}")
                    font_path = cached_arial_path
                else:
                    print("Downloading Arial font from GitHub...")
                    arial_url = "https://github.com/kavin808/arial.ttf/raw/refs/heads/master/arial.ttf"
                    
                    # Download the font
                    response = requests.get(arial_url, timeout=30)
                    response.raise_for_status()
                    
                    # Save the font to cache location
                    with open(cached_arial_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Use the cached font file
                    font_path = cached_arial_path
                    print(f"Arial font downloaded successfully to: {cached_arial_path}")
                
            except Exception as e:
                print(f"Failed to download Arial font: {e}")
                print("Continuing with default font settings...")
                font_path = None
        
        if font_path is not None:
            try:
                # 1) Create a brand-new manager
                fm.fontManager = fm.FontManager()
                
                # 2) Add your file
                fm.fontManager.addfont(font_path)
                
                # 3) Now find out what name it uses
                name = fm.FontProperties(fname=font_path).get_name()
                print("Registered as:", name)
                
                # 4) Point rcParams at that name
                mpl.rcParams['font.family'] = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
                
            except Exception as e:
                print(f"Failed to set custom font: {e}")
                print("Continuing with default font settings...")

    # 4) suppress user/future/deprecation warnings
    #print(f"{EMOJI['warnings']} Suppressing common warnings")
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)
    #print(f"{EMOJI['done']} Warnings suppressed")

    # 5) GPU detection
    print(f"{EMOJI['gpu']} Detecting GPU devices…")
    gpu_found = False
    
    # Check CUDA devices
    if torch is not None and torch.cuda.is_available():
        try:
            cuda_count = torch.cuda.device_count()
            print(f"{EMOJI['done']} NVIDIA CUDA GPUs detected: {cuda_count}")
            for idx in range(cuda_count):
                props = torch.cuda.get_device_properties(idx)
                print(f"    • [CUDA {idx}] {props.name}")
                print(f"      Memory: {props.total_memory/1024**3:.1f} GB | Compute: {props.major}.{props.minor}")
            gpu_found = True
        except Exception as e:
            print(f"{EMOJI['warnings']} CUDA detection failed: {e}")
    
    # Check Apple Silicon MPS
    if torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            print(f"{EMOJI['done']} Apple Silicon MPS detected")
            print(f"    • [MPS] Apple Silicon GPU - Metal Performance Shaders available")
            gpu_found = True
        except Exception as e:
            print(f"{EMOJI['warnings']} MPS detection failed: {e}")
    
    # Check AMD ROCm
    if torch is not None and hasattr(torch.version, 'hip') and torch.version.hip is not None:
        try:
            print(f"{EMOJI['done']} AMD ROCm GPU detected")
            print(f"    • [ROCm] AMD GPU - HIP version: {torch.version.hip}")
            gpu_found = True
        except Exception as e:
            print(f"{EMOJI['warnings']} ROCm detection failed: {e}")
    
    # Check Intel XPU
    if torch is not None and hasattr(torch, 'xpu') and torch.xpu.is_available():
        try:
            xpu_count = torch.xpu.device_count()
            print(f"{EMOJI['done']} Intel XPU detected: {xpu_count}")
            for idx in range(xpu_count):
                print(f"    • [XPU {idx}] Intel GPU")
            gpu_found = True
        except Exception as e:
            print(f"{EMOJI['warnings']} Intel XPU detection failed: {e}")
    
    if not gpu_found:
        if torch is None:
            print(f"{EMOJI['warnings']} PyTorch not available - GPU detection skipped")
        else:
            print(f"{EMOJI['warnings']} No GPU devices found (CUDA/MPS/ROCm/XPU)")

    # 6) print logo & version only once
    if not _has_printed_logo:
        #print(f"{EMOJI['logo']} OmicVerse Logo:")
        today = datetime.now()
        chinese_new_year = spring_festival.get(today.year)
        if today.month == 12 and today.day in (24, 25):
            print(days_christmas)
        elif chinese_new_year and (chinese_new_year - timedelta(days=1) <= today <= chinese_new_year + timedelta(days=3)):
            print(days_chinese_new_year)
        else:
            print(omics)
        print(f"🔖 Version: {__version__}   📚 Tutorials: https://omicverse.readthedocs.io/")
        _has_printed_logo = True

    import matplotlib.pyplot as plt
    plt.rcParams['axes.grid'] = False

    print(f"{EMOJI['done']} plot_set complete.\n")


# Create aliases for backward compatibility
plotset = plot_set
ov_plot_set = plot_set




def pyomic_palette()->list:
    r"""Returns the default OmicVerse color palette.
    
    Returns:
        List of hex color codes for plotting
    """ 
    return sc_color

def palette()->list:
    r"""Returns the default OmicVerse color palette.
    
    Returns:
        List of hex color codes for plotting
    """ 
    return sc_color

def red_palette()->list:
    r"""Returns a red-themed color palette.
    
    Returns:
        List of red-themed hex color codes
    """ 
    return red_color

def green_palette()->list:
    r"""Returns a green-themed color palette.
    
    Returns:
        List of green-themed hex color codes
    """ 
    return green_color

def orange_palette()->list:
    r"""Returns an orange-themed color palette.
    
    Returns:
        List of orange-themed hex color codes
    """ 
    return orange_color

def blue_palette()->list:
    r"""Returns a blue-themed color palette.
    
    Returns:
        List of blue-themed hex color codes
    """ 
    return blue_color

def plot_text_set(text, text_knock=2, text_maxsize=20):
    r"""Format text for plotting by adding line breaks.
    
    Arguments:
        text: Text string to format
        text_knock: Number of words between line breaks (2)
        text_maxsize: Maximum text length before formatting (20)
        
    Returns:
        Formatted text string with line breaks
    """
    if len(text) <= text_maxsize:
        return text
    
    words = text.split(' ')
    formatted_text = []
    for i, word in enumerate(words):
        if i > 0 and i % text_knock == 0:
            formatted_text.append('\n')
        formatted_text.append(word)
    
    return ' '.join(formatted_text).strip()

    
def ticks_range(x,width):
    r"""Generate tick positions for multi-group plots.
    
    Arguments:
        x: Number of ticks
        width: Width spacing between ticks
        
    Returns:
        List of tick positions
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
    r"""Create boxplot with jittered points for grouped data.
    
    Arguments:
        data: DataFrame containing the data to plot
        hue: Column name for grouping variable
        x_value: Column name for x-axis categories
        y_value: Column name for y-axis values
        width: Width of boxplots (0.6)
        title: Plot title ('')
        figsize: Figure size ((6,3))
        palette: List of colors (None)
        fontsize: Font size for labels (10)
        legend_bbox: Legend bounding box ((1, 0.55))
        legend_ncol: Number of legend columns (1)
        
    Returns:
        Tuple of (figure, axes) objects
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
    ax.set_title(title,fontsize=fontsize+1)
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
    r"""Plot network graph with customizable node and edge properties.
    
    Arguments:
        G: NetworkX graph object
        G_type_dict: Dictionary mapping nodes to types
        G_color_dict: Dictionary mapping nodes to colors
        pos_type: Layout algorithm - 'spring' or 'kamada_kawai' ('spring')
        pos_dim: Layout dimension - 2 or 3 (2)
        figsize: Figure size ((4,4))
        pos_scale: Layout scale factor (10)
        pos_k: Spring layout k parameter (None)
        pos_alpha: Edge transparency (0.4)
        node_size: Base node size (50)
        node_alpha: Node transparency (0.6)
        node_linewidths: Node border width (1)
        plot_node: Specific nodes to label (None)
        plot_node_num: Number of top degree nodes to label (20)
        label_verticalalignment: Label vertical alignment ('center_baseline')
        label_fontsize: Label font size (12)
        label_fontfamily: Label font family ('Arial')
        label_fontweight: Label font weight ('bold')
        label_bbox: Label bounding box properties (None)
        legend_bbox: Legend position ((0.7, 0.05))
        legend_ncol: Legend columns (3)
        legend_fontsize: Legend font size (12)
        legend_fontweight: Legend font weight ('bold')
        
    Returns:
        Tuple of (figure, axes) objects
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
    #for i in pos.keys():
    #    pos1[i]=np.array([-1000,-1000])
    for i in hub_gene:
        pos1[i]=pos[i]
    #label_options = {"ec": "white", "fc": "white", "alpha": 0.6}
    #nx.draw_networkx_labels(
    #    G,pos1,verticalalignment=label_verticalalignment,
    #    font_size=label_fontsize,font_family=label_fontfamily,
    #    font_weight=label_fontweight,bbox=label_bbox,
    #)
    from adjustText import adjust_text
    import adjustText
    texts=[ax.text(pos1[i][0], 
               pos1[i][1],
               i,
               fontdict={'size':label_fontsize,'weight':label_fontweight,'color':'black'}
               ) for i in hub_gene if 'ENSG' not in i]
    if adjustText.__version__<='0.8':
        adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)
    else:
        adjust_text(texts,only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                    arrowprops=dict(arrowstyle='->', color='red'))
   #adjust_text(texts,only_move={'text': 'xy'},arrowprops=dict(arrowstyle='->', color='red'),)

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
    r"""Plot stacked bar chart showing cell type proportions across groups.
    
    Arguments:
        adata: AnnData object
        celltype_clusters: Column name for cell types
        visual_clusters: Column name for grouping variable
        visual_li: List of groups to plot (None)
        visual_name: Label for x-axis ('')
        figsize: Figure size ((4,6))
        ticks_fontsize: Font size for tick labels (12)
        labels_fontsize: Font size for axis labels (12)
        legend: Whether to show legend (False)
        
    Returns:
        Tuple of (figure, axes) objects
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
    r"""Create combined embedding plot with cell type legend and counts.
    
    Arguments:
        adata: AnnData object
        figsize: Figure size ((6,4))
        basis: Embedding basis name ('umap')
        celltype_key: Column name for cell types ('major_celltype')
        title: Plot title (None)
        celltype_range: Grid range for cell type panel ((2,9))
        embedding_range: Grid range for embedding panel ((3,10))
        xlim: X-axis limit for counts (-1000)
        
    Returns:
        Tuple of (figure, [embedding_axis, celltype_axis])
    """

    adata.obs[celltype_key]=adata.obs[celltype_key].astype('category')
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
        ax2.plot((100,cell_num_pd.loc[cell,'count']),(idx,idx),
                c=cell_color_dict[cell],lw=4)
        ax2.text(100,idx+0.2,
                cell+'('+str("{:,}".format(cell_num_pd.loc[cell,'count']))+')',fontsize=11)
    ax2.set_xlim(xlim,cell_num_pd.iloc[1].values[0]) 
    ax2.text(xlim,idx+1,title,fontsize=12)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.axis('off')

    return fig,[ax1,ax2]

@register_function(
    aliases=["标签生成", "gen_mpl_labels", "cluster_labels", "添加标签", "embedding_labels"],
    category="utils",
    description="Add cluster labels at median positions in embedding plots with automatic text positioning",
    examples=[
        "# Basic cluster labeling",
        "fig, ax = plt.subplots()",
        "ov.utils.embedding(adata, basis='X_umap', color='leiden', ax=ax)",
        "ov.utils.gen_mpl_labels(adata, groupby='leiden', ax=ax)",
        "# Custom basis and text styling",
        "ov.utils.gen_mpl_labels(adata, groupby='celltype', basis='X_tsne',",
        "                        text_kwargs={'fontsize': 12, 'weight': 'bold'})",
        "# Exclude specific clusters",
        "ov.utils.gen_mpl_labels(adata, groupby='leiden', exclude=['0', '1'])",
        "# With automatic text adjustment",
        "ov.utils.gen_mpl_labels(adata, groupby='leiden', ax=ax,", 
        "                        adjust_kwargs={'arrowprops': {'arrowstyle': '->'}})"
    ],
    related=["utils.embedding", "utils.plot_ConvexHull", "pl.umap"]
)
def gen_mpl_labels(
    adata, groupby, exclude=(), 
    basis='X_umap',ax=None, adjust_kwargs=None, text_kwargs=None
):
    """Add cluster labels at median positions in embedding plots with automatic text positioning.
    
    Arguments:
        adata: AnnData object containing single-cell data.
        groupby: Column name for grouping in adata.obs.
        exclude: Groups to exclude from labeling. Default: ().
        basis: Embedding basis name in adata.obsm. Default: 'X_umap'.
        ax: Matplotlib axes object. Default: None.
        adjust_kwargs: Parameters for adjustText text adjustment. Default: None.
        text_kwargs: Parameters for text styling (None)
        
    Returns:
        None
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
    r"""Create embedding plot with customizable colors.
    
    Arguments:
        adata: AnnData object
        basis: Embedding basis name
        color: Column name for coloring
        color_dict: Custom color mapping (None)
        figsize: Figure size ((4,4))
        **kwargs: Additional parameters for sc.pl.embedding
        
    Returns:
        Tuple of (figure, axes) objects
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

@register_function(
    aliases=["堆叠火山图", "stacking_vol", "multiple_volcano", "火山图组合", "比较火山图"],
    category="utils", 
    description="Create stacked volcano plots for comparing differential expression across multiple datasets",
    examples=[
        "# Basic stacked volcano plot",
        "data_dict = {'Dataset1': deg_df1, 'Dataset2': deg_df2}",
        "color_dict = {'Dataset1': 'red', 'Dataset2': 'blue'}",
        "fig, axes = ov.utils.stacking_vol(data_dict, color_dict)",
        "# Custom thresholds and styling",
        "fig, axes = ov.utils.stacking_vol(data_dict, color_dict,",
        "                                 pval_threshold=0.05, log2fc_threshold=1.5,",
        "                                 figsize=(10,6), plot_genes_num=15)",
        "# Three-way comparison",
        "data_dict = {'Control': deg1, 'Treatment1': deg2, 'Treatment2': deg3}",
        "color_dict = {'Control': '#1f77b4', 'Treatment1': '#ff7f0e', 'Treatment2': '#2ca02c'}",
        "fig, axes = ov.utils.stacking_vol(data_dict, color_dict)"
    ],
    related=["pl.volcano", "bulk.get_deg", "single.cosg"]
)
def stacking_vol(data_dict:dict,color_dict:dict,
                 pval_threshold:float=0.01,
                 log2fc_threshold:int=2,
                 figsize:tuple=(8,4),
                 sig_color:str='#a51616',
                 normal_color:str='#c7c7c7',
                 plot_genes_num:int=10,
                 plot_genes_fontsize:int=8,
                plot_genes_weight:str='bold')->tuple:
    """Create stacked volcano plots for comparing differential expression across multiple datasets.
    
    Arguments:
        data_dict: Dictionary with DataFrames containing 'logfoldchanges', 'pvals_adj', 'names' columns.
        color_dict: Dictionary mapping dataset names to colors
        pval_threshold: P-value significance threshold (0.01)
        log2fc_threshold: Log2 fold change threshold (2)
        figsize: Figure size ((8,4))
        sig_color: Color for significant points ('#a51616')
        normal_color: Color for non-significant points ('#c7c7c7')
        plot_genes_num: Number of top genes to label (10)
        plot_genes_fontsize: Font size for gene labels (8)
        plot_genes_weight: Font weight for gene labels ('bold')
        
    Returns:
        Tuple of (figure, axes_dict)
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
        axes[i].set_ylim(y_min,y_max)
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


@register_function(
    aliases=["凸包绘制", "plot_ConvexHull", "convex_hull", "凸包轮廓", "cluster_hull"],
    category="utils",
    description="Add convex hull outline for a specific cluster in embedding plot",
    examples=[
        "# Basic convex hull for a cluster",
        "fig, ax = plt.subplots()",
        "ov.utils.embedding(adata, basis='X_umap', color='leiden', ax=ax)",
        "ov.utils.plot_ConvexHull(adata, basis='X_umap', cluster_key='leiden',",
        "                         hull_cluster='0', ax=ax)",
        "# Custom color and transparency",
        "ov.utils.plot_ConvexHull(adata, basis='X_tsne', cluster_key='celltype',",
        "                         hull_cluster='T cells', ax=ax, color='red', alpha=0.3)",
        "# Multiple cluster hulls",
        "for cluster in ['0', '1', '2']:",
        "    ov.utils.plot_ConvexHull(adata, basis='X_umap', cluster_key='leiden',",
        "                             hull_cluster=cluster, ax=ax)"
    ],
    related=["utils.embedding", "pl.umap", "pl.tsne"]
)
def plot_ConvexHull(adata:anndata.AnnData,basis:str,cluster_key:str,
                    hull_cluster:str,ax,color=None,alpha:float=0.2):
    """Add convex hull outline for a specific cluster in embedding plot.
    
    Arguments:
        adata: AnnData object containing single-cell data.
        basis: Embedding basis name in adata.obsm (e.g., 'X_umap', 'X_tsne').
        cluster_key: Column name for cluster assignments in adata.obs.
        hull_cluster: Specific cluster identifier to outline.
        ax: Matplotlib axes object to draw on.
        color: Hull color. Default: None (automatic).
        alpha: Hull transparency level. Default: 0.2.
        
    Returns:
        ax: Modified matplotlib axes object with convex hull added.

    Examples:
        >>> import omicverse as ov
        >>> import matplotlib.pyplot as plt
        >>> # Create embedding plot with convex hull
        >>> fig, ax = plt.subplots()
        >>> ov.utils.embedding(adata, basis='X_umap', color='leiden', ax=ax)
        >>> ov.utils.plot_ConvexHull(adata, basis='X_umap', cluster_key='leiden',
        ...                          hull_cluster='0', ax=ax)
    """
    
    adata.obs[cluster_key]=adata.obs[cluster_key].astype('category')
    if '{}_colors'.format(cluster_key) in adata.uns.keys():
        print('{}_colors'.format(cluster_key))
        type_color_all=dict(zip(adata.obs[cluster_key].cat.categories,adata.uns['{}_colors'.format(cluster_key)]))
    else:
        if len(adata.obs[cluster_key].cat.categories)>28:
            type_color_all=dict(zip(adata.obs[cluster_key].cat.categories,sc.pl.palettes.default_102))
        else:
            type_color_all=dict(zip(adata.obs[cluster_key].cat.categories,sc.pl.palettes.zeileis_28))
    
    #color_dict=dict(zip(adata.obs[cluster_key].cat.categories,adata.uns[f'{cluster_key}_colors']))
    points=adata[adata.obs[cluster_key]==hull_cluster].obsm[basis]
    hull = ConvexHull(points)
    vert = np.append(hull.vertices, hull.vertices[0])  # close the polygon by appending the first point at the end
    if color==None:
        ax.plot(points[vert, 0], points[vert, 1], '--', c=type_color_all[hull_cluster])
        ax.fill(points[vert, 0], points[vert, 1], c=type_color_all[hull_cluster], alpha=alpha)
    else:
        ax.plot(points[vert, 0], points[vert, 1], '--', c=color)
        ax.fill(points[vert, 0], points[vert, 1], c=color, alpha=alpha)
    return ax





class geneset_wordcloud(object):

    def __init__(self,adata,cluster_key,pseudotime,resolution=1000,figsize=(4,10)):
        self.adata=adata
        self.cluster_key=cluster_key
        self.pseudotime=pseudotime
        self.figsize=figsize
        self.resolution=resolution

    def get(self,):
        from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
        #Get the DataFrame of anndata
        test_df=self.adata.to_df()
        
        #Calculate the mean of each cluster
        ct_pd=pd.DataFrame(columns=test_df.columns)
        for ct in self.adata.obs[self.cluster_key].unique():
            ct_pd.loc[ct]=test_df.loc[self.adata.obs.loc[self.adata.obs[self.cluster_key]==ct].index].mean(axis=0)
    
        # 遍历每个基因，找到最高表达的细胞类型
        max_expr_cell_types = []
        for gene in ct_pd.columns:
            max_expr_cell_type = ct_pd[gene].idxmax()
            max_expr_cell_types.append((gene, max_expr_cell_type))
        
        # 将结果转换为数据框
        result_df = pd.DataFrame(max_expr_cell_types, columns=['Gene', 'Max_Expression_Cell_Type'])
    
        
        size_dict=dict(result_df['Max_Expression_Cell_Type'].value_counts()/result_df.shape[0])

        self.adata.obs[self.cluster_key]=self.adata.obs[self.cluster_key].astype('category')
        if '{}_colors'.format(self.cluster_key) in self.adata.uns.keys():
            cell_color_dict=dict(zip(self.adata.obs[self.cluster_key].cat.categories.tolist(),
                            self.adata.uns['{}_colors'.format(self.cluster_key)]))
        else:
            if len(self.adata.obs[self.cluster_key].cat.categories)>28:
                cell_color_dict=dict(zip(self.adata.obs[self.cluster_key].cat.categories,sc.pl.palettes.default_102))
            else:
                cell_color_dict=dict(zip(self.adata.obs[self.cluster_key].cat.categories,sc.pl.palettes.zeileis_28))


        wc_dict={}
        for ct in self.adata.obs[self.cluster_key].unique():
            #print(ct)
            word_li=result_df.loc[result_df['Max_Expression_Cell_Type']==ct,'Gene'].values.tolist()
            print(ct,100*self.figsize[0],
                  int(100*size_dict[ct]*self.figsize[1]))
            wc = WordCloud(background_color="#FFFFFF",min_font_size=12,max_font_size=700, max_words=30,
                           width=100*self.figsize[0],
                           height=int(100*size_dict[ct]*self.figsize[1]),
                           contour_width=3, contour_color='firebrick')
            # 生成词云
            wc.generate(''.join([i.split(' (')[0] for i in word_li]))
            wc_dict[ct]=wc

        self.wc_dict=wc_dict.copy()
        self.size_dict=size_dict
        self.result_df=result_df
        self.color_dict=cell_color_dict
        return wc_dict

    def get_geneset(self):
        return self.result_df

    def get_wordcloud(self):
        return self.wc_dict

    def plot(self):
        fig = plt.figure(figsize=self.figsize)
        grid = plt.GridSpec(self.resolution, 10)
        
        import matplotlib.colors as mcolors
        
        last_idx=0
        
        for idx,ct in zip(range(len(self.adata.obs[self.cluster_key].unique())),
                          self.adata.obs.groupby(self.cluster_key)[self.pseudotime].mean().sort_values().index):
            next_idx=last_idx+self.size_dict[ct]
            print(ct,round(last_idx*self.resolution),round(next_idx*self.resolution))
            ax=fig.add_subplot(grid[round(last_idx*self.resolution):round(next_idx*self.resolution), :])      # 占据第二行的前两列
        
            colors=['#FFFFFF',self.color_dict[ct]]
            xcmap = mcolors.LinearSegmentedColormap.from_list('test_cmap', colors, N=100)
            
            ax.imshow(self.wc_dict[ct].recolor(colormap=xcmap), interpolation='bilinear')
            last_idx+=self.size_dict[ct]
            #ax.grid(False)
            if idx!=0:
                ax.axhline(y=0, c="#000000")
            ax.axis(False)
            # 绘制边框
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)

        return fig
    def plot_heatmap(self,n_convolve=10,figwidth=10,cmap='RdBu_r',
                     cbar=False,cbar_kws=None,cbar_fontsize=12):
        if cbar_kws==None:
            cbar_kws={'shrink':0.5,'location':'left'}

        fig = plt.figure(figsize=(figwidth,self.figsize[1]))
        grid = plt.GridSpec(self.resolution, 10)
        
        import matplotlib.colors as mcolors
        
        last_idx=0
        
        for idx,ct in zip(range(len(self.adata.obs[self.cluster_key].unique())),
                          self.adata.obs.groupby(self.cluster_key)[self.pseudotime].mean().sort_values().index):
            next_idx=last_idx+self.size_dict[ct]
            #print(ct,round(last_idx*self.resolution),round(next_idx*self.resolution))
            ax=fig.add_subplot(grid[round(last_idx*self.resolution):round(next_idx*self.resolution), figwidth-self.figsize[0]:])      # 占据第二行的前两列
        
            colors=['#FFFFFF',self.color_dict[ct]]
            xcmap = mcolors.LinearSegmentedColormap.from_list('test_cmap', colors, N=100)
            
            ax.imshow(self.wc_dict[ct].recolor(colormap=xcmap), interpolation='bilinear')
            last_idx+=self.size_dict[ct]
            #ax.grid(False)
            if idx!=0:
                ax.axhline(y=0, c="#000000")
            ax.axis(False)
            # 绘制边框
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)


        #sort time
        time = self.adata.obs[self.pseudotime].values
        time = time[np.isfinite(time)]
        
        from scipy.sparse import issparse
        X = self.adata.X
        if issparse(X):
            X = X.A
        df = pd.DataFrame(X[np.argsort(time)], columns=self.adata.var_names)

        #convolve
        
        if n_convolve is not None:
            weights = np.ones(n_convolve) / n_convolve
            for gene in self.adata.var_names:
                # TODO: Handle exception properly
                try:
                    df[gene] = np.convolve(df[gene].values, weights, mode="same")
                except ValueError as e:
                    print(f"Skipping variable {gene}: {e}")
        max_sort = np.argsort(np.argmax(df.values, axis=0))
        df = pd.DataFrame(df.values[:, max_sort], columns=df.columns[max_sort])

        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df)
        
        ax2=fig.add_subplot(grid[:, :figwidth-self.figsize[0]])
        sns.heatmap(normalized_data.T,cmap=cmap,ax=ax2,cbar=cbar,cbar_kws=cbar_kws)
        # use matplotlib.colorbar.Colorbar object
        if cbar!=False:
            cbar1 = ax2.collections[0].colorbar
            # here set the labelsize by 20
            cbar1.ax.tick_params(labelsize=cbar_fontsize)
        #ax2.imshow(normalized_data.T,cmap='RdBu_r',)
        ax2.grid(False)
        ax2.axis(False)
        
        #ax3=fig.add_subplot(grid[:10, :8])
        # 添加类别可视化（以不同颜色的矩形表示）
        category_colors = self.adata.obs[self.cluster_key].map(self.color_dict).values[np.argsort(time)]
        for i, color in enumerate(category_colors):
            rect = plt.Rectangle((i, 0), 2, 2, color=color)
            ax2.add_patch(rect)

        return fig


from scanpy.plotting._anndata import ranking
from scanpy.plotting._utils import savefig_or_show
@register_function(
    aliases=["主成分方差比", "plot_pca_variance_ratio", "pca_variance", "PCA方差", "主成分分析方差"],
    category="utils",
    description="Plot PCA variance ratio to determine optimal number of principal components",
    examples=[
        "# Basic PCA variance ratio plot",
        "ov.pp.pca(adata, n_pcs=50)",
        "ov.utils.plot_pca_variance_ratio(adata, n_pcs=30)",
        "# Custom variance ratios with log scale",
        "ov.utils.plot_pca_variance_ratio(adata, n_pcs=50, log=True)",
        "# Check different representation",
        "ov.utils.plot_pca_variance_ratio(adata, use_rep='scaled|original|pca_var_ratios',",
        "                                 n_pcs=20)"
    ],
    related=["pp.pca", "pp.scale", "utils.cluster"]
)
def plot_pca_variance_ratio(
    adata,
    use_rep='scaled|original|pca_var_ratios',
    n_pcs: int = 30,
    log: bool = False,
    show=None,
    save=None,
):
    ranking(
        adata,
        "uns",
        use_rep,
        n_points=n_pcs,
        #dictionary="pca",
        labels="PC",
        log=log,
    )
    savefig_or_show("pca_variance_ratio", show=show, save=save)

def plot_pca_variance_ratio1(adata,threshold=0.85):

    import matplotlib.pyplot as plt
    plt.scatter(range(len(adata.uns['scaled|original|pca_var_ratios'])),
                adata.uns['scaled|original|pca_var_ratios'])
    ratio_max=max(adata.uns['scaled|original|pca_var_ratios'])
    ratio_max_85=(1-threshold)*ratio_max
    pcs_85_num=len(adata.uns['scaled|original|pca_var_ratios'][adata.uns['scaled|original|pca_var_ratios']>ratio_max_85])
    plt.axhline(ratio_max_85)
    plt.title(f'PCs:{pcs_85_num}')
    plt.xlabel('ranking')


def check_dependencies(dependencies=None, check_full=False):
    r"""Check if installed package versions match requirements.
    
    Arguments:
        dependencies: List of dependency strings (None)
        check_full: Whether to check optional dependencies (False)
        
    Returns:
        None
    """
    if dependencies is None:
        try:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pyproject_path = os.path.join(current_dir, 'pyproject.toml')
            
            with open(pyproject_path, 'rb') as f:
                pyproject = tomli.load(f)
                dependencies = pyproject['project']['dependencies']

            # If check_full is True, also check full dependencies
            if check_full and 'project' in pyproject and 'optional-dependencies' in pyproject['project']:
                if 'full' in pyproject['project']['optional-dependencies']:
                    full_deps = pyproject['project']['optional-dependencies']['full']
                    dependencies.extend(full_deps)

        except Exception as e:
            print(f"Warning: Could not read dependencies from pyproject.toml: {e}")
            return

    # try:
    #     pkg_resources.require(dependencies)
    #     print("All dependencies are satisfied.")
    # except (DistributionNotFound, VersionConflict) as e:
    #     print(f"Dependency error: {e}")
    
    import importlib.metadata as importlib_metadata
    for req in dependencies:
        try:
            importlib_metadata.distribution(req)  # Validate package installation and version constraints
        except importlib_metadata.PackageNotFoundError as e:
            print(f"Missing dependency: {req!r}: {e}")
        except Exception as e:
            # Handle version conflicts or other issues
            print(f"Dependency error for {req!r}: {e}")
    else:
        print("All dependencies are satisfied.")




def set_rcParams_scanpy(fontsize=14, color_map=None):
    """Set matplotlib.rcParams to Scanpy defaults.

    Call this through :func:`scanpy.set_figure_params`.
    """
    # figure
    import matplotlib as mpl
    from cycler import cycler
    
    rcParams["figure.figsize"] = (4, 4)
    rcParams["figure.subplot.left"] = 0.18
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.91

    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]
    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = 0.92 * fontsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = fontsize

    # legend
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4

    # color cycles
    rcParams["axes.prop_cycle"] = cycler(color=sc_color)

    # lines
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = fontsize
    rcParams["ytick.labelsize"] = fontsize

    # axes grid
    rcParams["axes.grid"] = True
    rcParams["grid.color"] = ".8"

    # color map
    rcParams["image.cmap"] = rcParams["image.cmap"] if color_map is None else color_map


def set_rcParams_defaults():
    """Reset `matplotlib.rcParams` to defaults."""
    rcParams.update(mpl.rcParamsDefault)


omics=r"""
   ____            _     _    __                  
  / __ \____ ___  (_)___| |  / /__  _____________ 
 / / / / __ `__ \/ / ___/ | / / _ \/ ___/ ___/ _ \ 
/ /_/ / / / / / / / /__ | |/ /  __/ /  (__  )  __/ 
\____/_/ /_/ /_/_/\___/ |___/\___/_/  /____/\___/                                              
"""
days_christmas=r"""
      .
   __/ \__
   \     /
   /.'o'.\
    .o.'.         Merry Christmas!
   .'.'o'.
  o'.o.'.o.
 .'.o.'.'.o.       ____ 
.o.'.o.'.o.'.     / __ \____ ___  (_)___| |  / /__  _____________ 
   [_____]       / / / / __ `__ \/ / ___/ | / / _ \/ ___/ ___/ _ \ 
    \___/       / /_/ / / / / / / / /__ | |/ /  __/ /  (__  )  __/ 
                \____/_/ /_/ /_/_/\___/ |___/\___/_/  /____/\___/
"""
#Tua Xiong
days_chinese_new_year=""""""