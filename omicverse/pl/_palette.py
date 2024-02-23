import scanpy as sc

sc_color=['#7CBB5F','#368650','#A499CC','#5E4D9A','#78C2ED','#866017', '#9F987F','#E0DFED',
 '#EF7B77', '#279AD7','#F0EEF0', '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', '#FCBC10',
 '#EAEFC5', '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']

red_color=['#F0C3C3','#E07370','#CB3E35','#A22E2A','#5A1713','#D3396D','#DBC3DC','#85539B','#5C2B80','#5C4694']
green_color=['#91C79D','#8FC155','#56AB56','#2D5C33','#BBCD91','#6E944A','#A5C953','#3B4A25','#010000']
orange_color=['#EFBD49','#D48F3E','#AC8A3E','#7D7237','#745228','#E1C085','#CEBC49','#EBE3A1','#6C6331','#8C9A48','#D7DE61']
blue_color=['#347862','#6BBBA0','#81C0DD','#3E8CB1','#88C8D2','#52B3AD','#265B58','#B2B0D4','#5860A7','#312C6C']
purple_color=['#823d86','#825b94','#bb98c6','#c69bc6','#a69ac9','#c5a6cc','#caadc4','#d1c3d4']

ditto_color=[
            "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
            "#D55E00", "#CC79A7", "#666666", "#AD7700", "#1C91D4",
            "#007756", "#D5C711", "#005685", "#A04700", "#B14380",
            "#4D4D4D", "#FFBE2D", "#80C7EF", "#00F6B3", "#F4EB71",
            "#06A5FF", "#FF8320", "#D99BBD", "#8C8C8C"
        ]

def optim_palette(adata,basis,colors,palette=None,**kwargs):
    """
    Optimized palette for plotting

    Arguments:
        adata: AnnData object
        basis: basis for plotting, which is the key of adata.obsm
        colors: key of adata.obs for color
        palette: palette for plotting
        kwargs: kwargs for spaco.colorize

    Returns:
        palette_spaco: palette for plotting
    
    """

    try:
        import spaco
        #print('mofax have been install version:',mfx.__version__)
    except ImportError:
        raise ImportError(
            'Please install the spaco: `pip install spaco-release`.'
        )
    
    adata.obs[colors]=adata.obs[colors].astype('category')

    if (adata.uns[f'{colors}_colors'] is None) and (palette is None):
        if len(adata.obs[colors].cat.categories)>28:
            palette_t=sc.pl.palettes.default_102[:len(list(set(adata.obs[colors].tolist())))]
        else:
            palette_t=sc.pl.palettes.zeileis_28[:len(list(set(adata.obs[colors].tolist())))]
        #palette_t=ditto_color[:len(list(set(adata.obs[colors].tolist())))]
    elif palette!= None:
        palette_t=palette[:len(list(set(adata.obs[colors].tolist())))]
    else:
        palette_t=adata.uns[f'{colors}_colors']
        
    import spaco
    color_mapping = spaco.colorize(
        cell_coordinates=adata.obsm[basis],
        cell_labels=adata.obs[colors],
        colorblind_type='none',
        palette=palette_t,
        **kwargs
    )
    # Order colors by categories in adata
    color_mapping = {k: color_mapping[k] for k in adata.obs[colors].cat.categories}
    palette_spaco = list(color_mapping.values())
    return palette_spaco
    