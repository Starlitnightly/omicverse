import os 
import scanpy as sc
import pandas as pd
import numpy as np
from typing import Optional
from ._SCSA import Process,Annotator
import sys
import argparse
import gzip
import time
import requests
import anndata
from ..pp._preprocess import scale
from ..utils import gen_mpl_labels

from .._settings import add_reference, Colors, EMOJI
from ..utils.registry import register_function
from ..datasets import download_data_requests

DATA_DOWNLOAD_LINK_DICT = {
    'whole':{
        'figshare':'https://figshare.com/ndownloader/files/37262710',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pySCSA_2023.db',
    },
    'pySCSA_2023_v2_plus':{
        'figshare':'https://figshare.com/ndownloader/files/41369037',
        'stanford':'https://stacks.stanford.edu/file/cv694yk7414/pySCSA_2023_v2_plus.db',
    },

}


def get_anno_dataset_url(dataset_name: str, prefer_stanford: bool = True) -> str:
    """Get URL for an annotation dataset by name, preferring Stanford over Figshare.

    Args:
        dataset_name: Name of the dataset (e.g., 'whole', 'pySCSA_2023_v2_plus').
        prefer_stanford: Whether to prefer Stanford links over Figshare (default: True).

    Returns:
        URL string for the dataset.

    Raises:
        ValueError: If dataset name is not found.
    """
    if dataset_name not in DATA_DOWNLOAD_LINK_DICT:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATA_DOWNLOAD_LINK_DICT")

    dataset_urls = DATA_DOWNLOAD_LINK_DICT[dataset_name]

    if prefer_stanford and 'stanford' in dataset_urls:
        print(f"{Colors.CYAN}Using Stanford mirror for {dataset_name}{Colors.ENDC}")
        return dataset_urls['stanford']
    elif 'figshare' in dataset_urls:
        if prefer_stanford:
            print(f"{Colors.WARNING}{EMOJI['warning']} Stanford link not available for {dataset_name}, using Figshare{Colors.ENDC}")
        return dataset_urls['figshare']
    else:
        raise ValueError(f"No valid URL found for dataset '{dataset_name}'")

def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

metatime_install=False
def check_metatime():
    r"""Check if metatime is installed and import it.
    
    Returns:
        None: Raises ImportError if metatime is not installed
    """
    global metatime_install
    try:
        import metatime
        metatime_install=True
        print('metatime have been install version:',metatime.__version__)
    except ImportError:
        raise ImportError(
            'Please install the metatime: `pip install metatime`.'
        )


# Deprecated: data_downloader has been replaced by download_data_requests
# Use get_anno_dataset_url() and download_data_requests() instead


def data_preprocess(adata,clustertype='leiden',
                    path='temp/rna.csv',layer='scaled',rank_rep=False):
    r"""Data preprocess for SCSA.
    
    Arguments:
        adata: AnnData object
        clustertype: Clustering name used in scanpy. ('leiden')
        path: The save path of datasets. ('temp/rna.csv')
        layer: Layer to use for processing. ('scaled')
        rank_rep: Whether to repeat ranking. (False)

    Returns:
        dat: Preprocessed data as DataFrame
    """
    dirname, _ = os.path.split(path)
    try:
        if not os.path.isdir(dirname):
            print("......Creating directory {}".format(dirname))
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        print("......Unable to create directory {}. Reason {}".format(dirname,e))

    sc.settings.verbosity = 2  # reduce the verbosity
    if rank_rep==False and 'rank_genes_groups' not in adata.uns.keys():
        sc.tl.rank_genes_groups(adata, clustertype, method='wilcoxon')
    elif rank_rep==True:
        sc.tl.rank_genes_groups(adata, clustertype, method='wilcoxon')
    else:
        pass
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    dat = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'logfoldchanges','scores','pvals']})
    dat.to_csv(path)
    return dat

def __cell_annotate(data,
                foldchange=1.5,pvalue=0.05,
                output='temp/rna_anno.txt',
                outfmt='txt',Gensymbol=True,
                species='Human',weight=100,tissue='All',
                celltype='normal',norefdb=False,noprint=True,list_tissue=False):
    r"""Cell annotation by SCSA.
    
    Arguments:
        data: AnnData object
        foldchange: Foldchange threshold. (1.5)
        pvalue: Pvalue threshold. (0.05)
        output: The save path of annotation result. ('temp/rna_anno.txt')
        outfmt: The format of annotation result. ('txt')
        Gensymbol: Whether to use gene symbol. (True)
        species: The species of datasets. ('Human')
        weight: The weight of datasets. (100)
        tissue: The tissue of datasets. ('All')
        celltype: The celltype of datasets. ('normal')
        norefdb: Whether to use reference database. (False)
        noprint: Whether to print the result. (True)
        list_tissue: Whether to list the tissue of datasets. (False)
    
    Returns:
        result: The annotation result
    """
    data.to_csv('temp/rna.csv')

    # Get URL from dataset dict and download using Stanford mirror
    url = get_anno_dataset_url('whole')
    model_path = download_data_requests(url=url, file_path='whole.db', dir='./temp')
    print(f'{Colors.CYAN}......Auto annotate cell{Colors.ENDC}')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', default = "temp/rna.csv")
    parser.add_argument('-o', '--output',default=output)
    parser.add_argument('-d', '--db', default = model_path,)
    parser.add_argument('-s', '--source', default = "scanpy",)
    parser.add_argument('-c', '--cluster', default = "all",)
    parser.add_argument('-f',"--fc",default = "1.5",)
    parser.add_argument('-fc',"--foldchange",default =foldchange,)
    parser.add_argument('-p',"--pvalue",default = pvalue,)
    parser.add_argument('-w',"--weight",default = weight,)
    parser.add_argument('-g',"--species",default = species,)
    parser.add_argument('-k',"--tissue",default = tissue,)
    parser.add_argument('-m', '--outfmt', default = outfmt, )
    parser.add_argument('-T',"--celltype",default = celltype,)
    parser.add_argument('-t', '--target', default = "cellmarker",)
    parser.add_argument('-E',"--Gensymbol",action = "store_true",default=Gensymbol,)
    parser.add_argument('-N',"--norefdb",action = "store_true",default=norefdb,)
    parser.add_argument('-b',"--noprint",action = "store_true",default=noprint,)
    parser.add_argument('-l',"--list_tissue",action = "store_true",default = False,)
    parser.add_argument('-M', '--MarkerDB',)
    args = parser.parse_args()

    p = Process()
    if args.list_tissue:
        p.list_tissue(args)
    p.run_cmd(args)

    result=pd.read_csv('temp/rna_anno.txt',sep='\t')
    return result


def __cell_anno_print(anno):
    r"""Print the annotation result.

    Arguments:
        anno: The annotation result

    Returns:
        None
    """
    for i in set(anno['Cluster']):
        test=anno.loc[anno['Cluster']==i].iloc[:2]
        if len(test) >= 2 and pd.notna(test.iloc[0]['Z-score']) and pd.notna(test.iloc[1]['Z-score']):
            if test.iloc[0]['Z-score']>test.iloc[1]['Z-score']*2:
                print('Nice:Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,test.iloc[0]['Cell Type'],
                                                            np.around(test.iloc[0]['Z-score'],3)))
            else:
                print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,('|').join(test['Cell Type'].values.tolist()),
                                                            ('|').join(np.around(test['Z-score'].values,3).astype(str).tolist())))
        elif len(test) >= 1 and pd.notna(test.iloc[0]['Z-score']):
            print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,test.iloc[0]['Cell Type'],
                                                        np.around(test.iloc[0]['Z-score'],3)))
        else:
            print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,'Unknown','NaN'))

def scanpy_lazy(adata:anndata.AnnData,min_genes:int=200,min_cells:int=3,drop_doublet:bool=True,
                n_genes_by_counts:int=4300,pct_counts_mt:int=25,
                target_sum:float=1e4,min_mean:float=0.0125, max_mean:int=3, min_disp:float=0.5,max_value:int=10,
                n_comps:int=100, svd_solver:str="auto",
                n_neighbors:int=15, random_state:int = 112, n_pcs:int=50,
                )->anndata.AnnData:
    r"""Scanpy lazy analysis pipeline.
    
    Arguments:
        adata: AnnData object
        min_genes: The min number of genes. (200)
        min_cells: The min number of cells. (3)
        drop_doublet: Whether to drop doublet. (True)
        n_genes_by_counts: The max number of genes. (4300)
        pct_counts_mt: The max proportion of mito-genes. (25)
        target_sum: The max counts of total_counts. (1e4)
        min_mean: The min mean of genes. (0.0125)
        max_mean: The max mean of genes. (3)
        min_disp: The min dispersion of genes. (0.5)
        max_value: The max value of genes. (10)
        n_comps: The number of components. (100)
        svd_solver: The solver of svd. ('auto')
        n_neighbors: The number of neighbors. (15)
        random_state: The random state. (112)
        n_pcs: The number of pcs. (50)

    Returns:
        adata: AnnData object
    """
    #filter cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    #filter the doublets cells
    if drop_doublet:
        sc.external.pp.scrublet(adata) #estimates doublets
        adata = adata[adata.obs['predicted_doublet'] == False] #do the actual filtering
    #calculate the proportion of mito-genes
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < n_genes_by_counts, :]
    adata = adata[adata.obs.pct_counts_mt < pct_counts_mt, :]
    #save the raw counts
    adata.layers["counts"] = adata.X.copy()
    #normalization, the max counts of total_counts is 20000 means the amount is 10e4
    sc.pp.normalize_total(adata, target_sum=target_sum)
    #log
    sc.pp.log1p(adata)
    #select high-variable genes
    sc.pp.highly_variable_genes(adata, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)
    #save and filter
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    #scale
    #scale(adata, max_value=max_value)
    sc.pp.scale(adata, max_value=max_value)
    #pca analysis
    sc.tl.pca(adata, n_comps=n_comps, svd_solver=svd_solver)
    #pca(adata,layer='scaled',n_pcs=50)
    #cell neighbors graph construct
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state = random_state, n_pcs=n_pcs)
    #umap
    sc.tl.leiden(adata)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    sc.tl.umap(adata, init_pos='paga')
    return adata

@register_function(
    aliases=["字典注释", "scanpy_cellanno_from_dict", "manual_annotation", "手动注释", "字典映射注释"],
    category="single",
    description="Manual cell type annotation from cluster-to-celltype dictionary mapping",
    prerequisites={
        'functions': ['leiden']
    },
    requires={
        'obs': []  # Dynamic: requires user-specified clustertype column
    },
    produces={
        'obs': []  # Dynamic: creates {anno_name}_celltype column
    },
    auto_fix='none',
    examples=[
        "# Basic manual annotation from dictionary",
        "cluster2annotation = {",
        "    '0': 'T cell', '1': 'B cell', '2': 'Monocyte',",
        "    '3': 'NK cell', '4': 'Dendritic cell'",
        "}",
        "ov.single.scanpy_cellanno_from_dict(adata, anno_dict=cluster2annotation,",
        "                                    clustertype='leiden')",
        "# Custom annotation key",
        "ov.single.scanpy_cellanno_from_dict(adata, anno_dict=cluster2annotation,",
        "                                    clustertype='leiden',",
        "                                    key_added='manual_celltype')",
        "# Compare with automatic annotation",
        "ov.utils.embedding(adata, color=['manual_celltype', 'scsa_celltype'])"
    ],
    related=["single.pySCSA", "single.get_celltype_marker", "utils.embedding"]
)
def scanpy_cellanno_from_dict(adata:anndata.AnnData,
                               anno_dict:dict,
                               anno_name:str='major',
                               clustertype:str='leiden',
                               )->None:
    r"""Add cell type annotation from dict to anndata object.

    Arguments:
        adata: AnnData object of scRNA-seq after preprocessing
        anno_dict: Dict of cell type annotation. key is the cluster name, value is the cell type name.like `{'0':'B cell','1':'T cell'}`
        anno_name: The name of annotation. ('major')
        clustertype: Clustering name used in scanpy. ('leiden')

    Returns:
        None
    """

    adata.obs[anno_name+'_celltype'] = adata.obs[clustertype].map(anno_dict).astype('category')
    print('...cell type added to {}_celltype on obs of anndata'.format(anno_name))

@register_function(
    aliases=["细胞类型标记基因", "get_celltype_marker", "celltype_markers", "标记基因", "差异基因"],
    category="single",
    description="Extract cell type-specific marker genes from differential expression analysis",
    prerequisites={
        'functions': ['leiden']
    },
    requires={
        'obs': []  # Dynamic: requires user-specified clustertype column
    },
    produces={
        'uns': ['rank_genes_groups']
    },
    auto_fix='escalate',
    examples=[
        "# Get markers for all cell types",
        "marker_dict = ov.single.get_celltype_marker(adata,",
        "                                           clustertype='leiden')",
        "print(marker_dict.keys())  # Show available cell types",
        "# Get markers for specific cell type annotation",
        "marker_dict = ov.single.get_celltype_marker(adata,",
        "                                           clustertype='scsa_celltype')",
        "# View markers for specific cell type",
        "b_cell_markers = marker_dict['B cell']",
        "print(b_cell_markers[:10])  # Top 10 markers",
        "# Use with plotting",
        "sc.pl.dotplot(adata, marker_dict, groupby='leiden')"
    ],
    related=["single.pySCSA", "single.scanpy_cellanno_from_dict", "pl.dotplot"]
)
def get_celltype_marker(adata:anndata.AnnData,
                            clustertype:str='leiden',
                            log2fc_min:int=2,scores_type='scores',
                            pval_cutoff:float=0.05,rank:bool=True,
                            key='rank_genes_groups',method='wilcoxon',
                            foldchange=None,topgenenumber=10,unique=True,
                            global_unique=False,use_raw:Optional[bool]=None,
                            layer:Optional[str]=None,**kwargs)->dict:
        r"""Get marker genes for each clusters.
        
        Arguments:
            adata: anndata object
            clustertype: Clustering name used in scanpy. (leiden)
            log2fc_min: Minimum log2 fold change of marker genes. (2)
            pval_cutoff: Maximum p value of marker genes. (0.05)
            rank: Whether to rank genes by wilcoxon test. (True)
            scores_type: The type of scores. can be selected from `scores` and `logfoldchanges`
            unique: Whether to remove duplicates within each cell type. (True)
            global_unique: Whether to remove duplicates across all cell types. (False)

        Returns:
            cellmarker: A dictionary of marker genes for each clusters.
        """
        print('...get cell type marker')
        celltypes = sorted(adata.obs[clustertype].unique())
        cell_marker_dict={}
        if key not in adata.uns.keys() or rank:
            rg_kwargs = dict(method=method, key_added=key)
            if use_raw is not None:
                rg_kwargs['use_raw'] = use_raw
            if layer is not None:
                rg_kwargs['layer'] = layer
            rg_kwargs.update(kwargs)
            sc.tl.rank_genes_groups(adata, clustertype, **rg_kwargs)
        for celltype in celltypes:
            degs = sc.get.rank_genes_groups_df(adata, group=celltype, key=key, log2fc_min=log2fc_min, 
                                            pval_cutoff=pval_cutoff)
            foldp=np.histogram(degs[scores_type])
            if foldchange is None:
                try:
                    foldchange=(foldp[1][np.where(foldp[1]>0)[0][-5]]+foldp[1][np.where(foldp[1]>0)[0][-6]])/2
                except:
                    foldchange=degs[scores_type].mean()
                    
            cellmarker=degs.loc[degs[scores_type]>foldchange]['names'].values[:topgenenumber]
            cell_marker_dict[celltype]=cellmarker
        if unique==True:
            for key in cell_marker_dict.keys():
                cell_marker_dict[key]=list(set(cell_marker_dict[key]))
        
        # Global uniqueness across all cell types
        if global_unique:
            used_genes = set()
            for celltype in celltypes:
                if celltype in cell_marker_dict:
                    # Filter out genes that have been used in previous cell types
                    unique_genes = [gene for gene in cell_marker_dict[celltype] if gene not in used_genes]
                    cell_marker_dict[celltype] = unique_genes
                    used_genes.update(unique_genes)
        
        return cell_marker_dict






@register_function(
    aliases=["单细胞注释", "pySCSA", "cell_annotation", "细胞类型注释", "自动注释"],
    category="single",
    description="Automated cell type annotation using SCSA (Single Cell Signature Analysis) with multiple databases. IMPORTANT: Use 'clustertype' parameter (NOT 'cluster') in cell_auto_anno()!",
    prerequisites={
        'optional_functions': ['preprocess', 'leiden']
    },
    requires={
        'var': [],  # Flexible - works with raw or processed data
        'obs': []   # Clustering recommended but not required
    },
    produces={
        'obs': ['scsa_celltype']
    },
    auto_fix='none',
    examples=[
        "# CRITICAL: Use clustertype='leiden', NOT cluster='leiden'!",
        "# Step 1: Initialize pySCSA",
        "scsa = ov.single.pySCSA(adata, foldchange=1.5, pvalue=0.01,",
        "                        species='Human', tissue='All', target='cellmarker')",
        "",
        "# Step 2: Run annotation - NOTE: parameter is 'clustertype'!",
        "anno = scsa.cell_anno(clustertype='leiden', cluster='all')",
        "",
        "# Step 3: Add annotations to adata.obs",
        "scsa.cell_auto_anno(adata, clustertype='leiden', key='scsa_celltype')",
        "# Results are now in adata.obs['scsa_celltype']",
        "",
        "# WRONG - DO NOT USE:",
        "# scsa.cell_auto_anno(adata, cluster='leiden')  # ERROR! 'cluster' is NOT valid!",
        "",
        "# Using PanglaoDB database",
        "scsa = ov.single.pySCSA(adata, target='panglaodb', tissue='All')",
        "anno = scsa.cell_anno(clustertype='leiden', cluster='all')",
        "scsa.cell_auto_anno(adata, clustertype='leiden', key='scsa_panglaodb')"
    ],
    related=["single.scanpy_cellanno_from_dict", "single.get_celltype_marker", "utils.embedding"]
)
class pySCSA(object):

    def __init__(self,adata:anndata.AnnData,
                foldchange:float=1.5,pvalue:float=0.05,
                output:str='temp/rna_anno.txt',
                model_path:str='',
                outfmt:str='txt',Gensymbol:bool=True,
                species:str='Human',weight:int=100,tissue:str='All',target:str='cellmarker',
                celltype:str='normal',norefdb:bool=False,cellrange:str=None,
                noprint:bool=True,list_tissue:bool=False,
                # Compatibility aliases used by older prompts/agents
                tissuename:str=None,speciename:str=None) -> None:

        r"""Initialize the pySCSA class.

        Arguments:
            adata: AnnData object of scRNA-seq after preprocessing
            foldchange: Fold change threshold for marker filtering. (1.5)
            pvalue: P-value threshold for marker filtering. (0.05)
            output: Output file for marker annotation. ('temp/rna_anno.txt')
            model_path: Path to the Database for annotation. If not provided, the model will be downloaded from the internet. ('')
            outfmt: Output format for marker annotation. ('txt')
            Gensymbol: Using gene symbol ID instead of ensembl ID in input file for calculation. (True)
            species: Species for annotation. Only used for cellmarker database. ('Human')
            weight: Weight threshold for marker filtering from cellranger v1.0 results. (100)
            tissue: Tissue for annotation. you can use `get_model_tissue` to see the available tissues. ('All')
            target: Target to annotation class in Database. ('cellmarker')
            celltype: Cell type for annotation. ('normal')
            norefdb: Only using user-defined marker database for annotation. (False)
            cellrange: Cell sub_type for annotation. (if you input T cell, it will only provide T helper cell, T cytotoxic cell, T regulatory cell, etc.) (None)
            noprint: Do not print any detail results. (True)
            list_tissue: List all available tissues in the database. (False)
        
        Returns:
            None
        """

        #create temp directory
        try:
            if not os.path.isdir('temp'):
                print("...Creating directory {}".format('temp'))
                os.makedirs('temp', exist_ok=True)
        except OSError as e:
            print("...Unable to create directory {}. Reason {}".format('temp',e))

        self.adata=adata
        self.foldchange=foldchange
        self.pvalue=pvalue
        self.output=output
        self.outfmt=outfmt
        self.Gensymbol=Gensymbol
        # Backwards compatibility: accept alias field names some agents produce
        if speciename is not None:
            species = speciename
        if tissuename is not None and tissue == 'All':
            tissue = tissuename

        self.species=species
        self.weight=weight
        self.tissue=tissue
        self.celltype=celltype
        self.norefdb=norefdb
        self.noprint=noprint
        self.list_tissue=list_tissue
        self.target=target
        self.cellrange=cellrange
        if model_path =='':
            url = get_anno_dataset_url('pySCSA_2023_v2_plus')
            self.model_path = download_data_requests(url=url, file_path='pySCSA_2023_v2_plus.db', dir='./temp')
        else:
            self.model_path=model_path

    def get_model_tissue(self,species:str="Human")->None:
        r"""List all available tissues in the database.
        
        Arguments:
            species: Species for annotation. Only used for cellmarker database. ('Human')

        Returns:
            None
        """
        
        anno = Annotator(foldchange=self.foldchange,
                    weight=self.weight,
                    pvalue=self.pvalue,
                    tissue=self.tissue,
                    species=self.species,
                    target=self.target,
                    norefdb=self.norefdb,
                    MarkerDB=None,
                    db=self.model_path,
                    noprint=self.noprint,
                    input="temp/rna.csv",
                    output=self.output,
                    source="scanpy",
                    cluster='all',
                    fc=self.foldchange,
                    outfmt=self.outfmt,
                    celltype=self.celltype,
                    Gensymbol=self.Gensymbol,
                    list_tissue=self.list_tissue,
                    cellrange=self.cellrange)
        anno.load_pickle_module(self.model_path)
        anno.get_list_tissue(species)


    def cell_anno(self,clustertype:str='leiden',
                  cluster:str='all',rank_rep=False)->pd.DataFrame:
        r"""Annotate cell type for each cluster.
        
        Arguments:
            clustertype: Clustering name used in scanpy. ('leiden')
            cluster: Only deal with one cluster of marker genes. ('all')
            rank_rep: Whether to repeat ranking. (False)
        
        Returns:
            result: Annotation result as DataFrame
        """

        dat=data_preprocess(self.adata,clustertype=clustertype,path='temp/rna.csv',rank_rep=rank_rep)
        dat.to_csv('temp/rna.csv')

        print('...Auto annotate cell')
        
        output_path = self.output or 'temp/rna_anno.txt'

        p = Process()
        p.run_cmd_p(foldchange=self.foldchange,
                    weight=self.weight,
                    pvalue=self.pvalue,
                    tissue=self.tissue,
                    species=self.species,
                    target=self.target,
                    norefdb=self.norefdb,
                    MarkerDB=None,
                    db=self.model_path,
                    noprint=self.noprint,
                    input="temp/rna.csv",
                    output=output_path,
                    source="scanpy",
                    cluster=cluster,
                    fc=self.foldchange,
                    outfmt=self.outfmt,
                    celltype=self.celltype,
                    Gensymbol=self.Gensymbol,
                    list_tissue=self.list_tissue,
                    cellrange=self.cellrange)

        # Load results from the requested output path (fallback to default)
        result_path = output_path if output_path else 'temp/rna_anno.txt'
        result=pd.read_csv(result_path,sep='\t')
        self.result=result
        add_reference(self.adata,'pySCSA','cell annotation with SCSA')
        return result
    
    def cell_anno_print(self)->None:
        r"""Print the annotation result.

        Returns:
            None
        """
        for i in set(self.result['Cluster']):
            test=self.result.loc[self.result['Cluster']==i].iloc[:2]
            if len(test) >= 2 and pd.notna(test.iloc[0]['Z-score']) and pd.notna(test.iloc[1]['Z-score']):
                if test.iloc[0]['Z-score']>test.iloc[1]['Z-score']*2:
                    print('Nice:Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,test.iloc[0]['Cell Type'],
                                                                np.around(test.iloc[0]['Z-score'],3)))
                else:
                    print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,('|').join(test['Cell Type'].values.tolist()),
                                                                ('|').join(np.around(test['Z-score'].values,3).astype(str).tolist())))
            elif len(test) >= 1 and pd.notna(test.iloc[0]['Z-score']):
                print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,test.iloc[0]['Cell Type'],
                                                            np.around(test.iloc[0]['Z-score'],3)))
            else:
                print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,'Unknown','NaN'))

    def cell_auto_anno(self,adata:anndata.AnnData,
                       clustertype:str='leiden',key='scsa_celltype')->None:
        r"""Add cell type annotation to anndata.obs['scsa_celltype'].
        
        Arguments:
            adata: anndata object
            clustertype: Clustering name used in scanpy. ('leiden')
            key: Key to store cell type annotation. ('scsa_celltype')
        
        Returns:
            None
        """
        # If annotation results are not present, run cell_anno first
        if not hasattr(self, "result") or self.result is None:
            self.cell_anno(clustertype=clustertype, cluster='all')

        test_li=[]
        for i in adata.obs[clustertype].value_counts().index:
            if int(i) in self.result['Cluster'].values:
                test_li.append(self.result.loc[self.result['Cluster']==int(i)].iloc[0]['Cell Type'])
            else:
                test_li.append('Unknown')
        scsa_anno=dict(zip([str(i) for i in adata.obs[clustertype].value_counts().index],
            test_li))
        adata.obs[key] = adata.obs[clustertype].map(scsa_anno).astype('category')
        print('...cell type added to {} on obs of anndata'.format(key))

    def get_celltype_marker(self,adata:anndata.AnnData,
                            clustertype:str='leiden',
                            log2fc_min:int=2,scores_type='scores',
                            pval_cutoff:float=0.05,rank:bool=True,
                            unique:bool=True,global_unique:bool=False)->dict:
        r"""Get marker genes for each clusters.
        
        Arguments:
            adata: anndata object
            clustertype: Clustering name used in scanpy. (leiden)
            log2fc_min: Minimum log2 fold change of marker genes. (2)
            pval_cutoff: Maximum p value of marker genes. (0.05)
            rank: Whether to rank genes by wilcoxon test. (True)
            scores_type: The type of scores. can be selected from `scores` and `logfoldchanges`
            unique: Whether to remove duplicates within each cell type. (True)
            global_unique: Whether to remove duplicates across all cell types. (False)

        Returns:
            cellmarker: A dictionary of marker genes for each clusters.
        """
        print('...get cell type marker')
        cell_marker_dict=get_celltype_marker(adata=adata,
                            clustertype=clustertype,
                            log2fc_min=log2fc_min,scores_type=scores_type,
                            pval_cutoff=pval_cutoff,rank=rank,
                            unique=unique,global_unique=global_unique)

        return cell_marker_dict
    


class MetaTiME(object):
    """
    MetaTiME: Meta-components in Tumor immune MicroEnvironment

    Github: https://github.com/yi-zhang/MetaTiME/
    
    """
    
    def __init__(self,adata:anndata.AnnData,mode:str='table'):
        """
        Initialize MetaTiME model

        Arguments:
            adata: anndata object
            mode: choose from ['mecnamedict', 'table', 'meciddict']
                    load manual assigned name for easy understanding of assigned names
                    from file: MeC_anno_name.tsv under mecDIR.
                    Required columns: `['MeC_id', 'Annotation', 'UseForCellStateAnno']` 
                    Required seperator: tab
                    Annotation column NA will be filtered.
                    If you want to use your own annotation, please follow the format of MeC_anno_name.tsv

        """
        check_metatime()
        global metatime_install
        if metatime_install==True:
            global mecs
            global mecmapper
            global annotator
            global config
            from metatime import mecs
            from metatime import mecmapper
            from metatime import annotator
            from metatime import config

        self.adata=adata
        # Load the pre-trained MeCs
        print('...load pre-trained MeCs')
        self.mecmodel = mecs.MetatimeMecs.load_mec_precomputed()

        # Load functional annotation for MetaTiME-TME
        print('...load functional annotation for MetaTiME-TME')
        self.mectable = mecs.load_mecname(mecDIR = config.SCMECDIR, mode =mode )
        self.mecnamedict = mecs.getmecnamedict_ct(self.mectable) 
        
        
    
    def overcluster(self,resolution : float=8, 
                random_state: int= 0, 
                clustercol :str = 'overcluster'):
        """
        Overcluster single cell data to get cluster level cell state annotation

        Arguments:
            resolution: resolution for leiden clustering
            random_state: random state for leiden clustering
            clustercol: column name for cluster level cell state annotation
        
        """

        print('...overclustering using leiden')
        sc.tl.leiden(self.adata, resolution=resolution, key_added = clustercol, random_state=random_state)
        self.clustercol=clustercol
        #self.adata = annotator.overcluster(self.adata,resolution,random_state,clustercol) # this generates a 'overcluster' columns in adata.obs
        
    def predictTiME(self,save_obs_name:str='MetaTiME'):
        """
        Predict TiME celtype for each cell

        Arguments:
            save_obs_name: column name for cell type annotation in adata.obs
        
        """
        print('...projecting MeC scores')
        self.pdata=mecmapper.projectMecAnn(self.adata, self.mecmodel.mec_score)
        projmat, mecscores = annotator.pdataToTable(self.pdata, self.mectable, gcol = self.clustercol)
        projmat, gpred, gpreddict = annotator.annotator(projmat,  self.mecnamedict, gcol = self.clustercol)
        self.adata = annotator.saveToAdata( self.adata, projmat)
        #self.pdata = annotator.saveToPdata( self.pdata, self.adata, projmat )
        self.adata.obs[save_obs_name] = self.adata.obs['{}_{}'.format(save_obs_name,self.clustercol)].str.split(': ').str.get(1)
        #self.pdata.obs[save_obs_name] = self.pdata.obs['{}_{}'.format(save_obs_name,self.clustercol)].str.split(': ').str.get(1)
        self.adata.obs['MetaTiME']=self.adata.obs['MetaTiME'].fillna('Unknown')
        self.adata.obs['Major_{}'.format(save_obs_name)]=[i.split('_')[0] for i in self.adata.obs[save_obs_name]]
        for i in self.adata.obs.columns:
            if 'MeC_' in i:
                del self.adata.obs[i]
        #self.pdata.obs['Major_{}'.format(save_obs_name)]=[i.split('_')[0] for i in self.pdata.obs[save_obs_name]]
        print('......The predicted celltype have been saved in obs.{}'.format(save_obs_name))
        print('......The predicted major celltype have been saved in obs.Major_{}'.format(save_obs_name))
        add_reference(self.adata,'MetaTiME','cell annotation with MetaTiME')
        return self.adata
    
    def plot(self,basis:str='X_umap',cluster_key:str='MetaTiME',fontsize:int=8, 
             min_cell:int=5, title=None,figsize:tuple=(6,6),
             dpi:int=80,frameon:bool=False,legend_loc=None,palette=None):
        """
        Plot annotated cells with  non-overlapping fonts.

        Arguments:
            basis: basis for plotting
            cluster_key: column name for cell type annotation in adata.obs
            fontsize: fontsize for plotting
            min_cell: minimum number of cells for plotting
            title: title for plotting
            figsize: figure size for plotting
            dpi: dpi for plotting
            frameon: frameon for plotting
            legend_loc: legend_loc for plotting
            palette: palette for plotting

        Returns:
            fig: figure object
            ax: axis object
        
        """
        import matplotlib.pyplot as plt
        if not title:
            title = cluster_key

        if( min_cell >0 ):
            groupcounts = self.adata.obs.groupby( cluster_key ).count()
            groupcounts = groupcounts[groupcounts.columns[0]]
            group_with_good_counts = groupcounts[groupcounts>= min_cell ].index.tolist()
            self.adata = self.adata[ self.adata.obs[ cluster_key ].isin( group_with_good_counts ) ]

        if palette ==None:
            palette=plt.cycler("color",plt.cm.tab20(np.linspace(0,1,20)))

        with plt.rc_context({"figure.figsize": figsize, "figure.dpi": dpi, "figure.frameon": frameon}):
            #ax = sc.pl.umap(pdata, color="MetaTiME_overcluster", show=False, legend_loc=None, frameon=False, size=30)
            ax = sc.pl.embedding(self.adata, basis=basis,color= cluster_key , show=False, legend_loc=None, add_outline=False, 
                    #legend_loc='on data',legend_fontsize=6, legend_fontoutline=2,
                    title= title, 
                    palette=palette, 
                    #palette=plt.cycler("color",plt.cm.Set1(np.linspace(0,1,9))), 
                    frameon=frameon
                    )
            gen_mpl_labels(
                self.adata,
                cluster_key,
                exclude=("None",),  
                basis=basis,
                ax=ax,
                adjust_kwargs=dict(arrowprops=dict(arrowstyle='-', color='black')),
                text_kwargs=dict(fontsize= fontsize ,weight='bold'),
            )
            fig = ax.get_figure()
            fig.tight_layout()
            #plt.show()
            return( fig,ax )

        
