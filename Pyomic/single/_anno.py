import os 
import scanpy as sc
import pandas as pd
import numpy as np
from ._SCSA import Process,Annotator
import sys
import argparse
import gzip
import time
import requests
import anndata

def data_downloader(url,path,title):
    r"""datasets downloader
    
    Parameters
    ----------
    - url: `str`
        the download url of datasets
    - path: `str`
        the save path of datasets
    - title: `str`
        the name of datasets
    
    Returns
    -------
    - path: `str`
        the save path of datasets
    """
    if os.path.isfile(path):
        print("......Loading dataset from {}".format(path))
        return path
    else:
        print("......Downloading dataset save to {}".format(path))
        
    dirname, _ = os.path.split(path)
    try:
        if not os.path.isdir(dirname):
            print("......Creating directory {}".format(dirname))
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        print("......Unable to create directory {}. Reason {}".format(dirname,e))
    
    
    start = time.time()
    size = 0
    res = requests.get(url, stream=True)

    chunk_size = 102400
    content_size = int(res.headers["content-length"]) 
    if res.status_code == 200:
        print('......[%s Size of file]: %0.2f MB' % (title, content_size/chunk_size/10.24))
        with open(path, 'wb') as f:
            for data in res.iter_content(chunk_size=chunk_size):
                f.write(data)
                size += len(data) 
                print('\r'+ '......[Downloader]: %s%.2f%%' % ('>'*int(size*50/content_size), float(size/content_size*100)), end='')
        end = time.time()
        print('\n' + ".......Finish！%s.2f s" % (end - start))
    
    return path


def data_preprocess(adata,clustertype='leiden',path='temp/rna.csv'):
    r"""data preprocess for SCSA
    
    Parameters
    ----------
    - adata: `AnnData`
        AnnData object
    - path: `str`   
        the save path of datasets

    Returns
    -------
    - adata: `AnnData`
        AnnData object
    """
    dirname, _ = os.path.split(path)
    try:
        if not os.path.isdir(dirname):
            print("......Creating directory {}".format(dirname))
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        print("......Unable to create directory {}. Reason {}".format(dirname,e))

    sc.settings.verbosity = 2  # reduce the verbosity
    if 'rank_genes_groups' not in adata.uns.keys():
        sc.tl.rank_genes_groups(adata, clustertype, method='wilcoxon')
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
    r"""cell annotation by SCSA
    
    Parameters
    ----------
    - data: `AnnData`
        AnnData object
    - foldchange: `float`
        foldchange threshold
    - pvalue: `float`
        pvalue threshold
    - output: `str`
        the save path of annotation result
    - outfmt: `str`
        the format of annotation result
    - Gensymbol: `bool`
        whether to use gene symbol
    - species: `str`
        the species of datasets
    - weight: `int`
        the weight of datasets
    - tissue: `str`
        the tissue of datasets
    - celltype: `str`
        the celltype of datasets
    - norefdb: `bool`
        whether to use reference database
    - noprint: `bool`
        whether to print the result
    - list_tissue: `bool`
        whether to list the tissue of datasets
    
    Returns
    -------
    - result: `pandas.DataFrame`
        the annotation result
    """
    data.to_csv('temp/rna.csv')

    #https://figshare.com/ndownloader/files/37262710
    model_path = data_downloader(url='https://figshare.com/ndownloader/files/37262710',
                                    path='temp/whole.db',title='whole')
    print('......Auto annotate cell')
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
    r"""print the annotation result
    
    Parameters
    ----------
    - anno: `pandas.DataFrame`
        the annotation result
    Returns
    -------

    """
    for i in set(anno['Cluster']):
        test=anno.loc[anno['Cluster']==i].iloc[:2]
        if test.iloc[0]['Z-score']>test.iloc[1]['Z-score']*2:
            print('Nice:Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,test.iloc[0]['Cell Type'],
                                                        np.around(test.iloc[0]['Z-score'],3)))
        else:
            print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,('|').join(test['Cell Type'].values.tolist()),
                                                        ('|').join(np.around(test['Z-score'].values,3).astype(str).tolist())))

def scanpy_lazy(adata:anndata.AnnData,min_genes:int=200,min_cells:int=3,drop_doublet:bool=True,
                n_genes_by_counts:int=4300,pct_counts_mt:int=25,
                target_sum:float=1e4,min_mean:float=0.0125, max_mean:int=3, min_disp:float=0.5,max_value:int=10,
                n_comps:int=100, svd_solver:str="auto",
                n_neighbors:int=15, random_state:int = 112, n_pcs:int=50,
                )->anndata.AnnData:
    r"""scanpy lazy analysis
    
    Arguments:
        adata: AnnData object
        min_genes: the min number of genes
        min_cells: the min number of cells
        drop_doublet: whether to drop doublet
        n_genes_by_counts: the max number of genes
        pct_counts_mt: the max proportion of mito-genes
        target_sum: the max counts of total_counts
        min_mean: the min mean of genes
        max_mean: the max mean of genes
        min_disp: the min dispersion of genes
        max_value: the max value of genes
        n_comps: the number of components
        svd_solver: the solver of svd
        n_neighbors: the number of neighbors
        random_state: the random state
        n_pcs: the number of pcs

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
    sc.pp.scale(adata, max_value=max_value)
    #pca analysis
    sc.tl.pca(adata, n_comps=n_comps, svd_solver=svd_solver)
    #cell neighbors graph construct
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state = random_state, n_pcs=n_pcs)
    #umap
    sc.tl.leiden(adata)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    sc.tl.umap(adata, init_pos='paga')
    return adata

def scanpy_cellanno_from_dict(adata:anndata.AnnData,
                               anno_dict:dict,
                               anno_name:str='major',
                               clustertype:str='leiden',
                               ):
    r"""add cell type annotation from dict to anndata object

    Arguments:
        adata: AnnData object of scRNA-seq after preprocessing
        anno_dict: dict of cell type annotation. key is the cluster name, value is the cell type name.like `{'0':'B cell','1':'T cell'}`
        anno_name: the name of annotation
        clustertype: Clustering name used in scanpy. (leiden)

    """

    adata.obs[anno_name+'_celltype'] = adata.obs[clustertype].map(anno_dict).astype('category')
    print('...cell type added to {}_celltype on obs of anndata'.format(anno_name))


class pySCSA(object):

    def __init__(self,adata:anndata.AnnData,
                foldchange:float=1.5,pvalue:float=0.05,
                output:str='temp/rna_anno.txt',
                model_path:str='',
                outfmt:str='txt',Gensymbol:bool=True,
                species:str='Human',weight:int=100,tissue:str='All',target:str='cellmarker',
                celltype:str='normal',norefdb:bool=False,noprint:bool=True,list_tissue:bool=False) -> None:

        r"""Initialize the pySCSA class

        Arguments:
            adata: AnnData object of scRNA-seq after preprocessing
            foldchange: Fold change threshold for marker filtering. (2.0)
            pvalue: P-value threshold for marker filtering. (0.05)
            output: Output file for marker annotation.(temp/rna_anno.txt)
            model_path: Path to the Database for annotation. If not provided, the model will be downloaded from the internet.
            outfmt: Output format for marker annotation. (txt)
            Gensymbol: Using gene symbol ID instead of ensembl ID in input file for calculation.
            species: Species for annotation. Only used for cellmarker database. ('Human',['Mouse'])
            weight: Weight threshold for marker filtering from cellranger v1.0 results. (100)
            tissue: Tissue for annotation. you can use `get_model_tissue` to see the available tissues. ('All')
            target: Target to annotation class in Database. (cellmarker,[cancersea])
            celltype: Cell type for annotation. (normal,[cancer])
            norefdb: Only using user-defined marker database for annotation.
            noprint: Do not print any detail results.
            list_tissue: List all available tissues in the database.
        
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
        self.species=species
        self.weight=weight
        self.tissue=tissue
        self.celltype=celltype
        self.norefdb=norefdb
        self.noprint=noprint
        self.list_tissue=list_tissue
        self.target=target
        if model_path =='':
            self.model_path=data_downloader(url='https://figshare.com/ndownloader/files/40053640',
                                            path='temp/pySCSA_2023.db',title='whole')
        else:
            self.model_path=model_path

    def get_model_tissue(self,species:str="Human")->None:
        r"""List all available tissues in the database.
        
        Arguments:
            species: Species for annotation. Only used for cellmarker database. ('Human',['Mouse'])

        """
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('-i', '--input', default = "temp/rna.csv")
        parser.add_argument('-o', '--output',default=self.output)
        parser.add_argument('-d', '--db', default = self.model_path,)
        parser.add_argument('-s', '--source', default = "scanpy",)
        parser.add_argument('-c', '--cluster', default = 'all',)
        parser.add_argument('-f',"--fc",default = self.foldchange,)
        parser.add_argument('-fc',"--foldchange",default =self.foldchange,)
        parser.add_argument('-p',"--pvalue",default = self.pvalue,)
        parser.add_argument('-w',"--weight",default = self.weight,)
        parser.add_argument('-g',"--species",default = self.species,)
        parser.add_argument('-k',"--tissue",default = self.tissue,)
        parser.add_argument('-m', '--outfmt', default = self.outfmt, )
        parser.add_argument('-T',"--celltype",default = self.celltype,)
        parser.add_argument('-t', '--target', default = self.target,)
        parser.add_argument('-E',"--Gensymbol",action = "store_true",default=self.Gensymbol,)
        parser.add_argument('-N',"--norefdb",action = "store_true",default=self.norefdb,)
        parser.add_argument('-b',"--noprint",action = "store_true",default=self.noprint,)
        parser.add_argument('-l',"--list_tissue",action = "store_true",default = 'True',)
        parser.add_argument('-M', '--MarkerDB',)

        args = parser.parse_args()
        
        anno = Annotator(args)
        anno.load_pickle_module(self.model_path)
        anno.get_list_tissue(species)


    def cell_anno(self,clustertype:str='leiden',cluster:str='all')->pd.DataFrame:
        r"""Annotate cell type for each cluster.
        
        Arguments:
            clustertype: Clustering name used in scanpy. (leiden)
            cluster: Only deal with one cluster of marker genes. (all,[1],[1,2,3],[...])
        """

        dat=data_preprocess(self.adata,clustertype=clustertype,path='temp/rna.csv')
        dat.to_csv('temp/rna.csv')

        print('...Auto annotate cell')
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('-i', '--input', default = "temp/rna.csv")
        parser.add_argument('-o', '--output',default=self.output)
        parser.add_argument('-d', '--db', default = self.model_path,)
        parser.add_argument('-s', '--source', default = "scanpy",)
        parser.add_argument('-c', '--cluster', default = cluster,)
        parser.add_argument('-f',"--fc",default = self.foldchange,)
        parser.add_argument('-fc',"--foldchange",default =self.foldchange,)
        parser.add_argument('-p',"--pvalue",default = self.pvalue,)
        parser.add_argument('-w',"--weight",default = self.weight,)
        parser.add_argument('-g',"--species",default = self.species,)
        parser.add_argument('-k',"--tissue",default = self.tissue,)
        parser.add_argument('-m', '--outfmt', default = self.outfmt, )
        parser.add_argument('-T',"--celltype",default = self.celltype,)
        parser.add_argument('-t', '--target', default = self.target,)
        parser.add_argument('-E',"--Gensymbol",action = "store_true",default=self.Gensymbol,)
        parser.add_argument('-N',"--norefdb",action = "store_true",default=self.norefdb,)
        parser.add_argument('-b',"--noprint",action = "store_true",default=self.noprint,)
        parser.add_argument('-l',"--list_tissue",action = "store_true",default = self.list_tissue,)
        parser.add_argument('-M', '--MarkerDB',)
        args = parser.parse_args()

        p = Process()
        p.run_cmd(args)

        result=pd.read_csv('temp/rna_anno.txt',sep='\t')
        self.result=result
        return result
    
    def cell_anno_print(self)->None:
        r"""print the annotation result
        
        """
        for i in set(self.result['Cluster']):
            test=self.result.loc[self.result['Cluster']==i].iloc[:2]
            if test.iloc[0]['Z-score']>test.iloc[1]['Z-score']*2:
                print('Nice:Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,test.iloc[0]['Cell Type'],
                                                            np.around(test.iloc[0]['Z-score'],3)))
            else:
                print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,('|').join(test['Cell Type'].values.tolist()),
                                                            ('|').join(np.around(test['Z-score'].values,3).astype(str).tolist())))

    def cell_auto_anno(self,adata:anndata.AnnData,clustertype:str='leiden')->None:
        r"""Add cell type annotation to anndata.obs['scsa_celltype']
        
        Arguments:
            adata: anndata object
            clustertype: Clustering name used in scanpy. (leiden)
        """
        scsa_anno=dict(zip([str(i) for i in range(len(adata.obs[clustertype].value_counts().index))],
            [self.result.loc[self.result['Cluster']==i].iloc[0]['Cell Type'] for i in range(len(adata.obs[clustertype].value_counts().index))]))
        adata.obs['scsa_celltype'] = adata.obs['leiden'].map(scsa_anno).astype('category')
        print('...cell type added to scsa_celltype on obs of anndata')

    def get_celltype_marker(self,adata:anndata.AnnData,
                            clustertype:str='leiden',
                            log2fc_min:int=2,
                            pval_cutoff:float=0.05)->dict:
        r"""Get marker genes for each clusters.
        
        Arguments:
            adata: anndata object
            clustertype: Clustering name used in scanpy. (leiden)
        """
        print('...get cell type marker')
        celltypes = sorted(adata.obs[clustertype].unique())
        cell_marker_dict={}
        if 'rank_genes_groups' not in adata.uns.keys():
            sc.tl.rank_genes_groups(adata, clustertype, method='wilcoxon')
        for celltype in celltypes:
            degs = sc.get.rank_genes_groups_df(adata, group=celltype, key='rank_genes_groups', log2fc_min=log2fc_min, 
                                            pval_cutoff=pval_cutoff)
            foldp=np.histogram(degs['scores'])
            foldchange=(foldp[1][np.where(foldp[1]>0)[0][-5]]+foldp[1][np.where(foldp[1]>0)[0][-6]])/2
            
            cellmarker=degs.loc[degs['scores']>foldchange]['names'].values
            cell_marker_dict[celltype]=cellmarker

        return cell_marker_dict