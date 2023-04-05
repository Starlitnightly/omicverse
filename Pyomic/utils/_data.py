r"""
Pyomic data (Pyomic.utils._data)
"""


import time
import requests
import os
import pandas as pd
import scanpy as sc


def read(path):
    if path.split('.')[-1]=='h5ad':
        return sc.read(path)
    elif path.split('.')[-1]=='csv':
        return pd.read_csv(path)
    elif path.split('.')[-1]=='tsv':
        return pd.read_csv(path,sep='\t')
    else:
        raise ValueError('The type is not supported.')
    
def read_csv(**kwargs):
    return pd.read_csv(**kwargs)

def read_10x_mtx(**kwargs):
    return sc.read_10x_mtx(**kwargs)

def read_h5ad(**kwargs):
    return sc.read_h5ad(**kwargs)

def read_10x_h5(**kwargs):
    return sc.read_10x_h5(**kwargs)


def data_downloader(url,path,title):
    r"""datasets downloader
    
    Arguments
    ---------
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

    chunk_size = 1024000
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

def download_CaDRReS_model():
    r"""load CaDRReS_model
    
    Parameters
    ---------

    Returns
    -------

    """
    _datasets = {
        'cadrres-wo-sample-bias_output_dict_all_genes':'https://figshare.com/ndownloader/files/39753568',
        'cadrres-wo-sample-bias_output_dict_prism':'https://figshare.com/ndownloader/files/39753571',
        'cadrres-wo-sample-bias_param_dict_all_genes':'https://figshare.com/ndownloader/files/39753574',
        'cadrres-wo-sample-bias_param_dict_prism':'https://figshare.com/ndownloader/files/39753577',
    }
    for datasets_name in _datasets.keys():
        print('......CaDRReS model download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],path='models/{}.pickle'.format(datasets_name),title=datasets_name)
    print('......CaDRReS model download finished!')

def download_GDSC_data():
    r"""load GDSC_data
    
    Parameters
    ---------

    Returns
    -------

    """
    _datasets = {
        'masked_drugs':'https://figshare.com/ndownloader/files/39753580',
        'GDSC_exp':'https://figshare.com/ndownloader/files/39744025',
    }
    for datasets_name in _datasets.keys():
        print('......GDSC data download start:',datasets_name)
        if datasets_name == 'masked_drugs':
            data_downloader(url=_datasets[datasets_name],path='models/{}.csv'.format(datasets_name),title=datasets_name)
        elif datasets_name == 'GDSC_exp':
            data_downloader(url=_datasets[datasets_name],path='models/{}.tsv.gz'.format(datasets_name),title=datasets_name)
    print('......GDSC data download finished!')

def download_pathway_database():
    r"""load pathway_database

    """
    _datasets = {
        'GO_Biological_Process_2021':'https://figshare.com/ndownloader/files/39820720',
        'GO_Cellular_Component_2021':'https://figshare.com/ndownloader/files/39820714',
        'GO_Molecular_Function_2021':'https://figshare.com/ndownloader/files/39820711',
        'WikiPathway_2021_Human':'https://figshare.com/ndownloader/files/39820705',
        'WikiPathways_2019_Mouse':'https://figshare.com/ndownloader/files/39820717',
        'Reactome_2022':'https://figshare.com/ndownloader/files/39820702',
    }
     
    for datasets_name in _datasets.keys():
        print('......Pathway Geneset download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],path='genesets/{}.txt'.format(datasets_name),title=datasets_name)
    print('......Pathway Geneset download finished!')

def download_geneid_annotation_pair():
    r"""load geneid_annotation_pair

    """
    _datasets = {
        'pair_GRCm39':'https://figshare.com/ndownloader/files/39820684',
        'pair_T2TCHM13':'https://figshare.com/ndownloader/files/39820687',
        'pair_GRCh38':'https://figshare.com/ndownloader/files/39820690',
        'pair_GRCh37':'https://figshare.com/ndownloader/files/39820693',
        'pair_danRer11':'https://figshare.com/ndownloader/files/39820696',
        'pair_danRer7':'https://figshare.com/ndownloader/files/39820699',
    }
     
    for datasets_name in _datasets.keys():
        print('......Geneid Annotation Pair download start:',datasets_name)
        model_path = data_downloader(url=_datasets[datasets_name],path='genesets/{}.tsv'.format(datasets_name),title=datasets_name)
    print('......Geneid Annotation Pair download finished!')

def geneset_prepare(geneset_path,organism='Human'):
    r"""load geneset

    Parameters
    ----------
    - geneset_path: `str`
        Path of geneset file.
    - organism: `str`
        Organism of geneset file. Default: 'Human'

    Returns
    -------
    - go_bio_dict: `dict`
        A dictionary of geneset.
    """
    go_bio_geneset=pd.read_csv(geneset_path,sep='\t\t',header=None)
    go_bio_dict={}
    if (organism == 'Mouse') or (organism == 'mouse') or (organism == 'mm'):
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i.lower().capitalize() for i in go_bio_geneset.loc[i,1].split('\t')]
    elif (organism == 'Human') or (organism == 'human') or (organism == 'hs'):
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i.upper() for i in go_bio_geneset.loc[i,1].split('\t')]
    else:
        for i in go_bio_geneset.index:
            go_bio_dict[go_bio_geneset.loc[i,0]]=[i.upper() for i in go_bio_geneset.loc[i,1].split('\t')]
    return go_bio_dict


    