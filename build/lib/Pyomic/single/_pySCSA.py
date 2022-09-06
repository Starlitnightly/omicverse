import os 
import scanpy as sc
import pandas as pd
import numpy as np
from ._SCSA import Process
import sys
import argparse
import gzip
import time
import requests

def data_downloader(url,path,title):
    r"""datasets downloader
    
    Arguments
    ---------
    url
        the download url of datasets
    path
        the save path of datasets
    title
        the name of datasets
    
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


def data_preprocess(adata,path='temp/rna.csv'):
    dirname, _ = os.path.split(path)
    try:
        if not os.path.isdir(dirname):
            print("......Creating directory {}".format(dirname))
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        print("......Unable to create directory {}. Reason {}".format(dirname,e))

    sc.settings.verbosity = 2  # reduce the verbosity
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    dat = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'logfoldchanges','scores','pvals']})
    dat.to_csv(path)
    return dat

def cell_annotate(data,
                foldchange=1.5,pvalue=0.05,
                output='temp/rna_anno.txt',
                outfmt='txt',Gensymbol=True,
                species='Human',weight=100,tissue='All',
                celltype='normal',norefdb=False,noprint=True,list_tissue=False):
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

def cell_anno_print(anno):
    for i in set(anno['Cluster']):
        test=anno.loc[anno['Cluster']==i].iloc[:2]
        if test.iloc[0]['Z-score']>test.iloc[1]['Z-score']*2:
            print('Nice:Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,test.iloc[0]['Cell Type'],
                                                        np.around(test.iloc[0]['Z-score'],3)))
        else:
            print('Cluster:{}\tCell_type:{}\tZ-score:{}'.format(i,('|').join(test['Cell Type'].values.tolist()),
                                                        ('|').join(np.around(test['Z-score'].values,3).astype(str).tolist())))