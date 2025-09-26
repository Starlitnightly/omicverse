import pdb,sys,os
import argparse
import anndata
import scanpy as sc
import numpy as np
from scipy import sparse



def get_eg_representatives(name:str) -> None:
    """
    Used for acquiring representatives' single-cell data in the example. Automatically check the latest representatives and store their single-cell data as /representative_sc.h5ad under the project's directory. 
    
    Parameters
    ----------
    name 
        Project name

    Returns
    -------
        None

    Example
    -------
    >>> name = 'project_name'
    >>> scSemiProfiler.get_eg_representatives(name)

    """
    
    
    scdata = anndata.read_h5ad('example_data/scdata.h5ad')
    sids = []
    f = open(name + '/sids.txt', 'r')
    lines = f.readlines()
    for l in lines:
        sids.append(l.strip())
    f.close()
    
    # get the latest round
    representatives = []
    files = os.listdir(name+'/status/')
    rounds = [0]
    for file in files:
        if 'representative' in file:
            f = open(name + '/status/' + file, 'r')
            lines = f.readlines()
            if len(lines) > len(representatives):
                representatives = []
                for l in lines:
                    representatives.append(int(l.strip()))
            f.close()
    
    rsids=[]
    for r in representatives:
        sid = sids[r]
        rsids.append(sid)
    
    rmask=[]    
    for i in range(len(scdata.obs.index)):
        sid = scdata.obs['sample_ids'][i]
        if sid in rsids:
            rmask.append(True)
        else:
            rmask.append(False)
    rmask = np.array(rmask)
    
    repredata = scdata[rmask,:]
    
    X = repredata.X
    X = np.array(X.todense())
    X = np.exp(X) - 1
    X = sparse.csr_matrix(X)
    adata = anndata.AnnData(X)
    adata.obs = repredata.obs
    adata.var = repredata.var
    
    adata.write(name + '/representative_sc.h5ad')
    
    print('Obtained single-cell data for representatives.')
    
    return









def main():
    parser = argparse.ArgumentParser(description="scSemiProfiler initsetup")
    #parser._action_groups.pop()
    parser.add_argument('--name', help='Project name.')
    args = parser.parse_args()
    #required = parser.add_argument_group('required arguments')
    #optional = parser.add_argument_group('optional arguments')
    
    #required.add_argument('--name',required=True,help="Project name.")
    
    #optional.add_argument('--na',required=False, default='new_project', help="Pe.")

    name = args.name
    
    get_eg_representatives(name)

if __name__=="__main__":
    main()

