import os
import pandas as pd
import numpy as np
from tqdm import trange
import scanpy as sc
import scipy.sparse as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from .utils import var_stabilize
import math



def plot_gene(adata, platform="visium", sorted_by='PI', input_data=None, size=2, 
              top_n=25, ncols_each_sheet=5, nrows_each_sheet=5,save_path=None):
    if sp.issparse(adata.X):
        gene_data = adata.X.A.T
    else:
        gene_data = adata.X.T
    if platform=="visium":
        gene_data = var_stabilize(gene_data)

    x = adata.obsm["spatial"][:,1] 
    y = adata.obsm["spatial"][:,0]
    
    ncols = ncols_each_sheet
    nrows = nrows_each_sheet
    
    if platform=="visium" or platform=="ST":
        cmp = mpl.colors.LinearSegmentedColormap.from_list('pink_green', ["#3AB370","#EAE7CC","#FD1593"], N=256)
    else:
        nodes = [0.0, 0.05, 0.3, 1.0]
        cmp = mpl.colors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, ["#EAE7CC","#EAE7CC","#FD1593","#FD1593"])))

    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Arial:italic:bold'
    mpl.rcParams['mathtext.it'] = 'Arial:italic:bold'     
    
    if sorted_by in ["SEP", "SIG", "PI"]:
        pi = pd.DataFrame(adata.var[sorted_by].reset_index())
        # sort gene according to 
        sorted_list = pi.sort_values(by = sorted_by, ascending = False)
        sorted_list.columns = ["geneID",sorted_by]
        sorted_idx = sorted_list.index.values
        gene_data = gene_data[sorted_idx, :]        
        gene_data = gene_data[:top_n]
        sorted_genename = sorted_list["geneID"].reset_index(drop = True)
        sorted_pi = sorted_list[sorted_by].reset_index(drop = True)
    else:
        adata.var_names = [i.upper() for i in list(adata.var_names)]
        assert input_data["genename"] is not None or input_data["qvalue"] is not None,"please set \" input_data.columns=['genename','qvalue']\" "
        input_data = input_data.sort_values("qvalue")
        sorted_genename = input_data["genename"]
        sorted_genename = pd.Series([i.upper() for i in sorted_genename])
        sorted_idx = np.ones(sorted_genename.shape).astype(int)
        for j in range(sorted_genename.shape[0]):
            sorted_idx[j] = np.where(adata.var_names==sorted_genename[j])[0]       
        gene_data = gene_data[sorted_idx, :]
        gene_data = gene_data[:top_n]
        sorted_qval = input_data["qvalue"]
        sorted_qval = sorted_qval.reset_index(drop = True)
    print("\nDrawing pictures:")    
    for figs in trange(0, len(gene_data), ncols * nrows):
        if figs < len(gene_data):
            n = figs
        else:
            n = int(len(gene_data)/ ncols/ nrows) * ncols * nrows + 1
        figure, axs = plt.subplots(nrows, ncols, figsize=(12,16))
        
        for i in range(ncols):
            for j in range(nrows):
                label = gene_data[n, :]
                if sorted_by == None:
                    genename = adata.var_names[n]
                elif sorted_by in ["SEP", "SIG", "PI"]:
                    genename = sorted_genename[n]
                    pi_value = sorted_pi[n]
                else:
                    genename = sorted_genename[n]
                    qval = sorted_qval[n]
                # axs = axs.T
                ax = axs[i][j]

                ax.scatter(y, x, s = size, c = label, cmap = cmp)

                if sorted_by == None:   
                    ax.set_title('${}$'.format(genename))
                elif sorted_by in ["SEP", "SIG", "PI"]:
                    ax.set_title('${}$'.format(genename) + ' ({}={:.3f})'.format(sorted_by,pi_value), fontdict = {'family': 'Arial','weight': 'bold','style': 'normal','size': 11})
                else:
                    ax.set_title('${}$'.format(genename) + ' (q-value={:g})'.format(qval), fontdict = {'family': 'Arial','weight': 'bold','style': 'normal','size': 11})
                ax.set_xlim(min(y)-20, max(y)+20)
                ax.set_ylim(min(x)-20, max(x)+20)
            
                x_major_locator = MultipleLocator(2000)
                y_major_locator = MultipleLocator(2000)                
                ax.xaxis.set_major_locator(x_major_locator)
                ax.yaxis.set_major_locator(y_major_locator)
                if platform=="visium" or platform=="SeqFISH" or platform=="ST":
                    ax.invert_yaxis()               
                ax.set_aspect('equal', 'box')                
                n += 1
               
                if n >= len(gene_data):
                    break
            if n >= len(gene_data):
                    break
                
        for i in range(nrows):
            for j in range(ncols):
                ax = axs[i][j]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False) 
                ax.spines['bottom'].set_visible(False) 
                ax.spines['left'].set_visible(False)  
                ax.spines['right'].set_visible(False)
                
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust()
        if save_path:
            if not os.path.isdir("{}/SVGs".format(save_path)):
                os.makedirs("{}/SVGs".format(save_path))
            plt.savefig("{}/SVGs/P{}.png".format(save_path, math.ceil(n/(ncols*nrows))), dpi=600, bbox_inches='tight')       
    print("\nDrawing completed !!")
    

    
    