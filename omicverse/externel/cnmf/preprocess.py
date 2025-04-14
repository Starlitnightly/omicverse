import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Collection
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from scipy.sparse import hstack, issparse

def moe_correct_ridge(Z_orig, Z_cos, Z_corr, R, W, K, Phi_Rk, Phi_moe, lamb):
    Z_corr = Z_orig.copy()
    for i in range(K):
        Phi_Rk = np.multiply(Phi_moe, R[i,:])
        x = np.dot(Phi_Rk, Phi_moe.T) + lamb
        W = np.dot(np.dot(np.linalg.inv(x), Phi_Rk), Z_orig.T)
        W[0,:] = 0 # do not remove the intercept
        Z_corr -= np.dot(W.T, Phi_Rk)
    Z_cos = Z_corr / np.linalg.norm(Z_corr, ord=2, axis=0)
    return Z_cos, Z_corr, W, Phi_Rk


def stdscale_quantile_celing(_adata, max_value=None, quantile_thresh=None):
    sc.pp.scale(_adata, zero_center=False, max_value=max_value)    
    if quantile_thresh is not None:
        if issparse(_adata.X):
            threshval = np.quantile(np.array(_adata.X.todense()).reshape(-1), quantile_thresh)
        else:
            threshval = np.quantile(_adata.X.reshape(-1), quantile_thresh)
            
        _adata.X[_adata.X > threshval] = threshval        
        
def make_count_hist(adata, num_cells=1000):
    z = adata.X[:num_cells,:].todense()
    y = np.array(z).reshape(-1)
    (fig,ax) = plt.subplots()
    _ = ax.hist(y[y>0], bins=100)
    del(z)
    del(y)
    ax.set_title('Quantile thresholded normalized count distribution')

    
class Preprocess():
    
    def __init__(self,  random_seed = None):
        """
        Class for preprocessing data for cNMF, optionally performing batch correction using Harmony to 
        correcting genes prior to cNMF. In addition, can split RNA and ADT data to allow learning the contribution
        of ADT features to GEPs learned based on RNA.
    
        Parameters
        ----------
    
        random_seed : int, optional (default=None)
            Optionally set the random seed prior to running pipeline for reproducibility
            
        """        

            
        np.random.seed(random_seed)            
    
    def filter_adata(self, _adata, filter_mito_thresh=None, min_cells_per_gene = 10,
                   min_counts_per_cell = 500, filter_mito_genes = False, filter_dot_genes = True,
                   makeplots=True):
        """
        Optional step to filter an AnnData object. 

        Parameters
        ----------
        
        _adata : AnnData object to filter. 

        filter_mito_thresh : float [0,1], optional (default=None)
            If provided, filter cells with a proportion of mitochondrial reads greater than filter_mito_thresh.
            Mitochondrial genes are defined by starting with MT-
            
        min_cells_per_RNA_gene :  float, optional (default=10)
            If provided, filter genes detected in fewer than min_cells_per_RNA_gene cells
            
        min_rna_counts_per_cell : float, optional (default=500)
            If provided, filter cells with fewer than min_rna_counts_per_cell RNA counts
            
        filter_mito_genes : boolean, optional (default=False)
            If provided, filter mitochondrial genes from the data
                
        filter_dot_genes : boolean, optional (default=True)
            If provided, filter genes containing "." in their gene_name from the data

        makeplots : boolean, optional (default=True)
            If provided, make plot of log10 counts and pct_mitochondria
        """    
        
        if min_cells_per_gene is not None:
            sc.pp.filter_genes(_adata, min_cells=min_cells_per_gene)

        _adata.obs['n_counts'] = np.asarray(_adata.X.sum(axis=1)).squeeze()
            
        if makeplots:
            (fig,ax) = plt.subplots()
            _ = ax.hist(_adata.obs['n_counts'].apply(np.log10), bins=100)
            ax.set_title('log10 n_counts')
            ylim = ax.get_ylim()            
            ax.vlines(x=np.log10(min_cells_per_gene), ymin=ylim[0], ymax=ylim[1])
            ax.set_ylim(ylim)
            
        if min_counts_per_cell is not None:
            sc.pp.filter_cells(_adata, min_counts=min_counts_per_cell)
        

        
        mt_genes = [x for x in _adata.var.index if 'MT-' in x]
        if filter_mito_thresh is not None:
            num_mito = np.asarray(_adata[:,mt_genes].X.sum(axis=1)).squeeze()
            pct_mito = num_mito / _adata.obs['n_counts']
            _adata.obs['pct_mito'] = pct_mito
            
            if makeplots:    
                (fig,ax) = plt.subplots()
                _ = ax.hist(_adata.obs['pct_mito'], bins=100)
                ax.title('pct_mito')            
            
            _adata = _adata[_adata.obs['pct_mito'] < filter_mito_thresh, :]

        tofilter = []
        if filter_dot_genes:
            dot_genes = [x for x in _adata.var.index if '.' in x]
            tofilter = dot_genes
            
        if filter_mito_genes:
            tofilter += mt_genes
            
        ind = ~_adata.var.index.isin(tofilter)
        _adata = _adata[:, ind]
        return(_adata)
        

    def preprocess_for_cnmf(self, _adata, feature_type_col = None, adt_feature_name = 'Antibody Capture',
                            harmony_vars= None, n_top_rna_genes = 2000, librarysize_targetsum= 1e4,
                            max_scaled_thresh = None, quantile_thresh = .9999, makeplots=True, theta=1,
                            save_output_base=None, max_iter_harmony=20):
        """
        Runs minimal preprocessing for cNMF, specifically preparing an HVG filtered, normalized, optionally batch corrected, output file
        from the RNA to use as counts input for cNMF as well as a library-size normalized file potentially including both RNA and ADT to
        use as the tpm input for cNMF.
    
        Parameters
        ----------

        _adata : AnnData object or list of 2 AnnData objects
            If a single AnnData object is provided, it is assumed to contain RNA data unless
            feature_type_col is not None, in which case it is assumed to contain both RNA and
            ADT data which can be distinguished by the value of the feature_type_col
            
        feature_type_col : string, optional (default=None)
            Column in adata.var that can be used to split  data into separate RNA and ADT objects. 
            Cells with feature_type_col == adt_feature_name are ADT, otherwise RNA. If  None (default),
            assumes data contains a single modality and does not do any splitting.
            
        adt_feature_name : string, optional (default='Antibody Capture')
            If data contains a single AnnData object, this value is used to distinguish ADT
            from RNA features in _adata.var['feature_type_col'] 
            
        harmony_vars : string, or list of strings, optional (default=None)
            Variables to use for batch correction. Must correspond to columns in _adata.obs. If
            not provided, no batch correction is done. 
            
        n_top_rna_genes : int (default=2000)
        
        librarysize_targetsum : float (default=1e4),
                            
        max_scaled_thresh : float (default=None)
        
        quantile_thresh : float (default=.9999)
            After variance normalizing genes, sets a filter on the max normed value equal to this quantile.
            I.e. by default the top 0.01% of values are thresholded to the 99.99th quantile. 
        
        makeplots : boolean (default=True)
        
        theta : float (default=1)
        
        max_iter_harmony : int (default=20)
            Maximum number of Harmony iterations to use
        
        save_output_base : str (default=None)
            If provided, saves output variables to disk with the following paths
             adata_RNA : [save_output_base].Corrected.HVG.Varnorm.h5ad
             adata_tp10k : [save_output_base].TP10K.h5ad
             hvgs : [save_output_base].Corrected.HVGs.txt
                
            
        Returns
        ----------
        adata_RNA : AnnData
            Normalized RNA data subsetted to high-variance genes, optionally batch corrected
            
        tp10k : AnnData
            Normalized ADT data containing all genes. If ADT data is included, RNA and ADT
            are library size normalized separately
            
        hvg : list
            List of high variance genes that can be used as input for cNMF
        """
        
        if (not isinstance(_adata, Collection)) and (feature_type_col is not None):
            ## Split RNA and ADT
            adata_ADT = _adata[:,_adata.var[feature_type_col]==adt_feature_name]
            adata_RNA = _adata[:,_adata.var[feature_type_col]!=adt_feature_name]
        elif not isinstance(_adata, Collection):
            ## Only RNA provided
            adata_RNA = _adata
            adata_RNA.var_names_make_unique()
            adata_RNA.var['features_renamed'] = adata_RNA.var.index
            adata_ADT = None
        elif len(_adata) == 2:
            ## RNA and ADT provided as distinct elements of list
            adata_RNA = _adata[0]
            adata_ADT = _adata[1]
            if adata_ADT.shape[0] != adata_RNA.shape[0]:
                raise Exception("ADT and RNA AnnDatas don't have the same number of cells")
            elif np.sum(adata_ADT.obs.index != adata_RNA.obs.index) > 0:
                raise Exception("Inconsistency of the index for the ADT and RNA AnnDatas")                            
            
        else:
            raise Exception('data should either be an AnnData object or a list of 2 AnnData objects')

        tp10k = sc.pp.normalize_per_cell(adata_RNA, counts_per_cell_after=librarysize_targetsum, copy=True)
        adata_RNA, hvgs = self.normalize_batchcorrect(adata_RNA, harmony_vars=harmony_vars,
                                                n_top_genes = n_top_rna_genes, 
                                                librarysize_targetsum= librarysize_targetsum,
                                                max_scaled_thresh = max_scaled_thresh,
                                                quantile_thresh = quantile_thresh, theta=theta,
                                                makeplots=makeplots, max_iter_harmony=max_iter_harmony)
        
        if adata_ADT is not None:            
            adata_ADT = adata_ADT[adata_RNA.obs.index, :]
            sc.pp.normalize_per_cell(adata_ADT, counts_per_cell_after=librarysize_targetsum)
            
            merge_var = pd.concat([tp10k.var, adata_ADT.var], axis=0)            
            tp10k = sc.AnnData(hstack((tp10k.X, adata_ADT.X)), obs=tp10k.obs, var=merge_var)
            del(adata_ADT)
            
            
        if save_output_base is not None:
            sc.write(save_output_base+'.Corrected.HVG.Varnorm.h5ad', adata_RNA)
            sc.write(save_output_base+'.TP10K.h5ad', tp10k)
            with open(save_output_base+'.Corrected.HVGs.txt', 'w') as F:
                F.write('\n'.join(hvgs))
            
        return(adata_RNA, tp10k, hvgs)
        
        
    def normalize_batchcorrect(self, _adata, normalize_librarysize=False,
                               harmony_vars=None, n_top_genes = None,
                               librarysize_targetsum= 1e4, max_scaled_thresh = None,
                               quantile_thresh = .9999, theta=1, makeplots=True,
                               max_iter_harmony=20):
        """
        Normalizes, filters high-variance genes, and optionally batch corrects an AnnData object containing a
        single data modality

        Parameters
        ----------
        
        _adata : AnnData object to normalize
        
        normalize_librarysize : boolean (default=None)
            Option to library_size normalize data for each cell to have total counts in librarysize_targetsum.
            Should not typically be used prior to cNMF. Does not impact behavior for running Harmony to learn
            the adjusted PCA basis which is done on library-size normalized data, only impacts the final normalized
            expression data.
        
        harmony_vars : String or list of strings, optional (default=None)
            The variables contained in adata.obs to use for Harmony correction. If None, does not correct

        n_top_rna_genes : float, optional (default=2000)
            Number of high-variance genes to use for PCA and batch correction
            
        RNA_normalize_targetsum :  float, optional (default=1e4)
            If provided, normalizes cells to this many total RNA counts prior to further normalization
            
        max_scaled_thresh : float, optional (default=None)
            If provided, sets a ceiling on this value after variance scaling genes. This can prevent outlier genes
            from obtaining disproportionately high values
            
        quantile_thresh : float, optional (default=.9999)
            If provided, sets a ceiling on this quantile after variance scaling genes. I.e. calculates this quantile value
            and sets a ceiling on this value. This can prevent outlier genes from obtaining disproportionately high values
    
        makeplots : boolean, optional (default=True)
            If provided, makes a histogram of quantile thresholded count distribution
            
        max_iter_harmony : int, optional (default=20)
            Maximum number of Harmony iterations to use
        """  
                
        if n_top_genes is not None:
            sc.pp.highly_variable_genes(_adata, flavor='seurat_v3', n_top_genes=n_top_genes)
        elif 'highly_variable' not in _adata.var.columns:
            raise Exception("If a numeric value for n_top_genes is not provided, you must include a highly_variable column in _adata")                            
            
        if harmony_vars is not None:
            anorm = sc.pp.normalize_per_cell(_adata, counts_per_cell_after=librarysize_targetsum, copy=True)
            anorm = anorm[:, _adata.var['highly_variable']]
            stdscale_quantile_celing(anorm, max_value=max_scaled_thresh, quantile_thresh=quantile_thresh)
            
            _adata = _adata[:, _adata.var['highly_variable']]
            stdscale_quantile_celing(_adata, max_value=max_scaled_thresh, quantile_thresh=quantile_thresh)
            
            if makeplots:
                make_count_hist(anorm, num_cells=1000)
                    
            sc.pp.pca(anorm, use_highly_variable=True, zero_center=True)
            if makeplots:
                sc.pl.pca_variance_ratio(anorm, log=True, n_pcs=50)
                
            _adata.obsm['X_pca'] = anorm.obsm['X_pca']
            
            if not normalize_librarysize:
                del(anorm)
                _adata.X, _adata.obsm['X_pca_harmony'] = self.harmony_correct_X(_adata.X.todense(), _adata.obs, _adata.obsm['X_pca'],
                                                                                harmony_vars, max_iter_harmony=max_iter_harmony)
            else:
                _adata.X, _adata.obsm['X_pca_harmony'] = self.harmony_correct_X(anorm.X.todense(), anorm.obs, anorm.obsm['X_pca'],
                                                                                harmony_vars, max_iter_harmony=max_iter_harmony) 
                
        else:
            if normalize_librarysize:
                sc.pp.normalize_per_cell(_adata, counts_per_cell_after=librarysize_targetsum)
            
            
            _adata = _adata[:, _adata.var['highly_variable']]
            stdscale_quantile_celing(_adata, max_value=max_scaled_thresh, quantile_thresh=quantile_thresh)
            if makeplots:
                make_count_hist(_adata, num_cells=1000)
                
        hvgs = list(_adata.var.index)
                
        return(_adata, hvgs)
            

        
    def harmony_correct_X(self, X, obs, pca, harmony_vars, theta=1, max_iter_harmony=20):
        """
        Runs batch correction on the provided data. Specifically it uses
        Harmony to learn the batch correction parameters but rather than just correcting
        the PCs, it applies the MOE ridge correction to normalized counts data.

        Parameters
        ----------

        X : array-like
            
        obs : panda.DataFrame
            Must include the columns specified in harmony_vars
    
        pca : array-like
            If provided, sets a ceiling on this quantile after variance scaling genes. I.e. calculates this quantile value
            and sets a ceiling on this value. This can prevent outlier genes from obtaining disproportionately high values
    
        harmony_vars : list
            List of columns included in the obs DataFrame to use for correction
            
        Returns
        ----------
        X_corr : array-like
            Corrected input data
 
         X_pca_harmony : array-like
            Corrected PCs
 
        """   

        try:
            import harmonypy
        except:
            raise ImportError("harmonypy is not installed. Please install it using 'pip install harmonypy' before proceeding.")
            
        harmony_res = harmonypy.run_harmony(pca, obs, harmony_vars, max_iter_harmony = max_iter_harmony,
                                            theta=theta)
        
        X_pca_harmony = harmony_res.Z_corr.T       
        _, X_corr, _, _ = moe_correct_ridge(X.T, None, None, harmony_res.R, None, harmony_res.K,
                                            None, harmony_res.Phi_moe, harmony_res.lamb)
        X_corr = np.array(X_corr.T)
        
        X_corr[X_corr<0] = 0
            
        return(X_corr, X_pca_harmony)

    
    def select_features_MI(self, _adata, cluster, max_scaled_thresh = None, quantile_thresh = .9999,
                          n_top_features = 70, makeplots=True):
        """
        Runs batch correction on the self.adata_RNA object. Specifically it uses
        Harmony to learn the batch correction parameters but rather than just correcting
        the PCs, it applies the MOE ridge correction to normalized counts data.

        Parameters
        ----------
            
        harmony_vars : String or list of strings
            The variables contained in adata.obs to use for Harmony correction
    
        quantile_thresh : float, optional (default=.9999)
            If provided, sets a ceiling on this quantile after variance scaling genes. I.e. calculates this quantile value
            and sets a ceiling on this value. This can prevent outlier genes from obtaining disproportionately high values
    
        makeplots : boolean, optional (default=True)
            If True, makes a scatter plot of values pre and post correction
        """
        sc.pp.normalize_per_cell(_adata)  
        stdscale_quantile_celing(_adata, max_value=max_scaled_thresh, quantile_thresh=quantile_thresh)

        if issparse(_adata.X):        
            res = mutual_info_classif(_adata.X.toarray(), cluster, discrete_features='auto',
                    n_neighbors=3, copy=True, random_state=None)
        else:
            res = mutual_info_classif(_adata.X, cluster, discrete_features='auto',
                    n_neighbors=3, copy=True, random_state=None)
        
        res = pd.Series(res, index=_adata.var.index)
        res = res.sort_values(ascending=False)
        resdf = pd.DataFrame([res.values, np.arange(res.shape[0])], columns=res.index, index=['MI', 'MI_Rank']).T
        resdf['MI_diff'] = resdf['MI'].diff()
        
        if makeplots:
            (fig,ax) = plt.subplots(1,1, figsize=(10,3), dpi=100)
            ax.scatter(resdf['MI_Rank'], resdf['MI'])
            ax.set_ylabel('MI', fontsize=11)
            ax.set_xlabel('MI Rank', fontsize=11)

            ylim = ax.get_ylim()
            ax.vlines(x=n_top_features, ymin=ylim[0], ymax=ylim[1], linestyle='--', color='k')
            ax.set_ylim(ylim)
        
        for v in resdf.columns:
            _adata.var[v] = resdf[v]
        _adata.var['highly_variable'] = _adata.var['MI_Rank']<n_top_features
        return(_adata)