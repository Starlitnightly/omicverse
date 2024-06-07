#!/usr/bin/env python

import numpy as np
import pandas as pd
import os, errno
import datetime

import itertools
import yaml
import subprocess
import scipy.sparse as sp

from scipy.spatial.distance import squareform
from sklearn.decomposition import non_negative_factorization
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import sparsefuncs


from scipy.cluster.hierarchy import leaves_list

import matplotlib.pyplot as plt

import scanpy as sc

from multiprocessing import Pool 


def save_df_to_npz(obj, filename):
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)

def save_df_to_text(obj, filename):
    obj.to_csv(filename, sep='\t')

def load_df_from_npz(filename):
    with np.load(filename, allow_pickle=True) as f:
        obj = pd.DataFrame(**f)
    return obj

def check_dir_exists(path):
    """
    Checks if directory already exists or not and creates it if it doesn't
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def worker_filter(iterable, worker_index, total_workers):
    return (p for i,p in enumerate(iterable) if (i-worker_index)%total_workers==0)

def fast_ols_all_cols(X, Y):
    pinv = np.linalg.pinv(X)
    beta = np.dot(pinv, Y)
    return(beta)

def fast_ols_all_cols_df(X,Y):
    beta = fast_ols_all_cols(X, Y)
    beta = pd.DataFrame(beta, index=X.columns, columns=Y.columns)
    return(beta)

def var_sparse_matrix(X):
    mean = np.array(X.mean(axis=0)).reshape(-1)
    Xcopy = X.copy()
    Xcopy.data **= 2
    var = np.array(Xcopy.mean(axis=0)).reshape(-1) - (mean**2)
    return(var)


def get_highvar_genes_sparse(expression, expected_fano_threshold=None,
                       minimal_mean=0.5, numgenes=None):
    # Find high variance genes within those cells
    gene_mean = np.array(expression.mean(axis=0)).astype(float).reshape(-1)
    E2 = expression.copy(); E2.data **= 2; gene2_mean = np.array(E2.mean(axis=0)).reshape(-1)
    gene_var = pd.Series(gene2_mean - (gene_mean**2))
    del(E2)
    gene_mean = pd.Series(gene_mean)
    gene_fano = gene_var / gene_mean

    # Find parameters for expected fano line
    top_genes = gene_mean.sort_values(ascending=False)[:20].index
    A = (np.sqrt(gene_var)/gene_mean)[top_genes].min()
    
    w_mean_low, w_mean_high = gene_mean.quantile([0.10, 0.90])
    w_fano_low, w_fano_high = gene_fano.quantile([0.10, 0.90])
    winsor_box = ((gene_fano > w_fano_low) &
                    (gene_fano < w_fano_high) &
                    (gene_mean > w_mean_low) &
                    (gene_mean < w_mean_high))
    fano_median = gene_fano[winsor_box].median()
    B = np.sqrt(fano_median)

    gene_expected_fano = (A**2)*gene_mean + (B**2)
    fano_ratio = (gene_fano/gene_expected_fano)

    # Identify high var genes
    if numgenes is not None:
        highvargenes = fano_ratio.sort_values(ascending=False).index[:numgenes]
        high_var_genes_ind = fano_ratio.index.isin(highvargenes)
        T=None


    else:
        if not expected_fano_threshold:
            T = (1. + gene_fano[winsor_box].std())
        else:
            T = expected_fano_threshold

        high_var_genes_ind = (fano_ratio > T) & (gene_mean > minimal_mean)

    gene_counts_stats = pd.DataFrame({
        'mean': gene_mean,
        'var': gene_var,
        'fano': gene_fano,
        'expected_fano': gene_expected_fano,
        'high_var': high_var_genes_ind,
        'fano_ratio': fano_ratio
        })
    gene_fano_parameters = {
            'A': A, 'B': B, 'T':T, 'minimal_mean': minimal_mean,
        }
    return(gene_counts_stats, gene_fano_parameters)



def get_highvar_genes(input_counts, expected_fano_threshold=None,
                       minimal_mean=0.5, numgenes=None):
    # Find high variance genes within those cells
    gene_counts_mean = pd.Series(input_counts.mean(axis=0).astype(float))
    gene_counts_var = pd.Series(input_counts.var(ddof=0, axis=0).astype(float))
    gene_counts_fano = pd.Series(gene_counts_var/gene_counts_mean)

    # Find parameters for expected fano line
    top_genes = gene_counts_mean.sort_values(ascending=False)[:20].index
    A = (np.sqrt(gene_counts_var)/gene_counts_mean)[top_genes].min()

    w_mean_low, w_mean_high = gene_counts_mean.quantile([0.10, 0.90])
    w_fano_low, w_fano_high = gene_counts_fano.quantile([0.10, 0.90])
    winsor_box = ((gene_counts_fano > w_fano_low) &
                    (gene_counts_fano < w_fano_high) &
                    (gene_counts_mean > w_mean_low) &
                    (gene_counts_mean < w_mean_high))
    fano_median = gene_counts_fano[winsor_box].median()
    B = np.sqrt(fano_median)

    gene_expected_fano = (A**2)*gene_counts_mean + (B**2)

    fano_ratio = (gene_counts_fano/gene_expected_fano)

    # Identify high var genes
    if numgenes is not None:
        highvargenes = fano_ratio.sort_values(ascending=False).index[:numgenes]
        high_var_genes_ind = fano_ratio.index.isin(highvargenes)
        T=None


    else:
        if not expected_fano_threshold:
            T = (1. + gene_counts_fano[winsor_box].std())
        else:
            T = expected_fano_threshold

        high_var_genes_ind = (fano_ratio > T) & (gene_counts_mean > minimal_mean)

    gene_counts_stats = pd.DataFrame({
        'mean': gene_counts_mean,
        'var': gene_counts_var,
        'fano': gene_counts_fano,
        'expected_fano': gene_expected_fano,
        'high_var': high_var_genes_ind,
        'fano_ratio': fano_ratio
        })
    gene_fano_parameters = {
            'A': A, 'B': B, 'T':T, 'minimal_mean': minimal_mean,
        }
    return(gene_counts_stats, gene_fano_parameters)


def compute_tpm(input_counts):
    """
    Default TPM normalization
    """
    tpm = input_counts.copy()
    sc.pp.normalize_per_cell(tpm, counts_per_cell_after=1e6)
    return(tpm)


def factorize_mp_signature(args):
    """
    wrapper around factorize to be able to use mp pool.
    args is a list:
    worker-i: int
    total_workers: int
    pointer to nmf object.
    """
    args[2].factorize(worker_i=args[0],  total_workers=args[1])


class cNMF():


    def __init__(self, adata, components, n_iter = 100, densify=False, tpm_fn=None, seed=None,
                        beta_loss='frobenius',num_highvar_genes=2000, genes_file=None,
                        alpha_usage=0.0, alpha_spectra=0.0, init='random',output_dir="./tmp", name=None):
        """
        Arguments:
            output_dir: path, optional (default="."). Output directory for analysis files.
            name: string, optional (default=None). A name for this analysis. Will be prefixed to all output files. If set to None, will be automatically generated from date (and random string).
        """

        self.output_dir = output_dir
        if name is None:
            import uuid
            now = datetime.datetime.now()
            rand_hash =  uuid.uuid4().hex[:6]
            name = '%s_%s' % (now.strftime("%Y_%m_%d"), rand_hash)
        self.name = name
        self.paths = None
        self._initialize_dirs()
        self.prepare(adata, components, n_iter, densify, tpm_fn, seed,
                        beta_loss, num_highvar_genes, genes_file,
                        alpha_usage, alpha_spectra, init)


    def _initialize_dirs(self):
        if self.paths is None:
            # Check that output directory exists, create it if needed.
            check_dir_exists(self.output_dir)
            check_dir_exists(os.path.join(self.output_dir, self.name))
            check_dir_exists(os.path.join(self.output_dir, self.name, 'cnmf_tmp'))

            self.paths = {
                'normalized_counts' : os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.norm_counts.h5ad'),
                'nmf_replicate_parameters' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.nmf_params.df.npz'),
                'nmf_run_parameters' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.nmf_idvrun_params.yaml'),
                'nmf_genes_list' :  os.path.join(self.output_dir, self.name, self.name+'.overdispersed_genes.txt'),

                'tpm' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.tpm.h5ad'),
                'tpm_stats' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.tpm_stats.df.npz'),

                'iter_spectra' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.spectra.k_%d.iter_%d.df.npz'),
                'iter_usages' :  os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.usages.k_%d.iter_%d.df.npz'),
                'merged_spectra': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.spectra.k_%d.merged.df.npz'),

                'local_density_cache': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.local_density_cache.k_%d.merged.df.npz'),
                'consensus_spectra': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.spectra.k_%d.dt_%s.consensus.df.npz'),
                'consensus_spectra__txt': os.path.join(self.output_dir, self.name, self.name+'.spectra.k_%d.dt_%s.consensus.txt'),
                'consensus_usages': os.path.join(self.output_dir, self.name, 'cnmf_tmp',self.name+'.usages.k_%d.dt_%s.consensus.df.npz'),
                'consensus_usages__txt': os.path.join(self.output_dir, self.name, self.name+'.usages.k_%d.dt_%s.consensus.txt'),

                'consensus_stats': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.stats.k_%d.dt_%s.df.npz'),

                'clustering_plot': os.path.join(self.output_dir, self.name, self.name+'.clustering.k_%d.dt_%s.png'),
                'gene_spectra_score': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.gene_spectra_score.k_%d.dt_%s.df.npz'),
                'gene_spectra_score__txt': os.path.join(self.output_dir, self.name, self.name+'.gene_spectra_score.k_%d.dt_%s.txt'),
                'gene_spectra_tpm': os.path.join(self.output_dir, self.name, 'cnmf_tmp', self.name+'.gene_spectra_tpm.k_%d.dt_%s.df.npz'),
                'gene_spectra_tpm__txt': os.path.join(self.output_dir, self.name, self.name+'.gene_spectra_tpm.k_%d.dt_%s.txt'),

                'k_selection_plot' :  os.path.join(self.output_dir, self.name, self.name+'.k_selection.png'),
                'k_selection_stats' :  os.path.join(self.output_dir, self.name, self.name+'.k_selection_stats.df.npz'),
            }


    def prepare(self, adata, components, n_iter = 100, densify=False, tpm_fn=None, seed=None,
                        beta_loss='frobenius',num_highvar_genes=2000, genes_file=None,
                        alpha_usage=0.0, alpha_spectra=0.0, init='random'):
        """
        Load input counts, reduce to high-variance genes, and variance normalize genes.
        Prepare file for distributing jobs over workers.

    Arguments:
        counts_fn: Path to input counts matrix
        components: Values of K to run NMF for
        n_iter: Number of iterations for factorization. If several "k" are specified, this many iterations will be run for each value of "k".
        densify: Convert sparse data to dense
        tpm_fn: If provided, load tpm data from file. Otherwise will compute it from the counts file
        seed: Seed for sklearn random state.
        beta_loss: 
        num_highvar_genes: If provided and genes_file is None, will compute this many highvar genes to use for factorization
        genes_file: If provided will load high-variance genes from a list of these genes
        alpha_usage: Regularization parameter for NMF corresponding to alpha_W in scikit-learn
        alpha_spectra: Regularization parameter for NMF corresponding to alpha_H in scikit-learn
        """
        
        '''
        if counts_fn.endswith('.h5ad'):
            input_counts = sc.read(counts_fn)
        else:
            ## Load txt or compressed dataframe and convert to scanpy object
            if counts_fn.endswith('.npz'):
                input_counts = load_df_from_npz(counts_fn)
            else:
                input_counts = pd.read_csv(counts_fn, sep='\t', index_col=0)
                
            if densify:
                input_counts = sc.AnnData(X=input_counts.values,
                                       obs=pd.DataFrame(index=input_counts.index),
                                       var=pd.DataFrame(index=input_counts.columns))
            else:
                input_counts = sc.AnnData(X=sp.csr_matrix(input_counts.values),
                                       obs=pd.DataFrame(index=input_counts.index),
                                       var=pd.DataFrame(index=input_counts.columns))
             '''   
        if 'counts' not in adata.layers:
            raise Exception('Error: No counts data found in adata.layers. Quitting!')
        input_counts=adata.copy()
        input_counts.X = input_counts.layers['counts']

                
        if sp.issparse(input_counts.X) & densify:
            input_counts.X = np.array(input_counts.X.todense())
 
        if tpm_fn is None:
            tpm = compute_tpm(input_counts)
            sc.write(self.paths['tpm'], tpm)
        elif tpm_fn.endswith('.h5ad'):
            subprocess.call('cp %s %s' % (tpm_fn, self.paths['tpm']), shell=True)
            tpm = sc.read(self.paths['tpm'])
        else:
            if tpm_fn.endswith('.npz'):
                tpm = load_df_from_npz(tpm_fn)
            else:
                tpm = pd.read_csv(tpm_fn, sep='\t', index_col=0)
            
            if densify:
                tpm = sc.AnnData(X=tpm.values,
                            obs=pd.DataFrame(index=tpm.index),
                            var=pd.DataFrame(index=tpm.columns)) 
            else:
                tpm = sc.AnnData(X=sp.csr_matrix(tpm.values),
                            obs=pd.DataFrame(index=tpm.index),
                            var=pd.DataFrame(index=tpm.columns)) 

            sc.write(self.paths['tpm'], tpm)
        
        if sp.issparse(tpm.X):
            gene_tpm_mean = np.array(tpm.X.mean(axis=0)).reshape(-1)
            gene_tpm_stddev = var_sparse_matrix(tpm.X)**.5
        else:
            gene_tpm_mean = np.array(tpm.X.mean(axis=0)).reshape(-1)
            gene_tpm_stddev = np.array(tpm.X.std(axis=0, ddof=0)).reshape(-1)
            
            
        input_tpm_stats = pd.DataFrame([gene_tpm_mean, gene_tpm_stddev],
             index = ['__mean', '__std'], columns = tpm.var.index).T
        save_df_to_npz(input_tpm_stats, self.paths['tpm_stats'])
        
        if genes_file is not None:
            highvargenes = open(genes_file).read().rstrip().split('\n')
        else:
            highvargenes = None

        norm_counts = self.get_norm_counts(input_counts, tpm, num_highvar_genes=num_highvar_genes,
                                               high_variance_genes_filter=highvargenes)

        self.save_norm_counts(norm_counts)
        (replicate_params, run_params) = self.get_nmf_iter_params(ks=components, n_iter=n_iter, random_state_seed=seed,
                                                                  beta_loss=beta_loss, alpha_usage=alpha_usage,
                                                                  alpha_spectra=alpha_spectra, init=init)
        self.save_nmf_iter_params(replicate_params, run_params)
        
    
    def combine(self, components=None, skip_missing_files=False):
        """
        Combine NMF iterations for the same value of K
        Parameters
        ----------
        components : list or None
            Values of K to combine iterations for. Defaults to all.

        skip_missing_files : boolean
            If True, ignore iteration files that aren't found rather than crashing. Default: False
        """

        if type(components) is int:
            ks = [components]
        elif components is None:
            run_params = load_df_from_npz(self.paths['nmf_replicate_parameters'])
            ks = sorted(set(run_params.n_components))
        else:
            ks = components

        for k in ks:
            self.combine_nmf(k, skip_missing_files=skip_missing_files)    
    
    
    
    def get_norm_counts(self, counts, tpm,
                         high_variance_genes_filter = None,
                         num_highvar_genes = None
                         ):
        """
        Parameters
        ----------

        counts : anndata.AnnData
            Scanpy AnnData object (cells x genes) containing raw counts. Filtered such that
            no genes or cells with 0 counts
        
        tpm : anndata.AnnData
            Scanpy AnnData object (cells x genes) containing tpm normalized data matching
            counts

        high_variance_genes_filter : np.array, optional (default=None)
            A pre-specified list of genes considered to be high-variance.
            Only these genes will be used during factorization of the counts matrix.
            Must match the .var index of counts and tpm.
            If set to None, high-variance genes will be automatically computed, using the
            parameters below.

        num_highvar_genes : int, optional (default=None)
            Instead of providing an array of high-variance genes, identify this many most overdispersed genes
            for filtering

        Returns
        -------

        normcounts : anndata.AnnData, shape (cells, num_highvar_genes)
            A counts matrix containing only the high variance genes and with columns (genes)normalized to unit
            variance

        """

        if high_variance_genes_filter is None:
            ## Get list of high-var genes if one wasn't provided
            if sp.issparse(tpm.X):
                (gene_counts_stats, gene_fano_params) = get_highvar_genes_sparse(tpm.X, numgenes=num_highvar_genes)  
            else:
                (gene_counts_stats, gene_fano_params) = get_highvar_genes(np.array(tpm.X), numgenes=num_highvar_genes)
                
            high_variance_genes_filter = list(tpm.var.index[gene_counts_stats.high_var.values])
                
        ## Subset out high-variance genes
        norm_counts = counts[:, high_variance_genes_filter]

        ## Scale genes to unit variance
        if sp.issparse(tpm.X):
            sc.pp.scale(norm_counts, zero_center=False)
            if np.isnan(norm_counts.X.data).sum() > 0:
                print('Warning NaNs in normalized counts matrix')                       
        else:
            norm_counts.X /= norm_counts.X.std(axis=0, ddof=1)
            if np.isnan(norm_counts.X).sum().sum() > 0:
                print('Warning NaNs in normalized counts matrix')                    
        
        ## Save a \n-delimited list of the high-variance genes used for factorization
        open(self.paths['nmf_genes_list'], 'w').write('\n'.join(high_variance_genes_filter))

        ## Check for any cells that have 0 counts of the overdispersed genes
        zerocells = norm_counts.X.sum(axis=1)==0
        if zerocells.sum()>0:
            examples = norm_counts.obs.index[np.ravel(zerocells)]
            raise Exception('Error: %d cells have zero counts of overdispersed genes. E.g. %s. Filter those cells and re-run or adjust the number of overdispersed genes. Quitting!' % (zerocells.sum(), ', '.join(examples[:4])))
        
        return(norm_counts)

    
    def save_norm_counts(self, norm_counts):
        self._initialize_dirs()
        sc.write(self.paths['normalized_counts'], norm_counts)

        
    def get_nmf_iter_params(self, ks, n_iter = 100,
                               random_state_seed = None,
                               beta_loss = 'kullback-leibler',
                               alpha_usage=0.0, alpha_spectra=0.0,
                               init='random'):
        """
        Create a DataFrame with parameters for NMF iterations.


        Parameters
        ----------
        ks : integer, or list-like.
            Number of topics (components) for factorization.
            Several values can be specified at the same time, which will be run independently.

        n_iter : integer, optional (defailt=100)
            Number of iterations for factorization. If several ``k`` are specified, this many
            iterations will be run for each value of ``k``.

        random_state_seed : int or None, optional (default=None)
            Seed for sklearn random state.
            
        alpha_usage : float, optional (default=0.0)
            Regularization parameter for NMF corresponding to alpha_W in scikit-learn

        alpha_spectra : float, optional (default=0.0)
            Regularization parameter for NMF corresponding to alpha_H in scikit-learn
        """

        if type(ks) is int:
            ks = [ks]

        # Remove any repeated k values, and order.
        k_list = sorted(set(list(ks)))

        n_runs = len(ks)* n_iter

        np.random.seed(seed=random_state_seed)
        nmf_seeds = np.random.randint(low=1, high=(2**32)-1, size=n_runs)

        replicate_params = []
        for i, (k, r) in enumerate(itertools.product(k_list, range(n_iter))):
            replicate_params.append([k, r, nmf_seeds[i]])
        replicate_params = pd.DataFrame(replicate_params, columns = ['n_components', 'iter', 'nmf_seed'])

        _nmf_kwargs = dict(
                        alpha_W=alpha_usage,
                        alpha_H=alpha_spectra,
                        l1_ratio=0.0,
                        beta_loss=beta_loss,
                        solver='mu',
                        tol=1e-4,
                        max_iter=1000,
                        init=init
                        )
        
        ## Coordinate descent is faster than multiplicative update but only works for frobenius
        if beta_loss == 'frobenius':
            _nmf_kwargs['solver'] = 'cd'

        return(replicate_params, _nmf_kwargs)


    def save_nmf_iter_params(self, replicate_params, run_params):
        self._initialize_dirs()
        save_df_to_npz(replicate_params, self.paths['nmf_replicate_parameters'])
        with open(self.paths['nmf_run_parameters'], 'w') as F:
            yaml.dump(run_params, F)


    def _nmf(self, X, nmf_kwargs):
        """
        Parameters
        ----------
        X : pandas.DataFrame,
            Normalized counts dataFrame to be factorized.

        nmf_kwargs : dict,
            Arguments to be passed to ``non_negative_factorization``

        """
        (usages, spectra, niter) = non_negative_factorization(X, **nmf_kwargs)

        return(spectra, usages)

    def factorize_multi_process(self, total_workers):
        """
        multiproces wrapper for nmf.factorize()
        factorize_multi_process() is direct wrapper around factorize to be able to launch it form mp.
        total_workers: int; number of workers to use.
        """
        list_args = [(x, total_workers, self) for x in range(total_workers)]
        
        with Pool(total_workers) as p:
            
            p.map(factorize_mp_signature, list_args)
            p.close()
            p.join()    
    
    def factorize(self,
                worker_i=0, total_workers=1,
                ):
        """
        Iteratively run NMF with prespecified parameters.

        Use the `worker_i` and `total_workers` parameters for parallelization.

        Generic kwargs for NMF are loaded from self.paths['nmf_run_parameters'], defaults below::

            ``non_negative_factorization`` default arguments:
                alpha=0.0
                l1_ratio=0.0
                beta_loss='kullback-leibler'
                solver='mu'
                tol=1e-4,
                max_iter=200
                regularization=None
                init='random'
                random_state, n_components are both set by the prespecified self.paths['nmf_replicate_parameters'].


        Parameters
        ----------
        norm_counts : pandas.DataFrame,
            Normalized counts dataFrame to be factorized.
            (Output of ``normalize_counts``)

        run_params : pandas.DataFrame,
            Parameters for NMF iterations.
            (Output of ``prepare_nmf_iter_params``)

        """
        run_params = load_df_from_npz(self.paths['nmf_replicate_parameters'])
        norm_counts = sc.read(self.paths['normalized_counts'])
        _nmf_kwargs = yaml.load(open(self.paths['nmf_run_parameters']), Loader=yaml.FullLoader)

        jobs_for_this_worker = worker_filter(range(len(run_params)), worker_i, total_workers)
        from tqdm import tqdm
        for idx in tqdm(jobs_for_this_worker):

            p = run_params.iloc[idx, :]
            #print('[Worker %d]. Starting task %d.' % (worker_i, idx))
            _nmf_kwargs['random_state'] = p['nmf_seed']
            _nmf_kwargs['n_components'] = p['n_components']

            (spectra, usages) = self._nmf(norm_counts.X, _nmf_kwargs)
            spectra = pd.DataFrame(spectra,
                                   index=np.arange(1, _nmf_kwargs['n_components']+1),
                                   columns=norm_counts.var.index)
            save_df_to_npz(spectra, self.paths['iter_spectra'] % (p['n_components'], p['iter']))


    def combine_nmf(self, k, skip_missing_files=False, remove_individual_iterations=False):
        run_params = load_df_from_npz(self.paths['nmf_replicate_parameters'])
        print('Combining factorizations for k=%d.'%k)

        run_params_subset = run_params[run_params.n_components==k].sort_values('iter')
        combined_spectra = []

        for i,p in run_params_subset.iterrows():
            current_file = self.paths['iter_spectra'] % (p['n_components'], p['iter'])
            if not os.path.exists(current_file):
                if not skip_missing_files:
                    print('Missing file: %s, run with skip_missing=True to override' % current_file)
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), current_file)
                else:
                    print('Missing file: %s. Skipping.' % current_file)
            else:
                spectra = load_df_from_npz(current_file)
                spectra.index = ['iter%d_topic%d' % (p['iter'], t+1) for t in range(k)]
                combined_spectra.append(spectra)
                
        if len(combined_spectra)>0:        
            combined_spectra = pd.concat(combined_spectra, axis=0)
            save_df_to_npz(combined_spectra, self.paths['merged_spectra']%k)
        else:
            print('No spectra found for k=%d' % k)
        return combined_spectra
    
    
    def refit_usage(self, X, spectra):
        """
        Takes an input data matrix and a fixed spectra and uses NNLS to find the optimal
        usage matrix. Generic kwargs for NMF are loaded from self.paths['nmf_run_parameters'].
        If input data are pandas.DataFrame, returns a DataFrame with row index matching X and
        columns index matching index of spectra

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, cells X genes
            Non-negative expression data to fit spectra to

        spectra : pandas.DataFrame or numpy.ndarray, programs X genes
            Non-negative spectra of expression programs
        """

        refit_nmf_kwargs = yaml.load(open(self.paths['nmf_run_parameters']), Loader=yaml.FullLoader)
        if type(spectra) is pd.DataFrame:
            refit_nmf_kwargs.update(dict(n_components = spectra.shape[0], H = spectra.values, update_H = False))
        else:
            refit_nmf_kwargs.update(dict(n_components = spectra.shape[0], H = spectra, update_H = False))
            
        _, rf_usages = self._nmf(X, nmf_kwargs=refit_nmf_kwargs)
        if (type(X) is pd.DataFrame) and (type(spectra) is pd.DataFrame):
            rf_usages = pd.DataFrame(rf_usages, index=X.index, columns=spectra.index)
          
        return(rf_usages)
    
    
    def refit_spectra(self, X, usage):
        """
        Takes an input data matrix and a fixed usage matrix and uses NNLS to find the optimal
        spectra matrix. Generic kwargs for NMF are loaded from self.paths['nmf_run_parameters'].
        If input data are pandas.DataFrame, returns a DataFrame with row index matching X and
        columns index matching index of spectra

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray, cells X genes
            Non-negative expression data to fit spectra to

        usage : pandas.DataFrame or numpy.ndarray, cells X programs
            Non-negative spectra of expression programs
        """
        return(self.refit_usage(X.T, usage.T).T)


    def consensus(self, k, density_threshold=0.5, local_neighborhood_size = 0.30,show_clustering = True,
                  skip_density_and_return_after_stats = False, close_clustergram_fig=False,
                  refit_usage=True):
        """
        Obtain consensus estimates of spectra and usages from a cNMF run and output a clustergram of
        the consensus matrix. Assumes prepare, factorize, and combine steps have already been run.


        Parameters
        ----------
        k : int
            Number of programs (must be within the k values specified in previous steps)

        density_threshold : float (default: 0.5)
            Threshold for filtering outlier spectra. 2.0 or greater applies no filter.
            
        local_neighborhood_size : float (default: 0.3)
            Determines number of neighbors to use for calculating KNN distance as local_neighborhood_size X n_iters

        show_clustering : boolean (default=False)
            If True, generates the consensus clustergram filter

        skip_density_and_return_after_stats : boolean (default=False)
            True when running k_selection_plot to compute stability and error for input parameters without computing
            consensus spectra and usages
            
        close_clustergram_fig : boolean (default=False)
            If True, closes the clustergram figure from output after saving the image to a file
            
        refit_usage : boolean (default=True)
            If True, refit the usage matrix one final time after finalizing the spectra_tpm matrix. Has been shown
            to increase inference accuracy in simulations.
        """
        
        
        merged_spectra = load_df_from_npz(self.paths['merged_spectra']%k)
        norm_counts = sc.read(self.paths['normalized_counts'])

        density_threshold_str = str(density_threshold)
        if skip_density_and_return_after_stats:
            density_threshold_str = '2'
        density_threshold_repl = density_threshold_str.replace('.', '_')
        n_neighbors = int(local_neighborhood_size * merged_spectra.shape[0]/k)

        # Rescale topics such to length of 1.
        l2_spectra = (merged_spectra.T/np.sqrt((merged_spectra**2).sum(axis=1))).T

        if not skip_density_and_return_after_stats:
            # Compute the local density matrix (if not previously cached)
            topics_dist = None
            if os.path.isfile(self.paths['local_density_cache'] % k):
                local_density = load_df_from_npz(self.paths['local_density_cache'] % k)
            else:
                #   first find the full distance matrix
                topics_dist = euclidean_distances(l2_spectra.values)
                #   partition based on the first n neighbors
                partitioning_order  = np.argpartition(topics_dist, n_neighbors+1)[:, :n_neighbors+1]
                #   find the mean over those n_neighbors (excluding self, which has a distance of 0)
                distance_to_nearest_neighbors = topics_dist[np.arange(topics_dist.shape[0])[:, None], partitioning_order]
                local_density = pd.DataFrame(distance_to_nearest_neighbors.sum(1)/(n_neighbors),
                                             columns=['local_density'],
                                             index=l2_spectra.index)
                save_df_to_npz(local_density, self.paths['local_density_cache'] % k)
                del(partitioning_order)
                del(distance_to_nearest_neighbors)

            density_filter = local_density.iloc[:, 0] < density_threshold
            l2_spectra = l2_spectra.loc[density_filter, :]
            if l2_spectra.shape[0] == 0:
                raise RuntimeError("Zero components remain after density filtering. Consider increasing density threshold")

        kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
        kmeans_model.fit(l2_spectra)
        kmeans_cluster_labels = pd.Series(kmeans_model.labels_+1, index=l2_spectra.index)

        # Find median usage for each gene across cluster
        median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()

        # Normalize median spectra to probability distributions.
        median_spectra = (median_spectra.T/median_spectra.sum(1)).T

        # Compute the silhouette score
        if len(np.unique(kmeans_cluster_labels)) == len(kmeans_cluster_labels):
            stability = 0
        else:
            stability = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric='euclidean')

        # Obtain reconstructed count matrix by re-fitting usage and computing dot product: usage.dot(spectra)
        rf_usages = self.refit_usage(norm_counts.X, median_spectra)
        rf_usages = pd.DataFrame(rf_usages, index=norm_counts.obs.index, columns=median_spectra.index)        
        rf_pred_norm_counts = rf_usages.dot(median_spectra)
        
        # Re-order usage by total contribution
        norm_usages = rf_usages.div(rf_usages.sum(axis=1), axis=0)      
        reorder = norm_usages.sum(axis=0).sort_values(ascending=False)
        rf_usages = rf_usages.loc[:, reorder.index]
        norm_usages = norm_usages.loc[:, reorder.index]
        median_spectra = median_spectra.loc[reorder.index, :]
        rf_usages.columns = np.arange(1, rf_usages.shape[1]+1)
        norm_usages.columns = rf_usages.columns
        median_spectra.index = rf_usages.columns

        # Compute prediction error as a frobenius norm
        if sp.issparse(norm_counts.X):
            prediction_error = ((norm_counts.X.todense() - rf_pred_norm_counts)**2).sum().sum()
        else:
            prediction_error = ((norm_counts.X - rf_pred_norm_counts)**2).sum().sum()
        
        consensus_stats = pd.DataFrame([k, density_threshold, stability, prediction_error],
                    index = ['k', 'local_density_threshold', 'stability', 'prediction_error'],
                    columns = ['stats'])

        if skip_density_and_return_after_stats:
            return consensus_stats
        
        # Convert spectra to TPM units, and obtain results for all genes by running last step of NMF
        # with usages fixed and TPM as the input matrix
        tpm = sc.read(self.paths['tpm'])
        tpm_stats = load_df_from_npz(self.paths['tpm_stats'])
        spectra_tpm = self.refit_spectra(tpm.X, norm_usages.astype(tpm.X.dtype))
        spectra_tpm = pd.DataFrame(spectra_tpm, index=rf_usages.columns, columns=tpm.var.index)
        spectra_tpm = spectra_tpm.div(spectra_tpm.sum(axis=1), axis=0) * 1e6
        
        # Convert spectra to Z-score units, and obtain results for all genes by running last step of NMF
        # with usages fixed and Z-scored TPM as the input matrix
        if sp.issparse(tpm.X):
            norm_tpm = (np.array(tpm.X.todense()) - tpm_stats['__mean'].values) / tpm_stats['__std'].values
        else:
            norm_tpm = (tpm.X - tpm_stats['__mean'].values) / tpm_stats['__std'].values
        
        usage_coef = fast_ols_all_cols(rf_usages.values, norm_tpm)
        usage_coef = pd.DataFrame(usage_coef, index=rf_usages.columns, columns=tpm.var.index)
        
        if refit_usage:
            ## Re-fitting usage a final time on std-scaled HVG TPM seems to
            ## increase accuracy on simulated data
            hvgs = open(self.paths['nmf_genes_list']).read().split('\n')
            norm_tpm = tpm[:, hvgs]
            if sp.issparse(norm_tpm.X):
                sc.pp.scale(norm_tpm, zero_center=False)                       
            else:
                norm_tpm.X /= norm_tpm.X.std(axis=0, ddof=1)
                
            spectra_tpm_rf = spectra_tpm.loc[:,hvgs]

            spectra_tpm_rf = spectra_tpm_rf.div(tpm_stats.loc[hvgs, '__std'], axis=1)
            rf_usages = self.refit_usage(norm_tpm.X, spectra_tpm_rf)
            rf_usages = pd.DataFrame(rf_usages, index=norm_counts.obs.index, columns=spectra_tpm_rf.index)                                                                  
               
        save_df_to_npz(median_spectra, self.paths['consensus_spectra']%(k, density_threshold_repl))
        save_df_to_npz(rf_usages, self.paths['consensus_usages']%(k, density_threshold_repl))
        save_df_to_npz(consensus_stats, self.paths['consensus_stats']%(k, density_threshold_repl))
        save_df_to_text(median_spectra, self.paths['consensus_spectra__txt']%(k, density_threshold_repl))
        save_df_to_text(rf_usages, self.paths['consensus_usages__txt']%(k, density_threshold_repl))
        save_df_to_npz(spectra_tpm, self.paths['gene_spectra_tpm']%(k, density_threshold_repl))
        save_df_to_text(spectra_tpm, self.paths['gene_spectra_tpm__txt']%(k, density_threshold_repl))
        save_df_to_npz(usage_coef, self.paths['gene_spectra_score']%(k, density_threshold_repl))
        save_df_to_text(usage_coef, self.paths['gene_spectra_score__txt']%(k, density_threshold_repl))
        if show_clustering:
            if topics_dist is None:
                topics_dist = euclidean_distances(l2_spectra.values)
                # (l2_spectra was already filtered using the density filter)
            else:
                # (but the previously computed topics_dist was not!)
                topics_dist = topics_dist[density_filter.values, :][:, density_filter.values]


            spectra_order = []
            for cl in sorted(set(kmeans_cluster_labels)):

                cl_filter = kmeans_cluster_labels==cl

                if cl_filter.sum() > 1:
                    from fastcluster import linkage
                    cl_dist = squareform(topics_dist[cl_filter, :][:, cl_filter], checks=False)
                    cl_dist[cl_dist < 0] = 0 #Rarely get floating point arithmetic issues
                    cl_link = linkage(cl_dist, 'average')
                    cl_leaves_order = leaves_list(cl_link)

                    spectra_order += list(np.where(cl_filter)[0][cl_leaves_order])
                else:
                    ## Corner case where a component only has one element
                    spectra_order += list(np.where(cl_filter)[0])


            from matplotlib import gridspec
            import matplotlib.pyplot as plt

            width_ratios = [0.5, 9, 0.5, 4, 1]
            height_ratios = [0.5, 9]
            fig = plt.figure(figsize=(sum(width_ratios), sum(height_ratios)))
            gs = gridspec.GridSpec(len(height_ratios), len(width_ratios), fig,
                                    0.01, 0.01, 0.98, 0.98,
                                   height_ratios=height_ratios,
                                   width_ratios=width_ratios,
                                   wspace=0, hspace=0)

            dist_ax = fig.add_subplot(gs[1,1], xscale='linear', yscale='linear',
                                      xticks=[], yticks=[],xlabel='', ylabel='',
                                      frameon=True)

            D = topics_dist[spectra_order, :][:, spectra_order]
            dist_im = dist_ax.imshow(D, interpolation='none', cmap='viridis',
                                     aspect='auto', rasterized=True)

            left_ax = fig.add_subplot(gs[1,0], xscale='linear', yscale='linear', xticks=[], yticks=[],
                xlabel='', ylabel='', frameon=True)
            left_ax.imshow(kmeans_cluster_labels.values[spectra_order].reshape(-1, 1),
                            interpolation='none', cmap='Spectral', aspect='auto',
                            rasterized=True)


            top_ax = fig.add_subplot(gs[0,1], xscale='linear', yscale='linear', xticks=[], yticks=[],
                xlabel='', ylabel='', frameon=True)
            top_ax.imshow(kmeans_cluster_labels.values[spectra_order].reshape(1, -1),
                              interpolation='none', cmap='Spectral', aspect='auto',
                                rasterized=True)


            hist_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 3],
                                   wspace=0, hspace=0)

            hist_ax = fig.add_subplot(hist_gs[0,0], xscale='linear', yscale='linear',
                xlabel='', ylabel='', frameon=True, title='Local density histogram')
            hist_ax.hist(local_density.values, bins=np.linspace(0, 1, 50))
            hist_ax.yaxis.tick_right()

            xlim = hist_ax.get_xlim()
            ylim = hist_ax.get_ylim()
            if density_threshold < xlim[1]:
                hist_ax.axvline(density_threshold, linestyle='--', color='k')
                hist_ax.text(density_threshold  + 0.02, ylim[1] * 0.95, 'filtering\nthreshold\n\n', va='top')
            hist_ax.set_xlim(xlim)
            hist_ax.set_xlabel('Mean distance to k nearest neighbors\n\n%d/%d (%.0f%%) spectra above threshold\nwere removed prior to clustering'%(sum(~density_filter), len(density_filter), 100*(~density_filter).mean()))
            
            ## Add colorbar
            cbar_gs = gridspec.GridSpecFromSubplotSpec(8, 1, subplot_spec=hist_gs[1, 0],
                                   wspace=0, hspace=0)
            cbar_ax = fig.add_subplot(cbar_gs[4,0], xscale='linear', yscale='linear',
                xlabel='', ylabel='', frameon=True, title='Euclidean Distance')
            vmin = D.min().min()
            vmax = D.max().max()
            fig.colorbar(dist_im, cax=cbar_ax,
            ticks=np.linspace(vmin, vmax, 3),
            orientation='horizontal')
            
            
            #hist_ax.hist(local_density.values, bins=np.linspace(0, 1, 50))
            #hist_ax.yaxis.tick_right()            

            fig.savefig(self.paths['clustering_plot']%(k, density_threshold_repl), dpi=250)
            if close_clustergram_fig:
                plt.close(fig)

        self.topic_dist = topics_dist
        self.spectra_order=spectra_order
        self.local_density = local_density
        self.kmeans_cluster_labels=kmeans_cluster_labels


    def k_selection_plot(self, close_fig=False):
        '''
        Borrowed from Alexandrov Et Al. 2013 Deciphering Mutational Signatures
        publication in Cell Reports
        '''
        run_params = load_df_from_npz(self.paths['nmf_replicate_parameters'])
        stats = []
        for k in sorted(set(run_params.n_components)):

            stats.append(self.consensus(k, skip_density_and_return_after_stats=True,
                                        show_clustering=False, close_clustergram_fig=True).stats)

        stats = pd.DataFrame(stats)
        stats.reset_index(drop = True, inplace = True)

        save_df_to_npz(stats, self.paths['k_selection_stats'])

        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()


        ax1.plot(stats.k, stats.stability, 'o-', color='b')
        ax1.set_ylabel('Stability', color='b', fontsize=15)
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        #ax1.set_xlabel('K', fontsize=15)

        ax2.plot(stats.k, stats.prediction_error, 'o-', color='r')
        ax2.set_ylabel('Error', color='r', fontsize=15)
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        ax1.set_xlabel('Number of Components', fontsize=15)
        ax1.grid('on')
        plt.tight_layout()
        fig.savefig(self.paths['k_selection_plot'], dpi=250)
        if close_fig:
            plt.close(fig)
            
            
    def load_results(self, K, density_threshold, n_top_genes=100):
        """
        Loads normalized usages and gene_spectra_scores for a given choice of K and 
        local_density_threshold for the cNMF run. Additionally returns a DataFrame of
        the top genes linked to each program with the number of genes indicated by the
        `n_top_genes` parameter
        
        Returns
        usage - cNMF usages (cells X K) normalized to sum to 1
        spectra - Z-score regressed coeffecients for each program (K x genes) with higher values cooresponding
                    to better marker genes
        top_genes - ranked list of marker genes per GEP (n_top_genes X K)
        """
        scorefn = self.paths['gene_spectra_score__txt'] % (K, str(density_threshold).replace('.', '_'))
        tpmfn = self.paths['gene_spectra_tpm__txt'] % (K, str(density_threshold).replace('.', '_'))
        usagefn = self.paths['consensus_usages__txt'] % (K, str(density_threshold).replace('.', '_'))
        spectra_scores = pd.read_csv(scorefn, sep='\t', index_col=0).T
        spectra_tpm = pd.read_csv(tpmfn, sep='\t', index_col=0).T

        usage = pd.read_csv(usagefn, sep='\t', index_col=0)
        usage = usage.div(usage.sum(axis=1), axis=0)
        
        try:
            usage.columns = [int(x) for x in usage.columns]
        except:
            print('Usage matrix columns include non integer values')
    
        top_genes = []
        for gep in spectra_scores.columns:
            top_genes.append(list(spectra_scores.sort_values(by=gep, ascending=False).index[:n_top_genes]))
        
        top_genes = pd.DataFrame(top_genes, index=spectra_scores.columns).T
        usage.columns = ['cNMF_%d' % i for i in usage.columns]
        result_dict = {}
        result_dict['usage_norm'] = usage
        result_dict['gep_scores'] = spectra_scores
        result_dict['gep_tpm'] = spectra_tpm
        result_dict['top_genes'] = top_genes
        return result_dict
    
    def get_results(self,adata,result_dict):
        import pandas as pd
        if result_dict['usage_norm'].columns[0] in adata.obs.columns:
            #remove the columns if they already exist
            #remove columns name starts with 'cNMF'
            adata.obs = adata.obs.loc[:,~adata.obs.columns.str.startswith('cNMF')]
        adata.obs = pd.merge(left=adata.obs, right=result_dict['usage_norm'], 
                             how='left', left_index=True, right_index=True)
        adata.var = pd.merge(left=adata.var,right=result_dict['gep_scores'].loc[adata.var.index],
                             how='left', left_index=True, right_index=True)
        df=adata.obs[result_dict['usage_norm'].columns].copy()
        max_topic = df.idxmax(axis=1)
        # 将结果添加到DataFrame中
        adata.obs['cNMF_cluster'] = max_topic
        print('cNMF_cluster is added to adata.obs')
        print('gene scores are added to adata.var')

    def get_results_rfc(self,adata,result_dict,use_rep='STAGATE',cNMF_threshold=0.5):
        import pandas as pd
        if result_dict['usage_norm'].columns[0] in adata.obs.columns:
            #remove the columns if they already exist
            #remove columns name starts with 'cNMF'
            adata.obs = adata.obs.loc[:,~adata.obs.columns.str.startswith('cNMF')]
        adata.obs = pd.merge(left=adata.obs, right=result_dict['usage_norm'], 
                             how='left', left_index=True, right_index=True)
        adata.var = pd.merge(left=adata.var,right=result_dict['gep_scores'].loc[adata.var.index],
                             how='left', left_index=True, right_index=True)
        
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        new_array = []
        class_array = []
        for i in range(1, result_dict['usage_norm'].shape[1] + 1):
            data = adata[adata.obs[f'cNMF_{i}'] > cNMF_threshold].obsm[use_rep].toarray()
            new_array.append(data)
            class_array.append(np.full(data.shape[0], i))

        new_array = np.concatenate(new_array, axis=0)
        class_array = np.concatenate(class_array)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(new_array,class_array,test_size=0.3)
        clf = DecisionTreeClassifier(random_state=0)
        rfc = RandomForestClassifier(random_state=0)
        clf = clf.fit(Xtrain,Ytrain)
        rfc = rfc.fit(Xtrain,Ytrain)
        #查看模型效果
        score_c = clf.score(Xtest,Ytest)
        score_r = rfc.score(Xtest,Ytest)
        #打印最后结果
        print("Single Tree:",score_c)
        print("Random Forest:",score_r)

        adata.obs['cNMF_cluster_rfc']=[str(i) for i in rfc.predict(adata.obsm[use_rep])]
        adata.obs['cNMF_cluster_clf']=[str(i) for i in clf.predict(adata.obsm[use_rep])]
        print('cNMF_cluster_rfc is added to adata.obs')
        print('cNMF_cluster_clf is added to adata.obs')





