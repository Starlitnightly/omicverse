import numpy as np
import pandas as pd
import scanpy as sc
from typing import Union, Sequence

def get_top_nmf_genes(adata: sc.AnnData,
                      gems: Union[str, Sequence[str]],
                      n_genes: int,
                      gene_type: str = 'all',
                      model_organism: str = 'human'):
    import pkg_resources
    tf_path=pkg_resources.resource_filename("omicverse", 'data_files/TF/allTFs_' + model_organism + '.txt')
    gene_types = ['all', 'tf']

    if gene_type not in gene_types:
        raise ValueError ("Invalid gene type. Please select one of: %s" % gene_types)
    
    model_organisms = ['human', 'mouse']

    if model_organism not in model_organisms:
        raise ValueError ("Invalid model organism. Please select one of: %s" % model_organisms)
    
    # Load the TFs if we need to
    if gene_type == 'tf':
        tfs_list = pd.read_csv(tf_path, header=None)[0].tolist()

    n_gems = adata.uns['nmf_info']['n_gems']
    nmf_vars = np.array(adata.uns['nmf_info']['vars'], dtype=object)
    nmf_loadings = adata.uns['nmf_info']['loadings']

    all_gems = ['GEM-' + str(i + 1) for i in range(n_gems)]

    top_genes_in_gems = []
    gem_labels = []
    gem_weights = []

    for gem in gems:
         
        gem_index = all_gems.index(gem)

        # Get the loadings corresponding tot his gem
        gem_loadings = nmf_loadings[gem_index, :]

        # Sort the genes in their order by loading
        sorted_gem_loadings = gem_loadings[np.argsort(-gem_loadings)]
        ordered_genes = nmf_vars[np.argsort(-gem_loadings)]

        if gene_type == 'all':
              
            for i in range(n_genes):

                top_genes_in_gems.append(ordered_genes[i])
                gem_labels.append(gem)
                gem_weights.append(sorted_gem_loadings[i])

        else: # We only take TFs from each GEM
         
            ordered_tfs = [gene for gene in ordered_genes if gene in tfs_list]
            sorted_tf_loadings = [sorted_gem_loadings[i] for i, gene in enumerate(ordered_genes) if gene in tfs_list]

            for i in range(n_genes):

                top_genes_in_gems.append(ordered_tfs[i])
                gem_labels.append(gem)
                gem_weights.append(sorted_tf_loadings[i])

    top_nmf_genes_df = pd.DataFrame(data={'Gene': top_genes_in_gems,
                                          'GEM': gem_labels,
                                          'Weight': gem_weights})
    return top_nmf_genes_df

def get_top_pyliger_genes(adata: sc.AnnData,
                        gems: Union[str, Sequence[str]],
                        n_genes: int,
                        gene_type: str = 'all',
                        model_organism: str = 'human'):
    
    import pkg_resources
    tf_path=pkg_resources.resource_filename("omicverse", 'data_files/TF/allTFs_' + model_organism + '.txt')

    gene_types = ['all', 'tf']

    if gene_type not in gene_types:
        raise ValueError ("Invalid gene type. Please select one of: %s" % gene_types)
    
    model_organisms = ['human', 'mouse']

    if model_organism not in model_organisms:
        raise ValueError ("Invalid model organism. Please select one of: %s" % model_organisms)

    # Load the TFs if we need to
    if gene_type == 'tf':
        tfs_list = pd.read_csv(tf_path, header=None)[0].tolist()

    pyliger_conds = [key for key in adata.uns['pyliger_info'].keys() if key not in ['n_vars', 'vars','n_gems']]

    n_gems = adata.uns['pyliger_info']['n_gems']
    pyliger_vars = np.array(adata.uns['pyliger_info']['vars'], dtype=object)

    all_gems = ['GEM-' + str(i + 1) for i in range(n_gems)]

    top_genes_in_gems = []
    gem_labels = []
    gem_weights = []

    for gem in gems:
        
        gem_index = all_gems.index(gem)

        all_gem_loadings = {cond: adata.uns['pyliger_info'][cond]['W'][:, gem_index] \
                                    + adata.uns['pyliger_info'][cond]['V'][:, gem_index] for cond in pyliger_conds}

        stacked_gem_loadings = np.hstack([all_gem_loadings[cond] for cond in all_gem_loadings])
        stacked_gem_genes = np.hstack([pyliger_vars for cond in all_gem_loadings])     
        sorted_gem_loadings = stacked_gem_loadings[np.argsort(-stacked_gem_loadings)]
        init_top_gem_genes = stacked_gem_genes[np.argsort(-stacked_gem_loadings)]
        
        top_genes = []

        for i, gene in enumerate(init_top_gem_genes):

            if gene_type == 'tf':

                if (gene in tfs_list)&(gene not in top_genes):
                    
                    top_genes.append(gene) # This tracks repeats, given that we're stacking the list of genes 
                    top_genes_in_gems.append(gene)
                    gem_labels.append(gem)
                    gem_weights.append(sorted_gem_loadings[i])
                    
                    if len(top_genes) == n_genes:
                        break
            else:

                if gene not in top_genes:

                    top_genes.append(gene) # This tracks repeated names, given that we're stacking the list of genes
                    top_genes_in_gems.append(gene)
                    gem_labels.append(gem)
                    gem_weights.append(sorted_gem_loadings[i])
                    
                if len(top_genes) == n_genes:
                    break

        

    top_pyliger_genes_df = pd.DataFrame(data={'Gene': top_genes_in_gems,
                                          'GEM': gem_labels,
                                          'Weight': gem_weights})
    return top_pyliger_genes_df

def get_top_nsf_genes(adata: sc.AnnData,
                      gems: Union[str, Sequence[str]],
                      n_genes: int,
                      gene_type: str = 'all',
                      model_organism: str = 'human'):
    import pkg_resources
    tf_path=pkg_resources.resource_filename("omicverse", 'data_files/TF/allTFs_' + model_organism + '.txt')
    
    gene_types = ['all', 'tf']

    if gene_type not in gene_types:
        raise ValueError ("Invalid gene type. Please select one of: %s" % gene_types)
    
    model_organisms = ['human', 'mouse']

    if model_organism not in model_organisms:
        raise ValueError ("Invalid model organism. Please select one of: %s" % model_organisms)
    
    # Load the TFs if we need to
    if gene_type == 'tf':
        tfs_list = pd.read_csv(tf_path, header=None)[0].tolist()

    n_gems = adata.uns['nsf_info']['n_gems']
    nsf_vars = np.array(adata.uns['nsf_info']['vars'], dtype=object)
    nsf_loadings = adata.uns['nsf_info']['loadings'].T

    all_gems = ['GEM-' + str(i + 1) for i in range(n_gems)]

    top_genes_in_gems = []
    gem_labels = []
    gem_weights = []

    for gem in gems:
         
        gem_index = all_gems.index(gem)

        # Get the loadings corresponding tot his gem
        gem_loadings = nsf_loadings[gem_index, :]

        # Sort the genes in their order by loading
        sorted_gem_loadings = gem_loadings[np.argsort(-gem_loadings)]
        ordered_genes = nsf_vars[np.argsort(-gem_loadings)]

        if gene_type == 'all':
              
            for i in range(n_genes):

                top_genes_in_gems.append(ordered_genes[i])
                gem_labels.append(gem)
                gem_weights.append(sorted_gem_loadings[i])

        else: # We only take TFs from each GEM
         
            ordered_tfs = [gene for gene in ordered_genes if gene in tfs_list]
            sorted_tf_loadings = [sorted_gem_loadings[i] for i, gene in enumerate(ordered_genes) if gene in tfs_list]

            for i in range(n_genes):

                top_genes_in_gems.append(ordered_tfs[i])
                gem_labels.append(gem)
                gem_weights.append(sorted_tf_loadings[i])

    top_nsf_genes_df = pd.DataFrame(data={'Gene': top_genes_in_gems,
                                          'GEM': gem_labels,
                                          'Weight': gem_weights})
    return top_nsf_genes_df

def get_top_gem_genes(adata: sc.AnnData,
                      gems: Union[str, Sequence[str]],
                      n_genes: int,
                      method: str = 'pyliger',
                      gene_type: str = 'all',
                      model_organism: str = 'human'):
    
    # Perform a bunch of checks that we're specifying the right options
    methods = ['nmf', 'pyliger', 'nsf']
    if method not in methods:
                raise ValueError ("Invalid method. Please select one of: %s" % methods)
    
    gene_types = ['all', 'tf']

    if gene_type not in gene_types:
        raise ValueError ("Invalid gene type. Please select one of: %s" % gene_types)
    
    model_organisms = ['human', 'mouse']

    if model_organism not in model_organisms:
        raise ValueError ("Invalid model organism. Please select one of: %s" % model_organisms)
    
    if method == 'nmf':
          
          return get_top_nmf_genes(adata, gems, n_genes, gene_type, model_organism)
    
    elif method == 'pyliger':
          
        return get_top_pyliger_genes(adata, gems, n_genes, gene_type, model_organism)
    
    else: # Should be NSF
    
        return get_top_nsf_genes(adata, gems, n_genes, gene_type, model_organism)
