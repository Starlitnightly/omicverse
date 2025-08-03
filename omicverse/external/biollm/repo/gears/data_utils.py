import pandas as pd
import numpy as np
import scanpy as sc
from random import shuffle
sc.settings.verbosity = 0
from tqdm import tqdm
import requests
import os, sys

import warnings
warnings.filterwarnings("ignore")

from .utils import parse_single_pert, parse_combo_pert, parse_any_pert, print_sys

def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added='rank_genes_groups_cov',
    return_dict=False,
):

    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        #name of the control group in the groupby obs column
        control_group_cov = '_'.join([cov_cat, control_group])

        #subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate]==cov_cat]

        #compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False
        )

        #add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns['rank_genes_groups']['names'])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict

    
def get_DE_genes(adata, skip_calc_de):
    adata.obs.loc[:, 'dose_val'] = adata.obs.condition.apply(lambda x: '1+1' if len(x.split('+')) == 2 else '1')
    adata.obs.loc[:, 'control'] = adata.obs.condition.apply(lambda x: 0 if len(x.split('+')) == 2 else 1)
    adata.obs.loc[:, 'condition_name'] =  adata.obs.apply(lambda x: '_'.join([x.cell_type, x.condition, x.dose_val]), axis = 1) 
    
    adata.obs = adata.obs.astype('category')
    if not skip_calc_de:
        rank_genes_groups_by_cov(adata, 
                         groupby='condition_name', 
                         covariate='cell_type', 
                         control_group='ctrl_1', 
                         n_genes=len(adata.var),
                         key_added = 'rank_genes_groups_cov_all')
    return adata

def get_dropout_non_zero_genes(adata):
    
    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.toarray().shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]
    
    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    pert_full_id2pert = dict(adata.obs[['condition_name', 'condition']].values)

    gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
    gene_idx2id = dict(zip(range(len(adata.var)), adata.var.index.values))

    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}
    non_dropout_gene_idx = {}

    for pert in adata.uns['rank_genes_groups_cov_all'].keys():
        p = pert_full_id2pert[pert]
        X = np.mean(adata[adata.obs.condition == p].X, axis = 0)

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        top = adata.uns['rank_genes_groups_cov_all'][pert]
        gene_idx_top = [gene_id2idx[i] for i in top]

        non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
        non_dropout_20_gene_id = [gene_idx2id[i] for i in non_dropout_20]

        non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
        non_zero_20_gene_id = [gene_idx2id[i] for i in non_zero_20]

        non_zeros_gene_idx[pert] = np.sort(non_zero)
        non_dropout_gene_idx[pert] = np.sort(non_dropouts)
        top_non_dropout_de_20[pert] = np.array(non_dropout_20_gene_id)
        top_non_zero_de_20[pert] = np.array(non_zero_20_gene_id)
        
    non_zero = np.where(np.array(X)[0] != 0)[0]
    zero = np.where(np.array(X)[0] == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))
    
    adata.uns['top_non_dropout_de_20'] = top_non_dropout_de_20
    adata.uns['non_dropout_gene_idx'] = non_dropout_gene_idx
    adata.uns['non_zeros_gene_idx'] = non_zeros_gene_idx
    adata.uns['top_non_zero_de_20'] = top_non_zero_de_20
    
    return adata


class DataSplitter():
    """
    Class for handling data splitting. This class is able to generate new
    data splits and assign them as a new attribute to the data file.
    """
    def __init__(self, adata, split_type='single', seen=0):
        self.adata = adata
        self.split_type = split_type
        self.seen = seen

    def split_data(self, test_size=0.1, test_pert_genes=None,
                   test_perts=None, split_name='split', seed=None, val_size = 0.1,
                   train_gene_set_size = 0.75, combo_seen2_train_frac = 0.75, only_test_set_perts = False):
        """
        Split dataset and adds split as a column to the dataframe
        Note: split categories are train, val, test
        """
        np.random.seed(seed=seed)
        unique_perts = [p for p in self.adata.obs['condition'].unique() if
                        p != 'ctrl']
        
        if self.split_type == 'simulation':
            train, test, test_subgroup = self.get_simulation_split(unique_perts,
                                                                  train_gene_set_size,
                                                                  combo_seen2_train_frac, 
                                                                  seed, test_perts, only_test_set_perts)
            train, val, val_subgroup = self.get_simulation_split(train,
                                                                  0.9,
                                                                  0.9,
                                                                  seed)
            ## adding back ctrl to train...
            train.append('ctrl')
        elif self.split_type == 'simulation_single':
            train, test, test_subgroup = self.get_simulation_split_single(unique_perts,
                                                                  train_gene_set_size,
                                                                  seed, test_perts, only_test_set_perts)
            train, val, val_subgroup = self.get_simulation_split_single(train,
                                                                  0.9,
                                                                  seed)
        elif self.split_type == 'no_test':
            train, val = self.get_split_list(unique_perts,
                                          test_size=val_size)      
        else:
            train, test = self.get_split_list(unique_perts,
                                          test_pert_genes=test_pert_genes,
                                          test_perts=test_perts,
                                          test_size=test_size)
            
            train, val = self.get_split_list(train, test_size=val_size)

        map_dict = {x: 'train' for x in train}
        map_dict.update({x: 'val' for x in val})
        if self.split_type != 'no_test':
            map_dict.update({x: 'test' for x in test})
        map_dict.update({'ctrl': 'train'})

        self.adata.obs[split_name] = self.adata.obs['condition'].map(map_dict)

        if self.split_type == 'simulation':
            return self.adata, {'test_subgroup': test_subgroup, 
                                'val_subgroup': val_subgroup
                               }
        else:
            return self.adata
    
    def get_simulation_split_single(self, pert_list, train_gene_set_size = 0.85, seed = 1, test_set_perts = None, only_test_set_perts = False):
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        
        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)
        
        if only_test_set_perts and (test_set_perts is not None):
            ood_genes = np.array(test_set_perts)
            train_gene_candidates = np.setdiff1d(unique_pert_genes, ood_genes)
        else:
            ## a pre-specified list of genes
            train_gene_candidates = np.random.choice(unique_pert_genes,
                                                    int(len(unique_pert_genes) * train_gene_set_size), replace = False)

            if test_set_perts is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perts))
                train_gene_candidates = train_gene_candidates[~np.isin(train_gene_candidates, test_set_perts)]
                ood_genes_exclude_test_set = np.setdiff1d(unique_pert_genes, np.union1d(train_gene_candidates, test_set_perts))
                train_set_addition = np.random.choice(ood_genes_exclude_test_set, num_overlap, replace = False)
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))
                
            ## ood genes
            ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)  
        
        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list,'single')
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, 'single')
        assert len(unseen_single) + len(pert_single_train) == len(pert_list)
        
        return pert_single_train, unseen_single, {'unseen_single': unseen_single}
    
    def get_simulation_split(self, pert_list, train_gene_set_size = 0.85, combo_seen2_train_frac = 0.85, seed = 1, test_set_perts = None, only_test_set_perts = False):
        
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        
        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)
        
        if only_test_set_perts and (test_set_perts is not None):
            ood_genes = np.array(test_set_perts)
            train_gene_candidates = np.setdiff1d(unique_pert_genes, ood_genes)
        else:
            ## a pre-specified list of genes
            train_gene_candidates = np.random.choice(unique_pert_genes,
                                                    int(len(unique_pert_genes) * train_gene_set_size), replace = False)

            if test_set_perts is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perts))
                train_gene_candidates = train_gene_candidates[~np.isin(train_gene_candidates, test_set_perts)]
                ood_genes_exclude_test_set = np.setdiff1d(unique_pert_genes, np.union1d(train_gene_candidates, test_set_perts))
                train_set_addition = np.random.choice(ood_genes_exclude_test_set, num_overlap, replace = False)
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))
                
            ## ood genes
            ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)                
        
        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list,'single')
        pert_combo = self.get_perts_from_genes(train_gene_candidates, pert_list,'combo')
        pert_train.extend(pert_single_train)
        
        ## the combo set with one of them in OOD
        combo_seen1 = [x for x in pert_combo if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 1]
        pert_test.extend(combo_seen1)
        
        pert_combo = np.setdiff1d(pert_combo, combo_seen1)
        ## randomly sample the combo seen 2 as a test set, the rest in training set
        np.random.seed(seed=seed)
        pert_combo_train = np.random.choice(pert_combo, int(len(pert_combo) * combo_seen2_train_frac), replace = False)
       
        combo_seen2 = np.setdiff1d(pert_combo, pert_combo_train).tolist()
        pert_test.extend(combo_seen2)
        pert_train.extend(pert_combo_train)
        
        ## unseen single
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, 'single')
        combo_ood = self.get_perts_from_genes(ood_genes, pert_list, 'combo')
        pert_test.extend(unseen_single)
        
        ## here only keeps the seen 0, since seen 1 is tackled above
        combo_seen0 = [x for x in combo_ood if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 0]
        pert_test.extend(combo_seen0)
        assert len(combo_seen1) + len(combo_seen0) + len(unseen_single) + len(pert_train) + len(combo_seen2) == len(pert_list)

        return pert_train, pert_test, {'combo_seen0': combo_seen0,
                                       'combo_seen1': combo_seen1,
                                       'combo_seen2': combo_seen2,
                                       'unseen_single': unseen_single}
        
    def get_split_list(self, pert_list, test_size=0.1,
                       test_pert_genes=None, test_perts=None,
                       hold_outs=True):
        """
        Splits a given perturbation list into train and test with no shared
        perturbations
        """

        single_perts = [p for p in pert_list if 'ctrl' in p and p != 'ctrl']
        combo_perts = [p for p in pert_list if 'ctrl' not in p]
        unique_pert_genes = self.get_genes_from_perts(pert_list)
        hold_out = []

        if test_pert_genes is None:
            test_pert_genes = np.random.choice(unique_pert_genes,
                                        int(len(single_perts) * test_size))

        # Only single unseen genes (in test set)
        # Train contains both single and combos
        if self.split_type == 'single' or self.split_type == 'single_only':
            test_perts = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                   'single')
            if self.split_type == 'single_only':
                # Discard all combos
                hold_out = combo_perts
            else:
                # Discard only those combos which contain test genes
                hold_out = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                     'combo')
        
        elif self.split_type == 'no_test':
            if test_perts is None:
                test_perts = np.random.choice(pert_list,
                                    int(len(pert_list) * test_size))
            

        elif self.split_type == 'combo':
            if self.seen == 0:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 1 gene seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 0]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 1:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 2 genes seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 1]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 2:
                if test_perts is None:
                    test_perts = np.random.choice(combo_perts,
                                     int(len(combo_perts) * test_size))       
                else:
                    test_perts = np.array(test_perts)
        else:
            if test_perts is None:
                test_perts = np.random.choice(combo_perts,
                                    int(len(combo_perts) * test_size))
        
        train_perts = [p for p in pert_list if (p not in test_perts)
                                        and (p not in hold_out)]
        return train_perts, test_perts

    def get_perts_from_genes(self, genes, pert_list, type_='both'):
        """
        Returns all single/combo/both perturbations that include a gene
        """

        single_perts = [p for p in pert_list if ('ctrl' in p) and (p != 'ctrl')]
        combo_perts = [p for p in pert_list if 'ctrl' not in p]
        
        perts = []
        
        if type_ == 'single':
            pert_candidate_list = single_perts
        elif type_ == 'combo':
            pert_candidate_list = combo_perts
        elif type_ == 'both':
            pert_candidate_list = pert_list
            
        for p in pert_candidate_list:
            for g in genes:
                if g in parse_any_pert(p):
                    perts.append(p)
                    break
        return perts

    def get_genes_from_perts(self, perts):
        """
        Returns list of genes involved in a given perturbation list
        """

        if type(perts) is str:
            perts = [perts]
        gene_list = [p.split('+') for p in np.unique(perts)]
        gene_list = [item for sublist in gene_list for item in sublist]
        gene_list = [g for g in gene_list if g != 'ctrl']
        return np.unique(gene_list)