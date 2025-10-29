import numpy as np
from itertools import compress
import pandas as pd
import os

def norm_to_one(ref, pseudo_min):
    
    ref_index = ref.index
    ref_columns = ref.columns
    
    ref = ref.to_numpy()
    g = ref.shape[1]

    phi = ref / np.sum(ref, axis = 1, keepdims=True) * (1 - pseudo_min * g) + pseudo_min

    min_value = np.min(ref, axis = 1)
    which_row = min_value > 0
    if np.any(which_row):
            phi[which_row, :] = ref[which_row, :] / np.sum(ref[which_row, :], axis = 1, keepdims=True)
    
    phi = pd.DataFrame(phi, index = ref_index, columns = ref_columns)
    return phi


def collapse(ref, labels):

        assert ref.shape[0] == len(labels), "Error: nrow(ref) and length(labels) do not match!"

        non_na_idx = [x is not None for x in labels]
        if non_na_idx.count(False) > 0:
            print("Warning: NA found in the cell type/state labels. These cells will be excluded!")
        labels = list(compress(labels, non_na_idx))
        ref = ref.loc[non_na_idx, :]

        labels_seen = set()
        labels_uniq = [x for x in labels if not (x in labels_seen or labels_seen.add(x))]

        ref_collapsed = pd.DataFrame()
        for label_i in labels_uniq:
            indices = [label_i == i for i in labels]
            ref_label = ref.loc[indices].to_numpy()
            ref_collapsed = pd.concat([ref_collapsed, pd.Series(np.sum(ref_label, axis=0)).to_frame().T], ignore_index=True)
        ref_collapsed.index = labels_uniq
        ref_collapsed.columns = ref.columns

        return ref_collapsed


def validate_input(input):
        input_ndarray = input.to_numpy()
        if np.max(input_ndarray) <= 1:
            print("Warning: input seems to be normalized.")
        elif np.max(input_ndarray) < 20:
            print("Warning: input seems to be log-transformed. Please double \
                  check your input. Log transformation should be avoided")
        
        if np.any(input_ndarray < 0) or np.any(np.isinf(input_ndarray)):
            raise ValueError("Error: input contains negative, non-finite or \
                             non-numeric values. Please double check your \
                             input is unnormalized and untransformed raw count.")
        
        if input.columns.empty:
            raise ValueError("Error: please specify the colnames of mixture / \
                             reference using gene identifiers!")

        if not isinstance(input, pd.DataFrame):
            raise ValueError("Error: the type of mixture and reference need \
                             to be DataFrame!")
        

def filter_bulk_outlier(mixture, outlier_cut, outlier_fraction):

    mixture_ndarray = mixture.to_numpy()

    mixture_norm = mixture_ndarray / np.sum(mixture_ndarray, axis = 1, keepdims=True)

    outlier_idx = np.sum(mixture_norm > outlier_cut, axis = 0) / mixture_norm.shape[0] > outlier_fraction

    mixture = mixture.loc[:, ~outlier_idx]
    print("Number of outlier genes filtered from mixture =", np.sum(outlier_idx))
    return mixture


def assign_category(input_genes, species):
    assert len(species) == 2 and species in ["hs", "mm"]
    
    if species == "hs":
        gene_list = pd.read_table(os.path.join(os.path.split(__file__)[0], "txt", "genelist.hs.new.txt"), sep="\t", header=None)
    if species == "mm":
        gene_list = pd.read_table(os.path.join(os.path.split(__file__)[0], "txt", "genelist.mm.new.txt"), sep="\t", header=None)

    if sum(1 for gene in input_genes if gene[:3] == "ENS") > len(input_genes) * 0.8:
        print("EMSEMBLE IDs detected.")
        input_genes_short = [gene.split(".")[0] for gene in input_genes]
        gene_df = gene_list.iloc[:, [0, 1]]
    else:
        print("Gene symbols detected. Recommend to use EMSEMBLE IDs for more unique mapping.")
        input_genes_short = input_genes
        gene_df = gene_list.iloc[:, [0, 2]]
    
    logic = [np.isin(np.array(input_genes_short), gene_df.loc[gene_df.iloc[:,0]==i, 1]) for i in gene_df.iloc[:, 0].unique()]
    logic = np.column_stack(logic)
    gene_group_matrix = pd.DataFrame(logic)
    gene_group_matrix.columns = gene_df.iloc[:, 0].unique()
    gene_group_matrix.index = input_genes
    return gene_group_matrix


def cleanup_genes(input : pd.DataFrame, input_type, species, gene_group, exp_cells=1):
    
    assert species in ["hs", "mm"]
    a = ["other_Rb", "chrM", "chrX", "chrY", "Rb", "Mrp", "act", "hb", "MALAT1"]
    assert all([g in a for g in gene_group])
    assert input_type in ["GEP", "count.matrix"]
    
    if input_type == "GEP":
        exp_cells = min(exp_cells, 1)
        print("As the input is a collpased GEP, exp.cells is set to min(exp.cells,1)")
    
    category_matrix = assign_category(input_genes = input.columns, species = species)
    category_matrix = category_matrix.loc[:, gene_group]

    print("number of genes filtered in each category: ")
    print(np.sum(category_matrix, axis = 0))

    exclude_idx = np.sum(category_matrix, axis=1) > 0
    print("A total of", exclude_idx.sum(), "genes from", gene_group, "have been excluded")
    input_filtered = input.loc[:, ~exclude_idx]
    
    if exp_cells > 0:
        exclude_lowexp_idx = (input_filtered > 0).sum(axis=0) >= exp_cells
        print("A total of", (~exclude_lowexp_idx).sum(), "gene expressed in fewer than", exp_cells, "cells have been excluded")
        input_filtered = input_filtered.loc[:, exclude_lowexp_idx]
    else:
        print("A total of 0 lowly expressed genes have been excluded")
    
    return input_filtered


def select_gene_type(input : pd.DataFrame, gene_type):
    assert all([g in ["protein_coding", "pseudogene", "lincRNA"] for g in gene_type])
    
    input_genes = input.columns
    gene_tab_path = os.path.join(os.path.split(__file__)[0], "txt", "gencode.v22.broad.category.txt")
    gene_list = pd.read_table(gene_tab_path, sep="\t", header=None)
    
    if sum(1 for gene in input_genes if gene[:3] == "ENS") > len(input_genes) * 0.8:
        print("EMSEMBLE IDs detected.")
        input_genes = [gene.split(".")[0] for gene in input_genes]
        gene_list_7 = gene_list.iloc[:,7].tolist()
        gene_match = [gene_list_7.index(x) for x in input_genes if x in gene_list_7]
        gene_df = gene_list.iloc[gene_match, [7, 8]]
    else:
        print("Gene symbols detected. Recommend to use EMSEMBLE IDs for more unique mapping.")
        gene_list_4 = gene_list.iloc[:,4].tolist()
        gene_match = [gene_list_4.index(x) for x in input_genes if x in gene_list_4]
        gene_df = gene_list.iloc[gene_match, [4, 8]]

    gene_df.columns = ["gene_name", "category"]

    selected_gene_idx = gene_df["category"].isin(gene_type)
    
    print("number of genes retained in each category: ")
    print(gene_df.loc[selected_gene_idx, "category"].value_counts())
    input_filtered = input.loc[:, selected_gene_idx.tolist()]
    return input_filtered
