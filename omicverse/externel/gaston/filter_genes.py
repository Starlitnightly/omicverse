import numpy as np

def filter_genes(counts_mat, gene_labels, umi_threshold=500, exclude_prefix=['MT-', 'RPL', 'RPS']):
    idx_kept=np.where(np.sum(counts_mat,0) > umi_threshold)[0]
    idx_kept=np.array( [i for i in idx_kept if gene_labels[i][:3] not in exclude_prefix] )
    gene_labels_idx=gene_labels[idx_kept]
    return idx_kept, gene_labels_idx