import numpy as np


from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def optimal_kmeans(data, k_range):
    # Function to determine the optimal number of clusters for KMeans
    best_score = -1
    best_k = 2

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)

        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_k = k

    # Fitting KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    final_labels = kmeans.fit_predict(data)
    return final_labels

def compute_pathway(adata,adata_aggr,pathway,gene_num = 3,n_components=10):
    """
    Compute tensor similarities among pathways

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix
    adata_aggr: AnnData
        Aggregated data matrix
    db_name: str
        Name of the database
    gene_num: int
        Minimum number of genes in the pathway overlapped with STT multi-stable genes

    Returns
    -------
    None, but updates adata.uns with the following
    pathway_select: dict
        Selected pathways satisfying the gene_num condition
    pathway_embedding: np.ndarray
        UMAP embedding of the pathway similarities
    pathway_labels: np.ndarray
        Cluster labels of the pathway embedding
    
    """
    import scvelo as scv
    #pathway = gp.parser.download_library(name = db_name)
    tpm_dict = {}
    pathway_select = {}
    temp = []
    gene_select = [x in adata.uns['gene_subset'] for x in adata.var_names]
    #velo =  adata.obsm['tensor_v_aver'].copy()
    #adata_aggr.layers['vj'] = np.concatenate((velo[:,gene_select,0],velo[:,gene_select,1]),axis = 1)
    cor_matrix = np.zeros((len(pathway.keys()), len(pathway.keys())))

    idx = 0
    for key in pathway.keys():
        gene_list = [x for x in pathway[key]] 
        gene_select = [x for x in gene_list if x in adata_aggr.var_names]
        gene_pathway = gene_select+[x+'_u' for x in gene_select]
        if len(gene_select)>=gene_num and gene_select not in temp:
                adata_aggr_select = adata_aggr[:,gene_pathway]
                scv.tl.velocity_graph(adata_aggr_select, vkey = 'vj', xkey = 'Ms', n_jobs = -1)
                current_array = adata_aggr_select.uns['vj_graph'].toarray().reshape(-1)
                for prev_idx in range(idx):
                    prev_key = list(pathway_select.keys())[prev_idx]
                    prev_array = adata_aggr.uns['vj_graph_'+prev_key].toarray().reshape(-1)
                    cor = np.corrcoef(current_array, prev_array)[0, 1]
                    cor_matrix[idx][prev_idx] = cor
                    cor_matrix[prev_idx][idx] = cor
                adata_aggr.uns['vj_graph_'+key] = adata_aggr_select.uns['vj_graph']
                #tpm_dict[key] = adata_aggr_select.uns['vj_graph'].toarray().reshape(-1)
                pathway_select[key] = gene_select
                idx = idx+1
                temp.append(gene_select)
    
    adata.uns['pathway_select'] = pathway_select
    cor_matrix = cor_matrix[:idx,:idx]

    # compute correlation
    #arr = np.stack(list(tpm_dict.values()))
    #cor = np.corrcoef(arr)
    # dimensionality reduction
    
    pca = PCA(n_components=n_components)
    pca_embedding = pca.fit_transform(cor_matrix)
    # Perform UMAP on the PCA embedding
    import umap
    umap_reducer = umap.UMAP(random_state=42)
    umap_embedding = umap_reducer.fit_transform(pca_embedding)
    # Perform hierarchical clustering
    adata.uns['pathway_embedding'] = umap_embedding
    c_labels = optimal_kmeans(umap_embedding,range(3,n_components))
    adata.uns['pathway_labels'] = c_labels

    