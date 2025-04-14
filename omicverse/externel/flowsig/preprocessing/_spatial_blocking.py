import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans

def construct_spatial_blocks(adata: sc.AnnData,
                             n_blocks: int,
                             use_graph: bool = False,
                             graph_adjacency: str = 'spatial_connectivities',
                             resolution: float = None,
                             spatial_block_key: str = "spatial_block",
                             spatial_key: str = "spatial"):
    
    # If we want to construct spatial clusters from the graph directly, we use leiden clustering
    if use_graph: 

        if resolution is None:
            ValueError('Need to specify a clustering resolution to construct blocks from spatial graph.')

        else:
            sc.tl.leiden(adata, resolution=resolution, key_added=spatial_block_key, adjacency=graph_adjacency)

    else:         # Run k-means clustering on the spatial coordinates (produces more "even" blocks)

        kmeans = KMeans(n_clusters=n_blocks).fit(adata.obsm[spatial_key])
        adata.obs[spatial_block_key] = pd.Series(kmeans.labels_, dtype='category').values