import numpy as np
import scanpy as sc
import squidpy as sq
import pandas as pd    


def get_gaston_input_adata(data_folder, get_rgb=False, spot_umi_threshold=50):
    adata=sq.read.visium(data_folder)
    adata.obsm['spatial']=adata.obsm['spatial'].astype(float)
    sc.pp.filter_cells(adata, min_counts=spot_umi_threshold)

    gene_labels=adata.var.index.to_numpy()
    counts_mat=adata.X
    coords_mat=np.array(adata.obsm['spatial'])

    if not get_rgb:
        return counts_mat, coords_mat, gene_labels

    library_id = list(adata.uns['spatial'].keys())[0] # adata2.uns['spatial'] should have only one key
    scale=adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
    img = sq.im.ImageContainer(adata.uns['spatial'][library_id]['images']['hires'],
                               scale=scale,
                               layer="img1")
    print('calculating RGB')
    sq.im.calculate_image_features(adata, img, features="summary", key_added="features")
    columns = ['summary_ch-0_mean', 'summary_ch-1_mean', 'summary_ch-2_mean']
    RGB_mean = adata.obsm["features"][columns]
    RGB_mean = RGB_mean / 255
    
    # RGB_mean = RGB_mean[RGB_mean.index.isin(df_pos['barcode'])]

    return counts_mat, coords_mat, gene_labels, RGB_mean.to_numpy()

def get_top_pearson_residuals(num_pcs, counts_mat, coords_mat, gene_labels=None, n_top_genes=5000, clip=0.01):
    df=pd.DataFrame(counts_mat, columns=gene_labels)
    adata=sc.AnnData(df)
    adata.obsm["coords"] = coords_mat
    
    sc.experimental.pp.highly_variable_genes(
            adata, flavor="pearson_residuals", n_top_genes=n_top_genes
    )
    
    adata = adata[:, adata.var["highly_variable"]]
    adata.layers["raw"] = adata.X.copy()
    adata.layers["sqrt_norm"] = np.sqrt(
        sc.pp.normalize_total(adata, inplace=False)["X"]
    )
    
    theta=np.Inf
    sc.experimental.pp.normalize_pearson_residuals(adata, clip=clip, theta=theta)
    sc.pp.pca(adata, n_comps=num_pcs)
    return adata.obsm['X_pca']

def get_gaston_input_xenium(folder,filter_zero_cells=False):
    adata = sc.read_10x_h5(filename=folder+'cell_feature_matrix.h5')

    df = pd.read_csv(folder+"cells.csv")
    df.set_index(adata.obs_names, inplace=True)
    adata.obs = df.copy()

    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].copy().to_numpy()

    counts_mat=np.array(adata.X.todense())
    coords_mat=adata.obs[["x_centroid", "y_centroid"]].to_numpy()

    cts=pd.read_csv(folder+'Cell_Barcode_Type_Matrices.csv')
    cts=np.array(cts['Cluster'])

    gene_labels=np.array(adata.var.index)

    if filter_zero_cells:
        inds_to_keep=np.where(np.sum(counts_mat,1)>0)[0]
        counts_mat=counts_mat[inds_to_keep,:]
        coords_mat=coords_mat[inds_to_keep,:]
        cts=cts[inds_to_keep]

    return counts_mat, coords_mat, gene_labels, cts