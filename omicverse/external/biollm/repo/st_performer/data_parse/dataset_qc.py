import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_samples


def normalize(adata):
    adata.raw = adata
    if np.min(adata.X.toarray()) < 0:
        adata.uns["qc"] = {"expr_type": "scale"}
    elif np.max(adata.X.toarray()) < 20:
        adata.uns["qc"] = {"expr_type": "log_norm"}
    elif np.max(adata.X[0:10, :].toarray() - np.int32(adata.X[0:10, :].toarray())) == np.int32(0):
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata.uns["qc"] = {"expr_type": "raw"}
    else:
        sc.pp.log1p(adata)
        adata.uns["qc"] = {"expr_type": "norm"}


def calculate_sparsity(data):
    total_elements = np.prod(data.shape)
    non_zero_elements = data.nnz
    sparsity = 1 - (non_zero_elements / total_elements)
    sparsity_dict = {"sparcity": sparsity}
    return sparsity_dict


def expression_matrix_qc(adata):
    data = adata.X

    qc_dict = calculate_sparsity(data)
    cell_count = data.shape[0]
    gene_count = data.shape[1]
    gene_avg_cell = np.mean(np.sum(data > 0, axis=1) / cell_count)
    cell_avg_gene = np.mean(np.sum(data > 0, axis=0))

    expr_dict = {"cell_count": cell_count,
                 "gene_count": gene_count,
                 "gene_avg_cell": gene_avg_cell,
                 "cell_avg_gene": cell_avg_gene}

    qc_dict.update(expr_dict)
    adata.uns["qc"].update({"expr_qc": qc_dict})


def celltype_scratch(adata):
    if 'annotation_au' in adata.obsm:
        annotation_au_df = adata.obsm['annotation_au']
        if "celltype" in annotation_au_df.columns:
            adata.obs['celltype'] = annotation_au_df['celltype']


CL = pd.read_csv("../data/CL.csv")
CL = CL.drop_duplicates(["celltype"])
del CL["tissue"]


def celltype_curated(adata):
    cell_meta = adata.obs
    cell_meta = pd.merge(cell_meta, CL, on="celltype", how="left")

    for column in ["ontology_name", "ontology_id", "is_immune_cell"]:
        if cell_meta[column].dtype.name == 'category':
            cell_meta[column] = cell_meta[column].cat.add_categories("notAvailable")
        else:
            cell_meta[column] = cell_meta[column].astype('category').cat.add_categories("notAvailable")
        cell_meta[column] = cell_meta[column].fillna("notAvailable")
        cell_meta[column] = cell_meta[column].astype('str')

    cell_meta.index = adata.obs_names

    adata.obs = cell_meta


def celltype_check(adata):
    cell_meta = adata.obs
    cell_meta = cell_meta.loc[~cell_meta["celltype"].isin(["notAvailable"]), :]
    celltype_num = adata.obs["celltype"].unique().shape[0]
    ontology_num = adata.obs["ontology_name"].unique().shape[0]
    ontology_pct = 1 - adata.obs['ontology_name'].isin(["notAvailable"]).sum() / adata.shape[0]

    adata.uns["qc"].update(
        {"annotation_qc": {
            "celltype_num": celltype_num,
            "ontology_num": ontology_num,
            "ontology_pct": ontology_pct}})

    sc.pp.highly_variable_genes(adata)
    sc.pp.scale(adata)
    sc.pp.pca(adata)

    if celltype_num > 1:
        cluster_labels = adata.obs["celltype"]
        silhouette_celltype = silhouette_samples(adata.obsm["X_pca"], cluster_labels)
        adata.obs["silhouette_celltype"] = silhouette_celltype
        adata.uns["qc"]["annotation_qc"].update({
            "ASW_celltype": adata.obs["silhouette_celltype"].groupby(adata.obs["celltype"]).mean().to_dict()})
        adata.uns["qc"]["annotation_qc"]["ASW_celltype"].update(
            {"celltype_avg": adata.obs["silhouette_celltype"].mean().tolist()})

    if ontology_num > 1:
        cluster_labels = adata.obs["ontology_name"]
        silhouette_ontology = silhouette_samples(adata.obsm["X_pca"], cluster_labels)
        adata.obs["silhouette_ontology"] = silhouette_ontology
        adata.uns["qc"]["annotation_qc"]["ASW_ontology"] = adata.obs["silhouette_ontology"].groupby(
            adata.obs["ontology_name"]).mean().to_dict()
        adata.uns["qc"]["annotation_qc"]["ASW_ontology"].update(
            {"ontology_avg": adata.obs["silhouette_ontology"].mean().tolist()})


def dataset_qc(adata):
    normalize(adata)
    celltype_scratch(adata)
    celltype_curated(adata)
    celltype_check(adata)
