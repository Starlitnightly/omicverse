import pandas as pd
import numpy as np
from anndata import AnnData
from .helper import intersect, find_marker, extract_exp, process_anndata
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics.cluster import adjusted_rand_score
import scanpy

def pearson_cal(spatrio_map: pd.DataFrame,
    ann1: AnnData,
    ann2: AnnData,
    use_marker: bool = True,
    top_n: int = 100,
    run_marker_by: str = "type",
    scale = True,
    spot_type = True
    ):
    adata1 = ann1.copy()
    adata2 = ann2.copy()
    common_genes = intersect(adata1.var.index, adata2.var.index)
    adata1 = adata1[:, common_genes]
    adata2 = adata2[:, common_genes]
    
    # adata_merge = adata1.concatenate(adata2,index_unique=None)
    # adata_merge = process_anndata(adata_merge,scale=scale,pca=False)
    
    map = pd.DataFrame(spatrio_map[['spot','cell']])
    #n1 = adata1.shape[0]
    #data1 = extract_exp(adata1).iloc[:n1]
    #data2 = extract_exp(adata2).iloc[n1:]
    data2 = extract_exp(adata2)
    data2['cell'] = data2.index
    data2 = pd.merge(map, data2, on='cell',how="left")
    grouped = data2.groupby(['spot'])
    mapped_data = grouped[data2.columns.tolist()[2:]].sum()
    tmp = scanpy.AnnData(X = mapped_data)
    
    adata_merge = adata1.concatenate(tmp,index_unique=None)
    adata_merge = process_anndata(adata_merge,scale=scale,pca=False)
    n1 = adata1.shape[0]
    data1 = extract_exp(adata_merge).iloc[:n1]
    mapped_data = extract_exp(adata_merge).iloc[n1:]

    common_genes = intersect(data1.columns, mapped_data.columns)
    data1 = data1[common_genes]
    mapped_data = mapped_data[common_genes]
    

    if use_marker:
        #sc.pp.filter_cells(adata2_copy, min_genes=200)
        #sc.pp.filter_genes(adata2_copy, min_cells=3)
        marker1 = find_marker(ann1,top_n,maker_by=run_marker_by)
        marker2 = find_marker(ann2,top_n,maker_by=run_marker_by)
        marker1.extend(marker2)
        gene_list = np.unique(marker1).tolist()
        adata1 = adata1[:, gene_list]
        adata2 = adata2[:, gene_list]
        data1 = data1[gene_list]
        mapped_data = mapped_data[gene_list]

    common_index = np.intersect1d(data1.index, mapped_data.index, assume_unique=False, return_indices=False)
    data1 = data1.loc[common_index]
    mapped_data = mapped_data.loc[common_index]
    
    if spot_type:
        map2 = pd.DataFrame(spatrio_map[['spot','spot_type']])
        spot_type_map = dict(zip(map2['spot'], map2['spot_type']))
        data1.index = data1.index.map(spot_type_map)
        mapped_data.index = mapped_data.index.map(spot_type_map)
        data1 = data1.groupby(data1.index).agg('sum')
        mapped_data = mapped_data.groupby(mapped_data.index).agg('sum')

    pearson_s = []
    for spot in data1.index.tolist():
        #x=data1.loc[spot].values
        #y=mapped_data.loc[spot].values
        from sklearn.preprocessing import scale
        x=data1.loc[spot].values
        y=mapped_data.loc[spot].values
        pc = pearsonr(x,y)
        pearson_s.append(pc[0])
    pearson = np.mean(pearson_s)
    return pearson

def calculate_cell_counts(cell_ratios: pd.DataFrame, cell_num_dict: dict) -> pd.DataFrame:
    cell_counts = cell_ratios.copy()
    for spot, cell_num in cell_num_dict.items():
        cell_counts.loc[spot] *= cell_num
    return cell_counts

def accuracy_cal(cell_counts: pd.DataFrame, ref_counts: pd.DataFrame):
    # common_spots = cell_ratios.index.intersection(ref_ratios.index)
    # cell_ratios = cell_ratios.loc[common_spots]
    # ref_ratios = ref_ratios.loc[common_spots]
    
    # if top_num is not None:
    #     cell_num = {spot: top_num for spot in common_spots}
    # if expected_num is not None:
    #     expected_num = expected_num.loc[common_spots]
    #     cell_num = expected_num['cell_num'].to_dict()
    
    # columns_order = ref_ratios.columns
    # cell_ratios = cell_ratios.reindex(columns=columns_order).fillna(0)
    # cell_ratios = cell_ratios.reindex(ref_ratios.index).fillna(0)
    # cell_counts = calculate_cell_counts(cell_ratios, cell_num).round().astype(int)
    # ref_counts = calculate_cell_counts(ref_ratios, cell_num).round().astype(int)
    common_spots = cell_counts.index.intersection(ref_counts.index)
    cell_counts = cell_counts.loc[common_spots]
    ref_counts = ref_counts.loc[common_spots]
    
    columns_order = ref_counts.columns
    cell_counts = cell_counts.reindex(columns=columns_order).fillna(0)
    cell_counts = cell_counts.reindex(ref_counts.index).fillna(0)
    
    # accuracy = []
    # for i in cell_counts.index:
    #     A = cell_counts.loc[i]
    #     B = ref_counts.loc[i]
    #     tmp = pd.DataFrame({'c1' : A,
    #                         'c2' : B})
    #     common_cells = tmp.min(axis=1).sum()
    #     total_cells = tmp.sum(axis=0).max()
    #     matching_ratio = common_cells / total_cells
    #     accuracy.append(matching_ratio)
    # accuracy = np.mean(accuracy)
    
    common_cells_sum = 0
    total_cells_sum = 0
    for i in cell_counts.index:
        A = cell_counts.loc[i]
        B = ref_counts.loc[i]
        tmp = pd.DataFrame({'c1': A, 'c2': B})
        common_cells_sum += tmp.min(axis=1).sum()
        total_cells_sum += tmp['c2'].sum()
    accuracy = common_cells_sum / total_cells_sum
    return accuracy

def spearman_cal(spatrio_map: pd.DataFrame,
    ann1: AnnData,
    ann2: AnnData,
    use_marker: bool = True,
    top_n: int = 100,
    run_marker_by: str = "type",
    scale = True,
    spot_type = True
    ):
    adata1 = ann1.copy()
    adata2 = ann2.copy()
    common_genes = intersect(adata1.var.index, adata2.var.index)
    adata1 = adata1[:, common_genes]
    adata2 = adata2[:, common_genes]
    
    # adata_merge = adata1.concatenate(adata2,index_unique=None)
    # adata_merge = process_anndata(adata_merge,scale=scale,pca=False)
    
    map = pd.DataFrame(spatrio_map[['spot','cell']])
    #n1 = adata1.shape[0]
    #data1 = extract_exp(adata1).iloc[:n1]
    #data2 = extract_exp(adata2).iloc[n1:]
    data2 = extract_exp(adata2)
    data2['cell'] = data2.index
    data2 = pd.merge(map, data2, on='cell',how="left")
    grouped = data2.groupby(['spot'])
    mapped_data = grouped[data2.columns.tolist()[2:]].sum()
    tmp = scanpy.AnnData(X = mapped_data)
    
    adata_merge = adata1.concatenate(tmp,index_unique=None)
    adata_merge = process_anndata(adata_merge,scale=scale,pca=False)
    n1 = adata1.shape[0]
    data1 = extract_exp(adata_merge).iloc[:n1]
    mapped_data = extract_exp(adata_merge).iloc[n1:]

    common_genes = intersect(data1.columns, mapped_data.columns)
    data1 = data1[common_genes]
    mapped_data = mapped_data[common_genes]
    

    if use_marker:
        #sc.pp.filter_cells(adata2_copy, min_genes=200)
        #sc.pp.filter_genes(adata2_copy, min_cells=3)
        marker1 = find_marker(ann1,top_n,maker_by=run_marker_by)
        marker2 = find_marker(ann2,top_n,maker_by=run_marker_by)
        marker1.extend(marker2)
        gene_list = np.unique(marker1).tolist()
        adata1 = adata1[:, gene_list]
        adata2 = adata2[:, gene_list]
        data1 = data1[gene_list]
        mapped_data = mapped_data[gene_list]

    common_index = np.intersect1d(data1.index, mapped_data.index, assume_unique=False, return_indices=False)
    data1 = data1.loc[common_index]
    mapped_data = mapped_data.loc[common_index]
    
    if spot_type:
        map2 = pd.DataFrame(spatrio_map[['spot','spot_type']])
        spot_type_map = dict(zip(map2['spot'], map2['spot_type']))
        data1.index = data1.index.map(spot_type_map)
        mapped_data.index = mapped_data.index.map(spot_type_map)
        data1 = data1.groupby(data1.index).agg('sum')
        mapped_data = mapped_data.groupby(mapped_data.index).agg('sum')

    spearman_s = []
    for spot in data1.index.tolist():
        x=data1.loc[spot].values
        y=mapped_data.loc[spot].values
        sc, _ = spearmanr(x,y)
        spearman_s.append(sc)
    spearman = np.mean(spearman_s)
    return spearman

def ari_cal(map_data: pd.DataFrame) :
    accuracy_data = map_data.copy()
    ari = adjusted_rand_score(accuracy_data.spot_type, accuracy_data.cell_type)
    return ari

def rmse_cal(cell_ratios, ref_ratios):
    mse = ((cell_ratios - ref_ratios) ** 2).mean(axis=1)
    rmse = np.sqrt(mse.mean())
    return rmse
