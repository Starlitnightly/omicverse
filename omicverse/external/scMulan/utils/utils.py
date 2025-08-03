import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import scanpy as sc

def cell_type_smoothing(adata,use_rep='X_pca', N_neighbors=15,n_pcs=10,
                         weighted_smoothing=True, threshold=0.5,
                         col_name = 'cell_type_from_scMulan'):
    # 重新算一个nn，避免取的N_neighbors比画umap的时候选的参数大
    sc.pp.neighbors(adata,use_rep=use_rep, n_pcs=n_pcs, 
                    n_neighbors=N_neighbors, key_added="Smoothing")
    
    # smoothing_score是近邻平滑之后给的分数
    # cell_type_from_mulan_smoothing是平滑后的结果

    if type(threshold) == int:
        threshold = threshold/(N_neighbors-1)

    smoothing_score_list = []
    smoothing_celltype_list = []

    for idx in tqdm(range(adata.shape[0])):
        distances = np.array(adata.obsp['Smoothing_distances'][idx,:].todense()).reshape(-1)
        pre_labels = adata.obs[col_name][distances>0]

        if weighted_smoothing:
            weights = (1/distances[distances>0])/sum(1/distances[distances>0])
            max_score = 0
            result_celltype = ""
            for ct in pre_labels.unique():
                if weights[pre_labels==ct].sum()>max_score:
                    max_score = weights[pre_labels==ct].sum()
                    result_celltype = ct
            smoothing_celltype_list.append(result_celltype)
            smoothing_score_list.append(max_score)

        else:
            result_counter = pre_labels.value_counts()[pre_labels.value_counts()>0]
            smoothing_celltype_list.append(result_counter.index[result_counter.argmax()])
            smoothing_score_list.append(result_counter.max()/(N_neighbors-1))

    adata.obs['cell_type_from_mulan_smoothing'] = smoothing_celltype_list
    adata.obs['smoothing_score'] = smoothing_score_list
    adata.obs.cell_type_from_mulan_smoothing[adata.obs.smoothing_score<threshold] = "Unclassified"
    adata.obs[col_name]=adata.obs[col_name].astype('category')
    if not (col_name + '_colors' in adata.uns):
        if len(adata.obs[col_name].cat.categories)<102:
            adata.uns[col_name + '_colors'] = sc.pl.palettes.default_102[:len(adata.obs[col_name].cat.categories)]
        else:
            adata.uns[col_name + '_colors'] = ['grey' for _ in range(len(adata.obs[col_name].cat.categories))]
        #sc.pl.umap(adata,color=[col_name],show=False)
        #plt.close()
    
    adata.uns['cell_type_from_mulan_smoothing_colors'] = ["#666666" for _ in adata.obs['cell_type_from_mulan_smoothing'].unique()]
    adata.obs[col_name] = pd.Categorical(adata.obs[col_name])
    adata.obs.cell_type_from_mulan_smoothing = pd.Categorical(adata.obs.cell_type_from_mulan_smoothing)
    for ct in adata.obs['cell_type_from_mulan_smoothing'].unique():
        if ct != 'Unclassified':
            ct_index = np.where(adata.obs[col_name].cat.categories==ct)[0][0]
            r_index = np.where(adata.obs["cell_type_from_mulan_smoothing"].cat.categories==ct)[0][0]
            adata.uns['cell_type_from_mulan_smoothing_colors'][r_index] = adata.uns[col_name + '_colors'][ct_index]


def visualize_selected_cell_types(adata, selected_cell_types, smoothing=False, col_name = 'cell_type_from_scMulan', **kwargs):
    if smoothing:
        selected_column = "cell_type_from_mulan_smoothing"
        if not (selected_column in adata.obs.columns):
            "You set 'smoothing=True', please run cell_type_smoothing before visualization."
            return -1
    else:
        selected_column = col_name

    adata.obs["selected_celltype"] = adata.obs[selected_column].astype(str)
    adata.obs.loc[[not (x in selected_cell_types) for x in adata.obs[selected_column]],"selected_celltype"] = "others"
    adata.obs.selected_celltype = pd.Categorical(adata.obs["selected_celltype"])
    adata.uns['selected_celltype_colors'] = ["#666666" for _ in adata.obs['selected_celltype'].unique()]
    # adata.obs.cell_type = pd.Categorical(adata.obs[selected_column])
    
    for ct in adata.obs['selected_celltype'].unique():
        if ct != 'others':
            ct_index = np.where(adata.obs[selected_column].cat.categories==ct)[0][0]
            r_index = np.where(adata.obs["selected_celltype"].cat.categories==ct)[0][0]
            adata.uns['selected_celltype_colors'][r_index] = adata.uns['{}_colors'.format(selected_column)][ct_index]
    print(adata.obs.selected_celltype.value_counts())
    sc.pl.umap(adata,color=["selected_celltype"], **kwargs)