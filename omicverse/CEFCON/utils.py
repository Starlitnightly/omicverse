import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
import networkx as nx
import scanpy as sc
from typing import Optional, Union
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.pyplot import gcf
import matplotlib.ticker as ticker
import seaborn as sns

from .resources import TFs_human, TFs_mouse


def data_preparation(input_expData: Union[str, sc.AnnData, pd.DataFrame],
                     input_priorNet: Union[str, pd.DataFrame],
                     genes_DE: Optional[Union[str, pd.DataFrame, pd.Series]] = None,
                     additional_edges_pct: float = 0.01):
    print('[0] - Data loading and preprocessing...')

    ## [1] Single-cell RNA-seq data
    lineages = None
    if isinstance(input_expData, str):
        p = Path(input_expData)
        if p.suffix == '.csv':
            adata = sc.read_csv(input_expData, first_column_names=True)
        else:  # h5ad
            adata = sc.read_h5ad(input_expData)
    elif isinstance(input_expData, sc.AnnData):
        adata = input_expData
        lineages = adata.uns.get('lineages')
    elif isinstance(input_expData, pd.DataFrame):
        adata = sc.AnnData(X=input_expData)
    else:
        raise Exception("Invalid input! The input format must be '.csv' file or '.h5ad' "
                        "formatted file, or an 'AnnData' object!", input_expData)

    possible_species = 'mouse' if bool(re.search('[a-z]', adata.var_names[0])) else 'human'

    # Gene symbols are uniformly handled in uppercase
    adata.var_names = adata.var_names.str.upper()

    ## [2] Prior network data
    if isinstance(input_priorNet, str):
        netData = pd.read_csv(input_priorNet, index_col=None, header=0)
    elif isinstance(input_priorNet, pd.DataFrame):
        netData = input_priorNet.copy()
    else:
        raise Exception("Invalid input!", input_priorNet)
    # make sure the genes of prior network are in the input scRNA-seq data
    netData['from'] = netData['from'].str.upper()
    netData['to'] = netData['to'].str.upper()
    netData = netData.loc[netData['from'].isin(adata.var_names.values)
                          & netData['to'].isin(adata.var_names.values), :]
    netData = netData.drop_duplicates(subset=['from', 'to'], keep='first', inplace=False)

    # Transfer into networkx object
    priori_network = nx.from_pandas_edgelist(netData, source='from', target='to', create_using=nx.DiGraph)
    priori_network_nodes = np.array(priori_network.nodes())

    # in_degree, out_degree (centrality)
    in_degree = pd.DataFrame.from_dict(nx.in_degree_centrality(priori_network),
                                       orient='index', columns=['in_degree'])
    out_degree = pd.DataFrame.from_dict(nx.out_degree_centrality(priori_network),
                                        orient='index', columns=['out_degree'])
    centrality = pd.concat([in_degree, out_degree], axis=1)
    centrality = centrality.loc[priori_network_nodes, :]

    ## [3] A mapper for node index and gene name
    idx_GeneName_map = pd.DataFrame({'idx': range(len(priori_network_nodes)),
                                     'geneName': priori_network_nodes},
                                    index=priori_network_nodes)

    edgelist = pd.DataFrame({'from': idx_GeneName_map.loc[netData['from'].tolist(), 'idx'].tolist(),
                             'to': idx_GeneName_map.loc[netData['to'].tolist(), 'idx'].tolist()})

    ## [4] add TF information
    is_TF = np.ones(len(priori_network_nodes), dtype=int)
    if possible_species == 'human':
        TFs_df = TFs_human
    else:
        TFs_df = TFs_mouse
    TF_list = TFs_df.iloc[:, 0].str.upper()
    is_TF[~np.isin(priori_network_nodes, TF_list)] = 0

    # Only keep the genes that exist in both single cell data and the prior gene interaction network
    adata = adata[:, priori_network_nodes]
    if lineages is None:
        cells_in_lineage_dict = {'all': adata.obs_names}  # all cells are regarded as in on lineage if lineages is None
    else:
        cells_in_lineage_dict = {l: adata.obs_names[adata.obs[l].notna()] for l in lineages}

    print(f"Consider the input data with {len(cells_in_lineage_dict)} lineages:")
    adata_lineages = dict()
    for l, c in cells_in_lineage_dict.items():
        print(f"  Lineage - {l}:")

        adata_l = sc.AnnData(X=adata[c, :].to_df())
        adata_l.var['is_TF'] = is_TF
        adata_l.varm['centrality_prior_net'] = centrality
        adata_l.varm['idx_GeneName_map'] = idx_GeneName_map
        adata_l.uns['name'] = l

        ## [5] Additional edges with high spearman correlation
        if isinstance(adata_l.X, sparse.csr_matrix):
            gene_exp = pd.DataFrame(adata_l.X.A.T, index=priori_network_nodes)
        else:
            gene_exp = pd.DataFrame(adata_l.X.T, index=priori_network_nodes)

        ori_edgeNum = len(edgelist)
        edges_corr = np.absolute(np.array(gene_exp.T.corr('spearman')))
        np.fill_diagonal(edges_corr, 0.0)
        x, y = np.where(edges_corr > 0.6)
        addi_top_edges = pd.DataFrame({'from': x, 'to': y, 'weight': edges_corr[x, y]})
        addi_top_k = int(gene_exp.shape[0] * (gene_exp.shape[0] - 1) * additional_edges_pct)
        if len(addi_top_edges) > addi_top_k:
            addi_top_edges = addi_top_edges.sort_values(by=['weight'], ascending=False)
            addi_top_edges = addi_top_edges.iloc[0:addi_top_k, 0:2]
        edgelist = pd.concat([edgelist, addi_top_edges.iloc[:, 0:2]], ignore_index=True)
        edgelist = edgelist.drop_duplicates(subset=['from', 'to'], keep='first', inplace=False)
        print('    {} extra edges (Spearman correlation > 0.6) are added into the prior gene interaction network.\n'
              '    Total number of edges: {}.'.format((len(edgelist) - ori_edgeNum), len(edgelist)))

        adata_l.uns['edgelist'] = edgelist

        ## [6] Differential expression scores
        logFC = adata.var.get(l + '_logFC')
        if (genes_DE is None) and (logFC is None):
            pass
        else:
            if genes_DE is not None:
                if isinstance(genes_DE, str):
                    genes_DE = pd.read_csv(genes_DE, index_col=0, header=0)
            else:  # logFC is not None
                genes_DE = logFC

            genes_DE = pd.DataFrame(genes_DE).iloc[:, 0]
            genes_DE.index = genes_DE.index.str.upper()
            genes_DE = genes_DE[genes_DE.index.isin(priori_network_nodes)].abs().dropna()
            node_score_auxiliary = pd.Series(np.zeros(len(priori_network_nodes)), index=priori_network_nodes)
            node_score_auxiliary[genes_DE.index] = genes_DE.values
            node_score_auxiliary = np.array(node_score_auxiliary)
            adata_l.var['node_score_auxiliary'] = node_score_auxiliary
            genes_DE = None

        adata_lineages[l] = adata_l
        print(f"    n_genes × n_cells = {adata_l.n_vars} × {adata_l.n_obs}")

    return adata_lineages


def cluster_cell_by_RGM(auc_mtx, true_cell_label, method='ward', k=None):
    """
    Cluster cells based on RGM activity matrix by using hierarchical clustering.
    """
    assert method in {'ward', 'complete', 'average', 'single'}
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

    auc_mtx_Z = pd.DataFrame(index=auc_mtx.index, columns=list(auc_mtx.columns))
    for row in list(auc_mtx.index):
        auc_mtx_Z.loc[row, :] = (auc_mtx.loc[row, :] - auc_mtx.loc[row, :].mean()) / auc_mtx.loc[row, :].std(ddof=0)

    if k is not None:
        ac = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage=method).fit(auc_mtx_Z)
        predicted_cell_label = ac.labels_
        NMIs = normalized_mutual_info_score(true_cell_label, predicted_cell_label)
        ARIs = adjusted_rand_score(true_cell_label, predicted_cell_label)
        Silhouettes = silhouette_score(auc_mtx, predicted_cell_label, metric='euclidean')
        N_clus = k
    else:
        NMIs, ARIs, N_clus, Silhouettes = [], [], [], []
        max_cluster_num = len(set(true_cell_label)) * 2
        for i in range(2, max_cluster_num):
            ac = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage=method).fit(auc_mtx_Z)
            out = ac.labels_
            NMIs.append(normalized_mutual_info_score(true_cell_label, out))
            ARIs.append(adjusted_rand_score(true_cell_label, out))
            N_clus.append(i)
            Silhouettes.append(silhouette_score(auc_mtx, out, metric='euclidean'))

    return {'NMIs': NMIs, 'ARIs': ARIs, 'Silhouettes': Silhouettes, 'num_clusters': N_clus}


def network_topological_properties(prior_network: pd.DataFrame):
    import networkx as nx
    from sklearn.linear_model import LinearRegression as lr

    prior_nx = nx.from_pandas_edgelist(prior_network, source='from', target='to', edge_attr=None,
                                       create_using=nx.DiGraph)

    # Number of genes
    N_genes = prior_nx.number_of_nodes()

    # Number of edges
    N_edges = len(prior_network[['from', 'to']].drop_duplicates())

    # Number of source genes
    N_source_genes = len(prior_network['from'].unique())

    # Number of target genes
    N_target_genes = len(prior_network['to'].unique())

    # Density
    Density = nx.density(prior_nx)

    # Average degree / in_degree / out_degree
    degree = sum(dict(prior_nx.degree()).values()) / N_genes

    # Clustering coefficient (time-consuming)
    clustering_coefficient = nx.average_clustering(prior_nx)

    # Slope of degree distribution
    degree_sequence = pd.DataFrame(np.array(prior_nx.degree))
    degree_sequence.columns = ["ind", "degree"]
    degree_sequence = degree_sequence.set_index("ind")
    dist = degree_sequence.degree.value_counts() / degree_sequence.degree.value_counts().sum()
    dist.index = dist.index.astype(int)

    x = np.log(dist.index.values).reshape([-1, 1])
    y = np.log(dist.values).reshape([-1, 1])

    model = lr()
    model.fit(x, y)

    slope = model.coef_[0][0]

    print(  # f"Dataset-species: {dataset}-{species}\n"
        f"Number of genes: {N_genes}\nNumber of edges: {N_edges}\n"
        f"Number of source genes: {N_source_genes}\nNumber of target genes: {N_target_genes}\n"
        f"Network density: {Density}\nAverage degree: {degree:.4f}\n"
        f"Average clustering coefficient: {clustering_coefficient:.4f}\n"
        f"Slope of degree distribution: {slope:.4f}")


def prepare_data_for_R(adata: sc.AnnData,
                       temp_R_dir: str,
                       reducedDim: Optional[str] = None,
                       cluster_label: Optional[str] = None):
    """
    Process the AnnData object and save the necessary data to files.
    These data files are prepared for running the `slingshot_MAST_script.R` or `MAST_script.R` scripts.
    """
    if 'log_transformed' not in adata.layers:
        raise ValueError(
            f'Did not find `log_transformed` in adata.layers.'
        )

    if isinstance(adata.layers['log_transformed'], sparse.csr_matrix):
        exp_normalized = adata.layers['log_transformed'].A
    else:
        exp_normalized = adata.layers['log_transformed']

    # The normalized and log transformed data is used for MAST
    normalized_counts = pd.DataFrame(exp_normalized,
                                     index=adata.obs_names,
                                     columns=adata.var_names)
    normalized_counts.to_csv(temp_R_dir + '/exp_normalized.csv', sep=',')

    # The reduced dimension data is used for Slingshot
    if reducedDim is not None:
        reducedDim_data = pd.DataFrame(adata.obsm[reducedDim], dtype='float32', index=None)
        reducedDim_data.to_csv(temp_R_dir + '/data_reducedDim.csv', index=None)
    else:
        if 'lineages' not in adata.uns:
            raise ValueError(
                f'Did not find `lineages` in adata.uns.'
            )
        else:
            pseudotime_all = pd.DataFrame(index=adata.obs_names)
            for li in adata.uns['lineages']:
                pseudotime_all[li] = adata.obs[li]
            pseudotime_all.to_csv(temp_R_dir + '/pseudotime_lineages.csv', index=True)

    # Cluster Labels (Leiden)
    if cluster_label is not None:
        cluster_labels = pd.DataFrame(adata.obs[cluster_label])
        cluster_labels.to_csv(temp_R_dir + '/clusters.csv')


def process_Slingshot_MAST_R(temp_R_dir: str,
                             split_num: int = 4,
                             start_cluster: int = 0,
                             end_cluster: Optional[list] = None):
    """
    Run the `slingshot_MAST_script.R` to get pseudotim and differential expression information for each lineage.
    """
    import subprocess
    import importlib.resources as res

    R_script_path = 'slingshot_MAST_script.R'
    with res.path('cefcon', R_script_path) as datafile:
        R_script_path = datafile

    path = Path(temp_R_dir)
    path.mkdir(exist_ok=path.exists(), parents=True)

    args = f'Rscript {R_script_path} {temp_R_dir} {split_num} {start_cluster}'
    if end_cluster is not None:
        args += f' {end_cluster}'
    print('Running Slingshot and MAST using: \'{}\'\n'.format(args))
    print('It will take a few minutes ...')
    with subprocess.Popen(args,
                          stdout=None, stderr=subprocess.PIPE,
                          shell=True) as p:
        out, err = p.communicate()
        if p.returncode == 0:
            print(f'Done. The results are saved in \'{temp_R_dir}\'.')
            print(f'      Trajectory (pseudotime) information: \'pseudotime_lineages.csv\'.')
            lineages = pd.read_csv(temp_R_dir + '/pseudotime_lineages.csv', index_col=0)
            lineages.columns
            print('      Differential expression information: ', end='')
            for l in lineages.columns:
                print(f'\'DEgenes_MAST_sp{split_num}_{l}.csv\' ', end='')
        else:
            print(f'Something error: returncode={p.returncode}.')


def process_MAST_R(temp_R_dir: str, split_num: int = 4):
    """
    Run the `MAST_script.R` to get differential expression information for each lineage.
    """
    import subprocess
    import importlib.resources as res

    R_script_path = 'MAST_script.R'
    with res.path('cefcon', R_script_path) as datafile:
        R_script_path = datafile

    path = Path(temp_R_dir)
    path.mkdir(exist_ok=path.exists(), parents=True)

    args = f'Rscript {R_script_path} {temp_R_dir} {split_num}'
    print('Running MAST using: \'{}\'\n'.format(args))
    print('It will take a few minutes ...')
    with subprocess.Popen(args,
                          stdout=None, stderr=subprocess.PIPE,
                          shell=True) as p:
        out, err = p.communicate()
        if p.returncode == 0:
            print(f'Done. The results are saved in \'{temp_R_dir}\'.')
            print(f'      Differential expression information: \'DEgenes_MAST_sp{split_num}_<x>.csv\'')
        else:
            print(f'Something error: returncode={p.returncode}.')


def plot_controllability_metrics(cefcon_results: Union[dict, list], return_value: bool = False):
    """

    """
    # calculate metrics
    con_df = pd.DataFrame(columns=['MDS_controllability_score', 'MFVS_controllability_score',
                                   'Jaccard_index', 'Driver_regulators_coverage', 'Lineage'])
    for k in cefcon_results:
        if isinstance(k, str):
            result = cefcon_results[k]
        else:
            result = k
        drivers_df = result.driver_regulator

        MFVS_driver_set = set(drivers_df.loc[drivers_df['is_MFVS_driver']].index)
        MDS_driver_set = set(drivers_df.loc[drivers_df['is_MDS_driver']].index)
        driver_regulators = set(drivers_df.loc[drivers_df['is_driver_regulator']].index)
        top_ranked_genes = driver_regulators.union(
            (set(drivers_df.index) - MFVS_driver_set.union(MDS_driver_set)))
        N_genes = result.n_genes

        # MDS controllability score
        MDS_con = 1 - len(MDS_driver_set) / N_genes
        # MFVS controllability score
        MFVS_con = 1 - len(MFVS_driver_set) / N_genes
        # Jaccard index
        Jaccard_con = len(MDS_driver_set.intersection(MFVS_driver_set)) / len(MDS_driver_set.union(MFVS_driver_set))
        # driver regulators coverage
        Critical_con = len(driver_regulators) / len(MDS_driver_set.union(MFVS_driver_set))

        con_df.loc[len(con_df)] = {'MDS_controllability_score': MDS_con,
                                   'MFVS_controllability_score': MFVS_con,
                                   'Jaccard_index': Jaccard_con,
                                   'Driver_regulators_coverage': Critical_con,
                                   'Lineage': result.name}

    con_df = pd.melt(con_df, id_vars=['Lineage'])

    # plot
    fig = plt.figure(figsize=(9, 2))
    sns.set_theme(style="ticks", font_scale=1.0)
    surrent_palette = sns.color_palette("Set1")

    # Controallability score
    con_df1 = con_df.loc[con_df['variable'].isin(['MDS_controllability_score', 'MFVS_controllability_score'])]
    fig.add_subplot(1, 3, 1)
    ax1 = sns.barplot(x="variable", y="value", hue="Lineage", palette=surrent_palette,
                      data=con_df1)
    ax1.set_xlabel('')
    ax1.set_xticklabels(['MDS', 'MFVS'], rotation=45, ha="right")
    ax1.set_ylabel('Controllability Score')
    ax1.set_ylim(con_df1['value'].min().round(1)-0.1, 1.0)
    sns.despine()
    ax1.get_legend().remove()

    # Jaccard index
    con_df2 = con_df.loc[con_df['variable'].isin(['Jaccard_index'])]
    fig.add_subplot(1, 3, 2)
    ax2 = sns.barplot(x="variable", y="value", hue="Lineage", palette=surrent_palette,
                      data=con_df2)
    ax2.set_xlabel('')
    ax2.set_xticklabels('')
    ax2.set_ylabel(r'Jaccard Index\nbetween MFVS & MDS')
    sns.despine()
    ax2.get_legend().remove()

    # Driver regulators coverage
    con_df3 = con_df.loc[con_df['variable'].isin(['Driver_regulators_coverage'])]
    fig.add_subplot(1, 3, 3)
    ax3 = sns.barplot(x="variable", y="value", hue="Lineage", palette=surrent_palette,
                      data=con_df3)
    ax3.set_xlabel('')
    ax3.set_xticklabels('')
    ax3.set_ylabel('Driver Regulators Coverage')
    sns.despine()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    plt.subplots_adjust(wspace=0.45)

    if return_value:
        return con_df
