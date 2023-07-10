import numpy as np
import pandas as pd
import scanpy as sc
from termcolor import colored
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import umap

import os.path


import seaborn as sns

from .core import *
#from core_working_ import *

from .datasets_via import *
import matplotlib as mpl
#import pyVIA.core as via
#print(os.path.abspath(via.__file__))


sc.settings.set_figure_params(dpi=120, facecolor='white') #or whichever facecolor e.g. black, 'white'
def cellrank_Human(ncomps=80, knn=30, v0_random_seed=7):
    import scvelo as scv
    dict_abb = {'Basophils': 'BASO1', 'CD4+ Effector Memory': 'TCEL7', 'Colony Forming Unit-Granulocytes': 'GRAN1',
                'Colony Forming Unit-Megakaryocytic': 'MEGA1', 'Colony Forming Unit-Monocytes': 'MONO1',
                'Common myeloid progenitors': "CMP", 'Early B cells': "PRE_B2", 'Eosinophils': "EOS2",
                'Erythroid_CD34- CD71+ GlyA-': "ERY2", 'Erythroid_CD34- CD71+ GlyA+': "ERY3",
                'Erythroid_CD34+ CD71+ GlyA-': "ERY1", 'Erythroid_CD34- CD71lo GlyA+': 'ERY4',
                'Granulocyte/monocyte progenitors': "GMP", 'Hematopoietic stem cells_CD133+ CD34dim': "HSC1",
                'Hematopoietic stem cells_CD38- CD34+': "HSC2",
                'Mature B cells class able to switch': "B_a2", 'Mature B cells class switched': "B_a4",
                'Mature NK cells_CD56- CD16- CD3-': "Nka3", 'Monocytes': "MONO2",
                'Megakaryocyte/erythroid progenitors': "MEP", 'Myeloid Dendritic Cells': 'mDC', 'Naïve B cells': "B_a1",
                'Plasmacytoid Dendritic Cells': "pDC", 'Pro B cells': 'PRE_B3'}

    string_ = 'ncomp =' + str(ncomps) + ' knn=' + str(knn) + ' randseed=' + str(v0_random_seed)
    # print('ncomp =', ncomps, ' knn=', knn, ' randseed=', v0_random_seed)
    print(colored(string_, 'blue'))
    nover_labels = pd.read_csv('/home/shobi/Trajectory/Datasets/HumanCD34/Nover_Cor_PredFine_notLogNorm.csv')[
        'x'].values.tolist()
    nover_labels = [dict_abb[i] for i in nover_labels]

    for i in list(set(nover_labels)):
        print('the population of ', i, 'is ', nover_labels.count(i))

    ad = scv.read_loom('/home/shobi/Downloads/Human Hematopoietic Profiling homo_sapiens 2019-11-08 16.12.loom')
    print(ad)
    # ad = sc.read('/home/shobi/Trajectory/Datasets/HumanCD34/human_cd34_bm_rep1.h5ad')
    # ad.obs['nover_label'] = nover_labels
    print('start cellrank pipeline', time.ctime())

    # scv.utils.show_proportions(ad)
    scv.pl.proportions(ad)
    scv.pp.filter_and_normalize(ad, min_shared_counts=20, n_top_genes=2000)
    sc.tl.pca(ad, n_comps=ncomps)
    n_pcs = ncomps

    print('npcs', n_pcs, 'knn', knn)
    sc.pp.neighbors(ad, n_pcs=n_pcs, n_neighbors=knn)
    sc.tl.louvain(ad, key_added='clusters', resolution=1)

    scv.pp.moments(ad, n_pcs=n_pcs, n_neighbors=knn)
    scv.tl.velocity(ad)
    scv.tl.velocity_graph(ad)
    scv.pl.velocity_embedding_stream(ad, basis='umap', color='nover_label')


def adata_preprocess(adata, n_top_genes=1000, log=True):
    # this is a lot like the steps for scvelo.pp.filter_and_normalize() which also allows selection of top genes (see Pancreas)
    sc.pp.filter_genes(adata, min_counts=1)  # only consider genes with more than 1 count#1
    # print(adata)
    sc.pp.normalize_per_cell(  # normalize with total UMI count per cell #same as normalize_total()
        adata, key_n_counts='n_counts_all'
    )
    # select highly-variable genes
    filter_result = sc.pp.filter_genes_dispersion(adata.X, flavor='cell_ranger', n_top_genes=n_top_genes, log=False)
    adata = adata[:, filter_result.gene_subset]  # subset the genes
    sc.pp.normalize_per_cell(adata)  # renormalize after filtering
    if log: sc.pp.log1p(adata)  # log transform: adata.X = log(adata.X + 1)
    '''
    total = adata.X
    total = total.sum(axis=0).transpose()
    total = pd.DataFrame(total.transpose())
    print('total')
    print(total.shape)
    #total = total.sum(axis=0).transpose()
    total.columns = [i for i in adata.var_names]

    print(total)
    total.to_csv('/home/shobi/Trajectory/Datasets/HumanCD34/library_counts_500hvg.csv')
    sc.pp.scale(adata, max_value=10)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=499)  # estimate only 2 PCs
    X_new = pca.fit_transform(adata.X)
    print('variance explained')
    print(pca.explained_variance_ratio_)
    print('pca.components_ shape ncomp x nfeat')
    print()
    df = pd.DataFrame(abs(pca.components_))
    df.to_csv('/home/shobi/Trajectory/Datasets/HumanCD34/pca_components_importance_500hvg.csv')
    print('done saving')
    '''
    # sc.pp.scale(adata, max_value=10)zheng scales after the log, but this doesnt work well and is also not used in scvelo.pp.filter_and_normalize
    return adata


def main_Human(ncomps=80, knn=30, v0_random_seed=10, run_palantir_func=False):
    '''
    df = pd.read_csv('/home/shobi/Trajectory/Datasets/HumanCD34/pca_components_importance_500hvg.csv')
    print(df)
    df = df.set_index('Unnamed: 0')
    print(df)
    df = df.sort_values(by='totals', axis=1, ascending = False)
    df.to_csv('/home/shobi/Trajectory/Datasets/HumanCD34/pca_components_importance_sorted_500hvg.csv')
    print('saved')
    '''
    import random
    random.seed(100)
    dict_abb = {'Basophils': 'BASO1', 'CD4+ Effector Memory': 'TCEL7', 'Colony Forming Unit-Granulocytes': 'GRAN1',
                'Colony Forming Unit-Megakaryocytic': 'MEGA1', 'Colony Forming Unit-Monocytes': 'MONO1',
                'Common myeloid progenitors': "CMP", 'Early B cells': "PRE_B2", 'Eosinophils': "EOS2",
                'Erythroid_CD34- CD71+ GlyA-': "ERY2", 'Erythroid_CD34- CD71+ GlyA+': "ERY3",
                'Erythroid_CD34+ CD71+ GlyA-': "ERY1", 'Erythroid_CD34- CD71lo GlyA+': 'ERY4',
                'Granulocyte/monocyte progenitors': "GMP", 'Hematopoietic stem cells_CD133+ CD34dim': "HSC1",
                'Hematopoietic stem cells_CD38- CD34+': "HSC2",
                'Mature B cells class able to switch': "B_a2", 'Mature B cells class switched': "B_a4",
                'Mature NK cells_CD56- CD16- CD3-': "Nka3", 'Monocytes': "MONO2",
                'Megakaryocyte/erythroid progenitors': "MEP", 'Myeloid Dendritic Cells': 'mDC (cDC)',
                'Naïve B cells': "B_a1",
                'Plasmacytoid Dendritic Cells': "pDC", 'Pro B cells': 'PRE_B3'}
    # NOTE: Myeloid DCs are now called Conventional Dendritic Cells cDCs

    string_ = 'ncomp =' + str(ncomps) + ' knn=' + str(knn) + ' randseed=' + str(v0_random_seed)
    # print('ncomp =', ncomps, ' knn=', knn, ' randseed=', v0_random_seed)
    print(colored(string_, 'blue'))
    nover_labels = pd.read_csv('/home/shobi/Trajectory/Datasets/HumanCD34/Nover_Cor_PredFine_notLogNorm.csv')[
        'x'].values.tolist()
    nover_labels = [dict_abb[i] for i in nover_labels]
    df_nover = pd.DataFrame(nover_labels)
    # df_nover.to_csv('/home/shobi/Trajectory/Datasets/HumanCD34/noverLabelsforMonocle.csv')
    print('save nover')
    for i in list(set(nover_labels)):
        print('Cell type', i, 'has ', nover_labels.count(i), 'cells')


    '''
    parc53_labels = pd.read_csv('/home/shobi/Trajectory/Datasets/HumanCD34/Nover_Cor_Parc53_set1.csv')[
        'x'].values.tolist()

    parclabels_all = pd.read_csv('/home/shobi/Trajectory/Datasets/HumanCD34/parclabels_all_set1.csv')[
        'parc'].values.tolist()
    parc_dict_nover = {}
    for i, c in enumerate(parc53_labels):
        parc_dict_nover[i] = dict_abb[c]
    parclabels_all = [parc_dict_nover[ll] for ll in parclabels_all]
    # print('all', len(parclabels_all))
    '''
    ad = sc.read(
        '/home/shobi/Trajectory/Datasets/HumanCD34/human_cd34_bm_rep1.h5ad') #https://drive.google.com/file/d/1ZSZbMeTQQPfPBGcnfUNDNL4om98UiNcO/view
    # 5780 cells x 14651 genes Human Replicate 1. Male african american, 38 years
    print('h5ad  ad size', ad)

    print(ad[:, 'MPO'].X.flatten().tolist())
    colors = pd.Series(ad.uns['cluster_colors'])
    colors['10'] = '#0b128f'
    ct_colors = pd.Series(ad.uns['ct_colors'])
    list_var_names = ad.var_names
    # print(list_var_names)
    ad.uns['iroot'] = np.flatnonzero(ad.obs_names == ad.obs['palantir_pseudotime'].idxmin())[0]
    print('iroot', np.flatnonzero(ad.obs_names == ad.obs['palantir_pseudotime'].idxmin())[0])

    tsne = pd.DataFrame(ad.obsm['tsne'], index=ad.obs_names, columns=['x', 'y'])
    tsnem = ad.obsm['tsne']
    palantir_tsne_df = pd.DataFrame(tsnem)
    # palantir_tsne_df.to_csv('/home/shobi/Trajectory/Datasets/HumanCD34/palantir_tsne.csv')
    revised_clus = ad.obs['clusters'].values.tolist().copy()
    loc_DCs = [i for i in range(5780) if ad.obs['clusters'].values.tolist()[i] == '7']
    for loc_i in loc_DCs:
        if ad.obsm['palantir_branch_probs'][loc_i, 5] > ad.obsm['palantir_branch_probs'][
            loc_i, 2]:  # if prob that cDC > pDC, then relabel as cDC
            revised_clus[loc_i] = '10'
    revised_clus = [int(i) for i in revised_clus]
    # magic_df = ad.obsm['MAGIC_imputed_data']

    # ad.X: Filtered, normalized and log transformed count matrix
    # ad.raw.X: Filtered raw count matrix
    # print('before extra filtering' ,ad.shape)
    # sc.pp.filter_genes(ad, min_cells=10)
    # print('after extra filtering', ad.shape)
    adata_counts = sc.AnnData(ad.X)

    # df_X = pd.DataFrame(ad.raw.X.todense(), columns = ad.var_names)

    # df_X.columns = [i for i in ad.var_names]
    # print('starting to save .X')
    # df_X.to_csv("/home/shobi/Trajectory/Datasets/HumanCD34/expression_matrix_raw.csv")

    # (ad.X)  # ad.X is filtered, lognormalized,scaled// ad.raw.X is the filtered but not pre-processed
    adata_counts.obs_names = ad.obs_names
    adata_counts.var_names = ad.var_names

    adata_counts.obs['clusters']= ad.obs['clusters']
    # adata_counts_raw = adata_preprocess(adata_counts_raw, n_top_genes=500, log=True) # when using HVG and no PCA
    # sc.tl.pca(adata_counts_raw,svd_solver='arpack', n_comps=ncomps)
    adata_counts.obs['nover_labels'] =[i for i in nover_labels]
    sc.tl.pca(adata_counts, svd_solver='arpack', n_comps=ncomps)

    #sc.pp.neighbors(adata_counts, n_pcs=80, n_neighbors=30)
    #sc.tl.umap(adata_counts, min_dist=0.8)
    #sc.pl.embedding(adata_counts, basis='X_umap', color=['nover_labels', 'clusters'], size=10)

    marker = ['x', '+', (5, 0), '>', 'o', (5, 2)]
    import colorcet as cc

    gene_list = [
        'ITGAX']  # ['GATA1', 'GATA2', 'ITGA2B', 'CSF1R', 'MPO', 'CD79B', 'SPI1', 'IRF8', 'CD34', 'IL3RA', 'ITGAX', 'IGHD',
    # 'CD27', 'CD14', 'CD22', 'ITGAM', 'CLC', 'MS4A3', 'FCGR3A', 'CSF1R']

    true_label = nover_labels  # revised_clus
    root_user = ['HSC1']# [4823]
    dataset = 'group'
    print('v0 random seed', v0_random_seed)
    # df_temp_write  = pd.DataFrame(adata_counts.obsm['X_pca'][:, 0:200])
    # df_temp_write.to_csv("/home/shobi/Trajectory/Datasets/HumanCD34/Human_CD34_200PCA.csv")
    Xin = adata_counts.obsm['X_pca'][:, 0:ncomps]
    # Xin = adata_counts_raw.obsm['X_pca'][:, 0:ncomps]
    # Xin = adata_counts_raw.X.todense()
    print(time.ctime())

    #phate_op = phate.PHATE(n_pca=None)
    #embedding = phate_op.fit_transform(adata_counts.obsm['X_pca'][:, 0:ncomps])
    #plot_scatter(embedding=embedding,labels=true_label,title='phate' )
    plt.show()
    print(time.ctime())
    v0 = VIA(Xin, true_label, jac_std_global=0.15, dist_std_local=1, knn=30,
             too_big_factor=0.3,
             root_user=root_user, dataset=dataset, preserve_disconnected=True, random_seed=v0_random_seed,
              is_coarse=True, pseudotime_threshold_TS=10,
             neighboring_terminal_states_threshold=3, piegraph_arrow_head_width=0.1, edgebundle_pruning_twice=True)#, embedding=tsnem)#, user_defined_terminal_group=['pDC','ERY1', 'ERY3', 'MONO1','mDC (cDC)','PRE_B2'] )# do_compute_embedding=True, embedding_type='via-mds')#,#embedding=adata_counts.obsm['X_umap'])  # *.4 root=1,
    #user_defined_terminal_group=['pDC','MONO1','MEGA1']
    v0.run_VIA()
    via_mds1 = via_mds(via_object=v0, double_diffusion=True, n_milestones=3000)
    f, ax = plot_scatter(embedding=via_mds1, labels=v0.true_label,title='dens sampling')

    via_mds1 = via_mds(via_object=v0, double_diffusion=False, n_milestones=3000)
    f, ax = plot_scatter(embedding=via_mds1, labels=v0.true_label, title='dens sampling')
    plt.show()

    draw_sc_lineage_probability(v0, embedding = tsnem)
    plt.show()
    #draw_sc_lineage_probability(v0, embedding=tsnem, marker_lineages=[2])
    #plt.show()
    plot_edge_bundle(via_object=v0, lineage_pathway=[2,3,5,6,9,12,16,17], linewidth_bundle=0.5, headwidth_bundle=1)


    v0.hammerbundle_milestone_dict = make_edgebundle_milestone(via_object=v0,
                                                               n_milestones=100)  # optional, but just showing how to recompute with different n_milestones
    plot_edge_bundle(via_object=v0, lineage_pathway=[2,3,5,6,9,12,16,17], linewidth_bundle=0.5, headwidth_bundle=1)
    plt.show()
    print('type of labels', type(v0.labels))
    f, ax, ax1 = draw_piechart_graph(via0=v0, linewidth_edge=3)
    ax1.set_title('pseudotime')
    ax.set_title('reference cell types')
    plt.show()
    v0.embedding = tsnem

    #start magic
    adata_counts_raw = sc.AnnData(ad.raw.X) #raw counts
    adata_counts_raw.var_names = [i for i in ad.var_names]

    df_ = pd.DataFrame(adata_counts_raw.X.todense())
    print('shape adata raw df', df_.shape)
    df_.columns = [i for i in adata_counts_raw.var_names]


    print('start magic')
    gene_list_magic = ['IL3RA', 'IRF8', 'GATA1', 'GATA2', 'ITGA2B', 'MPO', 'CD79B', 'SPI1', 'CD34', 'CSF1R', 'ITGAX']
    df_magic = v0.do_impute(df_, magic_steps=3, gene_list=gene_list_magic)
    df_magic_cluster = df_magic.copy()
    df_magic_cluster['parc'] = v0.labels
    df_magic_cluster = df_magic_cluster.groupby('parc', as_index=True).mean()
    print('end magic', df_magic.shape)
    #ad[:,'CD34'].X.flatten().tolist()


    plot_edge_bundle(via_object=v0)
    plt.show()

    marker_genes = ['ITGA2B', 'IL3RA', 'IRF8', 'MPO', 'CSF1R', 'GATA2', 'CD79B', 'CD34', 'GATA1']
    print('dfmagic', df_magic.head())

    plot_gene_trend_heatmaps(via_object=v0, df_gene_exp=df_magic, cmap='plasma', marker_lineages=[6,3,5])

    get_gene_expression(via0=v0, gene_exp=df_magic, marker_genes=marker_genes)
    plt.show()
    get_gene_expression(via0=v0, gene_exp=df_magic, marker_genes=marker_genes, marker_lineages=[6,3,5])
    #plot_gene_trend_heatmaps(via_object=v0, df_gene_exp=df_magic, cmap='plasma', marker_lineages=[16], normalize=False) #looks bad if you dont normalize since some genes are more highly expressed
    plt.show()

    plot_scatter(embedding=tsnem, labels=df_magic['MPO'].tolist(), categorical=False, title='MPO') #CHECK THIS
    plt.show()
    plot_edge_bundle(via_object=v0, sc_labels_expression=df_magic['MPO'].tolist(), n_milestones=100, extra_title_text='MPO')
    plt.show()

    via_streamplot(v0, embedding=tsnem, scatter_size=50, title='original tsne')
    plt.show()



    draw_trajectory_gams(v0, embedding=tsnem, draw_all_curves=False)
    plt.show()


    animated_streamplot(v0, embedding=v0.embedding, scatter_size=800, scatter_alpha=0.15, density_grid=1,
                        saveto='/home/shobi/Trajectory/Datasets/human_stream_test.gif', facecolor_='white', )

    print(f"{datetime.now()}\tAnimate milestone hammer with white bg")
    animate_edge_bundle(via_object=v0, hammerbundle_dict=v0.hammerbundle_milestone_dict,
                        time_series_labels=v0.time_series_labels,
                        linewidth_bundle=2, cmap='rainbow', facecolor='white',
                        extra_title_text='test animation', alpha_scatter=0.1, size_scatter=10,
                        saveto='/home/shobi/Trajectory/Datasets/human_edgebundle_test.gif')

    print('changing the level of visual pruning on the clustergraph and replotting')
    make_edgebundle_viagraph(via_object=v0, edgebundle_pruning=1)
    plot_edgebundle_viagraph(via_object=v0, plot_clusters=True, title='solo viagraph with bundling')
    plt.show()


    print(f"{datetime.now()}\tPlot CD34 milestone hammer external with different params ")
    hammerbundle_milestone_dict = make_edgebundle_milestone(via_object=v0, global_visual_pruning=0.5, decay=0.7, initial_bandwidth=0.02, milestone_labels=v0.labels)

    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=1.5, alpha_bundle_factor=2,
                     cmap='rainbow', facecolor='white', size_scatter=15, alpha_scatter=0.2,
                     extra_title_text='external edgebundle plot', headwidth_bundle=0.3, scale_scatter_size_pop=True)


    print(f"{datetime.now()}\tPlot CD34 milestone hammer external")
    hammerbundle_milestone_dict = make_edgebundle_milestone(via_object=v0, global_visual_pruning=0.5, decay=0.7, initial_bandwidth=0.05)

    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=1.5, alpha_bundle_factor=2,
                     cmap='rainbow', facecolor='white', size_scatter=15, alpha_scatter=0.2,
                     extra_title_text='external edgebundle plot', headwidth_bundle=0.3)

    hammerbundle_milestone_dict = make_edgebundle_milestone(via_object=v0, global_visual_pruning=0.5, decay=0.7, initial_bandwidth=0.05, n_milestones=300)

    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=1.5, alpha_bundle_factor=2,
                     cmap='rainbow', facecolor='white', size_scatter=15, alpha_scatter=0.2,
                     extra_title_text='initial bw'+str(0.05), headwidth_bundle=0.3)


    hammerbundle_milestone_dict = make_edgebundle_milestone(via_object=v0, global_visual_pruning=0.5, decay=0.7,
                                                            initial_bandwidth=0.02, n_milestones=300)

    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=1.5, alpha_bundle_factor=2,
                     cmap='rainbow', facecolor='white', size_scatter=15, alpha_scatter=0.2,
                     extra_title_text='initial bw'+str(0.02), headwidth_bundle=0.3)

    plt.show()



    #MAKE JSON for interactive graph
    # #v0.make_JSON(filename='scRNA_Hema_temp_Feb2022.js')
    super_labels = v0.labels

    print('starting to save selected genes')
    genes_save = ['ITGAX', 'GATA1', 'GATA2', 'ITGA2B', 'CSF1R', 'MPO', 'CD79B', 'SPI1', 'IRF8', 'CD34', 'IL3RA',
                  'ITGAX', 'IGHD',
                  'CD27', 'CD14', 'CD22', 'ITGAM', 'CLC', 'MS4A3', 'FCGR3A', 'CSF1R']
    #df_selected_genes = pd.DataFrame(adata_counts.X, columns=[cc for cc in adata_counts.var_names])
    #df_selected_genes = df_selected_genes[genes_save]
    # df_selected_genes.to_csv("/home/shobi/Trajectory/Datasets/HumanCD34/selected_genes.csv")
    #df_ = pd.DataFrame(ad.X)
    #df_.columns = [i for i in ad.var_names]



    # DC markers https://www.cell.com/pb-assets/products/nucleus/nucleus-phagocytes/rnd-systems-dendritic-cells-br.pdf
    gene_name_dict = {'GATA1': 'GATA1', 'GATA2': 'GATA2', 'ITGA2B': 'CD41 (Mega)', 'MPO': 'MPO (Mono)',
                      'CD79B': 'CD79B (B)', 'IRF8': 'IRF8 (DC)', 'SPI1': 'PU.1', 'CD34': 'CD34',
                      'CSF1R': 'CSF1R (cDC Up. Up then Down in pDC)', 'IL3RA': 'CD123 (pDC)', 'IRF4': 'IRF4 (pDC)',
                      'ITGAX': 'ITGAX (cDCs)', 'CSF2RA': 'CSF2RA (cDC)'}

    marker_genes = ['ITGA2B', 'IL3RA', 'IRF8', 'MPO', 'CSF1R', 'GATA2', 'CD79B', 'CD34', 'GATA1']
    get_gene_expression(via0=v0, gene_exp=df_magic, cmap='rainbow', marker_genes=marker_genes)
    plt.show()
    draw_piechart_graph(via0=v0, type_data='gene', gene_exp=df_magic_cluster['GATA1'].values, title='GATA1', cmap='coolwarm')
    plt.show()




    ad.obs['via0_label'] = [str(i) for i in super_labels]
    magic_ad = ad.obsm['MAGIC_imputed_data']
    magic_ad = sc.AnnData(magic_ad)
    magic_ad.obs_names = ad.obs_names
    magic_ad.var_names = ad.var_names
    magic_ad.obs['via0_label'] = [str(i) for i in super_labels]
    marker_genes_matrix_plot = {"ERY": ['GATA1', 'GATA2', 'ITGA2B'], "BCell": ['IGHD', 'CD22'],
                    "DC": ['IRF8', 'IL3RA', 'IRF4', 'CSF2RA', 'ITGAX'],
                    "MONO": ['CD14', 'SPI1', 'MPO', 'IL12RB1', 'IL13RA1', 'C3AR1', 'FCGR3A'], 'HSC': ['CD34']}

    sc.pl.matrixplot(magic_ad, marker_genes_matrix_plot, groupby='via0_label', dendrogram=True)
    '''

    sc.tl.rank_genes_groups(ad, groupby='via0_label', use_raw=True,
                            method='wilcoxon', n_genes=10)  # compute differential expression
    sc.pl.rank_genes_groups_heatmap(ad, n_genes=10, groupby="via0_label", show_gene_labels=True, use_raw=False)
    sc.pl.rank_genes_groups_tracksplot(ad, groupby='via0_label', n_genes = 3)  # plot the result
    '''
    v1 = VIA(Xin, true_label, jac_std_global=0.15, dist_std_local=1, knn=knn,
             too_big_factor=0.1, root_user=root_user,
             x_lazy=0.95, alpha_teleport=0.99, dataset='humanCD34', preserve_disconnected=True,
             super_terminal_clusters=v0.terminal_clusters, is_coarse=False,
             random_seed=v0_random_seed, pseudotime_threshold_TS=10, via_coarse=v0, edgebundle_pruning_twice=True)
    v1.run_VIA()



    draw_trajectory_gams(v0,v1,embedding= tsnem)
    plt.show()
    # DRAW Lineage pathways
    draw_sc_lineage_probability(v0, v1, embedding=tsnem)
    plt.show()

    get_gene_expression(via0=v1, gene_exp=df_magic, marker_genes=marker_genes)
    plt.show()

    #JSON interactive graphs
    #v1.make_JSON(filename='scRNA_Hema_via1_temp.js')
    df_magic_cluster = df_magic.copy()
    df_magic_cluster['via1'] = v1.labels
    df_magic_cluster = df_magic_cluster.groupby('via1', as_index=True).mean()


    ad.obs['parc1_label'] = [str(i) for i in v1.labels]

    sc.tl.rank_genes_groups(ad, groupby='parc1_label', use_raw=True,
                            method='wilcoxon', n_genes=10)  # compute differential expression

    sc.pl.matrixplot(ad, marker_genes_matrix_plot, groupby='parc1_label', use_raw=False)
    sc.pl.rank_genes_groups_heatmap(ad, n_genes=3, groupby="parc1_label", show_gene_labels=True, use_raw=False)


def main_Toy_comparisons(ncomps=10, knn=30, random_seed=42, dataset='Toy3', root_user='M1',
                         foldername="/home/shobi/Trajectory/Datasets/Toy3/"):
    print('dataset, ncomps, knn, seed', dataset, ncomps, knn, random_seed)

    # root_user = ["T1_M1", "T2_M1"]  # "M1"  # #'M1'  # "T1_M1", "T2_M1"] #"T1_M1"

    if dataset == "Toy3":
        print('dataset Toy3')
        df_counts = pd.read_csv(foldername + "toy_multifurcating_M8_n1000d1000.csv",
                                delimiter=",")

        #df_counts = pd.read_csv(foldername + "Toy3_noise_100genes_thinfactor8.csv", delimiter=",")
        df_ids = pd.read_csv(foldername + "toy_multifurcating_M8_n1000d1000_ids_with_truetime.csv",
                             delimiter=",")
        #counts = palantir.io.from_csv("/home/shobi/Trajectory/Datasets/Toy3/toy_multifurcating_M8_n1000d1000.csv")
        print('df_ids', df_ids.columns)
        root_user = ['M1']
        paga_root = "M1"
        palantir_root = 'C107'
    if dataset == "Toy4":  # 2 disconnected components
        df_counts = pd.read_csv(foldername + "toy_disconnected_M9_n1000d1000.csv",
                                delimiter=",")

        # df_counts = pd.read_csv(foldername + "Toy4_noise_500genes.csv",     delimiter=",")

        df_ids = pd.read_csv(foldername + "toy_disconnected_M9_n1000d1000_ids_with_truetime.csv", delimiter=",")
        #counts = palantir.io.from_csv("/home/shobi/Trajectory/Datasets/Toy4/toy_disconnected_M9_n1000d1000.csv")

        print(df_counts.shape, 'df_counts shape')
        root_user = ['T1_M1', 'T2_M1']  # 'T1_M1'
        paga_root = 'T2_M1'
        palantir_root = 'C107'
    if dataset == "Connected":
        df_counts = pd.read_csv(foldername + "ToyConnected_M9_n2000d1000.csv", delimiter=",")

        # df_counts = pd.read_csv(foldername + "ToyConnected_noise_500genes.csv",  delimiter=",")
        df_ids = pd.read_csv(foldername + "ToyConnected_M9_n2000d1000_ids_with_truetime.csv",
                             delimiter=",")
        #counts = palantir.io.from_csv("/home/shobi/Trajectory/Datasets/ToyConnected/ToyConnected_M9_n2000d1000.csv")
        root_user = ['M1']
        paga_root = "M1"
        palantir_root = 'C1'
    if dataset == "Connected2":
        df_counts = pd.read_csv(foldername + "Connected2_n1000d1000.csv",
                                delimiter=",")
        # df_counts = pd.read_csv(foldername + "ToyConnected2_noise_500genes.csv", 'rt',delimiter=",")

        df_ids = pd.read_csv(foldername + "Connected2_n1000d1000_ids_with_truetime.csv",
                             delimiter=",")
        #counts = palantir.io.from_csv("/home/shobi/Trajectory/Datasets/ToyConnected2/Connected2_n1000d1000.csv")
        root_user = ['M1']
        paga_root = "M1"
        palantir_root = 'C11'
        # suggest to use visual jaccard pruning of 1 (this doesnt alter underlying graph, just the display. can also use "M2" as the starting root,
    if dataset == "ToyMultiM11":
        df_counts = pd.read_csv(foldername + "Toymulti_M11_n3000d1000.csv",
                                delimiter=",")
        # df_counts = pd.read_csv(foldername + "ToyMulti_M11_noised.csv",             delimiter=",")

        df_ids = pd.read_csv(foldername + "Toymulti_M11_n3000d1000_ids_with_truetime.csv",
                             delimiter=",")
        #counts = palantir.io.from_csv(            "/home/shobi/Trajectory/Datasets/ToyMultifurcating_M11/Toymulti_M11_n3000d1000.csv")
        root_user = ['M1']
        paga_root = "M1"
        palantir_root = 'C1005'
    if dataset == "Cyclic":  # 4 milestones
        df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyCyclic/ToyCyclic_M4_n1000d1000.csv",
                                delimiter=",")
        df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyCyclic/ToyCyclic_noise_100genes_thinfactor3.csv",
                                delimiter=",")

        df_ids = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyCyclic/ToyCyclic_M4_n1000d1000_ids_with_truetime.csv",
                             delimiter=",")
        #counts = palantir.io.from_csv("/home/shobi/Trajectory/Datasets/ToyCyclic/ToyCyclic_M4_n1000d1000.csv")
        root_user = ['M1']  # 'T1_M1'
        paga_root = 'M1'
        palantir_root = 'C1'
    if dataset == "Cyclic2":  # 4 milestones
        df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyCyclic2/Cyclic2_n1000d1000.csv",
                                delimiter=",")
        # df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyCyclic2/ToyCyclic2_noise_500genes.csv",              delimiter=",")
        df_ids = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyCyclic2/Cyclic2_n1000d1000_ids_with_truetime.csv",
                             delimiter=",")
        #counts = palantir.io.from_csv("/home/shobi/Trajectory/Datasets/ToyCyclic2/Cyclic2_n1000d1000.csv")
        root_user = ['M1']  # 'T1_M1'
        paga_root = 'M1'
        palantir_root = 'C107'
    if dataset == 'Bifurc2':
        df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyBifurcating2/Bifurc2_M4_n2000d1000.csv",
                                delimiter=",")
        df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyBifurcating2/ToyBifurc2_noised.csv", delimiter=",")

        df_ids = pd.read_csv(            "/home/shobi/Trajectory/Datasets/ToyBifurcating2/Bifurc2_M4_n2000d1000_ids_with_truetime.csv",delimiter=",")
        #counts = palantir.io.from_csv("/home/shobi/Trajectory/Datasets/ToyBifurcating2/Bifurc2_M4_n2000d1000.csv")
        root_user = ['M1']  # 'T1_M1'
        paga_root = 'M1'
        palantir_root = 'C1006'

    if dataset == 'Disconnected2':
        df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyDisconnected2/Disconnected2_n1000d1000.csv",
                                delimiter=",")
        df_counts = pd.read_csv("/home/shobi/Trajectory/Datasets/ToyDisconnected2/ToyDisconnected2_noise_500genes.csv",
                                delimiter=",")
        df_ids = pd.read_csv(
            "/home/shobi/Trajectory/Datasets/ToyDisconnected2/Disconnected2_n1000d1000_ids_with_truetime.csv",
            delimiter=",")
        #counts = palantir.io.from_csv("/home/shobi/Trajectory/Datasets/ToyDisconnected2/Disconnected2_n1000d1000.csv")
        root_user = ['T1_M1', 'T1_M2', 'T1_M3']  # 'T1_M1'
        paga_root = 'T1_M1'
        palantir_root = 'C125'
    df_ids['cell_id_num'] = [int(s[1::]) for s in df_ids['cell_id']]
    print("shape", df_counts.shape, df_ids.shape)
    df_counts = df_counts.drop('Unnamed: 0', 1)
    df_ids = df_ids.sort_values(by=['cell_id_num'])
    df_ids = df_ids.reset_index(drop=True)
    true_label = df_ids['group_id'].tolist()
    print("shape", df_counts.index, df_ids.index)
    adata_counts = sc.AnnData(df_counts, obs=df_ids)
    sc.tl.pca(adata_counts, svd_solver='arpack', n_comps=ncomps)

    # comparisons

    adata_counts.uns['iroot'] = np.flatnonzero(adata_counts.obs['group_id'] == paga_root)[0]  # 'T1_M1'#'M1'
    do_paga = False  #
    do_palantir = False  #
    # comparisons
    if do_paga == True:
        sc.pp.neighbors(adata_counts, n_neighbors=knn, use_rep='X', )  # n_pcs=ncomps)  # 4
        sc.tl.draw_graph(adata_counts)
        # sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data')  # force-directed layout
        start_dfmap = time.time()
        #    sc.tl.diffmap(adata_counts, n_comps=ncomps)
        sc.tl.diffmap(adata_counts, n_comps=200)  # default retains n_comps = 15
        print('time taken to get diffmap given knn', time.time() - start_dfmap)
        sc.pp.neighbors(adata_counts, n_neighbors=knn, use_rep='X_diffmap')  # 4
        sc.tl.draw_graph(adata_counts)
        sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data')

        sc.tl.leiden(adata_counts, resolution=1.0, random_state=10)
        sc.tl.paga(adata_counts, groups='leiden')
        # sc.pl.paga(adata_counts, color=['leiden','group_id'])

        sc.tl.dpt(adata_counts, n_dcs=ncomps)
        df_paga = pd.DataFrame()
        df_paga['paga_dpt'] = adata_counts.obs['dpt_pseudotime'].values

        correlation = df_paga['paga_dpt'].corr(df_ids['true_time'])
        print('corr paga knn', knn, correlation)
        sc.pl.paga(adata_counts, color=['leiden', 'group_id', 'dpt_pseudotime'],
                   title=['leiden (knn:' + str(knn) + ' ncomps:' + str(ncomps) + ')',
                          'group_id (ncomps:' + str(ncomps) + ')', 'pseudotime (ncomps:' + str(ncomps) + ')'])
        # X = df_counts.values

    '''
    # palantir
    if do_palantir == True:
        print(palantir.__file__)  # location of palantir source code

        str_true_label = true_label.tolist()
        str_true_label = [(i[1:]) for i in str_true_label]

        str_true_label = pd.Series(str_true_label, index=counts.index)
        norm_df = counts  # palantir.preprocess.normalize_counts(counts)

        # pca_projections, _ = palantir.utils.run_pca(norm_df, n_components=ncomps) #normally use
        pca_projections = counts

        dm_res = palantir.utils.run_diffusion_maps(pca_projections, knn=knn,
                                                   n_components=300)  ## n_components=ncomps, knn=knn)

        ms_data = palantir.utils.determine_multiscale_space(dm_res)  # n_eigs is determined using eigengap

        tsne = palantir.utils.run_tsne(ms_data)

        palantir.plot.plot_cell_clusters(tsne, str_true_label)

        # C108 for M12 connected' #M8n1000d1000 start - c107 #c1001 for bifurc n2000d1000 #disconnected n1000 c108, "C1 for M10 connected" # c10 for bifurcating_m4_n2000d1000
        # c107 for T1_M1, C42 for T2_M1 disconnected
        # C1 for M8_connected, C1005 for multi_M11 , 'C1006 for bifurc2'
        pr_res = palantir.core.run_palantir(ms_data, early_cell=palantir_root, num_waypoints=500, knn=knn)
        df_palantir = pd.read_csv(
            '/home/shobi/Trajectory/Datasets/Toy3/palantir_pt.csv')  # /home/shobi/anaconda3/envs/ViaEnv/lib/python3.7/site-packages/palantir

        pt = df_palantir['pt']
        correlation = pt.corr(true_time)
        print('corr Palantir', correlation)
        print('')
        palantir.plot.plot_palantir_results(pr_res, tsne, n_knn=knn, n_comps=pca_projections.shape[1])
        plt.show()
    '''
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=ncomps)
    # pc = pca.fit_transform(df_counts)

    Xin = adata_counts.obsm['X_pca'][:, 0:ncomps]
    # Xin = adata_counts.X
    if dataset == 'Toy4':
        jac_std_global = .15  # .15
    else:
        jac_std_global = 0.15  # .15#0.15 #bruge 1 til cyclic2, ellers 0.15
    #
    v0 = VIA(Xin, true_label, jac_std_global=jac_std_global, dist_std_local=1,
             knn=knn, cluster_graph_pruning_std=1,
             too_big_factor=0.3, root_user=root_user, preserve_disconnected=True, dataset='toy',
             visual_cluster_graph_pruning=1, max_visual_outgoing_edges=2,
             random_seed=random_seed)  # *.4 root=2,
    v0.run_VIA()
    super_labels = v0.labels
    df_ids['pt'] = v0.single_cell_pt_markov
    correlation = df_ids['pt'].corr(df_ids['true_time'])
    print('corr via knn', knn, correlation)
    super_edges = v0.edgelist
    # v0.make_JSON(filename = 'Toy3_ViaOut_temp.js')

    print('Granular VIA iteration')
    v1 = VIA(Xin, true_label, jac_std_global=jac_std_global, dist_std_local=1,
             knn=knn,     too_big_factor=0.1, root_user=root_user, is_coarse=False,
             x_lazy=0.95, alpha_teleport=0.99, preserve_disconnected=True, dataset='toy',
             visual_cluster_graph_pruning=1, max_visual_outgoing_edges=2, cluster_graph_pruning_std=1,
             random_seed=random_seed, via_coarse=v0)

    v1.run_VIA()
    df_ids['pt1'] = v1.single_cell_pt_markov
    correlation = df_ids['pt1'].corr(df_ids['true_time'])
    print('corr via1 knn', knn, correlation)
    labels = v1.labels


    embedding = adata_counts.obsm['X_pca'][:,  0:2]  # umap.UMAP().fit_transform(adata_counts.obsm['X_pca'][idx, 0:5])


    draw_trajectory_gams(via_coarse=v0, via_fine=v1, embedding=embedding)
    plt.show()

    num_group = len(set(true_label))
    line = np.linspace(0, 1, num_group)

    f, (ax1, ax3) = plt.subplots(1, 2, sharey=True)

    for color, group in zip(line, set(true_label)):
        where = np.where(np.asarray(true_label) == group)[0]

        ax1.scatter(embedding[where, 0], embedding[where, 1], label=group,
                    c=np.asarray(plt.cm.jet(color)).reshape(-1, 4))
    ax1.legend(fontsize=6)
    ax1.set_title('true labels')

    ax3.set_title("Pseudotime using ncomps:" + str(Xin.shape[1]) + '. knn:' + str(knn))
    ax3.scatter(embedding[:, 0], embedding[:, 1], c=v1.single_cell_pt_markov, cmap='viridis_r')
    plt.show()
    df_genes = pd.DataFrame(adata_counts.obsm['X_pca'][:, 0:5], columns=['Gene0', 'Gene1', 'Gene2', 'Gene3', 'Gene4']) #dummy gene values for illustration
    v1.get_gene_expression(gene_exp=df_genes, marker_genes=['Gene0', 'Gene1', 'Gene2'])


    draw_sc_lineage_probability(via_coarse=v0, via_fine=v1, embedding=embedding)

    plt.show()


def main_Toy(ncomps=10, knn=30, random_seed=41, dataset='Toy3', root_user=['M1'],
             cluster_graph_pruning_std=1, foldername="/home/shobi/Trajectory/Datasets/"):
    print('dataset, ncomps, knn, seed', dataset, ncomps, knn, random_seed)
    import phate
    if dataset == "Toy3":
        print('inside Toy3')
        #df_counts = pd.read_csv(foldername + "toy_multifurcating_M8_n1000d1000.csv", delimiter=",")
        #df_ids = pd.read_csv(foldername + "toy_multifurcating_M8_n1000d1000_ids_with_truetime.csv", delimiter=",")


        adata_counts = toy_multifurcating()

        #root_user, dataset  = None, ''#['M1'], 'group' #alternative root setting
        root_user, dataset= ['M1'], 'group'
        #root_user, dataset = [16], ''
        paga_root = "M1"
        dataset_name = 'Toy3'
    if dataset == "Toy4":  # 2 disconnected components
        print('inside toy4')
        #df_counts = pd.read_csv(foldername + "toy_disconnected_M9_n1000d1000.csv", delimiter=",")
        #df_ids = pd.read_csv(foldername + "toy_disconnected_M9_n1000d1000_ids_with_truetime.csv", delimiter=",")
        adata_counts = toy_disconnected()
        #root_user, dataset =  [136,4], ''
        root_user, dataset = ['T2_M1','T1_M1'],'group'#  #alternative root settings:   None, '' OR [136,4],''
        paga_root = 'T1_M1'
        dataset_name = 'Toy4'
    #df_ids['cell_id_num'] = [int(s[1::]) for s in df_ids['cell_id']]

    #df_counts = df_counts.drop('Unnamed: 0', 1)
    #df_ids = df_ids.sort_values(by=['cell_id_num'])
    #df_ids = df_ids.reset_index(drop=True)
    true_label = adata_counts.obs['group_id'].tolist()#df_ids['group_id'].tolist()
    #true_time = df_ids['true_time']
    #adata_counts = sc.AnnData(df_counts, obs=df_ids)

    sc.tl.pca(adata_counts, svd_solver='arpack', n_comps=ncomps)

    adata_counts.uns['iroot'] = np.flatnonzero(adata_counts.obs['group_id'] == paga_root)[0]  # 'T1_M1'#'M1'

    if dataset == 'Toy4':
        jac_std_global = 0.15#0.15  # 1
    else:
        jac_std_global = 0.05#0.15#0.15
    import umap

    embedding = umap.UMAP(min_dist=0.5).fit_transform(adata_counts.obsm['X_pca'][:, 0:10])  # 50

    # embedding = adata_counts.obsm['X_pca'][:, 0:2]
    # plt.scatter(embedding[:,0],embedding[:,1])
    # plt.show()
    do_phate = False
    if do_phate == True:
        phate_op = phate.PHATE(n_pca=None)
        embedding = phate_op.fit_transform(adata_counts.obsm['X_pca'][:, 0:ncomps])
        plt.scatter(embedding[:, 0], embedding[:, 1], c=[int(i[-1]) for i in true_label], cmap='rainbow', s=2)
        plt.show()
        plt.scatter(embedding[:, 0], embedding[:, 1], c=[int(i[-1]) for i in true_label], cmap='rainbow', s=2)
        plt.show()
    X_pca= adata_counts.obsm['X_pca'][:, 0:ncomps]
    print('root user', root_user, print(true_label))
    #optionally do kmeans or pass in cluster labels
    #from sklearn.cluster import KMeans
    #kmeans = KMeans(n_clusters=20, random_state=1).fit(X_pca)
    #labels=kmeans.labels_

    #self.labels = kmeans.labels_.flatten()

    v0 = VIA(X_pca, true_label, jac_std_global=jac_std_global, dist_std_local=1,
             knn=knn,
             cluster_graph_pruning_std=cluster_graph_pruning_std,
             too_big_factor=0.3, root_user=root_user, preserve_disconnected=True, dataset=dataset,
             visual_cluster_graph_pruning=1, max_visual_outgoing_edges=2,
             random_seed=random_seed, piegraph_arrow_head_width=0.2,
             piegraph_edgeweight_scalingfactor=1.0,  resolution_parameter=1)#, do_compute_embedding=True, embedding_type='via-umap') #, embedding=embedding)  # *.4 root=2,embedding=embedding user_defined_terminal_group=['M8','M6'] #embedding_type='via-mds', do_compute_embedding=True,, user_defined_terminal_group=['M6','M8','M2','M7']
    v0.run_VIA()

    e1=via_umap(via_object=v0, init_pos='via', random_state=v0.random_seed)#, n_epochs=100, spread=1,                                                      distance_metric='euclidean', min_dist=0.1, saveto='',                                                      random_state=v0.random_seed)
    plot_scatter(e1, labels=v0.labels, title='via init')
    e2 = via_umap(via_object=v0)  # , n_epochs=100, spread=1,                                                      distance_metric='euclidean', min_dist=0.1, saveto='',                                                      random_state=v0.random_seed)
    plot_scatter(e1, labels=v0.labels, title='spectral init')
    plt.show()
    draw_trajectory_gams(via_object=v0, embedding=embedding)
    plt.show()

    draw_sc_lineage_probability(v0, embedding=embedding)
    plt.show()
    via_mds1 = via_mds(via_object=v0)
    f, ax = plot_scatter(embedding=via_mds1, labels=v0.true_label, title='viamds')
    plt.show()
    via_mds1 = via_mds(via_object=v0)
    f, ax = plot_scatter(embedding=via_mds1, labels=v0.single_cell_pt_markov,title='viamds')
    plt.show()

    print('make new edgbune bundle')
    #v0.embedding = None #testing automatic embedding computation
    #v0.hammerbundle_milestone_dict = None #testing automatic hammerbundling computation
    v0.hammerbundle_milestone_dict=make_edgebundle_milestone(via_object=v0, n_milestones=40) #optional, but just showing how to recompute with different n_milestones
    plot_edge_bundle(via_object=v0, lineage_pathway=v0.terminal_clusters, linewidth_bundle=0.5, headwidth_bundle=1, cmap='plasma', text_labels=True, show_milestones=True, scale_scatter_size_pop=False)
    plot_edge_bundle(via_object=v0, lineage_pathway=v0.terminal_clusters[0:2], linewidth_bundle=0.5, headwidth_bundle=1, cmap='plasma',text_labels=True, show_milestones=False, scale_scatter_size_pop=True, fontsize_labels=4)
    plot_edge_bundle(via_object=v0, linewidth_bundle=0.5, headwidth_bundle=2)
    plot_edge_bundle(via_object=v0, lineage_pathway=[v0.terminal_clusters[0]], linewidth_bundle=0.5, headwidth_bundle=1)
    plt.show()
    draw_sc_lineage_probability(v0, embedding=embedding, marker_lineages=v0.terminal_clusters[0:2])  # [10,5,1,0,2]) #Toy3)
    plt.show()
    hammerbundle_milestone_dict = make_edgebundle_milestone(via_object=v0, global_visual_pruning=1,
                                                            initial_bandwidth=0.02, decay=0.7, milestone_labels=v0.labels)
    print(f"{datetime.now()}\tPlot milestone hammer external")
    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=0.3, alpha_bundle_factor=2,
                     cmap='rainbow', facecolor='white', size_scatter=15, alpha_scatter=0.5,scale_scatter_size_pop=True,
                     extra_title_text='VIA graph clusters', headwidth_bundle=2.5)

    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=0.3, alpha_bundle_factor=2,
                     cmap='viridis', facecolor='white', size_scatter=15, alpha_scatter=0.5, scale_scatter_size_pop=True,
                     extra_title_text='Using VIA clusters for lineage pathways', headwidth_bundle=2.5, lineage_pathway=v0.terminal_clusters)
    plt.show()
    '''
    via_umap1 = via_umap(via_object = v0)
    f, ax = plot_scatter(embedding=via_umap1, labels=v0.true_label)
    plt.show()
    
    via_mds1 = via_mds(via_object=v0)
    f, ax = plot_scatter(embedding=via_mds1, labels=v0.true_label)
    plt.show()
    via_umap1 = via_umap(X_input=v0.data, graph=v0.csr_full_graph)
    f, ax = plot_scatter(embedding=via_umap1, labels=v0.true_label)
    plt.show()
    '''
    print(f'{datetime.now()}\tFor random seed: {random_seed}, the first sc markov pt are', v0.single_cell_pt_markov[0:10])

    #draw_sc_lineage_probability(v0, embedding=embedding,marker_lineages=[5,10])
    #plt.show()

    #v0.embedding = embedding


    hammerbundle_milestone_dict = make_edgebundle_milestone(via_object=v0, global_visual_pruning=1, initial_bandwidth=0.02, decay=0.7)

    print(f"{datetime.now()}\tPlot milestone hammer external")
    #edges can be colored by time-series numeric labels, pseudotime, or gene expression. If not specificed then time-series is chosen if available, otherwise falls back to pseudotime. to use gene expression the sc_labels_expression is provided as a list
    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=1.5, alpha_bundle_factor=2,
                     cmap='plasma_r', facecolor='white', size_scatter=15, alpha_scatter=0.2, scale_scatter_size_pop=True,
                     extra_title_text='Gene0 Expression', headwidth_bundle=0.15, sc_labels_expression = adata_counts.obsm['X_pca'][:, 0].tolist(), text_labels=True, sc_labels=true_label)

    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=1.5, alpha_bundle_factor=2,
                     cmap='plasma_r', facecolor='white', size_scatter=15, alpha_scatter=0.2,
                     scale_scatter_size_pop=True,
                     extra_title_text='', headwidth_bundle=0.15,
                    text_labels=True,
                     sc_labels=true_label)

    plt.show()
    df_genes = pd.DataFrame(adata_counts.obsm['X_pca'][:, 0:5], columns=['Gene0', 'Gene1', 'Gene2', 'Gene3', 'Gene4'])

    if dataset_name=='Toy3':
        f, axlist = plot_gene_trend_heatmaps(via_object=v0, df_gene_exp=df_genes, marker_lineages=v0.terminal_clusters)
        axlist[-1].set_xlabel("pseudotime", fontsize=20)
        plt.show()
    if dataset_name=='Toy4':
        f, axlist = plot_gene_trend_heatmaps(via_object=v0, df_gene_exp=df_genes, marker_lineages=[])
        axlist[-1].set_xlabel("pseudotime", fontsize=20)
        plt.show()
    get_gene_expression(v0, gene_exp=df_genes, marker_genes=['Gene0', 'Gene1', 'Gene2'])
    plt.show()

    print('draw piechart graph')
    draw_piechart_graph(via_object=v0)
    plt.show()


    plot_edgebundle_viagraph(via_object=v0, plot_clusters=True, title='viagraph with bundling', fontsize=10)
    plt.show()

    via_streamplot(via_object=v0, embedding=embedding, scatter_size=20)  # embedding
    plt.show()
    print('making animated stream plot. This may take a few minutes when the streamline count is high')
    animated_streamplot(v0, embedding, scatter_size=800, scatter_alpha=0.15, density_grid=1,
                        saveto='/home/shobi/Trajectory/Datasets/Toy3/test_framerates.gif', facecolor_='white',
                        cmap_stream='Blues')
    plt.show()


    hammerbundle_milestone_dict = make_edgebundle_milestone(via_object=v0, global_visual_pruning=1,
                                                            initial_bandwidth=0.02, decay=0.7, n_milestones=300)
    print(f"{datetime.now()}\tPlot milestone hammer external")
    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=1.5, alpha_bundle_factor=2,
                     cmap='rainbow', facecolor='white', size_scatter=15, alpha_scatter=0.2,
                     extra_title_text='edgebundle plot', headwidth_bundle=0.15)


    hammerbundle_milestone_dict = make_edgebundle_milestone(via_object=v0, global_visual_pruning=1,
                                                            initial_bandwidth=0.05, decay=0.7, n_milestones=300)
    print(f"{datetime.now()}\tPlot milestone hammer external")
    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=1.5, alpha_bundle_factor=2,
                     cmap='rainbow', facecolor='white', size_scatter=15, alpha_scatter=0.2,
                     extra_title_text='edgebundle plot', headwidth_bundle=0.15)

    hammerbundle_milestone_dict = make_edgebundle_milestone(via_object=v0, global_visual_pruning=1,
                                                            initial_bandwidth=0.05, decay=0.7, n_milestones=50)
    print(f"{datetime.now()}\tPlot milestone hammer external")
    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=1.5, alpha_bundle_factor=2,
                     cmap='rainbow', facecolor='white', size_scatter=15, alpha_scatter=0.2,
                     extra_title_text='edgebundle plot', headwidth_bundle=0.15)

    hammerbundle_milestone_dict = make_edgebundle_milestone(via_object=v0, global_visual_pruning=1,
                                                            initial_bandwidth=0.02, decay=0.7, n_milestones=50)
    print(f"{datetime.now()}\tPlot milestone hammer external")
    plot_edge_bundle(hammerbundle_dict=hammerbundle_milestone_dict,
                     linewidth_bundle=1.5, alpha_bundle_factor=2,
                     cmap='rainbow', facecolor='white', size_scatter=15, alpha_scatter=0.2,
                     extra_title_text='edgebundle plot', headwidth_bundle=0.15)

    plt.show()


    print(f"{datetime.now()}\tAnimate milestone hammer with white bg")
    animate_edge_bundle(via_object=v0, hammerbundle_dict=v0.hammerbundle_milestone_dict,
                        time_series_labels=v0.time_series_labels,
                        linewidth_bundle=2, cmap='rainbow', facecolor='white',
                        extra_title_text='test animation', alpha_scatter=0.1, size_scatter=10,
                        saveto='/home/shobi/Trajectory/Datasets/testing_oct26.gif')
    make_edgebundle_viagraph(via_object=v0, edgebundle_pruning=1)
    plot_edgebundle_viagraph(via_object=v0, plot_clusters=True, title='viagraph with bundling', fontsize=10)
    plt.show()
    #embedding = via_umap(X_input=adata_counts.obsm['X_pca'][:, 0:10], graph=(v0.csr_full_graph), n_components=2, spread=1.0, min_dist=0.5,           init_pos='spectral', random_state=1, n_epochs=100)
    #mbedding = via_mds(graph = v0.csr_full_graph)




    #knn_struct =construct_knn_utils(X_pca[idx_sub,:])
    #embedding = mds_milestone_knn_new( X_pca=X_pca,viagraph_full=v0.csr_full_graph, k=10, n_milestones=500)

    #embedding = via.via_umap(X_input=v0.data, graph=row_stoch, n_components=2, spread=spread, min_dist=min_dist,   init_pos='spectral', random_state=random_seed_umap, n_epochs=50)

    #embedding = sgd_mds(via_graph=v0.csr_full_graph,X_pca=adata_counts.obsm['X_pca'][:, 0:ncomps], t_diff_op = 3, ndims= 2, random_seed=random_seed)




    print('changing edge color on viagraphs')
    draw_piechart_graph(v0, edge_color='gray')
    plt.show()

    super_labels = v0.labels
    print('super labels', type(super_labels))
    df_ids = pd.DataFrame()
    df_ids['true_time'] =adata_counts.obs['true_time'].tolist()
    df_ids['pt'] = v0.single_cell_pt_markov
    correlation = df_ids['pt'].corr(df_ids['true_time'])
    print('corr via knn', knn, correlation)

    # v0.make_JSON(filename = 'Toy3_ViaOut_temp.js')
    #draw_sc_lineage_probability(via_coarse=v0, via_fine=v0, embedding=embedding)

    plt.show()

    v1 = VIA(adata_counts.obsm['X_pca'][:, 0:ncomps], true_label, jac_std_global=jac_std_global, dist_std_local=1,
             knn=knn,             too_big_factor=0.1,             cluster_graph_pruning_std=cluster_graph_pruning_std,
             super_cluster_labels=super_labels, root_user=root_user, is_coarse=False,
             x_lazy=0.95, alpha_teleport=0.99, preserve_disconnected=True, dataset=dataset,
             visual_cluster_graph_pruning=1, max_visual_outgoing_edges=2,random_seed=random_seed, via_coarse=v0)
    v1.run_VIA()

    df_ids['pt1'] = v1.single_cell_pt_markov
    correlation = df_ids['pt1'].corr(df_ids['true_time'])
    print('corr via knn', knn, correlation)


    draw_trajectory_gams(v0,v1, embedding)
    plt.show()

    df_genes = pd.DataFrame(adata_counts.obsm['X_pca'][:, 0:5], columns=['Gene0', 'Gene1', 'Gene2', 'Gene3', 'Gene4'])

    get_gene_expression(v0, gene_exp=df_genes, marker_genes=['Gene0', 'Gene1', 'Gene2'])
    plt.show()
    draw_sc_lineage_probability(via_object = v0, via_fine=v1, embedding=embedding)
    plt.show()


def main_Bcell(ncomps=50, knn=20, random_seed=0, cluster_graph_pruning_std=.15,path='/home/shobi/Trajectory/Datasets/Bcell/'):
    print('Input params: ncomp, knn, random seed', ncomps, knn, random_seed)

    # https://github.com/STATegraData/STATegraData
    def run_zheng_Bcell(adata, min_counts=3, n_top_genes=500, do_HVG=True):
        sc.pp.filter_genes(adata, min_counts=min_counts)
        # sc.pp.filter_genes(adata, min_cells=3)# only consider genes with more than 1 count
        '''
        sc.pp.normalize_per_cell(  # normalize with total UMI count per cell
            adata, key_n_counts='n_counts_all')
        '''
        sc.pp.normalize_total(adata, target_sum=1e4)
        if do_HVG == True:
            sc.pp.log1p(adata)
            '''
            filter_result = sc.pp.filter_genes_dispersion(  # select highly-variable genes
            adata.X, flavor='cell_ranger', n_top_genes=n_top_genes, log=False )
            adata = adata[:, filter_result.gene_subset]  # subset the genes
            '''
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, min_mean=0.0125, max_mean=3,
                                        min_disp=0.5)  # this function expects logarithmized data
            print('len hvg ', sum(adata.var.highly_variable))
            adata = adata[:, adata.var.highly_variable]
        sc.pp.normalize_per_cell(adata)  # renormalize after filtering
        # if do_log: sc.pp.log1p(adata)  # log transform: adata.X = log(adata.X + 1)
        if do_HVG == False: sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)  # scale to unit variance and shift to zero mean
        return adata
    '''
    def run_palantir_func_Bcell(ad1, ncomps, knn, tsne_X, true_label):
        ad = ad1.copy()
        tsne = pd.DataFrame(tsne_X, index=ad.obs_names, columns=['x', 'y'])
        norm_df_pal = pd.DataFrame(ad.X)
        new = ['c' + str(i) for i in norm_df_pal.index]
        norm_df_pal.columns = [i for i in ad.var_names]
        # print('norm df', norm_df_pal)

        norm_df_pal.index = new
        pca_projections, _ = palantir.utils.run_pca(norm_df_pal, n_components=ncomps)

        sc.tl.pca(ad, svd_solver='arpack')
        dm_res = palantir.utils.run_diffusion_maps(pca_projections, n_components=ncomps, knn=knn)

        ms_data = palantir.utils.determine_multiscale_space(dm_res)  # n_eigs is determined using eigengap
        print('ms data shape: determined using eigengap', ms_data.shape)
        # tsne =  pd.DataFrame(tsnem)#palantir.utils.run_tsne(ms_data)
        tsne.index = new
        # print(type(tsne))
        str_true_label = pd.Series(true_label, index=norm_df_pal.index)

        palantir.plot.plot_cell_clusters(tsne, str_true_label)

        start_cell = 'c42'  # '#C108 for M12 connected' #M8n1000d1000 start - c107 #c1001 for bifurc n2000d1000 #disconnected n1000 c108, "C1 for M10 connected" # c10 for bifurcating_m4_n2000d1000

        pr_res = palantir.core.run_palantir(ms_data, early_cell=start_cell, num_waypoints=1200, knn=knn)
        palantir.plot.plot_palantir_results(pr_res, tsne, n_knn=knn, n_comps=ncomps)
        imp_df = palantir.utils.run_magic_imputation(norm_df_pal, dm_res)
        Bcell_marker_gene_list = ['Igll1', 'Myc', 'Ldha', 'Foxo1', 'Lig4']  # , 'Slc7a5']#,'Slc7a5']#,'Sp7','Zfp629']
        gene_trends = palantir.presults.compute_gene_trends(pr_res, imp_df.loc[:, Bcell_marker_gene_list])
        palantir.plot.plot_gene_trends(gene_trends)
        plt.show()
    '''
    def run_paga_func_Bcell(adata_counts1, ncomps, knn, embedding):
        # print('npwhere',np.where(np.asarray(adata_counts.obs['group_id']) == '0')[0][0])
        adata_counts = adata_counts1.copy()
        sc.tl.pca(adata_counts, svd_solver='arpack', n_comps=ncomps)
        adata_counts.uns['iroot'] = 33  # np.where(np.asarray(adata_counts.obs['group_id']) == '0')[0][0]

        sc.pp.neighbors(adata_counts, n_neighbors=knn, n_pcs=ncomps)  # 4
        sc.tl.draw_graph(adata_counts)
        sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data')  # force-directed layout
        start_dfmap = time.time()
        sc.tl.diffmap(adata_counts, n_comps=ncomps)
        print('time taken to get diffmap given knn', time.time() - start_dfmap)
        sc.pp.neighbors(adata_counts, n_neighbors=knn, use_rep='X_diffmap')  # 4
        sc.tl.draw_graph(adata_counts)
        sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data')
        sc.tl.leiden(adata_counts, resolution=1.0)
        sc.tl.paga(adata_counts, groups='leiden')
        # sc.pl.paga(adata_counts, color=['louvain','group_id'])

        sc.tl.dpt(adata_counts, n_dcs=ncomps)
        sc.pl.paga(adata_counts, color=['leiden', 'group_id', 'dpt_pseudotime'],
                   title=['leiden (knn:' + str(knn) + ' ncomps:' + str(ncomps) + ')',
                          'group_id (ncomps:' + str(ncomps) + ')', 'pseudotime (ncomps:' + str(ncomps) + ')'])
        sc.pl.draw_graph(adata_counts, color='dpt_pseudotime', legend_loc='on data')
        print('dpt format', adata_counts.obs['dpt_pseudotime'])
        plt.scatter(embedding[:, 0], embedding[:, 1], c=adata_counts.obs['dpt_pseudotime'].values, cmap='viridis')
        plt.title('PAGA DPT')
        plt.show()

    def find_time_Bcell(s):
        start = s.find("Ik") + len("Ik")
        end = s.find("h")
        return int(s[start:end])

    def find_cellID_Bcell(s):
        start = s.find("h") + len("h")
        end = s.find("_")
        return s[start:end]

    Bcell = pd.read_csv(path + 'genes_count_table.txt', sep='\t')
    gene_name = pd.read_csv(path + 'genes_attr_table.txt', sep='\t')

    Bcell_columns = [i for i in Bcell.columns]
    adata_counts = sc.AnnData(Bcell.values[:, 1:].T)
    Bcell_columns.remove('tracking_id')

    print(gene_name.shape, gene_name.columns)
    Bcell['gene_short_name'] = gene_name['gene_short_name']
    adata_counts.var_names = gene_name['gene_short_name']
    adata_counts.obs['TimeCellID'] = Bcell_columns

    time_list = [find_time_Bcell(s) for s in Bcell_columns]

    print('time list set', set(time_list))
    adata_counts.obs['TimeStamp'] = [str(tt) for tt in time_list]

    ID_list = [find_cellID_Bcell(s) for s in Bcell_columns]
    adata_counts.obs['group_id'] = [str(i) for i in time_list]
    ID_dict = {}
    color_dict = {}
    for j, i in enumerate(list(set(ID_list))):
        ID_dict.update({i: j})
    print('timelist', list(set(time_list)))
    for j, i in enumerate(list(set(time_list))):
        color_dict.update({i: j})

    print('shape of raw data', adata_counts.shape)

    adata_counts_unfiltered = adata_counts.copy()

    Bcell_marker_gene_list = ['Myc', 'Igll1', 'Slc7a5', 'Ldha', 'Foxo1', 'Lig4']

    small_large_gene_list = ['Kit', 'Pcna', 'Ptprc', 'Il2ra', 'Vpreb1', 'Cd24a', 'Igll1', 'Cd79a', 'Cd79b', 'Mme',
                             'Spn']
    list_var_names = [s for s in adata_counts_unfiltered.var_names]
    matching = [s for s in list_var_names if "IgG" in s]

    for gene_name in Bcell_marker_gene_list:
        print('gene name', gene_name)
        loc_gata = np.where(np.asarray(adata_counts_unfiltered.var_names) == gene_name)[0][0]
    for gene_name in small_large_gene_list:
        print('looking at small-big list')
        print('gene name', gene_name)
        loc_gata = np.where(np.asarray(adata_counts_unfiltered.var_names) == gene_name)[0][0]
    # diff_list = [i for i in diff_list if i in list_var_names] #based on paper STable1 https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2006506#pbio.2006506.s007
    # adata_counts = adata_counts[:,diff_list] #if using these, then set do-HVG to False
    print('adata counts difflisted', adata_counts.shape)
    adata_counts = run_zheng_Bcell(adata_counts, n_top_genes=5000, min_counts=30,
                                   do_HVG=True)  # 5000 for better ordering
    print('adata counts shape', adata_counts.shape)
    # sc.pp.recipe_zheng17(adata_counts)

    # (ncomp=50, knn=20 gives nice results. use 10PCs for visualizing)

    marker_genes = {"small": ['Rag2', 'Rag1', 'Pcna', 'Myc', 'Ccnd2', 'Cdkn1a', 'Smad4', 'Smad3', 'Cdkn2a'],
                    # B220 = Ptprc, PCNA negative for non cycling
                    "large": ['Ighm', 'Kit', 'Ptprc', 'Cd19', 'Il2ra', 'Vpreb1', 'Cd24a', 'Igll1', 'Cd79a', 'Cd79b'],
                    "Pre-B2": ['Mme', 'Spn']}  # 'Cd19','Cxcl13',,'Kit'

    print('make the v0 matrix plot')
    mplot_adata = adata_counts_unfiltered.copy()  # mplot_adata is for heatmaps so that we keep all genes
    mplot_adata = run_zheng_Bcell(mplot_adata, n_top_genes=25000, min_counts=1, do_HVG=False)
    # mplot_adata.X[mplot_adata.X>10] =10
    # mplot_adata.X[mplot_adata.X< -1] = -1
    # sc.pl.matrixplot(mplot_adata, marker_genes, groupby='TimeStamp', dendrogram=True)

    sc.tl.pca(adata_counts, svd_solver='arpack', n_comps=200)  # ncomps
    # df_bcell_pc = pd.DataFrame(adata_counts.obsm['X_pca'])
    # print('df_bcell_pc.shape',df_bcell_pc.shape)
    # df_bcell_pc['time'] = [str(i) for i in time_list]
    # df_bcell_pc.to_csv('/home/shobi/Trajectory/Datasets/Bcell/Bcell_200PCs.csv')
    # sc.pl.pca_variance_ratio(adata_counts, log=True)

    jet = cm.get_cmap('viridis', len(set(time_list)))
    cmap_ = jet(range(len(set(time_list))))

    jet2 = cm.get_cmap('jet', len(set(ID_list)))
    cmap2_ = jet2(range(len(set(ID_list))))

    # color_dict = {"0": [0], "2": [1], "6": [2], "12": [3], "18": [4], "24": [5]}
    # sc.pl.heatmap(mplot_adata, var_names  = small_large_gene_list,groupby = 'TimeStamp', dendrogram = True)
    embedding = umap.UMAP(random_state=42, n_neighbors=15, init='random').fit_transform(
        adata_counts.obsm['X_pca'][:, 0:5])
    df_umap = pd.DataFrame(embedding)
    # df_umap.to_csv('/home/shobi/Trajectory/Datasets/Bcell/Bcell_umap.csv')

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    for i in list(set(time_list)):
        loc = np.where(np.asarray(time_list) == i)[0]
        ax4.scatter(embedding[loc, 0], embedding[loc, 1], c=cmap_[color_dict[i]], alpha=1, label=str(i))
        if i == 0:
            for xx in range(len(loc)):
                poss = loc[xx]
                ax4.text(embedding[poss, 0], embedding[poss, 1], 'c' + str(xx))

    ax4.legend()

    ax1.scatter(embedding[:, 0], embedding[:, 1], c=mplot_adata[:, 'Pcna'].X.flatten(), alpha=1)
    ax1.set_title('Pcna, cycling')
    ax2.scatter(embedding[:, 0], embedding[:, 1], c=mplot_adata[:, 'Vpreb1'].X.flatten(), alpha=1)
    ax2.set_title('Vpreb1')
    ax3.scatter(embedding[:, 0], embedding[:, 1], c=mplot_adata[:, 'Cd24a'].X.flatten(), alpha=1)
    ax3.set_title('Cd24a')

    # ax2.text(embedding[i, 0], embedding[i, 1], str(i))

    '''    
    for i, j in enumerate(list(set(ID_list))):
        loc = np.where(np.asarray(ID_list) == j)
        if 'r'in j: ax2.scatter(embedding[loc, 0], embedding[loc, 1], c=cmap2_[i], alpha=1, label=str(j), edgecolors = 'black' )
        else: ax2.scatter(embedding[loc, 0], embedding[loc, 1], c=cmap2_[i], alpha=1, label=str(j))
    '''
    # plt.show()

    true_label = time_list

    # run_paga_func_Bcell(adata_counts, ncomps, knn, embedding)

    #run_palantir_func_Bcell(adata_counts, ncomps, knn, embedding, true_label)

    print('input has shape', adata_counts.obsm['X_pca'].shape)
    input_via = adata_counts.obsm['X_pca'][:, 0:ncomps]

    df_input = pd.DataFrame(adata_counts.obsm['X_pca'][:, 0:200])
    df_annot = pd.DataFrame(['t' + str(i) for i in true_label])
    # df_input.to_csv('/home/shobi/Trajectory/Datasets/Bcell/Bcell_200PC_5000HVG.csv')
    # df_annot.to_csv('/home/shobi/Trajectory/Datasets/Bcell/Bcell_annots.csv')
    root_user = [42]
    v0 = VIA(input_via, true_label, jac_std_global=0.15, dist_std_local=1, knn=knn,
             too_big_factor=0.3, dataset='bcell',
             cluster_graph_pruning_std=cluster_graph_pruning_std,
             root_user=root_user, preserve_disconnected=True, random_seed=random_seed,
              )  # *.4#root_user = 34
    v0.run_VIA()

    super_labels = v0.labels

    #tsi_list = get_loc_terminal_states(via0=v0, X_input=adata_counts.obsm['X_pca'][:, 0:ncomps])
    v1 = VIA(adata_counts.obsm['X_pca'][:, 0:ncomps], true_label, jac_std_global=0.15, dist_std_local=1, knn=knn,
             too_big_factor=0.05, is_coarse=False,
             cluster_graph_pruning_std=cluster_graph_pruning_std,
             super_cluster_labels=super_labels, super_node_degree_list=v0.node_degree_list,
             super_terminal_cells=tsi_list, root_user=root_user, full_neighbor_array=v0.full_neighbor_array,
             full_distance_array=v0.full_distance_array, ig_full_graph=v0.ig_full_graph,
             csr_array_locally_pruned=v0.csr_array_locally_pruned,
             x_lazy=0.99, alpha_teleport=0.99, preserve_disconnected=True, dataset='bcell',
             super_terminal_clusters=v0.terminal_clusters, random_seed=random_seed)

    v1.run_VIA()


    # plot gene expression vs. pseudotime
    Bcell_marker_gene_list = ['Igll1', 'Myc', 'Slc7a5', 'Ldha', 'Foxo1', 'Lig4', 'Sp7', 'Zfp629']  # irf4 down-up
    df_ = pd.DataFrame(adata_counts_unfiltered.X)  # no normalization, or scaling of the gene count values
    df_.columns = [i for i in adata_counts_unfiltered.var_names]
    df_Bcell_marker = df_[Bcell_marker_gene_list]
    print(df_Bcell_marker.shape, 'df_Bcell_marker.shape')
    df_Bcell_marker.to_csv('/home/shobi/Trajectory/Datasets/Bcell/Bcell_markergenes.csv')

    df_magic = v0.do_impute(df_, magic_steps=3, gene_list=Bcell_marker_gene_list)
    v1.get_gene_expression(df_magic)
    plt.show()
    draw_trajectory_gams(via_coarse = v0, via_fine = v1, embedding =embedding)
    plt.show()

    draw_sc_lineage_probability(via_coarse = v0, via_fine = v1, embedding =embedding)

    plt.show()


def plot_EB():
    # genes along lineage cluster path
    df_groupby_p1 = pd.read_csv(
        '/home/shobi/Trajectory/Datasets/EB_Phate/df_groupbyParc1_knn20_pc100_seed20_allgenes.csv')

    path_clusters = [43, 38, 42, 56, 7,
                     3]  # NC[43,41,16,2,3,6]#SMP[43,41,16,14,11,18]#C[43,41,16,14,12,15]#NS3[43,38,42,56,7,3]
    target = "NS 3"  # 'NC 6' #'SMP 18'#' Cardiac 15'
    marker_genes_dict = {'Hermang': ['TAL1', 'HOXB4', 'SOX17', 'CD34', 'PECAM1'],
                         'NP': ['NES', 'MAP2'], 'NS': ['LHX2', 'NR2F1', 'DMRT3', 'LMX1A',
                                                       # 'KLF7', 'ISL1', 'DLX1', 'ONECUT1', 'ONECUT2', 'OLIG1','PAX6', 'ZBTB16','NPAS1', 'SOX1'
                                                       'NKX2-8', 'EN2'], 'NC': ['PAX3', 'FOXD3', 'SOX9', 'SOX10'],
                         'PostEn': ['CDX2', 'ASCL2', 'KLF5', 'NKX2-1'],
                         'EN': ['ARID3A', 'GATA3', 'SATB1', 'SOX15', 'SOX17', 'FOXA2'],
                         'Pre-NE': ['POU5F1', 'OTX2'], 'SMP': ['TBX18', 'SIX2', 'TBX15', 'PDGFRA'],
                         'Cardiac': ['TNNT2', 'HAND1', 'F3', 'CD82', 'LIFR'],
                         'EpiCard': ['WT1', 'TBX5', 'HOXD9', 'MYC', 'LOX'],
                         'PS/ME': ['T', 'EOMES', 'MIXL1', 'CER1', 'SATB1'],
                         'NE': ['GBX2', 'GLI3', 'LHX2', 'LHX5', 'SIX3', 'SIX6'],
                         # 'OLIG3','HOXD1', 'ZIC2', 'ZIC5','HOXA2','HOXB2'
                         'ESC': ['NANOG', 'POU5F1'], 'Pre-NE': ['POU5F1', 'OTX2'], 'Lat-ME': ['TBX5', 'HOXD9', 'MYC']}
    relevant_genes = []
    relevant_keys = ['ESC', 'Pre-NE', 'NE', 'NP',
                     'NS']  # NC['ESC', 'Pre-NE', 'NE', 'NC']#SMP['ESC','PS/ME','Lat-ME','SMP']#NS['ESC', 'Pre-NE', 'NE', 'NP', 'NS']
    dict_subset = {key: value for key, value in marker_genes_dict.items() if key in relevant_keys}
    print('dict subset', dict_subset)
    for key in relevant_keys:
        relevant_genes.append(marker_genes_dict[key])

    relevant_genes = [item for sublist in relevant_genes for item in sublist]

    print(relevant_genes)
    df_groupby_p1 = df_groupby_p1.set_index('parc1')
    df_groupby_p1 = df_groupby_p1.loc[path_clusters]
    df_groupby_p1 = df_groupby_p1[relevant_genes]

    df_groupby_p1 = df_groupby_p1.transpose()

    # print( df_groupby_p1.head)

    # print(df_groupby_p1)
    ax = sns.heatmap(df_groupby_p1, vmin=-1, vmax=1, yticklabels=True)

    ax.set_title('target ' + str(target))
    plt.show()

    # df_groupby_p1 = pd.concat([df_groupby_p1,df_groupby_p1])
    # adata = sc.AnnData(df_groupby_p1)
    # adata.var_names = df_groupby_p1.columns
    # print(adata.var_names)
    # adata.obs['parc1'] = ['43','38','42','56','7','3','43','38','42','56','7','3']
    # print(adata.obs['parc1'])
    # sc.pl.matrixplot(adata, dict_subset, groupby='parc1', vmax=1, vmin=-1, dendrogram=False)


def main_EB_clean(ncomps=30, knn=20, v0_random_seed=24, cluster_graph_pruning_std=.15,
                  foldername='/home/shobi/Trajectory/Datasets/EB_Phate/'):
    import phate
    true_time_labels = pd.read_csv(foldername+'EB_true_time_labels.csv')
    true_time_labels = true_time_labels.drop(['Unnamed: 0'], axis=1)
    true_time_labels = true_time_labels['true_time_labels']
    print('true time labels len', len(true_time_labels))

    #embedding_filename = 'viamds_singlediffusion_doExpTrue_k20_milestones5000_kprojectmilestones2t_stepmds2_knnmds50_kseqmds10_kseq10_nps30_tdiff1_randseed24_diffusionop5_rsMds42_615.csv' #geo corr k10 0.58348163 # euc corr 0.7580 #k5 geo corr 0.5724 # geocorr k3 -0.02
    #embedding_filename='viamds_singlediffusion_prescaled_doExpFalse_k10_milestones3000_kprojectmilestones2t_stepmds2_knnmds50_kseqmds5_kseq10_npc30_tdiff1_randseed24_diffusionop5_rsMds42_217.csv' #geocorr k10 #euc corr 0.6658 #: geo corr k15 0.5819
    embedding_filename='phate_scaled_npcphate_npc30_knn50.csv' #euc corr  0.6756920 #geo corr k15 0.58356
    #embedding_filename = 'phate_npcphate_npc50_knn20.csv' # geo corr 0.57782 k10 #euc corr 0.7508 #k5 geo corr: 0.5347 #=0.3
    #embedding_filename = 'viaumap_k50kseq10nps30tdiff1mindist0.1rs24stage087.csv' #euc corr 0.7403  #k5 geo corr:0.5174 #geocorr k3: -0.17
    #embedding_filename= 'umap_pc10_knn15.csv' #euc corr 0.63585 #geo corr k5 -0.287622 # geo corr k10 -0.3266 #geo corr k3 -0.0632
    print('embedding filename', embedding_filename)
    embedding = pd.read_csv(foldername+embedding_filename)
    embedding = embedding.drop(['Unnamed: 0'], axis=1)
    embedding=embedding.values
    print(embedding.shape)
    root_index = 1
    root_coords = np.array([[embedding[root_index,0], embedding[root_index,1]]])
    from scipy.spatial.distance import cdist
    U_distances_ = cdist(root_coords, embedding, metric='euclidean')
    print(f'U_distance {U_distances_.shape}')  # ,U_distances_.flatten().tolist())
    df_ = pd.DataFrame()
    df_['Udistances'] = U_distances_.flatten().tolist()
    df_['Udistances'] = df_['Udistances'].fillna(0)
    df_['true_time'] = true_time_labels

    correlation = df_['Udistances'].corr(df_['true_time'])
    print(f'correlation euclidean distances, {correlation}')
    geo_k = 15
    corr_val = corr_geodesic_distance_lowdim(embedding = embedding , knn=geo_k, time_labels=true_time_labels, root=root_index)
    print(f'geodesic distance corr val {corr_val} geo_knn {geo_k}')

    marker_genes_dict = {'Hermang': ['TAL1', 'HOXB4', 'SOX17', 'CD34', 'PECAM1'],
                         'NP': ['NES', 'MAP2'],
                         'NS': ['KLF7', 'ISL1', 'DLX1', 'ONECUT1', 'ONECUT2', 'OLIG1', 'NPAS1', 'LHX2', 'NR2F1',
                                'NPAS1', 'DMRT3', 'LMX1A',
                                'NKX2-8', 'EN2', 'SOX1', 'PAX6', 'ZBTB16'], 'NC': ['PAX3', 'FOXD3', 'SOX9', 'SOX10'],
                         'PostEn': ['CDX2', 'ASCL2', 'KLF5', 'NKX2-1'],
                         'EN': ['ARID3A', 'GATA3', 'SATB1', 'SOX15', 'SOX17', 'FOXA2'], 'Pre-NE': ['POU5F1', 'OTX2'],
                         'SMP': ['TBX18', 'SIX2', 'TBX15', 'PDGFRA'],
                         'Cardiac': ['TNNT2', 'HAND1', 'F3', 'CD82', 'LIFR'],
                         'EpiCard': ['WT1', 'TBX5', 'HOXD9', 'MYC', 'LOX'],
                         'PS/ME': ['T', 'EOMES', 'MIXL1', 'CER1', 'SATB1'],
                         'NE': ['GBX2', 'OLIG3', 'HOXD1', 'ZIC2', 'ZIC5', 'GLI3', 'LHX2', 'LHX5', 'SIX3', 'SIX6',
                                'HOXA2', 'HOXB2'], 'ESC': ['NANOG', 'POU5F1', 'OTX2'], 'Pre-NE': ['POU5F1', 'OTX2']}
    marker_genes_list = []
    for key in marker_genes_dict:
        for item in marker_genes_dict[key]:
            marker_genes_list.append(item)


    n_var_genes = 'no filtering for HVG'  # 15000


    # TI_pcs = pd.read_csv(foldername+'PCA_TI_200_final.csv')
    # TI_pcs is PCA run on data that has been: filtered (remove cells with too large or small library count - can directly use all cells in EBdata.mat), library normed, sqrt transform, scaled to unit variance/zero mean
    # TI_pcs = TI_pcs.values[:, 1:]

    from scipy.io import loadmat
    annots = loadmat(
        foldername + 'EBdata.mat')  # has been filtered but not yet normed (by library size) nor other subsequent pre-processing steps
    # print('annots', annots)
    data = annots['data'].toarray()  # (16825, 17580) (cells and genes have been filtered)
    # print('data min max', np.max(data), np.min(data), data[1, 0:20], data[5, 250:270], data[1000, 15000:15050])
    loc_ = np.where((data < 1) & (data > 0))
    temp = data[(data < 1) & (data > 0)]
    # print('temp non int', temp)
    print('annots', annots)
    print('annots', annots.keys())
    time_labels = annots['cells'].flatten().tolist()
    # df_timelabels = pd.DataFrame(time_labels, columns=['true_time_labels'])
    # df_timelabels.to_csv(foldername+'EB_true_time_labels.csv')

    gene_names_raw = annots['EBgenes_name']  # (17580, 1) genes

    adata = sc.AnnData(data)

    gene_names = []
    for i in gene_names_raw:
        gene_names.append(i[0][0])
    adata.var_names = gene_names
    adata.obs['time'] = ['Day' + str(i) for i in time_labels]

    adata.X = sc.pp.normalize_total(adata, inplace=False)['X']  # normalize by library after filtering
    adata.X = np.sqrt(adata.X)  # follow Phate paper which doesnt take log1() but instead does sqrt() transformation

    #Y_phate = pd.read_csv(foldername + 'EB_phate_embedding.csv')
    #Y_phate = Y_phate.values



    #Y_phate = phate_operator.fit_transform(adata.X)  # before scaling. as done in PHATE

    scale = False  # scaling mostly improves the cluster-graph heatmap of genes vs clusters. doesnt sway VIA performance
    if scale == True:  # we scale before VIA. scaling not needed for PHATE
        print('pp scaled')
        adata.X = (adata.X - np.mean(adata.X, axis=0)) / np.std(adata.X, axis=0)
        print('data max min after SCALED', np.max(adata.X), np.min(adata.X))
    else:

        print('not pp scaled')
    sc.tl.pca(adata, svd_solver='arpack', n_comps=200, random_state=0)

    v0_too_big = 0.15
    v1_too_big = 0.05
    print('ncomps, knn, n_var_genes, v0big, p1big, randomseed, time', ncomps, knn, n_var_genes, v0_too_big, v1_too_big,
          v0_random_seed, time.ctime())
    # adata.obsm['X_pca'] = TI_pcs
    for knn in [10,20,50]:
        for ncomps in [100]:
            input_data = adata.obsm['X_pca'][:, 0:ncomps]
            do_phate=False
            if do_phate:
                print('do phate')
                phate_operator = phate.PHATE(n_jobs=-1, n_pca=None,knn=knn)
                embedding_phate = phate_operator.fit_transform(input_data)
                df_phate = pd.DataFrame(embedding_phate)
                save_str = 'phate_npc'+str(ncomps)+'_knn'+str(knn)
                df_phate.to_csv('/home/shobi/Trajectory/Datasets/EB_Phate/phate_scaled_npc'+save_str+'.csv')
                f1, ax=plot_scatter(embedding=embedding_phate, labels=time_labels, title=save_str)
                f1.savefig('/home/shobi/Trajectory/Datasets/EB_Phate/phate_npc'+save_str+'.png', facecolor='white', transparent=False, )
                plt.show()

            print('do v0')
            root_user = [1]
            v0 = VIA(input_data, time_labels, jac_std_global=0.15, dist_std_local=1, knn=knn,
                     cluster_graph_pruning_std=cluster_graph_pruning_std, resolution_parameter=2,
                     too_big_factor=v0_too_big, root_user=root_user, dataset='EB', random_seed=v0_random_seed,
                      is_coarse=True, preserve_disconnected=True, do_compute_embedding=True, embedding_type='via-mds', time_series=True, time_series_labels=time_labels, do_gaussian_kernel_edgeweights=False, t_diff_step=1)  # *.4 root=1,
            v0.run_VIA()

            draw_piechart_graph(via_object = v0)
            plt.show()



    via_streamplot(v0, Y_phate)

    v1 = VIA(input_data, time_labels, jac_std_global=0.15, dist_std_local=1, knn=knn,
             cluster_graph_pruning_std=cluster_graph_pruning_std,
             too_big_factor=v1_too_big,  root_user=root_user, is_coarse=False,    x_lazy=0.95, alpha_teleport=0.99, preserve_disconnected=True, dataset='EB',
            random_seed=21, via_coarse=v0)

    v1.run_VIA()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.scatter(Y_phate[:, 0], Y_phate[:, 1], c=time_labels, s=5, cmap='viridis', alpha=0.5)
    ax2.scatter(Y_phate[:, 0], Y_phate[:, 1], c=v1.single_cell_pt_markov, s=5, cmap='viridis', alpha=0.5)
    ax1.set_title('Embryoid: Annotated Days')
    ax2.set_title('Embryoid VIA Pseudotime (Randomseed' + str(v0_random_seed) + ')')
    plt.show()

    draw_trajectory_gams(via_coarse=v0, via_fine=v1, embedding=Y_phate)
    plt.show()


    draw_sc_lineage_probability(via_coarse=v0, via_fine=v1, embedding=Y_phate)
    plt.show()

    adata.obs['via0'] = [str(i) for i in v0.labels]
    adata.obs['parc1'] = [str(i) for i in v1.labels]
    adata.obs['terminal_state'] = ['True' if i in v1.terminal_clusters else 'False' for i in v1.labels]
    adata.X = (adata.X - np.mean(adata.X, axis=0)) / np.std(adata.X,
                                                            axis=0)  # to improve scale of the matrix plot we will scale
    sc.pl.matrixplot(adata, marker_genes_dict, groupby='parc1', vmax=1, vmin=-1, dendrogram=True, figsize=[20, 10])


def main_EB(ncomps=30, knn=20, v0_random_seed=24):
    marker_genes_dict = {'Hermang': ['TAL1', 'HOXB4', 'SOX17', 'CD34', 'PECAM1'],
                         'NP': ['NES', 'MAP2'],
                         'NS': ['KLF7', 'ISL1', 'DLX1', 'ONECUT1', 'ONECUT2', 'OLIG1', 'NPAS1', 'LHX2', 'NR2F1',
                                'NPAS1', 'DMRT3', 'LMX1A',
                                'NKX2-8', 'EN2', 'SOX1', 'PAX6', 'ZBTB16'], 'NC': ['PAX3', 'FOXD3', 'SOX9', 'SOX10'],
                         'PostEn': ['CDX2', 'ASCL2', 'KLF5', 'NKX2-1'],
                         'EN': ['ARID3A', 'GATA3', 'SATB1', 'SOX15', 'SOX17', 'FOXA2'], 'Pre-NE': ['POU5F1', 'OTX2'],
                         'SMP': ['TBX18', 'SIX2', 'TBX15', 'PDGFRA'],
                         'Cardiac': ['TNNT2', 'HAND1', 'F3', 'CD82', 'LIFR'],
                         'EpiCard': ['WT1', 'TBX5', 'HOXD9', 'MYC', 'LOX'],
                         'PS/ME': ['T', 'EOMES', 'MIXL1', 'CER1', 'SATB1'],
                         'NE': ['GBX2', 'OLIG3', 'HOXD1', 'ZIC2', 'ZIC5', 'GLI3', 'LHX2', 'LHX5', 'SIX3', 'SIX6',
                                'HOXA2', 'HOXB2'], 'ESC': ['NANOG', 'POU5F1', 'OTX2'], 'Pre-NE': ['POU5F1', 'OTX2']}
    marker_genes_list = []
    for key in marker_genes_dict:
        for item in marker_genes_dict[key]:
            marker_genes_list.append(item)

    v0_too_big = 0.3
    v1_too_big = 0.05
    root_user = 1

    n_var_genes = 'no filtering for HVG'  # 15000
    print('ncomps, knn, n_var_genes, v0big, p1big, randomseed, time', ncomps, knn, n_var_genes, v0_too_big, v1_too_big,
          v0_random_seed, time.ctime())

    # data = data.drop(['Unnamed: 0'], axis=1)

    TI_pcs = pd.read_csv(
        '/home/shobi/Trajectory/Datasets/EB_Phate/PCA_TI_200_final.csv')  # filtered, library normed, sqrt transform, scaled to unit variance/zero mean
    TI_pcs = TI_pcs.values[:, 1:]

    umap_pcs = pd.read_csv('/home/shobi/Trajectory/Datasets/EB_Phate/PCA_umap_200_TuesAM.csv')
    umap_pcs = umap_pcs.values[:, 1:]
    # print('TI PC shape', TI_pcs.shape)
    from scipy.io import loadmat
    annots = loadmat(
        '/home/shobi/Trajectory/Datasets/EB_Phate/EBdata.mat')  # has been filtered but not yet normed (by library s
    data = annots['data'].toarray()  # (16825, 17580) (cells and genes have been filtered)
    # print('data min max', np.max(data), np.min(data), data[1, 0:20], data[5, 250:270], data[1000, 15000:15050])
    # loc_ = np.where((data < 1) & (data > 0))
    temp = data[(data < 1) & (data > 0)]
    # print('temp non int', temp)

    time_labels = annots['cells'].flatten().tolist()
    print('time labels set', set(time_labels))

    import scprep

    dict_labels = {'Day 00-03': 0, 'Day 06-09': 2, 'Day 12-15': 4, 'Day 18-21': 6, 'Day 24-27': 8}

    # print(annots.keys())  # (['__header__', '__version__', '__globals__', 'EBgenes_name', 'cells', 'data'])
    gene_names_raw = annots['EBgenes_name']  # (17580, 1) genes

    print(data.shape)

    adata = sc.AnnData(data)
    # time_labels = pd.read_csv('/home/shobi/Trajectory/Datasets/EB_Phate/labels_1.csv')
    # time_labels = time_labels.drop(['Unnamed: 0'], axis=1)
    # time_labels = time_labels['time']
    # adata.obs['time'] = [str(i) for i in time_labels]

    gene_names = []
    for i in gene_names_raw:
        gene_names.append(i[0][0])
    adata.var_names = gene_names
    adata.obs['time'] = [str(i) for i in time_labels]

    # filter_result = sc.pp.filter_genes_dispersion(adata.X, flavor='cell_ranger', n_top_genes=5000, log=False) #dont take log
    adata_umap = adata.copy()
    # adata = adata[:, filter_result.gene_subset]  # subset the genes
    # sc.pp.normalize_per_cell(adata, min_counts=2)  # renormalize after filtering
    print('data max min BEFORE NORM', np.max(adata.X), np.min(adata.X), adata.X[1, 0:20])
    rowsums = adata.X.sum(axis=1)
    # adata.X = adata.X / rowsums[:, np.newaxis]
    # adata.X = sc.pp.normalize_total(adata, exclude_highly_expressed=True, max_fraction=0.05, inplace=False)['X']  #normalize after filtering
    adata.X = sc.pp.normalize_total(adata, inplace=False)['X']  # normalize after filtering
    print('data max min after NORM', np.max(adata.X), np.min(adata.X), adata.X[1, 0:20])
    adata.X = np.sqrt(adata.X)  # follow Phate paper which doesnt take log1() but instead does sqrt() transformation
    adata_umap.X = np.sqrt(adata_umap.X)
    print('data max min after SQRT', np.max(adata.X), np.min(adata.X), adata.X[1, 0:20])
    # sc.pp.log1p(adata)  # log transform: adata.X = log(adata.X + 1)
    '''
    phate_operator = phate.PHATE(n_jobs=-1)

    Y_phate = phate_operator.fit_transform(adata.X)
    scprep.plot.scatter2d(Y_phate, c=time_labels, figsize=(12, 8), cmap="Spectral",
                          ticks=False, label_prefix="PHATE")
    plt.show()
    '''
    Y_phate = pd.read_csv('/home/shobi/Trajectory/Datasets/EB_Phate/EB_phate_embedding.csv')
    Y_phate = Y_phate.values
    scale = True
    if scale == True:
        print('pp scaled')
        # sc.pp.scale(adata)
        adata.X = (adata.X - np.mean(adata.X, axis=0)) / np.std(adata.X, axis=0)
        sc.pp.scale(adata_umap)
        print('data max min after SCALED', np.max(adata.X), np.min(adata.X))
    else:

        print('not pp scaled')

    print('sqrt transformed')
    # sc.pp.recipe_zheng17(adata, n_top_genes=15000) #expects non-log data
    # g = sc.tl.rank_genes_groups(adata, groupby='time', use_raw=True, n_genes=10)#method='t-test_overestim_var'
    # sc.pl.rank_genes_groups_heatmap(adata, n_genes=3, standard_scale='var')

    '''
    pcs = pd.read_csv('/home/shobi/Trajectory/Datasets/EB_Phate/umap_200_matlab.csv')
    pcs = pcs.drop(['Unnamed: 0'], axis=1)
    pcs = pcs.values
    print(time.ctime())
    ncomps = 50
    input_data =pcs[:, 0:ncomps]
    '''

    print('v0_toobig, p1_toobig, v0randomseed', v0_too_big, v1_too_big, v0_random_seed)
    print('do pca')
    # sc.tl.pca(adata, svd_solver='arpack', n_comps=200, random_state = 0)
    # sc.tl.pca(adata_umap, svd_solver='arpack', n_comps=200)
    # df_pca_TI_200 = pd.DataFrame(adata.obsm['X_pca'])
    # df_pca_TI_200.to_csv('/home/shobi/Trajectory/Datasets/EB_Phate/PCA_TI_200_TuesAM.csv')

    # df_pca_umap_200 = pd.DataFrame(adata_umap.obsm['X_pca'])
    # df_pca_umap_200.to_csv('/home/shobi/Trajectory/Datasets/EB_Phate/PCA_umap_200_TuesAM.csv')
    adata.obsm['X_pca'] = TI_pcs
    adata_umap.obsm['X_pca'] = umap_pcs

    input_data = adata.obsm['X_pca'][:, 0:ncomps]
    '''
    #plot genes vs clusters for each trajectory

    df_plot_gene = pd.DataFrame(adata.X, columns=[i for i in adata.var_names])
    df_plot_gene = df_plot_gene[marker_genes_list]

    previous_p1_labels = pd.read_csv('/home/shobi/Trajectory/Datasets/EB_Phate/df_labels_knn20_pc100_seed20.csv')

    title_str = 'Terminal state 27 (Cardiac)'
    gene_groups = ['ESC', 'PS/ME','EN','Cardiac']
    clusters = [43,41,16,14,12,27]
    '''

    u_knn = 15
    repulsion_strength = 1
    n_pcs = 10
    print('knn and repel', u_knn, repulsion_strength)
    #U = pd.read_csv('/home/shobi/Trajectory/Datasets/EB_Phate/umap_pc10_knn15.csv')
    #U = U.values[:, 1:]
    U = Y_phate

    print('do v0')
    v0 = VIA(input_data, time_labels, jac_std_global=0.15, dist_std_local=1, knn=knn,
             too_big_factor=v0_too_big, root_user=root_user, dataset='EB', random_seed=v0_random_seed, is_coarse=True, preserve_disconnected=True)  # *.4 root=1,
    v0.run_VIA()
    super_labels = v0.labels
    v0_labels_df = pd.DataFrame(super_labels, columns=['v0_labels'])
    #v0_labels_df.to_csv('/home/shobi/Trajectory/Datasets/EB_Phate/p0_labels.csv')
    adata.obs['via0'] = [str(i) for i in super_labels]
    '''
    df_temp1 = pd.DataFrame(adata.X, columns = [i for i in adata.var_names])
    df_temp1 = df_temp1[marker_genes_list]
    df_temp1['via0']=[str(i) for i in super_labels]
    df_temp1 = df_temp1.groupby('via0').mean()
    '''
    # sns.clustermap(df_temp1, vmin=-1, vmax=1,xticklabels=True, yticklabels=True, row_cluster= False, col_cluster=True)

    # sc.pl.matrixplot(adata, marker_genes_dict, groupby='via0', vmax=1, vmin =-1, dendrogram=True)
    '''
    sc.tl.rank_genes_groups(adata, groupby='via0', use_raw=True,
                            method='t-test_overestim_var', n_genes=5)  # compute differential expression
    sc.pl.rank_genes_groups_heatmap(adata, groupby='via0',vmin=-3, vmax=3)  # plot the result
    '''


    v1 = VIA(input_data, time_labels, jac_std_global=0.15, dist_std_local=1, knn=knn,
             too_big_factor=v1_too_big, is_coarse=False, root_user=root_user,
             x_lazy=0.95, alpha_teleport=0.99, preserve_disconnected=True, dataset='EB',
            random_seed=v0_random_seed, via_coarse=v0)

    v1.run_VIA()
    # adata.obs['parc1'] = [str(i) for i in v1.labels]
    # sc.pl.matrixplot(adata, marker_genes, groupby='parc1', dendrogram=True)
    labels = v1.labels


    adata.obs['parc1'] = [str(i) for i in labels]
    # df_ts = pd.DataFrame(adata.X, columns = [i for i in adata.var_names])
    # df_ts = df_ts[marker_genes_list]
    # df_ts['parc1'] =  [str(i) for i in labels]
    adata.obs['terminal_state'] = ['True' if i in v1.terminal_clusters else 'False' for i in labels]
    # df_ts = df_ts[df_ts['terminal_state']=='True']
    adata_TS = adata[adata.obs['terminal_state'] == 'True']
    # sns.clustermap(df_temp1, vmin=-1, vmax=1, xticklabels=True, yticklabels=True, row_cluster=False, col_cluster=True)
    sc.pl.matrixplot(adata, marker_genes_dict, groupby='parc1', vmax=1, vmin=-1, dendrogram=True)
    # sc.pl.matrixplot(adata_TS, marker_genes_dict, groupby='parc1', vmax=1, vmin=-1, dendrogram=True)

    # U = umap.UMAP(n_neighbors=10, random_state=0, repulsion_strength=repulsion_strength).fit_transform(input_data[:, 0:n_pcs])
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.scatter(U[:, 0], U[:, 1], c=time_labels, s=5, cmap='viridis', alpha=0.5)
    ax2.scatter(U[:, 0], U[:, 1], c=v1.single_cell_pt_markov, s=5, cmap='viridis', alpha=0.5)
    plt.title('repulsion and knn and pcs ' + str(repulsion_strength) + ' ' + str(u_knn) + ' ' + str(
        n_pcs) + ' randseed' + str(v0_random_seed))
    plt.show()

    super_clus_ds_PCA_loc = sc_loc_ofsuperCluster_PCAspace(v0, v1, np.arange(0, len(labels)))
    draw_trajectory_gams(via_coarse=v0, via_fine=v1, embedding = U)
    plt.show()

    draw_sc_lineage_probability(via_coarse=v0, via_fine=v1, embedding = U)

    plt.show()

    marker_genes = ['CD44', 'GATA4', 'PDGFRa', 'EpCAM']
    df_genes = pd.DataFrame(adata[:, marker_genes].X)
    df_genes.columns = marker_genes
    v0.get_gene_expression(gene_exp=df_genes)
    plt.show()
def main_mESC_timeseries(knn=40, cluster_graph_pruning_std = 0.15, random_seed = 0, knn_sequential=15,jac_std_global=0.5):
    root = [0.0]

    U= pd.read_csv('/home/shobi/Trajectory/Datasets/mESC/mESC_7000perDay_noscaling_meso_timeseries_viaumap_knn40_knnseq15_locallypruned.csv')
    U = U.values[:,1:]
    plt.scatter(U[:, 0], U[:,1], s=4, alpha=0.7)
    plt.show()
    data = pd.read_csv('/home/shobi/Trajectory/Datasets/mESC/mESC_7000perDay_noscaling_meso.csv')

    marker_meso = ['Sca-1', 'CD41', 'Nestin', 'Desmin', 'CD24', 'FoxA2', 'Oct4', 'CD45', 'Ki67', 'Vimentin',
                         'Cdx2', 'Nanog', 'pStat3-705', 'Sox2', 'Flk-1', 'Tuj1', 'H3K9ac', 'Lin28', 'PDGFRa', 'EpCAM',
                         'CD44', 'GATA4', 'Klf4', 'CCR9', 'p53', 'SSEA1', 'bCatenin', 'IdU']
    true_labels_numeric= data['day'].tolist()
    print('set true_labels', set(true_labels_numeric))
    data = data[marker_meso] #using the subset of markers in the original paper

    scale_arcsinh = 5
    raw = data.values
    raw = raw.astype(np.float)
    raw = raw / scale_arcsinh
    raw = np.arcsinh(raw)

    adata = sc.AnnData(raw)
    adata.var_names = data.columns
    adata.obs['day_str'] = [str(i) for i in true_labels_numeric]
    print(f"anndata shape {adata.shape}")
    print(f"jac std global {jac_std_global}")
    v0 = VIA(adata.X, true_labels_numeric, jac_std_global=jac_std_global, dist_std_local=1, knn=knn,
             cluster_graph_pruning_std=cluster_graph_pruning_std,
             too_big_factor=0.3, resolution_parameter=2,
             root_user=root, dataset='group', random_seed=random_seed,
             is_coarse=True, preserve_disconnected=False, pseudotime_threshold_TS=40, x_lazy=0.99,
             alpha_teleport=0.99, time_series=True, time_series_labels=true_labels_numeric, edgebundle_pruning=cluster_graph_pruning_std, edgebundle_pruning_twice=False, knn_sequential=knn_sequential, t_diff_step=2)

    v0.run_VIA()
    via_streamplot(v0, embedding=U[:, 0:2], scatter_size=50, scatter_alpha=0.2, marker_edgewidth=0.01,
                   density_stream=1, density_grid=0.5, smooth_transition=1, smooth_grid=0.3, use_sequentially_augmented=True)
    plt.show()
    marker_genes = ['CD44', 'GATA4', 'PDGFRa', 'EpCAM']
    df_genes = pd.DataFrame(adata[:, marker_genes].X)
    df_genes.columns = marker_genes
    v0.get_gene_expression(gene_exp=df_genes)
    plt.show()
    '''
    U = via_umap(X_input=v0.data, graph=v0.csr_full_graph, n_components=2, spread=1.0, min_dist=0.3,
                      init_pos='spectral', random_state=1, n_epochs=100)

    U_df = pd.DataFrame(U[:, 0:2])
    U_df.to_csv('/home/shobi/Trajectory/Datasets/mESC/mESC_7000perDay_noscaling_meso_timeseries_umap.csv')
    '''
    plt.scatter(U[:, 0], U[:, 1], c=v0.time_series_labels, cmap='jet', s=4, alpha=0.7)
    plt.show()
    draw_trajectory_gams(via_coarse=v0, via_fine=v0, embedding=U, draw_all_curves=False)
    plt.show()




def main_mESC(knn=30, v0_random_seed=42, cluster_graph_pruning_std=.0, run_palantir_func=False):
    import random
    rand_str = 950  # random.randint(1, 999)
    print('rand string', rand_str)
    print('knn', knn)

    data_random_seed = 20
    root = ['0.0'] #corresponds to the group label '0.0' days, not an index!!
    type_germ = 'Meso'
    normalize = True
    data = pd.read_csv('/home/shobi/Trajectory/Datasets/mESC/mESC_' + type_germ + '_markers.csv')
    print('counts', data.groupby('day').count())
    # print(data.head())
    print(data.shape)
    n_sub = 7000
    print('type,', type_germ, 'nelements', n_sub, 'v0 randseed', v0_random_seed)
    title_string = 'randstr:' + str(rand_str) + ' Knn' + str(knn) + ' nelements:' + str(n_sub) + ' ' + 'meso'
    # data = data[data['day']!=0]

    v0_too_big = 0.3
    p1_too_big = 0.15  # .15
    print('v0 and p1 too big', v0_too_big, p1_too_big)
    data_sub = data[data['day'] == 0.0]
    np.random.seed(data_random_seed)
    idx_sub = np.random.choice(a=np.arange(0, data_sub.shape[0]), size=min(n_sub, data_sub.shape[0]), replace=False,
                               p=None)  # len(true_label)
    data_sub = data_sub.values[idx_sub, :]
    data_sub = pd.DataFrame(data_sub, columns=data.columns)
    for i in [1.0, 2, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]:
        sub = data[data['day'] == i]
        print(sub.shape[0])
        np.random.seed(data_random_seed)
        idx_sub = np.random.choice(a=np.arange(0, sub.shape[0]), size=min(n_sub, sub.shape[0]), replace=False,
                                   p=None)  # len(true_label)
        sub = sub.values[idx_sub, :]
        print('size of subpopulation',i,'day', sub.shape)
        sub = pd.DataFrame(sub, columns=data.columns)
        data_sub = pd.concat([data_sub, sub], axis=0, ignore_index=True, sort=True)

    true_label = data_sub['day']

    true_type = data_sub['type']
    #data_sub = data_sub.drop([ 'Unnamed: 0'], axis=1)
    #data_sub.to_csv('/home/shobi/Trajectory/Datasets/mESC/mESC_7000perDay_noscaling_meso.csv')
    data = data_sub.drop(['day', 'Unnamed: 0', 'type'], axis=1)
    # print('after subbing', data.head)
    cols = ['Sca-1', 'CD41', 'Nestin', 'Desmin',
            'CD24', 'FoxA2', 'Oct4', 'CD45', 'Ki67', 'Vimentin',
            'Nanog', 'pStat3-705', 'Sox2', 'Flk-1', 'Tuj1',
            'H3K9ac', 'Lin28', 'PDGFRa', 'EpCAM', 'CD44',
            'GATA4', 'Klf4', 'CCR9', 'p53', 'SSEA1', 'IdU', 'Cdx2']  # 'bCatenin'

    meso_good = ['CD24', 'FoxA2', 'Oct4', 'CD45', 'Ki67', 'Vimentin', 'Cdx2', 'CD54', 'pStat3-705', 'Sox2', 'Flk-1',
                 'Tuj1', 'SSEA1', 'H3K9ac', 'Lin28', 'PDGFRa', 'bCatenin', 'EpCAM', 'CD44', 'GATA4', 'Klf4', 'CCR9',
                 'p53']
    marker_genes_ecto = ['Oct4', 'Nestin', 'CD45', 'Vimentin', 'Cdx2', 'Flk-1', 'PDGFRa', 'CD44',
                         'GATA4', 'CCR9', 'CD54', 'CD24', 'CD41', 'Tuji']
    marker_genes_meso_paper_sub = ['Oct4', 'CD54', 'SSEA1', 'Lin28', 'Cdx2', 'CD45', 'Nanog', 'Sox2', 'Flk-1', 'Tuj1',
                                   'PDGFRa', 'EpCAM', 'CD44', 'CCR9', 'GATA4']

    marker_genes_meso_paper = ['Nestin', 'FoxA2', 'Oct4', 'CD45', 'Sox2', 'Flk-1', 'Tuj1', 'PDGFRa', 'EpCAM', 'CD44',
                               'GATA4', 'CCR9', 'Nanog', 'Cdx2', 'Vimentin']  # 'Nanog''Cdx2','Vimentin'
    marker_genes_endo = ['Sca-1''Nestin', 'CD45', 'Vimentin', 'Cdx2', 'Flk-1', 'PDGFRa', 'CD44',
                         'GATA4', 'CCR9', 'CD54', 'CD24', 'CD41', 'Oct4']
    marker_genes_meso = ['Sca-1', 'CD41', 'Nestin', 'Desmin', 'CD24', 'FoxA2', 'Oct4', 'CD45', 'Ki67', 'Vimentin',
                         'Cdx2', 'Nanog', 'pStat3-705', 'Sox2', 'Flk-1', 'Tuj1', 'H3K9ac', 'Lin28', 'PDGFRa', 'EpCAM',
                         'CD44', 'GATA4', 'Klf4', 'CCR9', 'p53', 'SSEA1', 'bCatenin', 'IdU']  # ,'c-Myc'

    marker_dict = {'Ecto': marker_genes_ecto, 'Meso': marker_genes_meso, 'Endo': marker_genes_meso}
    marker_genes = marker_dict[type_germ]
    data = data[marker_genes]

    print('marker genes ', marker_genes)

    pre_fac_scale = [4, 1,
                     1]  # 4,1,1 (4,1,1 is used in the paper but actually no scaling factor is really required, the results are unperturbed
    pre_fac_scale_genes = ['H3K9ac', 'Lin28', 'Oct4']
    for pre_scale_i, pre_gene_i in zip(pre_fac_scale, pre_fac_scale_genes):
        data[pre_gene_i] = data[pre_gene_i] / pre_scale_i
        print('prescaled gene', pre_gene_i, 'by factor', pre_scale_i)

    scale_arcsinh = 5
    raw = data.values
    raw = raw.astype(np.float)
    raw_df = pd.DataFrame(raw, columns=data.columns)

    raw = raw / scale_arcsinh
    raw = np.arcsinh(raw)
    # print(data.shape, raw.shape)

    adata = sc.AnnData(raw)
    adata.var_names = data.columns
    # print(adata.shape, len(data.columns))

    true_label_int = [i for i in true_label]
    adata.obs['day'] = ['0' + str(i) if i < 10 else str(i) for i in true_label_int]
    true_label_str = [str(i) for i in
                      true_label_int]  # the way find_root works is to match any part of root-user to majority truth


    if normalize == True:
        sc.pp.scale(adata, max_value=5)
        print(colored('normalized', 'blue'))
    else:
        print(colored('NOT normalized', 'blue'))
    print('adata', adata.shape)
    # ncomps = 30

    # sc.tl.pca(adata, svd_solver='arpack', n_comps=ncomps)
    n_umap = adata.shape[0]

    np.random.seed(data_random_seed)

    udata = adata.X[:, :][0:n_umap]

    # U = umap.UMAP().fit_transform(udata)
    # U_df = pd.DataFrame(U, columns=['x', 'y'])
    # U_df.to_csv('/home/shobi/Trajectory/Datasets/mESC/umap_89782cells_meso.csv')
    idx = np.arange(0, adata.shape[  0])  # np.random.choice(a=np.arange(0, adata.shape[0]), size=adata.shape[0], replace=False, p=None)  # len(true_label)
    # idx=np.arange(0, len(true_label_int))
    U = pd.read_csv(        '/home/shobi/Trajectory/Datasets/mESC/umap_89782cells_meso.csv')  # umap_89782cells_7000each_Randseed20_meso.csv')
    # U = pd.read_csv('/home/shobi/Trajectory/Datasets/mESC/phate_89782cells_mESC.csv')
    U = U.values[0:len(true_label), 1:]
    plt.scatter(U[:, 0], U[:, 1], c=true_label, cmap='jet', s=4, alpha=0.7)
    plt.show()
    '''
    for gene_i in ['CD44', 'GATA4', 'PDGFRa', 'EpCAM']:
        # subset = adata[[gene_i]].values #scale is not great so hard to visualize on the raw data expression
        subset = adata[:, gene_i].X.flatten()

        plt.scatter(U[:, 0], U[:, 1], c=subset, cmap='viridis', s=4, alpha=0.7)
        plt.title(gene_i)
        plt.show()
    '''
    print(U.shape)
    # U_df = pd.read_csv('/home/shobi/Trajectory/Datasets/mESC/phate_89782cells_mESC.csv')
    # U = U_df.drop('Unnamed: 0', 1)
    U = U[idx, :]
    # subsample start
    n_subsample = len(true_label_int)  # 50000 #palantir wont scale
    U = U[0:n_subsample, :]

    # phate_operator = phate.PHATE(n_jobs=-1)

    # Y_phate = phate_operator.fit_transform(adata.X)
    # phate_df = pd.DataFrame(Y_phate)
    # phate_df.to_csv('/home/shobi/Trajectory/Datasets/mESC/phate_89782cells_mESC.csv')
    true_label_int0 = list(np.asarray(true_label_int))
    # Start Slingshot data prep
    '''
    slingshot_annots = true_label_int0[0:n_umap]
    slingshot_annots = [int(i) for i in slingshot_annots]
    Slingshot_annots = pd.DataFrame(slingshot_annots,columns = ['label'])
    Slingshot_annots.to_csv('/home/shobi/Trajectory/Datasets/mESC/Slingshot_annots_int_10K_sep.csv')


    Slingshot_data = pd.DataFrame(adata.X[0:n_umap], columns=marker_genes)
    Slingshot_data.to_csv('/home/shobi/Trajectory/Datasets/mESC/Slingshot_input_data_10K_sep.csv')
    # print('head sling shot data', Slingshot_data.head)
    # print('head sling shot annots', Slingshot_annots.head)

    print('slingshot data shape', Slingshot_data.shape)
    # sling_adata =sc.AnnData(Slingshot_data)
    '''
    # end Slingshot data prep
    adata = adata[idx]
    true_label_int = list(np.asarray(true_label_int)[idx])
    true_label_int = true_label_int[0:n_subsample]

    true_label_str = list(np.asarray(true_label_str)[idx])
    true_label_str = true_label_str[0:n_subsample]
    true_type = list(np.asarray(true_type)[idx])
    true_type = list(np.asarray(true_type)[idx])[0:n_subsample]
    sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
    # plt.scatter(sling_adata.obsm['X_pca'][:,0],sling_adata.obsm['X_pca'][:,1], c = Slingshot_annots['label'])
    plt.show()
    print('time', time.ctime())
    loc_start = np.where(np.asarray(true_label_int) == 0)[0][0]
    adata.uns['iroot'] = loc_start
    print('iroot', loc_start)
    # Start PAGA
    '''
    sc.pp.neighbors(adata, n_neighbors=knn, n_pcs=28)  # 4
    sc.tl.draw_graph(adata)
    # sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data')  # force-directed layout
    start_dfmap = time.time()
    sc.tl.diffmap(adata, n_comps=28)
    print('time taken to get diffmap given knn', time.time() - start_dfmap)
    #sc.pp.neighbors(adata, n_neighbors=knn, use_rep='X_diffmap')  # 4
    #sc.tl.draw_graph(adata)
    sc.tl.leiden(adata, resolution=1.0, random_state=10)
    sc.tl.paga(adata, groups='leiden')
    adata.obs['group_id'] = true_label_int
    # sc.pl.paga(adata_counts, color=['louvain','group_id'])

    sc.tl.dpt(adata, n_dcs=28)
    print('time paga end', time.ctime())
    plt.show()

    df_paga_dpt = pd.DataFrame()
    df_paga_dpt['paga_dpt'] = adata.obs['dpt_pseudotime'].values
    df_paga_dpt['days'] = true_label_int
    df_paga_dpt.to_csv('/home/shobi/Trajectory/Datasets/mESC/paga_dpt_knn' + str(knn) + '.csv')

    sc.pl.paga(adata, color=['leiden', 'group_id', 'dpt_pseudotime'],
               title=['leiden',  'group_id', 'pseudotime'])
    plt.show()
    # sc.pl.matrixplot(adata, marker_genes_meso, groupby='day', dendrogram=True)
    '''
    # end PAGA
    '''
    #start palantir run
    t_pal_start = time.time()
    run_palantir_mESC(adata[0:n_subsample:], knn=knn, tsne=U, str_true_label = true_label_str)
    print('palantir run time', round(time.time() - t_pal_start))
    df_palantir = pd.read_csv('/home/shobi/Trajectory/Datasets/mESC/palantir_pt.csv')
    df_palantir['days'] = true_label_int
    df_palantir.to_csv('/home/shobi/Trajectory/Datasets/mESC/palantir_pt.csv')
    '''
    # df_ = pd.DataFrame(adata.X)
    # df_.columns = [i for i in adata.var_names]
    # df_.to_csv('/home/shobi/Trajectory/Datasets/mESC/transformed_normalized_input.csv')
    #df_ = pd.DataFrame(true_label_int, columns=['days'])
    #    df_.to_csv('/home/shobi/Trajectory/Datasets/mESC/annots_days.csv')

    time_series_labels = np.asarray(true_label_int)
    print('time series labels',time_series_labels.shape, time_series_labels[3])
    print('check root is in annots', root[0] in true_label_str)
    v0 = VIA(adata.X, true_label_str, jac_std_global=0.3, dist_std_local=1, knn=knn,
             cluster_graph_pruning_std=cluster_graph_pruning_std,
             too_big_factor=v0_too_big, resolution_parameter=2,
             root_user=root, dataset='mESC', random_seed=v0_random_seed,
             is_coarse=True, preserve_disconnected=False, pseudotime_threshold_TS=40, x_lazy=0.99,
             alpha_teleport=0.99, time_series=True, time_series_labels=true_label_int, edgebundle_pruning=cluster_graph_pruning_std, edgebundle_pruning_twice=False) # visual_cluster_graph_pruning=1, max_visual_outgoing_edges=3,
    v0.run_VIA()

    draw_trajectory_gams(via_coarse=v0, via_fine=v0, embedding=U)
    plt.show()
    via_streamplot(v0, embedding=U[:, 0:2], scatter_size=50, scatter_alpha=0.2, marker_edgewidth=0.01,
                      density_stream=1, density_grid=0.5, smooth_transition=1, smooth_grid=0.3)
    plt.show()


    U = via_umap(X_input=v0.data, graph=v0.csr_full_graph, n_components=2, spread=1.0, min_dist=0.1,
                           init_pos='spectral', random_state=1, n_epochs=100)
    plt.scatter(U[:, 0], U[:, 1], c=v0.time_series_labels,cmap='jet', s=4, alpha=0.7)
    plt.show()

    draw_trajectory_gams(via_coarse=v0, via_fine=v0, embedding=U)
    plt.show()
    via_streamplot(v0, embedding=U[:, 0:2], scatter_size=50, scatter_alpha=0.2, marker_edgewidth=0.01,
                   density_stream=1, density_grid=0.5, smooth_transition=1, smooth_grid=0.3)
    plt.show()

    df_pt = v0.single_cell_pt_markov
    f, (ax1, ax2,) = plt.subplots(1, 2, sharey=True)
    s_genes = ''
    for s in marker_genes:
        s_genes = s_genes + ' ' + s
    plt.title(str(len(true_label)) + 'cells ' + str(title_string) + '\n marker genes:' + s_genes, loc='left')
    ax1.scatter(U[:, 0], U[:, 1], c=true_label_int, cmap='jet', s=4, alpha=0.7)
    ax2.scatter(U[:, 0], U[:, 1], c=df_pt, cmap='jet', s=4, alpha=0.7)

    df_pt = pd.DataFrame()
    df_pt['via_knn'] = v0.single_cell_pt_markov
    df_pt['days'] = true_label_int
    #df_pt.to_csv('/home/shobi/Trajectory/Datasets/mESC/noMCMC_nolazynotele_via_pt_knn_Feb2021' + str(    knn) + 'resolution2jacp15.csv')
    adata.obs['via0'] = [str(i) for i in v0.labels]
    # show geneplot
    # sc.pl.matrixplot(adata, marker_genes, groupby='via0', dendrogram=True)
    marker_genes = ['CD44', 'GATA4', 'PDGFRa', 'EpCAM']
    df_genes = pd.DataFrame(adata[:, marker_genes].X)
    df_genes.columns = marker_genes
    v0.get_gene_expression(gene_exp=df_genes)
    plt.show()
    '''
    #show geneplot
    for gene_i in ['CD44', 'GATA4', 'PDGFRa', 'EpCAM']:
        # subset = data[[gene_i]].values
        subset = adata[:, gene_i].X.flatten()
        print('gene expression for', gene_i)
        v0.get_gene_expression(subset, gene_i)
    plt.show()
    '''
    draw_trajectory_gams(via_coarse=v0, via_fine=v0, embedding=U)
    plt.show()
    v1 = VIA(adata.X, true_label_str, jac_std_global=0.15, dist_std_local=1, knn=knn,
             cluster_graph_pruning_std=cluster_graph_pruning_std,
             too_big_factor=p1_too_big, root_user=root, is_coarse=False,
             x_lazy=0.99, alpha_teleport=0.99, preserve_disconnected=True, dataset='mESC',
             visual_cluster_graph_pruning=1, max_visual_outgoing_edges=3,
             random_seed=v0_random_seed,    pseudotime_threshold_TS=40, via_coarse=v0)
    v1.run_VIA()
    df_pt['via_v1'] = v1.single_cell_pt_markov
    #df_pt.to_csv('/home/shobi/Trajectory/Datasets/mESC/noMCMC_nolazynotele_via_pt_knn_Feb2021' + str(        knn) + 'resolution2jacp15.csv')
    adata.obs['parc1'] = [str(i) for i in v1.labels]
    sc.pl.matrixplot(adata, marker_genes, groupby='parc1', dendrogram=True)

    v1.get_gene_expression(gene_exp=df_genes)
    # X = adata.obsm['X_pca'][:,0:2]
    # print(X.shape)

    c_pt = v1.single_cell_pt_markov[0:n_umap]
    c_type = true_type[0:n_umap]
    dict_type = {'EB': 0, 'Endo': 5, "Meso": 10, 'Ecto': 15}
    c_type = [dict_type[i] for i in c_type]
    u_truelabel = true_label_int[0:n_umap]

    # U = umap.UMAP().fit_transform(adata.obsm['X_pca'][idx, 0:ncomps])
    # U = Y_phate[idx,:]

    print('umap done', rand_str, time.ctime())
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    s_genes = ''
    for s in marker_genes:
        s_genes = s_genes + ' ' + s
    plt.title(str(len(true_label)) + 'cells ' + str(title_string) + '\n marker genes:' + s_genes, loc='left')
    ax1.scatter(U[:, 0], U[:, 1], c=true_label_int, cmap='jet', s=4, alpha=0.7)
    ax2.scatter(U[:, 0], U[:, 1], c=c_pt, cmap='jet', s=4, alpha=0.7)
    ax3.scatter(U[:, 0], U[:, 1], c=c_type, cmap='jet', s=4, alpha=0.7)

    plt.show()

    draw_trajectory_gams(via_coarse=v0, via_fine=v1, embedding = U)
    # draw_sc_lineage_probability(v1, U, knn_hnsw, v0.full_graph_shortpath, np.arange(0, n_umap))
    plt.show()


def main_scATAC_zscores(knn=20, ncomps=30, cluster_graph_pruning_std=.15):
    # datasets can be downloaded from the link below
    # https://nbviewer.jupyter.org/github/pinellolab/STREAM/blob/master/tutorial/archives/v0.4.1_and_earlier_versions/4.STREAM_scATAC-seq_k-mers.ipynb?flush_cache=true

    # these are the kmers based feature matrix
    # https://www.dropbox.com/sh/zv6z7f3kzrafwmq/AACAlU8akbO_a-JOeJkiWT1za?dl=0
    # https://github.com/pinellolab/STREAM_atac

    ##KMER START

    df = pd.read_csv("/home/shobi/Trajectory/Datasets/scATAC_Hemato/zscore_scaled_kmer.tsv",
                     sep='\t')  # TF Zcores from STREAM NOT the original buenrostro corrected PCs
    df = df.transpose()
    print('df kmer size', df.shape)
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row
    df.columns = new_header  # set the header row as the df header
    df = df.apply(pd.to_numeric)  # CONVERT ALL COLUMNS
    true_label = pd.read_csv("/home/shobi/Trajectory/Datasets/scATAC_Hemato/cell_label.csv", sep='\t')
    true_label = true_label['cell_type'].values
    cell_types = ['GMP', 'HSC', 'MEP', 'CLP', 'CMP', 'LMPP', 'MPP', 'pDC', 'mono']
    cell_dict = {'pDC': 'purple', 'mono': 'yellow', 'GMP': 'orange', 'MEP': 'red', 'CLP': 'aqua',
                 'HSC': 'black', 'CMP': 'moccasin', 'MPP': 'darkgreen', 'LMPP': 'limegreen'}

    ### KMER end

    ### for MOTIFS start
    '''
    df = pd.read_csv("/home/shobi/Trajectory/Datasets/scATAC_Hemato/pinellolab_chromVAR_buenrostro_motifs_noHSC0828.csv",sep=',')  # TF Zcores from STREAM NOT the original buenrostro corrected PCs
    cell_annot = df["Unnamed: 0"].values
    df = df.drop('Unnamed: 0', 1)

    print('nans', np.sum(df.isna().sum()))
    df = df.interpolate()
    print('nans', df.isna().sum())
    #df = pd.read_csv("/home/shobi/Trajectory/Datasets/scATAC_Hemato/zscore_scaled_transpose.csv",sep=',')
    print(df.head, df.shape)

    cell_types = ['GMP', 'HSC', 'MEP', 'CLP', 'CMP', 'LMuPP', 'MPP', 'pDC', 'mono', 'UNK']
    cell_dict = {'pDC': 'purple', 'mono': 'yellow', 'GMP': 'orange', 'MEP': 'red', 'CLP': 'aqua',
                 'HSC': 'black', 'CMP': 'moccasin', 'MPP': 'darkgreen', 'LMuPP': 'limegreen','UNK':'gray'}

    true_label = []
    found_annot=False
    count = 0
    for annot in cell_annot:
        for cell_type_i in cell_types:
            if cell_type_i in annot:
                true_label.append(cell_type_i)
                if found_annot== True: print('double count', annot)
                found_annot = True

        if found_annot ==False:
            true_label.append('unknown')
            print('annot is unknown', annot)
            count = count+1
        found_annot=False
    '''
    ## FOR MOTIFS end

    print('true label', true_label)
    print('len true label', len(true_label))
    df_Buen = pd.read_csv('/home/shobi/Trajectory/Datasets/scATAC_Hemato/scATAC_hemato_Buenrostro.csv', sep=',')

    # df.to_csv("/home/shobi/Trajectory/Datasets/scATAC_Hemato/zscore_scaled_transpose.csv")

    df = df.reset_index(drop=True)
    df_num = df.values
    df_num = pd.DataFrame(df_num, columns=new_header)
    print('df_num', df_num.head)
    df_num = df_num.apply(pd.to_numeric)
    df_num['true'] = true_label
    print(df_num.groupby('true', as_index=False).mean())

    print('df', df.head(), df.shape)
    print(df.columns.tolist()[0:10])
    for i in ['AGATAAG', 'CCTTATC']:
        if i in df.columns: print(i, ' is here')

    ad = sc.AnnData(df)
    ad.var_names = df.columns
    ad.obs['cell_type'] = true_label
    sc.tl.pca(ad, svd_solver='arpack', n_comps=300)
    color = []
    for i in true_label:
        color.append(cell_dict[i])
    # PCcol = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
    # embedding = umap.UMAP(n_neighbors=15, random_state=2, repulsion_strength=0.5).fit_transform(ad.obsm['X_pca'][:, 1:5])
    # embedding = umap.UMAP(n_neighbors=20, random_state=2, repulsion_strength=0.5).fit_transform(df_Buen[PCcol])
    # df_umap = pd.DataFrame(embedding)
    # df_umap.to_csv('/home/shobi/Trajectory/Datasets/scATAC_Hemato/embedding_5PC.csv')
    embedding = pd.read_csv('/home/shobi/Trajectory/Datasets/scATAC_Hemato/embedding_5PC.csv')
    embedding = embedding.drop('Unnamed: 0', 1)
    embedding = embedding.values
    # idx = np.random.choice(a=np.arange(0, len(cell_annot)), size=len(cell_annot), replace=False,p=None)

    gene_dict = {'ENSG00000165702_LINE1058_GFI1B_D': 'GFI1B', 'ENSG00000162676_LINE1033_GFI1_D_N2': 'GFI1',
                 'ENSG00000114853_LINE833_ZBTB47_I': 'cDC', 'ENSG00000105610_LINE802_KLF1_D': 'KLF1 (MkP)',
                 'ENSG00000119919_LINE2287_NKX23_I': "NKX32", 'ENSG00000164330_LINE251_EBF1_D_N2': 'EBF1 (Bcell)',
                 'ENSG00000157554_LINE1903_ERG_D_N3': 'ERG', 'ENSG00000185022_LINE531_MAFF_D_N1': 'MAFF',
                 'ENSG00000124092_LINE881_CTCFL_D': 'CTCF', 'ENSG00000119866_LINE848_BCL11A_D': 'BCL11A',
                 'ENSG00000117318_LINE151_ID3_I': 'ID3', 'ENSG00000078399_LINE2184_HOXA9_D_N1': 'HOXA9',
                 'ENSG00000078399_LINE2186_HOXA9_D_N1': 'HOXA9', 'ENSG00000172216_LINE498_CEBPB_D_N8': 'CEBPB',
                 'ENSG00000123685_LINE392_BATF3_D': 'BATF3', 'ENSG00000102145_LINE2081_GATA1_D_N7': 'GATA1',
                 'ENSG00000140968_LINE2752_IRF8_D_N2': "IRF8", "ENSG00000140968_LINE2754_IRF8_D_N1": "IRF8"}

    gene_dict = {'ENSG00000164330_LINE251_EBF1_D_N2': 'EBF1 (Bcell)',
                 'ENSG00000123685_LINE392_BATF3_D': 'BATF3', 'ENSG00000102145_LINE2081_GATA1_D_N7': 'GATA1',
                 'ENSG00000140968_LINE2752_IRF8_D_N2': "IRF8", "ENSG00000140968_LINE2754_IRF8_D_N1": "IRF8"}

    print('end umap')
    '''
    fig, ax = plt.subplots()
    for key in cell_dict:
        loc = np.where(np.asarray(true_label) == key)[0]
        if key == 'LMPP':
            alpha = 0.7
        else:
            alpha = 0.55
        ax.scatter(embedding[loc, 0], embedding[loc, 1], c=cell_dict[key], alpha=alpha, label=key)
        if key == 'HSC':
            for loci in loc:
                if (loci % 10) == 0:
                    print(loci)
                    #ax.text(embedding[loci, 0], embedding[loci, 1], 'c' + str(loci))
    plt.legend()
    plt.show()
    '''
    knn = knn
    random_seed = 4  # 4
    # df[PCcol]
    ncomps = ncomps + 1
    start_ncomp = 1
    root = [1200]  # 1200#500nofresh
    # df_pinello = pd.DataFrame(ad.obsm['X_pca'][:, 0:200])
    # df_pinello.to_csv('/home/shobi/Trajectory/Datasets/scATAC_Hemato/pc200_streamTFZscore.csv')

    # run palantir

    X_in = ad.obsm['X_pca'][:, start_ncomp:ncomps]
    # X_in = df.values
    print('root, pc', root, ncomps)

    # palantir_scATAC_Hemato(X_in, knn,embedding, true_label)
    plt.show()
    jac_std_global = 0.3
    df_genes = pd.DataFrame()
    for key in gene_dict:
        subset_ = df_Buen[key].values
        subset_ = (subset_ - subset_.mean()) / subset_.std()
        # print('subset shape', subset_.shape)
        df_genes[key] = subset_
    df_genes.to_csv('/home/shobi/Trajectory/Datasets/scATAC_Hemato/df_magic_scatac_hemato.csv')

    # palantir_scATAC_Hemato(X_in, knn, embedding, true_label, df_genes)
    v0 = VIA(X_in, true_label, jac_std_global=0.3, dist_std_local=1, knn=knn,
             cluster_graph_pruning_std=cluster_graph_pruning_std,
             too_big_factor=0.3, root_user=root, dataset='scATAC', random_seed=random_seed,
             visual_cluster_graph_pruning=.15, max_visual_outgoing_edges=2,is_coarse=True, preserve_disconnected=False)  # *.4 root=1,
    v0.run_VIA()


    df['via0'] = v0.labels
    # df_v0 = pd.DataFrame(v0.labels)
    # df_v0['truth'] = cell_annot
    # df_v0.to_csv('/home/shobi/Trajectory/Datasets/scATAC_Hemato/viav0_labels.csv')
    df_mean = df.groupby('via0', as_index=False).mean()
    # v0.draw_piechart_graph( type_pt='gene', gene_exp=df_mean['AGATAAG'].values, title='GATA1')
    # plt.show()


    v1 = VIA(X_in, true_label, jac_std_global=0.15, dist_std_local=1, knn=knn,
             cluster_graph_pruning_std=cluster_graph_pruning_std,
             too_big_factor=0.05, root_user=root, is_coarse=False,
             preserve_disconnected=True, dataset='scATAC',
             visual_cluster_graph_pruning=.15, max_visual_outgoing_edges=2, random_seed=random_seed, via_coarse=v0)
    v1.run_VIA()
    df['via1'] = v1.labels
    df_Buen['via1'] = v1.labels
    df_mean = df_Buen.groupby('via1', as_index=False).mean()
    df_mean = df.groupby('via1', as_index=False).mean()

    '''
    for key in gene_dict:
        
        v1.draw_piechart_graph( type_pt='gene', gene_exp=df_mean[key].values, title=gene_dict[key])
        plt.show()
    '''
    # get knn-graph and locations of terminal states in the embedded space
    draw_trajectory_gams(v0,v1,embedding)
    plt.show()
    marker_genes =[key for key in gene_dict]
    df_genes = df_Buen[marker_genes]
    df_genes.columns = [gene_dict[key] for key in gene_dict]
    for col in df_genes.columns:
        df_genes[col] = (df_genes[col] - df_genes[col].mean()) / df_genes[col].std()
    v1.get_gene_expression(gene_exp=df_genes)
    '''
    for key in gene_dict:
        subset_ = df_Buen[key].values
        subset_ = (subset_ - subset_.mean()) / subset_.std()
        # print('subset shape', subset_.shape)
        df[key] = subset_
        v0.get_gene_expression(subset_, title_gene=gene_dict[key])
    '''

    plt.show()
    # draw trajectory and evolution probability for each lineage
    draw_sc_lineage_probability(via_coarse=v0, via_fine=v1, embedding=embedding)
    plt.show()

'''
def palantir_scATAC_Hemato(pc_array, knn, tsne, str_true_label, df_magic):
    t0 = time.time()
    norm_df_pal = pd.DataFrame(pc_array)
    # print('norm df', norm_df_pal)

    new = ['c' + str(i) for i in norm_df_pal.index]

    # loc_start = np.where(np.asarray(str_true_label) == 'T1_M1')[0][0]
    start_cell = 'c1200'
    print('start cell', start_cell)
    norm_df_pal.index = new

    ncomps = norm_df_pal.values.shape[1]
    print('palantir ncomps', ncomps)
    dm_res = palantir.utils.run_diffusion_maps(norm_df_pal, knn=knn, n_components=ncomps)

    ms_data = palantir.utils.determine_multiscale_space(dm_res)  # n_eigs is determined using eigengap.
    ## In this case n_eigs becomes 1. to increase sensitivity then increase n_eigs in this dataset
    print('ms data', ms_data.shape)
    tsne = pd.DataFrame(tsne, columns=['x', 'y'])  # palantir.utils.run_tsne(ms_data)
    tsne.index = new

    str_true_label = pd.Series(str_true_label, index=norm_df_pal.index)
    print('c1200 in str true label', str_true_label['c1200'], str_true_label['c369'], str_true_label['c928'])
    palantir.plot.plot_cell_clusters(tsne, str_true_label)

    # start_cell = 'c4823'  # '#C108 for M12 connected' #M8n1000d1000 start - c107 #c1001 for bifurc n2000d1000 #disconnected n1000 c108, "C1 for M10 connected" # c10 for bifurcating_m4_n2000d1000
    num_waypoints = 1200  # 1200  # 1200 default
    pr_res = palantir.core.run_palantir(ms_data, early_cell=start_cell, num_waypoints=num_waypoints, knn=knn)
    print('time end palantir', round(time.time() - t0))
    palantir.plot.plot_palantir_results(pr_res, tsne, knn, ncomps)
    gene_dict = {'ENSG00000164330_LINE251_EBF1_D_N2': 'EBF1 (Bcell)',
                 'ENSG00000123685_LINE392_BATF3_D': 'BATF3', 'ENSG00000102145_LINE2081_GATA1_D_N7': 'GATA1',
                 "ENSG00000140968_LINE2754_IRF8_D_N1": "IRF8"}
    genes = [key for key in gene_dict]
    # norm_df_pal = (norm_df_pal - norm_df_pal.mean()) / norm_df_pal.std()
    # df_magic = (df_magic - df_magic.mean()) / df_magic.std()
    df_magic.index = new
    gene_trends = palantir.presults.compute_gene_trends(pr_res, df_magic.loc[:, genes])
    palantir.plot.plot_gene_trends(gene_trends)
    plt.show()
'''

def main_scATAC_Hemato(knn=20):
    # buenrostro PCs and TF scores that have been "adjusted" for nice output by Buenrostro
    # hematopoietic stem cells (HSC) also multipotent progenitors (MPP), lymphoid-primed multipotent progenitors (LMuPP),
    # common lymphoid progenitors (CLP), and megakaryocyte-erythrocyte progenitors (MEP)
    # common myeloid progenitors (CMP), granulocyte-monocyte progenitors (GMP), Unknown (UNK)
    # NOTE in the original "cellname" it was LMPP which overlaped in string detection with MPP. So we changed LMPP to LMuPP
    # cell_types = ['GMP', 'HSC-frozen','HSC-Frozen','HSC-fresh','HSC-LS','HSC-SIM','MEP','CLP', 'CMP','LMuPP','MPP','pDC','mono','UNK']
    # cell_dict = {'UNK':'aqua','pDC':'purple','mono':'orange','GMP':'orange','MEP':'red','CLP':'aqua', 'HSC-frozen':'black','HSC-Frozen':'black','HSC-fresh':'black','HSC-LS':'black','HSC-SIM':'black', 'CMP':'moccasin','MPP':'darkgreen','LMuPP':'limegreen'}
    cell_types = ['GMP', 'HSC', 'MEP', 'CLP', 'CMP', 'LMuPP', 'MPP', 'pDC', 'mono', 'UNK']
    cell_dict = {'UNK': 'gray', 'pDC': 'purple', 'mono': 'gold', 'GMP': 'orange', 'MEP': 'red', 'CLP': 'aqua',
                 'HSC': 'black', 'CMP': 'moccasin', 'MPP': 'darkgreen', 'LMuPP': 'limegreen'}
    df = pd.read_csv('/home/shobi/Trajectory/Datasets/scATAC_Hemato/scATAC_hemato_Buenrostro.csv', sep=',')
    print('df', df.shape)
    cell_annot = df['cellname'].values
    print('len of cellname', len(cell_annot))

    true_label = []
    count = 0
    found_annot = False
    for annot in cell_annot:
        for cell_type_i in cell_types:
            if cell_type_i in annot:
                true_label.append(cell_type_i)
                if found_annot == True: print('double count', annot)
                found_annot = True

        if found_annot == False:
            true_label.append('unknown')
            count = count + 1
        found_annot = False
    print('abbreviated annot', len(true_label), 'number of unknowns', count)

    df_true = pd.DataFrame(true_label)
    # df_true.to_csv('/home/shobi/Trajectory/Datasets/scATAC_Hemato/scATAC_hemato_Buenrostro_truelabel.csv')
    PCcol = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
    '''
    allcols = df.columns.tolist()
    for i in ['cellname', 'R_color', 'G_color','B_color','PC1','PC2','PC3','PC4','PC5']:
        allcols.remove(i)

    df_TF=df[allcols]
    print('df_TF shape', df_TF.shape)
    ad_TF = sc.AnnData(df_TF)
    ad_TF.obs['celltype']=true_label

    sc.pp.scale(ad_TF)
    #sc.pp.normalize_per_cell(ad_TF)
    sc.tl.pca(ad_TF, svd_solver='arpack', n_comps=300)
    df_PCA_TF = ad_TF.obsm['X_pca'][:, 0:300]
    df_PCA_TF=pd.DataFrame(df_PCA_TF)
    df_PCA_TF.to_csv('/home/shobi/Trajectory/Datasets/scATAC_Hemato/300PC_TF.csv')
    '''
    df['celltype'] = true_label
    print('counts', df.groupby('celltype').count())

    color = []
    for i in true_label:
        color.append(cell_dict[i])

    print('start UMAP')
    # embedding = umap.UMAP(n_neighbors=20, random_state=2, repulsion_strength=0.5).fit_transform(df[PCcol])
    # df_umap = pd.DataFrame(embedding)
    # df_umap.to_csv('/home/shobi/Trajectory/Datasets/scATAC_Hemato/embedding_5PC.csv')
    embedding = pd.read_csv('/home/shobi/Trajectory/Datasets/scATAC_Hemato/embedding_5PC.csv')
    embedding = embedding.drop('Unnamed: 0', 1)
    embedding = embedding.values
    # idx = np.random.choice(a=np.arange(0, len(cell_annot)), size=len(cell_annot), replace=False,p=None)
    print('end umap')

    fig, ax = plt.subplots()
    for key in cell_dict:
        loc = np.where(np.asarray(true_label) == key)[0]
        if key == 'LMuPP':
            alpha = 0.8
        else:
            alpha = 0.55
        ax.scatter(embedding[loc, 0], embedding[loc, 1], c=cell_dict[key], alpha=alpha, label=key, s=90)
        if key == 'HSC':
            for loci in loc:
                if (loci % 20) == 0:
                    print(loci)
                    # ax.text(embedding[loci, 0], embedding[loci, 1], 'c' + str(loci))
    plt.legend(fontsize='large', markerscale=1.3)
    plt.show()

    knn = knn
    random_seed = 3  # 2
    X_in = df[PCcol].values  # ad_TF.obsm['X_pca'][:, start_ncomp:ncomps]

    start_ncomp = 0
    root = [1200]  # 1200#500nofresh

    # palantir_scATAC_Hemato(X_in, knn, embedding, true_label)

    v0 = VIA(X_in, true_label, jac_std_global=0.5, dist_std_local=1, knn=knn,
             too_big_factor=0.3, root_user=root, dataset='scATAC', random_seed=random_seed,
             visual_cluster_graph_pruning=.15, max_visual_outgoing_edges=2,is_coarse=True, preserve_disconnected=False)  # *.4 root=1,
    v0.run_VIA()
    #Make JSON FILES for interactive graph
    #v0.make_JSON(filename='scATAC_BuenrostroPC_temp.js')
    df['via0'] = v0.labels
    df_mean = df.groupby('via0', as_index=False).mean()

    gene_dict_long = {'ENSG00000092067_LINE336_CEBPE_D_N1': 'CEBPE Eosophil (GMP/Mono)',
                 'ENSG00000102145_LINE2081_GATA1_D_N7': 'GATA1 (MEP)', 'ENSG00000105610_LINE802_KLF1_D': 'KLF1 (MkP)',
                 'ENSG00000119919_LINE2287_NKX23_I': "NKX32", 'ENSG00000164330_LINE251_EBF1_D_N2': 'EBF1 (Bcell)',
                 'ENSG00000157554_LINE1903_ERG_D_N3': 'ERG', 'ENSG00000185022_LINE531_MAFF_D_N1': 'MAFF',
                 'ENSG00000124092_LINE881_CTCFL_D': 'CTCF', 'ENSG00000119866_LINE848_BCL11A_D': 'BCL11A',
                 'ENSG00000117318_LINE151_ID3_I': 'ID3', 'ENSG00000078399_LINE2184_HOXA9_D_N1': 'HOXA9',
                 'ENSG00000078399_LINE2186_HOXA9_D_N1': 'HOXA9', 'ENSG00000172216_LINE498_CEBPB_D_N8': 'CEBPB',
                 'ENSG00000123685_LINE392_BATF3_D': 'BATF3', 'ENSG00000140968_LINE2752_IRF8_D_N2': "IRF8",
                 "ENSG00000140968_LINE2754_IRF8_D_N1": "IRF8"}
    gene_dict = {'ENSG00000164330_LINE251_EBF1_D_N2': 'EBF1 (Bcell)',
                 'ENSG00000123685_LINE392_BATF3_D': 'BATF3', 'ENSG00000102145_LINE2081_GATA1_D_N7': 'GATA1',
                 'ENSG00000140968_LINE2752_IRF8_D_N2': "IRF8_N2", "ENSG00000140968_LINE2754_IRF8_D_N1": "IRF8_N1"}
    marker_genes = [key for key in gene_dict]
    df_genes = df[marker_genes]
    df_genes.columns = [gene_dict[key] for key in gene_dict]
    #for col in df_genes.columns:
        #df_genes[col] = (df_genes[col] - df_genes[col].mean()) / df_genes[col].std()
    v0.get_gene_expression(gene_exp=df_genes)
    plt.show(    )
    draw_trajectory_gams(v0, v0, embedding)
    plt.show()
    gene_interest = 'ENSG00000123685_LINE392_BATF3_D'
    v0.draw_piechart_graph(data_type='gene', gene_exp=df_mean[gene_interest], title=gene_dict[gene_interest])
    plt.show()

    v1 = VIA(X_in, true_label, jac_std_global=0.15, dist_std_local=1, knn=knn,
             too_big_factor=0.1,root_user=root, is_coarse=False,
             preserve_disconnected=True, dataset='',  visual_cluster_graph_pruning=.15, max_visual_outgoing_edges=2,random_seed=random_seed, via_coarse=v0  )
    v1.run_VIA()
    df['via1'] = v1.labels
    df_mean = df.groupby('via1', as_index=False).mean()

    for key in gene_dict:
        v1.draw_piechart_graph(type_pt='gene', gene_exp=df_mean[key].values, title=gene_dict[key])
        plt.show()

    # draw overall pseudotime and main trajectories
    draw_trajectory_gams(v0,v1,embedding)
    # draw trajectory and evolution probability for each lineage
    draw_sc_lineage_probability(v0,v1,embedding)
    plt.show()
    return


def via_wrapper(adata, true_label=None, embedding=None, knn=20, jac_std_global=0.15, root=0, dataset='', random_seed=42,
                v0_toobig=0.3, v1_toobig=0.1, marker_genes=[], ncomps=20, preserve_disconnected=False,
                cluster_graph_pruning_std=0.15, draw_all_curves=True, piegraph_edgeweight_scalingfactor=1.5,
                piegraph_arrow_head_width=0.2,edgebundle_pruning=None,edgebundle_pruning_twice=False):
    v0 = VIA(adata.obsm['X_pca'][:, 0:ncomps], true_label, jac_std_global=jac_std_global, dist_std_local=1, knn=knn,
             too_big_factor=v0_toobig, root_user=root, dataset=dataset, random_seed=random_seed, is_coarse=True, preserve_disconnected=preserve_disconnected,
             cluster_graph_pruning_std=cluster_graph_pruning_std, piegraph_arrow_head_width=piegraph_arrow_head_width,
             piegraph_edgeweight_scalingfactor=piegraph_edgeweight_scalingfactor, visual_cluster_graph_pruning=0.15,
             max_visual_outgoing_edges=2, edgebundle_pruning_twice=edgebundle_pruning_twice, edgebundle_pruning=edgebundle_pruning)  # *.4 root=1,
    v0.run_VIA()
    if true_label is None:
        true_label = v0.true_label

    if len(marker_genes) > 0:
        df_ = pd.DataFrame(adata.X)
        df_.columns = [i for i in adata.var_names]
        df_magic = v0.do_impute(df_, magic_steps=3, gene_list=marker_genes)
        v0.get_gene_expression(gene_exp=df_magic[marker_genes])

    plt.show()
    draw_sc_lineage_probability(via_coarse=v0, via_fine=v0, embedding=embedding, )

    # plot coarse cluster heatmap
    draw_trajectory_gams(via_coarse=v0, via_fine=v0, embedding=embedding, draw_all_curves=draw_all_curves)
    plt.show()

    via_streamplot(v0, embedding=embedding[:, 0:2], scatter_size=400, scatter_alpha=0.2, marker_edgewidth=0.01,
                      density_stream=2, density_grid=0.5, smooth_transition=1, smooth_grid=0.3)
    plt.show()


    if len(marker_genes) > 0:
        adata.obs['via0'] = [str(i) for i in v0.labels]
        sc.pl.matrixplot(adata, marker_genes, groupby='via0', dendrogram=True)
        plt.show()


    v1 = VIA(adata.obsm['X_pca'][:, 0:ncomps], true_label, jac_std_global=jac_std_global, dist_std_local=1, knn=knn,
             too_big_factor=v1_toobig,
              root_user=root, is_coarse=False,
             preserve_disconnected=preserve_disconnected, dataset=dataset,
             random_seed=random_seed,
              cluster_graph_pruning_std=cluster_graph_pruning_std,
             piegraph_edgeweight_scalingfactor=piegraph_edgeweight_scalingfactor, via_coarse=v0, piegraph_arrow_head_width=piegraph_arrow_head_width)

    v1.run_VIA()
    # plot fine cluster heatmap

    if len(marker_genes) > 0:
        adata.obs['parc1'] = [str(i) for i in v1.labels]
        sc.pl.matrixplot(adata, marker_genes, groupby='parc1', dendrogram=True)
        plt.show()


    draw_trajectory_gams(via_coarse=v0, via_fine=v1, embedding=embedding,draw_all_curves=draw_all_curves)

    # draw trajectory and evolution probability for each lineage
    draw_sc_lineage_probability(v0,v1,embedding)

    if len(marker_genes) > 0:
        df_ = pd.DataFrame(adata.X)
        df_.columns = [i for i in adata.var_names]
        df_magic = v0.do_impute(df_, magic_steps=3, gene_list=marker_genes)
        v1.get_gene_expression(gene_exp=df_magic[marker_genes])
        plt.show()
    return


def via_wrapper_disconnected(adata, true_label=None, embedding=None, knn=20, jac_std_global=0.15, root=[1], dataset='',
                             random_seed=42, v0_toobig=0.3, marker_genes=[], ncomps=20, preserve_disconnected=True,
                             cluster_graph_pruning_std=0.15):
    v0 = VIA(adata.obsm['X_pca'][:, 0:ncomps], true_label, jac_std_global=jac_std_global, dist_std_local=1, knn=knn,
             too_big_factor=v0_toobig, root_user=root, dataset=dataset, random_seed=random_seed,is_coarse=True, preserve_disconnected=preserve_disconnected,
             cluster_graph_pruning_std=cluster_graph_pruning_std)  # *.4 root=1,
    v0.run_VIA()
    if true_label is None: true_label = v0.true_label
    # plot coarse cluster heatmap
    via_streamplot(v0, embedding)
    if len(marker_genes) > 0:
        adata.obs['via0'] = [str(i) for i in v0.labels]
        sc.pl.matrixplot(adata, marker_genes, groupby='via0', dendrogram=True)
        plt.show()

    '''
    # get the terminal states so that we can pass it on to the next iteration of Via to get a fine grained pseudotime
    tsi_list = get_loc_terminal_states(v0, adata.X)

    # get knn-graph and locations of terminal states in the embedded space
    knn_hnsw = make_knn_embeddedspace(embedding)
    super_clus_ds_PCA_loc = sc_loc_ofsuperCluster_PCAspace(v0, v0, np.arange(0, len(v0.labels)))
    # draw overall pseudotime and main trajectories
    draw_trajectory_gams(embedding, super_clus_ds_PCA_loc, v0.labels, v0.labels, v0.edgelist_maxout,
                         v0.x_lazy, v0.alpha_teleport, v0.single_cell_pt_markov, true_label, knn=v0.knn,
                         final_super_terminal=v0.terminal_clusters,
                         sub_terminal_clusters=v0.terminal_clusters,
                         title_str='Pseudotime', ncomp=ncomps)
    # draw trajectory and evolution probability for each lineage
    draw_sc_lineage_probability(v0, embedding, knn_hnsw, v0.full_graph_shortpath,
                                          np.arange(0, len(true_label)),
                                          adata.X)
    '''
    draw_trajectory_gams(via_coarse=v0, via_fine=v0, embedding = embedding)
    plt.show()
    draw_sc_lineage_probability(via_coarse=v0, via_fine=v0, embedding = embedding)
    plt.show()
    if len(marker_genes) > 0:
        df_ = pd.DataFrame(adata.X)
        df_.columns = [i for i in adata.var_names]
        df_magic = v0.do_impute(df_, magic_steps=3, gene_list=marker_genes)
        v0.get_gene_expression(df_magic)
        plt.show()
    return


def paga_faced(ad, knn, embedding, true_label, cell_line='mcf7'):
    from matplotlib.ticker import FormatStrFormatter
    print('inside paga ad.shape', ad.shape)
    print(ad.var_names)
    adata_counts = ad.copy()
    adata_counts.obs['group_id'] = true_label
    true_time = [int(i[-1]) for i in true_label]
    ncomps = adata_counts.X.shape[1]
    # print('adata X', adata_counts.X)
    # sc.tl.pca(adata_counts, svd_solver='arpack', n_comps=ncomps)
    if cell_line == 'mcf7':
        # root_label = 'T1_M1'
        root_label = 1592
    else:
        # root_label = 'T2_M1'
        root_label = 725
    adata_counts.uns[
        'iroot'] = root_label  # np.where(np.asarray(adata_counts.obs['group_id']) == root_label)[0][  0]  # 'T1_M1' for mcf7

    sc.pp.neighbors(adata_counts, use_rep='X',
                    n_neighbors=knn)  # use_rep = 'paga_X' when excludindg vol features  #n_pcs. Use this many PCs. If n_pcs==0 use .X if use_rep is None.
    # sc.tl.draw_graph(adata_counts)
    # sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data')  # force-directed layout
    start_dfmap = time.time()
    sc.tl.diffmap(adata_counts, n_comps=ncomps)
    print('time taken to get diffmap given knn', time.time() - start_dfmap)
    sc.pp.neighbors(adata_counts, n_neighbors=knn, use_rep='X_diffmap')  # 4
    # sc.tl.draw_graph(adata_counts)
    # sc.pl.draw_graph(adata_counts, color='group_id', legend_loc='on data')
    sc.tl.leiden(adata_counts, resolution=1.0)
    sc.tl.paga(adata_counts, groups='leiden')

    # sc.pl.paga(adata_counts, color=['louvain','group_id'])

    sc.tl.dpt(adata_counts, n_dcs=ncomps)
    # sc.pl.paga(adata_counts, node_size_scale=3, color=['group_id'])

    sc.pl.paga(adata_counts, node_size_scale=3, color=['leiden', 'group_id', 'dpt_pseudotime'],
               title=['leiden (knn:' + str(knn) + ' ncomps:' + str(ncomps) + ')',
                      'group_id (ncomps:' + str(ncomps) + ')', cell_line + ' pseudotime (ncomps:' + str(ncomps) + ')'])
    # sc.pl.draw_graph(adata_counts, color='dpt_pseudotime', legend_loc='on data')
    # print('dpt format', adata_counts.obs['dpt_pseudotime'])
    pt = adata_counts.obs['dpt_pseudotime'].values
    import scipy.stats as stats
    true_time = [int(i[-1]) for i in true_label]
    correlation, p_value = stats.pearsonr(pt, true_time)
    print('paga pearson', correlation, p_value)
    correlation, p_value = stats.kendalltau(pt, true_time)
    print('paga kendall', correlation, p_value)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=adata_counts.obs['dpt_pseudotime'].values, cmap='viridis')
    plt.title('PAGA DPT')
    plt.show()
    # start trend plotting
    genes = ['Volume']
    labels = genes
    gene_lineage_dict = {'Volume': 'G2'}

    paths_mcf7_ex_vol = [('G2', [20, 15, 5, 2, 9, 3, 0, 21, 6, 17, 13, 8, 1, 12, 11, 4, 7, 14, 18]),
                         ('G2.1', [20, 15, 5, 2, 9, 3, 0, 21, 6, 17, 13, 8, 1, 12, 11, 4, 7, 14])]
    paths_mb231_ex_vol = [('G2', [24, 21, 6, 12, 11, 8, 14, 3, 19, 15, 18, 10, 1, 2, 17, 23, 7, 0, 5, 25, 4, 20, 16]), (
    'GExtra.1', [24, 21, 6, 12, 11, 8, 14, 3, 19, 15, 18, 10, 1, 2, 17, 23, 7, 0, 5, 25, 4, 20, 16])]
    paths_mcf7 = [('G2', [24, 9, 13, 8, 4, 5, 16, 1, 6, 14, 19, 2, 21, 12, 18, 0, 3, 20, 15, 17, 7, 10, 22]),
                  ('GExtra.1', [24, 9, 13, 8, 4, 5, 16, 1, 6, 14, 19, 2, 21, 12, 18, 0, 3, 20, 15, 17, 7, 10, 22])]
    paths = [('G2', [8, 15, 2, 20, 1, 9, 7, 22, 0, 13, 6, 12, 3, 11, 25, 16, 24]),
             ('GExtra.1', [8, 15, 2, 20, 1, 9, 7, 22, 0, 13, 6, 12, 3, 11, 25, 16, 24])]
    trends = pd.Series()

    _, axs = plt.subplots(ncols=len(paths), figsize=(6, 2.5), gridspec_kw={'wspace': 0.05, 'left': 0.11})
    plt.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.2)
    for ipath, (descr, path) in enumerate(paths):
        _, data = sc.pl.paga_path(
            adata_counts, path, genes,
            show_node_names=False,
            ax=axs[ipath],
            ytick_fontsize=12,
            left_margin=0.15,
            n_avg=50,
            show_yticks=True if ipath == 0 else False,
            show_colorbar=False,
            color_map='Greys',
            title='path to\n{} fate'.format(descr[:-1]),
            return_data=True,
            show=False)
        trends[descr] = data
    plt.show()

    i = 1
    fig = plt.figure(figsize=[10, 10])
    for gene, label in zip(genes, labels):

        ax = plt.gca()

        for l in trends.keys():
            print('l', l)
            order = trends[l].distance.sort_values().index
            print('order', len(order), order)

            bins = np.ravel(trends[l].distance[order])
            t = np.ravel(trends[l].loc[order, gene])
            print(len(t), t)
            print('gene', gene)

            # Plot
            if gene_lineage_dict[gene] == l:
                ax = fig.add_subplot(1, 1, i);
                i = i + 1
                ax.scatter(bins, t, s=5)
                correlation, p_value = stats.pearsonr(bins, t)
                print('pearson', correlation, p_value)
                correlation, p_value = stats.kendalltau(bins, t)
                print('kendeall', correlation, p_value)
        ax.set_title(label)
        ax.set_xlabel('Pseudo-time ordering', fontsize=10)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        sns.despine()
    plt.show()

'''
def run_palantir_faced(ad, knn, tsne, str_true_label, cell_line, df_magic):
    import palantir
    t0 = time.time()
    norm_df_pal = pd.DataFrame(ad.X)
    # print('norm df', norm_df_pal)
    new = ['c' + str(i) for i in norm_df_pal.index]

    if cell_line == 'mcf7':
        loc_start = 1592  # 2905
    else:
        loc_start = 725  # 725 is a edge G1 cell in mb231 #500#np.where(np.asarray(str_true_label) == 'T1_M1')[0][0] c2905 G1 cell in mcf7 (T1_M1)
    start_cell = 'c' + str(loc_start)
    print('start cell', start_cell)
    norm_df_pal.index = new
    df_magic.index = new
    norm_df_pal.columns = [i for i in ad.var_names]
    ncomps = norm_df_pal.values.shape[1]
    print('palantir ncomps', ncomps)
    dm_res = palantir.utils.run_diffusion_maps(norm_df_pal, knn=knn, n_components=ncomps)

    ms_data = palantir.utils.determine_multiscale_space(dm_res)  # n_eigs is determined using eigengap
    print('ms data', ms_data.shape)
    tsne = pd.DataFrame(tsne, columns=['x', 'y'])  # palantir.utils.run_tsne(ms_data)
    tsne.index = new
    # print(type(tsne))
    tsne = palantir.utils.run_tsne(ms_data)
    str_true_label = pd.Series(str_true_label, index=norm_df_pal.index)
    palantir.plot.plot_cell_clusters(tsne, str_true_label)

    # start_cell = 'c4823'  # '#C108 for M12 connected' #M8n1000d1000 start - c107 #c1001 for bifurc n2000d1000 #disconnected n1000 c108, "C1 for M10 connected" # c10 for bifurcating_m4_n2000d1000
    num_waypoints = 1200  # 1200 default
    pr_res = palantir.core.run_palantir(ms_data, early_cell=start_cell, num_waypoints=num_waypoints, knn=knn)
    print('time end palantir', round(time.time() - t0))
    palantir.plot.plot_palantir_results(pr_res, tsne, knn, ncomps)
    df_palantir = pd.read_csv(
        '/home/shobi/Trajectory/Datasets/Toy3/palantir_pt.csv')  # /home/shobi/anaconda3/envs/ViaEnv/lib/python3.7/site-packages/palantir
    true_time = [int(i[-1]) for i in str_true_label]
    pt = df_palantir['pt']
    import scipy.stats as stats
    correlation, p_value = stats.pearsonr(pt, true_time)
    print('palantir pearson', correlation, p_value)
    correlation, p_value = stats.kendalltau(pt, true_time)
    print('palantir kendall', correlation, p_value)
    genes = ['Volume', 'Area', 'Phase Entropy Skewness']
    # norm_df_pal = (norm_df_pal - norm_df_pal.mean()) / norm_df_pal.std()
    df_magic = (df_magic - df_magic.mean()) / df_magic.std()
    gene_trends = palantir.presults.compute_gene_trends(pr_res, df_magic.loc[:, genes])
    palantir.plot.plot_gene_trends(gene_trends)
    plt.show()
'''

def main_faced(cell_line='mcf7', cluster_graph_pruning_std=1.):
    def faced_histogram():
        import seaborn as sns
        # penguins = sns.load_dataset('penguins')
        # print(penguins)

        df_output = pd.read_csv('/home/shobi/Trajectory/Datasets/FACED/pt_histogram_mcf7_all.csv')
        # bins = np.linspace(min(df_output['pt']), max(df_output['pt']), 100)

        df_output['pt'] = df_output['pt'] * 10
        pt_all = df_output['pt'].values
        g1 = df_output[df_output['stage'] == 'G1']['pt'].values
        s = df_output[df_output['stage'] == 'S']['pt'].values
        g2 = df_output[df_output['stage'] == 'G2']['pt'].values

        # sns.histplot(df_output, x = 'pt', kde=True, hue= 'stage',bins=50, stat='count', color='darkblue',multiple='stack',   kde_kws={'bw_adjust': 2, 'clip': (0.0, 1.0)})
        # plt.show()

        from scipy.stats import norm
        # plt.hist(pt_all, bins=50, alpha=0.8, label='all', color='lightgrey',density = True)# weights=np.ones_like(pt_all) / len(pt_all))
        plt.hist(g1, bins=50, alpha=0.8, label='G1', color='gold',
                 density=True)  # weights = np.ones_like(g1) / len(g1))
        mu, std = norm.fit(g1)
        # xmin, xmax = plt.xlim()
        g1min = min(g1)
        g1max = max(g1)
        x = np.linspace(g1min, g1max, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, color='gold', linewidth=2)
        plt.legend(loc='upper right')
        plt.show()

        plt.hist(s, bins=50, alpha=0.8, label='S', color='green', density=True)  # weights = np.ones_like(s) / len(s))
        mu, std = norm.fit(s)
        # xmin, xmax = plt.xlim()
        g1min = min(s)
        g1max = max(s)
        x = np.linspace(g1min, g1max, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, color='green')
        plt.legend(loc='upper right')
        plt.show()
        # plt.hist(s, bins, alpha=0.8, label='S',color='green',density=True)
        # plt.hist(g2, bins, alpha=0.9, label='G2/M',color='blue',density=True)

        plt.hist(g2, bins=50, alpha=0.8, label='G2', color='lightblue',
                 density=True)  # ,weights = np.ones_like(g2) / len(g2))
        mu, std = norm.fit(g2)
        # xmin, xmax = plt.xlim()
        g1min = min(g2)
        g1max = max(g2)
        x = np.linspace(g1min, g1max, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, color='blue')
        plt.legend(loc='upper right')
        plt.show()
        # plt.hist(s, bins, alpha=0.8, label='S',color='green',density=True)
        # plt.hist(g2, bins, alpha=0.9, label='G2/M',color='blue',density=True)

        '''
        bin_heights_g1, bin_borders_g1, _ = plt.hist(g1, bins='auto', label='G1', color='yellow')
        bin_heights_g2, bin_borders_g2, _ = plt.hist(g2, bins='auto', label='G2/M', color='yellow')
        bin_heights_s, bin_borders_s, _ = plt.hist(s, bins='auto', label='S', color='yellow')
        bin_centers_g1 = bin_borders_g1[:-1] + np.diff(bin_borders_g1) / 2
        bin_centers_g2 = bin_borders_g2[:-1] + np.diff(bin_borders_g2) / 2
        bin_centers_s = bin_borders_s[:-1] + np.diff(bin_borders_s) / 2
        from scipy.optimize import curve_fit
        def gaussian(x, mean, amplitude, standard_deviation):
            return amplitude * np.exp(- (x - mean) ** 2 / (2 * standard_deviation ** 2))
        popt_g1, _ = curve_fit(gaussian, bin_centers_g1, bin_heights_g1, p0=[1., 0., 1.])
        popt_g2, _ = curve_fit(gaussian, bin_centers_g2, bin_heights_g2, p0=[1., 0., 1.])
        popt_s, _ = curve_fit(gaussian, bin_centers_s, bin_heights_s, p0=[1., 0., 1.])

        x_interval_for_fit_g1 = np.linspace(bin_borders_g1[0], bin_borders_g1[-1], 10000)
        x_interval_for_fit_g2 = np.linspace(bin_borders_g2[0], bin_borders_g2[-1], 10000)
        x_interval_for_fit_s = np.linspace(bin_borders_s[0], bin_borders_s[-1], 10000)
        plt.plot(x_interval_for_fit_g1, gaussian(x_interval_for_fit_g1, *popt_g1), label='fit', color ='yellow')
        plt.plot(x_interval_for_fit_g2, gaussian(x_interval_for_fit_g2, *popt_g2), label='fit')
        plt.plot(x_interval_for_fit_s, gaussian(x_interval_for_fit_s, *popt_s), label='fit')
        '''
        # print('g1', g1)
        # plt.hist(g1, bins, alpha=0.7, label='G1', color='yellow', density=True)

    # faced_histogram()

    from scipy.io import loadmat

    def make_df(dataset='mcf7', prefix='T1_M', start_g1=700):
        foldername = '/home/shobi/Trajectory/Datasets/FACED/' + dataset + '.mat'
        data_mat = loadmat(foldername)
        features = data_mat['label']
        features = [i[0] for i in features.flatten()]
        phases = data_mat['phases']
        phases = list(phases.flatten())
        phases = [prefix + str(i) for i in phases]
        for i in [1, 2, 3]:
            count = phases.count(prefix + str(i))
            print(dataset, ' count of phase initial ', i, 'is', count)
        phases = phases[start_g1:len(phases)]
        for i in [1, 2, 3]:
            count = phases.count(prefix + str(i))
            print(dataset, ' count of phase final ', i, 'is', count)
        print('features_mat', features)
        print('phases_mat', len(phases))
        data_mat = data_mat['data']
        data_mat = data_mat[start_g1:]
        print('shape of data', data_mat.shape)
        df = pd.DataFrame(data_mat, columns=features)
        df['phases'] = phases
        df['CellID'] = df.index + start_g1
        phases = df['phases'].values.tolist()

        df = df.drop('phases', 1)

        for i in [1, 2, 3]:
            count = phases.count(prefix + str(i))
            print('count of phase', i, 'is', count)
        return df, phases

    df_mb231, phases_mb231 = make_df('mb231', prefix='T2_M', start_g1=0)
    df_mcf7, phases_mcf7 = make_df('mcf7', prefix='T1_M', start_g1=1000)  # 1000

    # cell_line = 'mb231'
    # cell_line = 'mcf7'
    print('cell line line', cell_line)
    if cell_line == 'mb231':
        phases = phases_mb231
        df = df_mb231
        df = df.drop("CellID", 1)
    else:
        phases = phases_mcf7
        df = df_mcf7
        df = df.drop("CellID", 1)
    print('df.shape', df.shape, df.head)
    df_mean = df.copy()
    df_mean['type_phase'] = phases

    df_mean = df_mean.groupby('type_phase', as_index=False).mean()
    print('dfmean')
    # df_mean.to_csv('/home/shobi/Trajectory/Datasets/FACED/mixed_mean.csv')
    ad = sc.AnnData(df)
    ad.var_names = df.columns
    sc.pp.scale(ad)

    # FEATURE SELECTION

    from sklearn.feature_selection import mutual_info_classif
    kepler_mutual_information = mutual_info_classif(ad.X, phases)  # cancer_type

    plt.subplots(1, figsize=(26, 1))
    sns.heatmap(kepler_mutual_information[:, np.newaxis].T, cmap='Blues', cbar=False, linewidths=1, annot=True)
    plt.yticks([], [])
    plt.gca().set_xticklabels(df.columns, rotation=45, ha='right', fontsize=12)
    plt.suptitle("Kepler Variable Importance (mutual_info_classif)", fontsize=18, y=1.2)
    plt.gcf().subplots_adjust(wspace=0.2)
    # plt.show()

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import phate
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(ad.X, phases, test_size=0.3, random_state=1)
    clf = DecisionTreeClassifier(criterion='gini', max_depth=3)
    clf.fit(X_train, y_train)  # cancer_type

    pd.Series(clf.feature_importances_, index=df.columns).plot.bar(color='steelblue', figsize=(12, 6))
    # plt.show()
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # FEATURE SELECTION END

    knn = 20  # 20
    random_seed = 1
    jac_std_global = .5  # normally use 0.5

    sc.tl.pca(ad, svd_solver='arpack')
    X_in = ad.X  # ad_TF.obsm['X_pca'][:, start_ncomp:ncomps]
    df_X = pd.DataFrame(X_in)
    df_X.columns = df.columns
    # corrMatrix = df_X.corr()
    print('corrMatrix')
    # corrMatrix.to_csv('/home/shobi/Trajectory/Datasets/FACED/corrMatrix_'+cell_line+'.csv')

    df_X['Area'] = df_X['Area'] * 3
    df_X['Dry Mass'] = df_X['Dry Mass'] * 3
    df_X['Volume'] = df_X['Volume'] * 20
    # df_X['Phase Entropy Mean'] = df_X['Phase Entropy Mean'] * 20#when excluding vol related

    # df_X = df_X.drop(['Area','Dry Mass','Dry Mass Density','Circularity', 'Eccentricity','Volume'], 1)
    # df_X = df_X.drop(['Area', 'Dry Mass', 'Volume'], 1) #when excluding vol related

    # df_X.to_csv('/home/shobi/Trajectory/Datasets/FACED/'+cell_line+'_notCorrVol.csv')

    X_in = df_X.values
    ad = sc.AnnData(df_X)
    print('ad.shape', df_X.shape, ad.X.shape, ad.shape)
    sc.tl.pca(ad, svd_solver='arpack')
    ad_paga = sc.AnnData(df)
    ad_paga.var_names = [i for i in df.columns]
    ad_paga.obsm['paga_X'] = df_X.values

    ad.var_names = df_X.columns
    print('ad.shape', ad.X.shape, ad.shape)
    print('features', df_X.columns)
    ncomps = df.shape[1]
    root = 1
    root_user = ['T1_M1', 'T2_M1']
    true_label = phases
    cancer_type = ['mb' if 'T2' in i else 'mc' for i in phases]

    print('cancer type', cancer_type)
    print(phases)
    print('root, pc', root, ncomps)

    # Start embedding

    f, ax = plt.subplots()

    embedding = umap.UMAP().fit_transform(ad.obsm['X_pca'][:, 0:20])
    phate_op = phate.PHATE()
    # embedding = phate_op.fit_transform(X_in)
    df_embedding = pd.DataFrame(embedding)
    # df_embedding.to_csv('/home/shobi/Trajectory/Datasets/FACED/umap_mcf7_38feat.csv')
    print('saving embedding umap')
    # df_embedding['true_label'] = phases
    # df_embedding.to_csv('/home/shobi/Trajectory/Datasets/FACED/phate_mb231.csv')

    cell_dict = {'T1_M1': 'red', 'T2_M1': 'yellowgreen', 'T1_M2': 'orange', 'T2_M2': 'darkgreen', 'T1_M3': 'yellow',
                 'T2_M3': 'blue'}
    # cell_dict = {'T1_M1': 'red', 'T2_M1': 'blue', 'T1_M2': 'red', 'T2_M2': 'blue', 'T1_M3': 'red',             'T2_M3': 'blue'}

    for key in list(set(true_label)):  # ['T1_M1', 'T2_M1','T1_M2', 'T2_M2','T1_M3', 'T2_M3']:
        loc = np.where(np.asarray(true_label) == key)[0]
        if '_M1' in key:
            alpha = 0.5
        else:
            alpha = 0.8
        ax.scatter(embedding[loc, 0], embedding[loc, 1], c=cell_dict[key], alpha=alpha, label=key)
    plt.legend()
    plt.show()

    # END embedding
    print('ad.shape', ad.shape, 'true label length', len(true_label))

    embedding_pal = embedding  # ad.obsm['X_pca'][:,0:2]
    #paga_faced(ad, knn, embedding, true_label, cell_line=cell_line)  # use ad_paga for excluding corr vol

    #run_palantir_faced(ad, knn, embedding_pal, true_label, cell_line=cell_line, df_magic=df)
    #plt.show()

    v0 = VIA(X_in, true_label, jac_std_global=jac_std_global, dist_std_local=3, knn=knn, resolution_parameter=1,
             cluster_graph_pruning_std=cluster_graph_pruning_std,
             too_big_factor=0.3, root_user=root_user, dataset='faced', random_seed=random_seed,
             visual_cluster_graph_pruning=1, max_visual_outgoing_edges=2,
             is_coarse=True, preserve_disconnected=True, preserve_disconnected_after_pruning=True,
             pseudotime_threshold_TS=40)  # *.4 root=1,
    v0.run_VIA()

    import scipy.stats as stats
    pt = v0.single_cell_pt_markov
    true_time = [int(i[-1]) for i in true_label]
    correlation, p_value = stats.pearsonr(pt, true_time)
    print('via0 pearson', correlation, p_value)
    correlation, p_value = stats.kendalltau(pt, true_time)
    print('via0 kendall', correlation, p_value)

    v0.make_JSON(filename='faced_v0_temp.js')

    df_output = pd.DataFrame(v0.single_cell_pt_markov, columns=['pt'])
    # df_output['pt']= (df_output['pt'] - df_output['pt'].mean()) / df_output['pt'].std()
    dict_stage = {'T2_M1': 'G1', 'T2_M2': 'S', 'T2_M3': 'G2', 'T1_M1': 'G1', 'T1_M2': 'S', 'T1_M3': 'G2'}
    df_output['stage'] = [dict_stage[i] for i in true_label]
    df_output.to_csv('/home/shobi/Trajectory/Datasets/FACED/pt_histogram_mb231_all.csv')
    faced_histogram()

    all_cols = ['Area', 'Volume', 'Circularity', 'Eccentricity', 'AspectRatio', 'Orientation', 'Dry Mass',
                'Dry Mass Density', 'Dry Mass var', 'Dry Mass Skewness', 'Peak Phase', 'Phase Var', 'Phase Skewness',
                'Phase Kurtosis', 'Phase Range', 'Phase Min', 'Phase Centroid Displacement', 'Phase STD Mean',
                'Phase STD Variance', 'Phase STD Skewness', 'Phase STD Kurtosis', 'Phase STD Centroid Displacement',
                'Phase STD Radial Distribution', 'Phase Entropy Mean', 'Phase Entropy Var', 'Phase Entropy Skewness',
                'Phase Entropy Kurtosis', 'Phase Entropy Centroid Displacement', 'Phase Entropy Radial Distribution',
                'Phase Fiber Centroid Displacement', 'Phase Fiber Radial Distribution',
                'Phase Fiber Pixel >Upper Percentile', 'Phase Fiber Pixel >Median', 'Mean Phase Arrangement',
                'Phase Arrangement Var', 'Phase Arrangement Skewness', 'Phase Orientation Var',
                'Phase Orientation Kurtosis']

    '''
    for pheno_i in ['Area']:#, 'Volume', 'Circularity', 'Eccentricity', 'AspectRatio', 'Orientation', 'Dry Mass', 'Dry Mass Density', 'Dry Mass var', 'Dry Mass Skewness', 'Peak Phase', 'Phase Var', 'Phase Skewness', 'Phase Kurtosis', 'Phase Range', 'Phase Min', 'Phase Centroid Displacement', 'Phase STD Mean', 'Phase STD Variance', 'Phase STD Skewness', 'Phase STD Kurtosis', 'Phase STD Centroid Displacement', 'Phase STD Radial Distribution', 'Phase Entropy Mean', 'Phase Entropy Var', 'Phase Entropy Skewness', 'Phase Entropy Kurtosis', 'Phase Entropy Centroid Displacement', 'Phase Entropy Radial Distribution', 'Phase Fiber Centroid Displacement', 'Phase Fiber Radial Distribution', 'Phase Fiber Pixel >Upper Percentile', 'Phase Fiber Pixel >Median', 'Mean Phase Arrangement', 'Phase Arrangement Var', 'Phase Arrangement Skewness', 'Phase Orientation Var', 'Phase Orientation Kurtosis']:
        subset_ = df[pheno_i].values
        print('subset shape', subset_.shape)
        v0.get_gene_expression(subset_, title_gene=pheno_i)
        plt.show()
    '''
    df_features = df[all_cols]
    for col in df_features.columns:
        df_features[col] = (df_features[col] - df_features[col].mean()) / df_features[col].std()
    v0.get_gene_expression(df_features)
    plt.show()
    df['via_coarse_cluster'] = v0.labels
    df['phases'] = true_label
    df['pt_coarse'] = v0.single_cell_pt_markov



    v1 = VIA(X_in, true_label, jac_std_global=jac_std_global, dist_std_local=1, knn=knn,
             cluster_graph_pruning_std=cluster_graph_pruning_std,
             too_big_factor=0.05, root_user=root_user, is_coarse=False,
             preserve_disconnected=True, dataset='faced',
             visual_cluster_graph_pruning=1, max_visual_outgoing_edges=2, random_seed=random_seed, pseudotime_threshold_TS=40, via_coarse=v0)
    v1.run_VIA()
    pt = v1.single_cell_pt_markov
    correlation, p_value = stats.pearsonr(pt, true_time)
    print('via1 pearson', correlation, p_value)
    correlation, p_value = stats.kendalltau(pt, true_time)
    print('via1 kendall', correlation, p_value)

    df['via_fine_cluster'] = v1.labels
    df['pt_fine'] = v1.single_cell_pt_markov
    df.to_csv('/home/shobi/Trajectory/Datasets/FACED/' + cell_line + 'pseudotime_gwinky.csv')

    all_cols = ['Volume', 'Dry Mass', 'Eccentricity', 'Orientation', 'Peak Phase', 'Phase Fiber Radial Distribution',
                'Phase STD Variance', 'Phase Entropy Skewness', 'Circularity', 'AspectRatio', 'Dry Mass Density',
                'Dry Mass var',
                'Dry Mass Skewness', 'Phase Var', 'Phase Skewness', 'Phase Kurtosis', 'Phase Range', 'Area',
                'Phase Min', 'Phase Centroid Displacement', 'Phase STD Mean',
                'Phase STD Skewness', 'Phase STD Kurtosis', 'Phase STD Centroid Displacement',
                'Phase STD Radial Distribution', 'Phase Entropy Mean', 'Phase Entropy Var', 'Phase Entropy Kurtosis',
                'Phase Entropy Centroid Displacement', 'Phase Entropy Radial Distribution',
                'Phase Fiber Centroid Displacement', 'Phase Fiber Pixel >Upper Percentile', 'Phase Fiber Pixel >Median',
                'Mean Phase Arrangement', 'Phase Arrangement Var', 'Phase Arrangement Skewness',
                'Phase Orientation Var', 'Phase Orientation Kurtosis']

    df_features = df[all_cols]
    for col in df_features.columns:
        df_features[col] = (df_features[col]-df_features.mean())/df_features.std()

    v1.get_gene_expression(df_features)


    # draw overall pseudotime and main trajectories

    draw_sc_lineage_probability(v0, v1, embedding=embedding)


    draw_trajectory_gams(v0, v1, embedding=embedding)
    plt.show()


def main1():
    knn = 10
    # df = pd.read_csv('/home/shobi/Trajectory/Datasets/mESC/via_pt_knn'+str(knn)+'.csv')
    # df = pd.read_csv('/home/shobi/Trajectory/Datasets/mESC/via_pt_knn_nolazynotele'+str(knn)+'.csv')
    # df = pd.read_csv('/home/shobi/Trajectory/Datasets/mESC/via_pt_knn_Feb2021' + str(knn) + '.csv')
    df = pd.read_csv('/home/shobi/Trajectory/Datasets/mESC/noMCMC_nolazynotele_via_pt_knn_Feb2021' + str(knn) + '.csv')

    truth = df['days']
    pt = df['via_knn']
    pt = df['via_v1']
    correlation = pt.corr(truth)
    print('corr via knn', knn, correlation)


def main():
    dataset = 'Toy3'  #
    # dataset = 'mESC'  # 'EB'#'mESC'#'Human'#,'Toy'#,'Bcell'  # 'Toy'
    if dataset == 'Human':

        main_Human(ncomps=80, knn=30, v0_random_seed=10,
                   run_palantir_func=False)  # 100 comps, knn30, seed=4 is great too// pc=100, knn20 rs=1// pc80,knn30,rs3//pc80,knn30,rs10
        # cellrank_Human(ncomps = 20, knn=30)
    elif dataset == 'Bcell':
        main_Bcell(ncomps=20, knn=10, random_seed=1)  # 0 is good
    elif dataset == 'faced':
        main_faced(cell_line='mcf7')

    elif dataset == 'mESC':
        # main_mESC(knn=20, v0_random_seed=9, run_palantir_func=False) works well  # knn=20 and randomseed =8 is good
        #main_mESC(knn=40, v0_random_seed=20, run_palantir_func=False) very good too
        main_mESC_timeseries(knn=40)

    elif dataset == 'EB':
        # main_EB(ncomps=30, knn=20, v0_random_seed=24)
        main_EB_clean(ncomps=30, knn=20, v0_random_seed=24)
        # plot_EB()
    elif dataset == 'scATAC_Hema':
        main_scATAC_Hemato(knn=20)
        # main_scATAC_zscores(knn=30, ncomps =10) #knn=20, ncomps = 30)
    elif dataset == 'Toy3':
        #main_Toy_comparisons(ncomps=10, knn=30, random_seed=2, dataset='Toy3',  foldername="/home/shobi/Trajectory/Datasets/Toy3/")

        main_Toy(ncomps=20, knn=30, random_seed=2, dataset='Toy3',     foldername="/home/shobi/Trajectory/Datasets/Toy3/", cluster_graph_pruning_std=1)
        # main_Toy_comparisons(ncomps=10, knn=10, random_seed=2, dataset='ToyMultiM11',              foldername="/home/shobi/Trajectory/Datasets/ToyMultifurcating_M11/")
        # main_Toy_comparisons(ncomps=10, knn=20, random_seed=2, dataset='Toy3',                             foldername="/home/shobi/Trajectory/Datasets/Toy3/")
    elif dataset=='Toy4':
        main_Toy(ncomps=10, knn=30, random_seed=0, dataset='Toy4',foldername="/home/shobi/Trajectory/Datasets/Toy4/")  # pc10/knn30/rs2 for Toy4
    elif dataset == 'wrapper':

        # Read two files 1) the first file contains 200PCs of the Bcell filtered and normalized data for the first 5000 HVG.
        # 2)The second file contains raw count data for marker genes

        data = pd.read_csv('/home/shobi/Trajectory/Datasets/Bcell/Bcell_200PCs.csv')
        data_genes = pd.read_csv('/home/shobi/Trajectory/Datasets/Bcell/Bcell_markergenes.csv')
        data_genes = data_genes.drop(['Unnamed: 0'], axis=1)
        true_label = data['time_hour']
        data = data.drop(['cell', 'time_hour'], axis=1)
        adata = sc.AnnData(data_genes)
        adata.obsm['X_pca'] = data.values

        # use UMAP or PHate to obtain embedding that is used for single-cell level visualization
        embedding = umap.UMAP(random_state=42, n_neighbors=15, init='random').fit_transform(data.values[:, 0:5])

        # list marker genes or genes of interest if known in advance. otherwise marker_genes = []
        marker_genes = ['Igll1', 'Myc', 'Slc7a5', 'Ldha', 'Foxo1', 'Lig4', 'Sp7']  # irf4 down-up
        # call VIA
        via_wrapper(adata, true_label, embedding, knn=15, ncomps=20, jac_std_global=0.15, root=[42], dataset='', random_seed=1,v0_toobig=0.3, v1_toobig=0.1, marker_genes=marker_genes, draw_all_curves=True, piegraph_edgeweight_scalingfactor=1.0, piegraph_arrow_head_width=0.05, edgebundle_pruning_twice = True)


if __name__ == '__main__':
    main()
