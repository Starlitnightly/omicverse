from typing import Optional

import pandas as pd
import numpy as np
import networkx as nx
import scanpy as sc
from sklearn.linear_model import LinearRegression as lr

from matplotlib import pyplot as plt
#from matplotlib_venn import venn3
import matplotlib.ticker as ticker
from matplotlib.pyplot import rc_context
import seaborn as sns
# import aucell and ctxcore
from ._aucell import aucell
from ..ctxcore.genesig import GeneSignature

from .driver_regulators import driver_regulators

venn_install = False

def matplotlib_venn():
    global venn_install
    try:
        import matplotlib_venn as venn3
        venn_install=True
    except ImportError:
        raise ImportError(
            'Please install the matplotlib_venn: `pip install matplotlib_venn`.'
        )
    
class CefconResults:

    def __init__(self,
                 adata: sc.AnnData,
                 network: nx.DiGraph,
                 gene_embedding: pd.DataFrame):
        self.name = adata.uns['name']
        self.network = network
        self.gene_embedding = gene_embedding

        genes = list(network.nodes())
        self.expression_data = adata[:, genes].to_df()
        self.TFs = adata[:, genes].var_names[adata[:, genes].var['is_TF'] == 1]
        if 'node_score_auxiliary' in adata.var:
            self.DEgenes = set(adata.var_names[adata.var['node_score_auxiliary'] > 0]).intersection(set(genes))
        else:
            self.DEgenes = set(genes)

        self.n_cells = adata.n_obs
        self.n_genes = len(genes)
        self.n_edges = network.number_of_edges()
        self.influence_score = None
        self.driver_regulator = None
        self.gene_cluster = None
        self.RGMs_AUCell_dict = None
        self._adata_gene = None
        self._out_critical_genes = None
        self._in_critical_genes = None

    def __repr__(self) -> str:
        descr = f"CefconResults object with n_cells * n_genes = {self.n_cells} * {self.n_genes}, n_edges = {self.n_edges}"
        descr += f"\n    name: {self.name}" \
                 f"\n    expression_data: yes" \
                 f"\n    network: {self.network}" \
                 f"\n    gene_embedding: # dimension = {self.gene_embedding.shape[1] if self.gene_embedding is not None else 'None'}" \
                 f"\n    influence_score: {'None' if self.influence_score is None else 'yes'}" \
                 f"\n    driver_regulator: {'None' if self.driver_regulator is None else 'yes'}" \
                 f"\n    gene_cluster: {'None' if self.gene_cluster is None else 'yes'}" \
                 f"\n    RGMs_AUCell_dict: {'None' if self.RGMs_AUCell_dict is None else 'yes'}"
        return descr

    def gene_influence_score(self):
        """
        Obtain gene influence score
        """
        influence_score = pd.DataFrame(np.zeros((len(self.network.nodes), 2)),
                                       index=sorted(self.network.nodes),
                                       columns=['out', 'in'])
        for i, v in enumerate(['in', 'out']):
            # The out-degree type of influence is obtained from the incoming network;
            # The in-degree type of influence is obtained from the outgoing network.
            gene_att_score = np.sum(nx.to_numpy_array(self.network,
                                                      nodelist=sorted(self.network.nodes),
                                                      dtype='float32',
                                                      weight='weights_{}'.format(v)),  # {}_weights
                                    axis=1 - i)
            influence_score.iloc[:, i] = np.log1p(gene_att_score).flatten().tolist()

        lam = 0.8
        influence_score['influence_score'] = lam * influence_score.loc[:, 'out'] + \
                                             (1 - lam) * influence_score.loc[:, 'in']
        influence_score.rename(columns={'out': 'score_out', 'in': 'score_in'}, inplace=True)
        influence_score = influence_score.sort_values(by='influence_score', ascending=False)
        self.influence_score = influence_score

    def driver_regulators(self, topK: int = 100, return_value: bool = False, output_file: Optional[str] = None, solver = 'GUROBI'):
        """
        Obtain driver regulators from the constructed cell-lineage-specific GRN
        """
        if self.influence_score is None:
            self.gene_influence_score()

        self.driver_regulator, self._out_critical_genes, self._in_critical_genes = driver_regulators(self.network,
                                                                                                     self.influence_score,
                                                                                                     topK=topK,
                                                                                                     driver_union=True,
                                                                                                     solver=solver)
        self.driver_regulator['is_TF'] = self.driver_regulator.index.isin(self.TFs)

        if isinstance(output_file, str):
            self.driver_regulator.to_csv(output_file)
        if return_value:
            return self.driver_regulator

    def RGM_activity(self, num_workers: int = 8, return_value: bool = False):
        """
        Select RGMs of driver regulators and calculate their activities in each cell.
        Activity score is calculated based on AUCell, which is from the `pyscenic` package.
        """
        print('[3] - Identifying regulon-like gene modules...')

        if self.driver_regulator is None:
            raise ValueError(
                f'Did not find the result of driver regulators. Run `cefcon.NetModel.driver_regulators` first.'
            )

        network = self.network
        DEgenes = self.DEgenes
        drivers = set(self.driver_regulator[self.driver_regulator['is_driver_regulator']].index)

        auc_mtx_out, auc_mtx_in, auc_mtx = None, None, None
        out_hub_nodes = self._out_critical_genes.intersection(drivers)
        out_RGMs = {i + '_out({})'.format(len(list(set(network.successors(i)).intersection(DEgenes)))):
                        list(set(network.successors(i)).intersection(DEgenes)) for i in out_hub_nodes
                    if len(set(network.successors(i)).intersection(DEgenes)) >= 10}
        in_hub_nodes = self._in_critical_genes.intersection(drivers)
        in_RGMs = {i + '_in({})'.format(len(list(set(network.predecessors(i)).intersection(DEgenes)))):
                       list(set(network.predecessors(i)).intersection(DEgenes)) for i in in_hub_nodes
                   if len(set(network.predecessors(i)).intersection(DEgenes)) >= 10}

        # n_cells x n_genes
        out_RGMs = [GeneSignature(name=k, gene2weight=v) for k, v in out_RGMs.items()]
        in_RGMs = [GeneSignature(name=k, gene2weight=v) for k, v in in_RGMs.items()]
        if len(out_RGMs) > 0:
            auc_mtx_out = aucell(self.expression_data, out_RGMs, num_workers=num_workers, auc_threshold=0.25,
                                 normalize=False)
            # Generate a Z-score for each RGM to enable comparison between RGMs
            auc_mtx_out_Z = pd.DataFrame(index=auc_mtx_out.index, columns=list(auc_mtx_out.columns), dtype=float)
            for col in list(auc_mtx_out.columns):
                auc_mtx_out_Z[col] = (auc_mtx_out[col] - auc_mtx_out[col].mean()) / auc_mtx_out[col].std(ddof=0)

        if len(in_RGMs) > 0:
            auc_mtx_in = aucell(self.expression_data, in_RGMs, num_workers=num_workers, auc_threshold=0.25,
                                normalize=False)
            # Generate a Z-score for each RGM to enable comparison between RGMs
            auc_mtx_in_Z = pd.DataFrame(index=auc_mtx_in.index, columns=list(auc_mtx_out.columns), dtype=float)
            for col in list(auc_mtx_in.columns):
                auc_mtx_in_Z[col] = (auc_mtx_in[col] - auc_mtx_in[col].mean()) / auc_mtx_in[col].std(ddof=0)

        if (len(out_RGMs) > 0) & (len(in_RGMs) > 0):
            auc_mtx = pd.merge(auc_mtx_out, auc_mtx_in, how='inner', left_index=True, right_index=True)
            # Generate a Z-score for each RGM to enable comparison between RGMs
            auc_mtx_Z = pd.DataFrame(index=auc_mtx.index, columns=list(auc_mtx.columns), dtype=float)
            for col in list(auc_mtx.columns):
                auc_mtx_Z[col] = (auc_mtx[col] - auc_mtx[col].mean()) / auc_mtx[col].std(ddof=0)

        RGMs = out_RGMs + in_RGMs
        RGMs_AUCell_dict = {'RGMs': RGMs, 'aucell': auc_mtx,
                            'aucell_out': auc_mtx_out, 'aucell_in': auc_mtx_in}
        self.RGMs_AUCell_dict = RGMs_AUCell_dict
        print('Done!')

        if return_value:
            return RGMs_AUCell_dict

    def plot_gene_embedding_with_clustering(self,
                                            n_neighbors: int = 30,
                                            resolution: float = 1,
                                            return_value: bool = False):
        """
        Cluster genes with CEFCON derived embeddings by using the Leiden algorithm and visualize in 2-D UMAP space.
        """
        if self._adata_gene is None:
            self._adata_gene = sc.AnnData(X=self.gene_embedding)
            sc.pp.neighbors(self._adata_gene, n_neighbors=n_neighbors, use_rep='X')
            # Higher resolutions lead to more communities, while lower resolutions lead to fewer communities.
            sc.tl.leiden(self._adata_gene, resolution=resolution)
            sc.tl.umap(self._adata_gene, n_components=2, min_dist=0.3)
            pos = self._adata_gene.obsm['X_umap']
            self.gene_cluster = self._adata_gene.obs['leiden']

        with rc_context({'figure.figsize': (4, 4)}):
            sc.pl.umap(self._adata_gene, color=['leiden'], legend_loc='on data',
                       legend_fontsize=8, legend_fontoutline=2, title='Leiden clustering using CEFCON derived gene embeddings')

        if return_value:
            return self.gene_cluster

    def plot_influence_score(self, topK: int = 20):
        """
        Plot the gene influence score of top-k driver regulators.
        """
        if self.driver_regulator is None:
            raise ValueError(
                f'Did not find the result of driver regulators. Run `cefcon.NetModel.driver_regulators` first.'
            )
        data_for_plot = self.driver_regulator[self.driver_regulator['is_driver_regulator']]
        data_for_plot = data_for_plot[0:topK]

        plt.figure(figsize=(1.5, topK * 0.15))
        #sns.set_theme(style='ticks', font_scale=0.5)

        ax = sns.barplot(x='influence_score', y=data_for_plot.index, data=data_for_plot, orient='h',
                         palette=sns.color_palette(f"ch:start=.5,rot=-.5,reverse=1,dark=0.4", n_colors=topK))
        ax.set_title(self.name)
        ax.set_xlabel('Influence score')
        ax.set_ylabel('Driver regulators')
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        sns.despine()

        plt.show()

    def plot_network(self, nodes: Optional[list] = None):
        """
        Plot network graph with circos layout.
        """
        try:
            import nxviz as nv
            from nxviz import annotate
        except ImportError:
            raise ImportError("install nxviz via `pip install nxviz`")

        if nodes is not None:
            sub_net = self.network.subgraph(nodes)
        else:
            sub_net = self.network

        nx.set_node_attributes(sub_net, self.driver_regulator['influence_score'].to_dict(), 'influence_score')
        nx.set_node_attributes(sub_net, {i: i for i in sub_net.nodes}, 'gene_name')

        #sns.set_theme(font_scale=0.7)
        ax = nv.circos(sub_net,
                       sort_by='influence_score',
                       # node_size_by='influence_score',
                       node_enc_kwargs={
                           'size_scale': 0.6,
                       },
                       edge_alpha_by='weights_combined',
                       edge_enc_kwargs={
                           'lw_scale': 0.5,
                           'alpha_scale': 0.7,
                       },
                       )
        annotate.circos_labels(sub_net, sort_by='influence_score', layout='rotate')

    def plot_driver_genes_Venn(self,figsize=(4,4)):
        """
        Plot Venn diagram of MDS, MFVS and top-ranked regulators.
        """
        if self.driver_regulator is None:
            raise ValueError(
                f'Did not find the result of driver regulators. Run `cefcon.NetModel.driver_regulators` first.'
            )
        else:
            drivers_df = self.driver_regulator

        MFVS_driver_set = set(drivers_df.loc[drivers_df['is_MFVS_driver']].index)
        MDS_driver_set = set(drivers_df.loc[drivers_df['is_MDS_driver']].index)
        top_ranked_genes = set(drivers_df.loc[drivers_df['is_driver_regulator']].index).union(
            (set(drivers_df.index) - MFVS_driver_set.union(MDS_driver_set)))

        f = plt.figure(figsize=figsize)
        #sns.set_theme(font_scale=f.get_dpi() / 100)

        matplotlib_venn()
        global venn_install
        if venn_install==True:
            global_imports_members('matplotlib_venn', members=['venn3'], asfunction=True)
            
        out = venn3(subsets=[MDS_driver_set, MFVS_driver_set, top_ranked_genes],
                    set_labels=('MDS driver genes({})'.format(len(MDS_driver_set)),
                                'MFVS driver genes({})'.format(len(MFVS_driver_set)),
                                'Top ranked genes({})'.format(len(top_ranked_genes))),
                    set_colors=('#0076FF', '#D74715', '#009000'),
                    alpha=0.45)
        for text in out.set_labels:
            if text is not None:
                text.set_fontsize(7)
        for text in out.subset_labels:
            if text is not None:
                text.set_fontsize(8)
        plt.tight_layout()
        # plt.savefig('drivers_Venn_plot.svg', dpi=600, format='svg', bbox_inches='tight')
        plt.show()

    def plot_RGM_activity_heatmap(self, cell_label: Optional[pd.DataFrame] = None,
                                   type: str = 'out',col_cluster = True, bbox_to_anchor=(1.35, 0.90)):
        """
        Plot clustermap of RGM activity matrix.
        If `cell_label` is provided, cells are ordered by cell clusters, else cells are ordered by hierarchical clustering.
        """
        if self.RGMs_AUCell_dict is None:
            raise ValueError(
                f'Did not find the result of RGMs. Run `cefcon.NetModel.RGM_activity` first.'
            )
        assert type in ['out', 'in', 'all']

        if type == 'all':
            auc_mtx = self.RGMs_AUCell_dict['aucell']
        elif type == 'out':
            auc_mtx = self.RGMs_AUCell_dict['aucell_out']
        else:  # in
            auc_mtx = self.RGMs_AUCell_dict['aucell_in']

        # Create a categorical palette to identify the networks
        if cell_label is not None:
            network_lable = cell_label
            network_pal = sns.husl_palette(len(network_lable.unique()), h=.5)
            network_lut = dict(zip(map(str, network_lable.unique()), network_pal))
            network_colors = pd.Series(list(network_lable), index=auc_mtx.index).map(network_lut)
            col_cluster = col_cluster
        else:
            network_colors = None
            col_cluster = col_cluster

        # plot clustermap (n_cell * n_gene)
        f = plt.figure()
        #sns.set_theme(font_scale=f.get_dpi()/150)
        g = sns.clustermap(auc_mtx.T, method='ward', square=False, linecolor='black',
                           z_score=0, vmin=-2.5, vmax=2.5,
                           col_cluster=col_cluster, col_colors=network_colors, cmap="RdBu_r",
                           figsize=(4.5, 0.15 * auc_mtx.shape[1]),
                           xticklabels=False, yticklabels=True, dendrogram_ratio=0.12,
                           cbar_pos=(0.75, 0.92, 0.2, 0.02),
                           cbar_kws={'orientation': 'horizontal'})
        g.cax.set_visible(True)
        g.ax_heatmap.set_ylabel('Regulon-like gene modules')
        g.ax_heatmap.set_xlabel('Cells')
        g.ax_heatmap.yaxis.set_minor_locator(ticker.NullLocator())
        if cell_label is not None:
            for label in network_lable.unique():
                g.ax_col_dendrogram.bar(0, 0, color=network_lut[label], label=label, linewidth=0)
            g.ax_col_dendrogram.legend(title='Cell types', loc="upper left", ncol=1,
                                       bbox_to_anchor=bbox_to_anchor, facecolor='white')
        g.ax_row_dendrogram.set_visible(False)
        plt.show()

    def plot_network_degree_distribution(self):
        """
        Plot degree distribution of the predicted network
        """
        network = self.network.copy()
        network.remove_edges_from(nx.selfloop_edges(network))
        network = network.subgraph(max(nx.weakly_connected_components(network), key=len)).copy()

        # Slope of degree distribution
        degree_sequence = pd.DataFrame(np.array(network.degree))
        degree_sequence.columns = ["ind", "degree"]
        degree_sequence['degree'] = degree_sequence['degree'].astype(int)
        degree_sequence = degree_sequence.loc[degree_sequence['degree'] != 0, :]
        degree_sequence = degree_sequence.set_index("ind")
        dist = degree_sequence.degree.value_counts() / degree_sequence.degree.value_counts().sum()
        dist.index = dist.index.astype(int)

        x = np.log(dist.index.values).reshape([-1, 1])
        y = np.log(dist.values).reshape([-1, 1])

        model = lr()
        model.fit(x, y)

        # Figure
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        #sns.set_theme(style="white", font_scale=fig.get_dpi() / 120)

        x_ = np.array([-1, 5]).reshape([-1, 1])
        y_ = model.predict(x_)

        ax.set_title("degree distribution (log scale)")
        ax.plot(x_.flatten(), y_.flatten(), c="black", alpha=0.5)

        ax.scatter(x.flatten(), y.flatten(), c="black")
        ax.text(0.45, 0.95,
                f"slope: {model.coef_[0][0] :.4g}, " + r"$R^2$: " + f"{model.score(x, y) :.4g}\n" +
                f"num_of_genes: {self.network.number_of_nodes()}\n" +
                f"num_of_edges: {self.network.number_of_edges()}\n" +
                f"clustering_coefficient: {nx.average_clustering(self.network) :.4g}\n",
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=8.5)
        ax.set_ylim([y.min() - 0.2, y.max() + 0.2])
        ax.set_xlim([-0.2, x.max() + 0.2])
        ax.set_xlabel("log k")
        ax.set_ylabel("log P(k)")
        ax.grid(None)

        plt.tight_layout()
        plt.show()


def global_imports(modulename,shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = __import__(modulename)

def global_imports_members(modulename, members=None, asfunction=False):
    if members is None:
        members = [modulename]  # Default to importing the entire module

    imported_module = __import__(modulename, fromlist=members)

    if asfunction:
        for member in members:
            globals()[member] = getattr(imported_module, member)
    else:
        globals()[modulename] = imported_module
