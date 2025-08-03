import os
import numpy as np
import pandas as pd
import scanpy as sc


import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib.pyplot import cm
from scipy.spatial.distance import cdist, euclidean
from sklearn.neighbors import NearestNeighbors
from ._starfysh import LOGGER


class ArchetypalAnalysis:
    # Todo: add assertion that input `adata` should be raw counts to ensure math assumptions
    def __init__(
        self,
        adata_orig,
        r=100,
        u=None,
        u_3d=None,
        verbose=True,
        outdir=None,
        filename=None,
        savefig=False,
    ):
        """
        Parameters
        ----------
        adata_orig : sc.AnnData
            ST raw count matrix
        
        r : int
            Resolution parameter to control granularity of major archetypes
            If two archetypes reside within r nearest neighbors, the latter
            one will be merged.
        """

        # Check adata raw counts


        self.adata = adata_orig.copy()
        
        # Perform dim. reduction with PCA, select the first 30 PCs
        if 'X_umap' in self.adata.obsm_keys():
            del self.adata.obsm['X_umap']
        sc.pp.normalize_total(self.adata, target_sum=1e6)
        sc.pp.pca(self.adata, n_comps=30)
        
        self.r = r  # granularity parameter
        self.count = self.adata.obsm['X_pca']
        self.n_spots = self.count.shape[0]
        
        self.verbose = verbose
        self.outdir = outdir
        self.filename = filename
        self.savefig = savefig

        self.archetype = None
        self.major_archetype = None
        self.major_idx = None
        self.arche_dict = None
        self.arche_df = None
        self.kmin = 0

        self.U = u
        self.U_3d = u_3d

    def compute_archetypes(
        self, 
        cn=30,
        n_iters=20, 
        converge=1e-3,
        display=False
    ):
        """
        Estimate the upper bound of archetype count (k) by calculating intrinsic dimension
        Compute hierarchical archetypes (major + raw) with given granularity

        Parameters
        ----------
        cn : int
            Conditional Number to choose PCs for intrinsic estimator as
            lower bound # archetype estimation. Please refer to:
            https://scikit-dimension.readthedocs.io/en/latest/skdim.id.FisherS.html#skdim.id.FisherS

        n_iters : int
            Max. # iterations of AA to find the best k estimation

        converge : int
            Convergence criteria for AA iteration with diff(explained variance)

        display : bool
            Whether to display Intrinsic Dimension (ID) estimation plots

        Returns
        -------
        archetype : np.ndarray (dim=[K, G])
            Raw archetypes as linear combination of subset of spot counts

        arche_dict : dict
            Hierarchical structure of major_archetype -> its fine-grained neighbor archetypes

        major_idx : int
            Index of major archetypes among `k` raw candidates after merging\

        evs : list
            Explained variance with different Ks
        """
        
        # TMP: across-sample comparison: fix # principle components for all samples
        from py_pcha import PCHA
        import skdim
        if self.verbose:
            LOGGER.info('Computing intrinsic dimension to estimate k...')

        # Estimate ID
        id_model = skdim.id.FisherS(conditional_number=cn,
                                    produce_plots=display,
                                    verbose=self.verbose)
        
        self.kmin = max(1, int(id_model.fit(self.count).dimension_))

        # Compute raw archetypes
        if self.verbose:
            LOGGER.info('Estimating lower bound of # archetype as {0}...'.format(self.kmin))
        X = self.count.T
        archetypes = []
        evs = []

        # TODO: speedup with multiprocessing
        for i, k in enumerate(range(self.kmin, self.kmin+n_iters, 2)):
            archetype, _, _, _, ev = PCHA(X, noc=k)
            evs.append(ev)
            archetypes.append(np.array(archetype).T)
            if i > 0 and ev - evs[i-1] < converge:
                # early stopping
                break
        self.archetype = archetypes[-1]

        # Merge raw archetypes to get major archetypes
        if self.verbose:
            LOGGER.info('{0} variance explained by raw archetypes.\n'
                        'Merging raw archetypes within {1} NNs to get major archetypes'.format(np.round(ev, 4), self.r))
            
        arche_dict, major_idx = self._merge_archetypes()
        self.major_archetype = self.archetype[major_idx]
        self.major_idx = np.array(major_idx)
        self.arche_dict = arche_dict

        # return all archetypes for Silhouette score calculation
        return archetypes, arche_dict, major_idx, evs
    
    def _merge_archetypes(self):
        """
        Merge raw archetypes into major ones by removing candidate with `r`-step distance
        from its previous identified neighbors
        """
        assert self.archetype is not None, "Please compute archetypes first!"

        n_archetypes = self.archetype.shape[0]
        X_concat = np.vstack([self.count, self.archetype])
        nbrs = NearestNeighbors(n_neighbors=self.r).fit(X_concat)
        nn_graph = nbrs.kneighbors(X_concat)[1][self.n_spots:, 1:]  # retrieve NN-graph of only archetype spots

        idxs_to_remove = set()
        arche_dict = {}
        for i in range(n_archetypes):
            if i not in idxs_to_remove:
                query = np.arange(self.n_spots+i, self.n_spots+n_archetypes)
                nbrs = np.setdiff1d(
                    nn_graph[i][np.isin(nn_graph[i], query)] - self.n_spots,
                    list(idxs_to_remove)  # avoid over-assign merged archetypes to multiple major archetypes
                )
                if len(nbrs) != 0:
                    arche_dict[i] = np.insert(nbrs, 0, i)
                    idxs_to_remove.update(nbrs)

        major_idx = np.setdiff1d(np.arange(n_archetypes), list(idxs_to_remove))
        return arche_dict, major_idx

    def find_archetypal_spots(self, major=True):
        """
        Assign N-nearest-neighbor spots to each archetype as `archetypal spots` (archetype community)

        Parameters
        ----------
        major : bool
            Whether to find NNs for only major archetypes

        Returns
        -------
        arche_df : pd.DataFrame
            Dataframe of archetypal spots
        """
        assert self.archetype is not None, "Please compute archetypes first!"
        if self.verbose:
            LOGGER.info('Finding {} nearest neighbors for each archetype...'.format(self.r))

        indices = self.major_idx if major else np.arange(self.archetype.shape[0])
        x_concat = np.vstack([self.count, self.archetype])
        nbrs = self._get_knns(x_concat, n_nbrs=self.r, indices=indices+self.n_spots)
        self.arche_df = pd.DataFrame({
            'arch_{}'.format(idx): g
            for (idx, g) in zip(indices, nbrs)
        })

        return self.arche_df

    def find_markers(self, n_markers=30, display=False):
        """
        Find marker genes for each archetype community via Wilcoxon rank sum test (in-group vs. out-of-group)

        Parameters
        ----------
        n_markers : int
            Number of top marker genes to find for each archetype community

        Returns
        -------
        marker_df : pd.DataFrame
            Dataframe of marker genes for each archetype community
        """
        assert self.arche_df is not None, "Please compute archetypes & assign nearest-neighbors first!"
        if self.verbose:
            LOGGER.info('Finding {} top marker genes for each archetype...'.format(n_markers))

        adata = self.adata.copy()
        markers = []
        for col in self.arche_df.columns:
            # Annotate in-group (current archetype) vs. out-of-group
            annots = np.zeros(self.n_spots, dtype=np.int64).astype(str)
            annots[self.arche_df[col]] = col
            adata.obs[col] = annots
            adata.obs[col] = adata.obs[col].astype('category')

            # Identify marker genes
            sc.tl.rank_genes_groups(adata, col, use_raw=False, method='wilcoxon')
            markers.append(adata.uns['rank_genes_groups']['names'][col][:n_markers])

            if display:
                plt.rcParams['figure.figsize'] = (8, 3)
                plt.rcParams['figure.dpi'] = 300
                sc.pl.rank_genes_groups_violin(adata, groups=[col], n_genes=n_markers)

        return pd.DataFrame(np.stack(markers, axis=1), columns=self.arche_df.columns)

    def assign_archetypes(self, anchor_df, r=30):
        """
        Stable-matching to obtain best 1-1 mapping of archetype community to its closest anchor community
        (cell-type specific anchor spots)

        Parameters
        ----------
        anchor_df : pd.DataFrame
            Dataframe of anchor spot indices

    `   r : int
            Resolution parameter to threshold archetype - anchor mapping

        Returns
        -------
        overlaps_df : pd.DataFrame
            DataFrame of overlapping spot ratio of each anchor `i` to archetype `j`

        map_dict : dict
            Dictionary of cell type -> mapped archetype
        """
        assert self.arche_df is not None, "Please compute archetypes & assign nearest-neighbors first!"

        x_concat = np.vstack([self.count, self.archetype])
        anchor_nbrs = anchor_df.values
        archetypal_nbrs = self._get_knns(
            x_concat, n_nbrs=r, indices=self.n_spots + self.major_idx).T  # r-nearest nbrs to each archetype

        overlaps = np.array(
            [
                [
                    len(np.intersect1d(anchor_nbrs[:, i], archetypal_nbrs[:, j]))
                    for j in range(archetypal_nbrs.shape[1])
                ]
                for i in range(anchor_nbrs.shape[1])
            ]
        )
        overlaps_df = pd.DataFrame(overlaps, index=anchor_df.columns, columns=self.arche_df.columns)

        # Stable marriage matching: archetype -> anchor clusters
        map_idx_df = self._stable_matching(overlaps)
        map_dict = {overlaps_df.index[k]: overlaps_df.columns[v] for k, v in map_idx_df.items()}
        return overlaps_df, map_dict

    def find_distant_archetypes(self, anchor_df, map_dict=None, n=3):
        """
        Sort and return top n archetypes that are unmapped and farthest from anchor spots of know cell types
        They are more likely to represent novel cell types / states

        Parameters
        ----------
        anchor_df : pd.DataFrame
            Dataframe of anchor spot indices

        map_dict : dict
            Dictionary of cell type -> mapped archetype

        n : int
            Number of distant archetypes to return

        Returns
        -------
        distant_archetypes : list
            List of archetype labels (farthest --> closest to anchors)
        """
        assert self.arche_df is not None, "Please compute archetypes & assign nearest-neighbors first!"

        cell_types = anchor_df.columns
        arche_lbls = self.arche_df.columns

        # Find the unmapped archetypes
        if map_dict is None:
            _, map_dict = self.assign_archetypes(anchor_df=anchor_df)
        unmapped_archetypes = np.setdiff1d(
            arche_lbls,
            list(set([v for k, v in map_dict.items()]))
        )

        # Sort unmapped archetypes in descending orders with avg. distance to its 2 closest anchor spot centroid
        if n > len(unmapped_archetypes):
            LOGGER.warning('Insufficient candidates to find {0} distant archetypes\nSet n={1}'.format(
                n, len(unmapped_archetypes)
            ))
        anchor_centroids = self.count[anchor_df[anchor_df.columns]].mean(0)
        arche_centroids = self.count[self.arche_df[self.arche_df.columns]].mean(0)
        dist_df = pd.DataFrame(
            cdist(anchor_centroids, arche_centroids),
            index=cell_types,
            columns=arche_lbls
        )
        dist_unmapped = dist_df[unmapped_archetypes].values  # subset only distance to `unmapped` archetypes
        dist_to_nbrs = np.sort(dist_unmapped, axis=0)[:2].mean(0)
        distant_arches = [unmapped_archetypes[idx] for idx in np.argsort(-dist_to_nbrs)][:n] # dist - Discending order

        return distant_arches

    def _get_knns(self, x, n_nbrs, indices):
        """Compute kNNs (actual spots) to each archetype"""
        assert 0 <= indices.min() < indices.max() < x.shape[0], \
            "Invalid indices of interest to compute k-NNs"
        nbrs = np.zeros((len(indices), n_nbrs), dtype=np.int32)
        for i, index in enumerate(indices):
            u = x[index]
            dist = np.ones(x.shape[0])*np.inf
            for j, v in enumerate(x[:self.n_spots]):
                dist[j] = np.linalg.norm(u-v)
            nbrs[i] = np.argsort(dist)[:n_nbrs]
        return nbrs

    def _stable_matching(self, A):
        matching = {}
        free_rows, free_cols = set(range(A.shape[0])), set(range(A.shape[1]))

        while free_rows:
            i = free_rows.pop()
            for j in np.argsort(A[i])[::-1]:  # iter cols in decreasing vals
                if j in free_cols:
                    matching[i] = j
                    free_cols.remove(j)
                    break
                else:  # Check matched cols & compare tie-breaking conditions
                    i_prime = next(k for k, v in matching.items() if v == j)
                    if A[i, j] > A[i_prime, j]:
                        matching[i] = j
                        free_rows.add(i_prime)
                        del matching[i_prime]
                        break

        return matching
    
    # -------------------
    # Plotting functions
    # -------------------
    
    def _get_umap(self, ndim=2, random_state=42):
        import umap
        assert ndim == 2 or ndim == 3, "Invalid dimension for UMAP: {}".format(ndim)
        LOGGER.info('Calculating UMAPs for counts + Archetypes...')
        reducer = umap.UMAP(n_neighbors=self.r+10, n_components=ndim, random_state=random_state)
        U = reducer.fit_transform(np.vstack([self.count, self.archetype]))
        return U

    def _save_fig(self, fig, lgds, default_name):
        filename = self.filename if self.filename is not None else default_name
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        
        fig.savefig(
            os.path.join(self.outdir, filename+'.svg'),
            bbox_extra_artists=lgds, bbox_inches='tight',
            format='svg'
        )

    def plot_archetypes(
        self, 
        major=True, 
        do_3d=False, 
        lgd_ncol=1, 
        figsize=(6, 4),
        disp_cluster=True, 
        disp_arche=True
    ):
        """
        Display archetype & archetypal spot communities
        """
        assert self.arche_df is not None, "Please compute archetypes & assign nearest-neighbors first!"
        n_archetypes = self.arche_df.shape[1]
        arche_indices = self.major_idx if major else np.arange(n_archetypes)
        U = self.U_3d if do_3d else self.U
        colors = cm.tab20(np.linspace(0, 1, n_archetypes))

        if do_3d:
            if self.U_3d is None:
                self.U_3d = self._get_umap(ndim=3)
            U = self.U_3d

            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200, subplot_kw=dict(projection='3d'))

            # Color background spots & archetypal spots
            
            ax.scatter(
                U[:self.n_spots, 0],
                U[:self.n_spots, 1],
                U[:self.n_spots, 2], 
                s=1, alpha=0.7, linewidth=.3,
                edgecolors='black', c='lightgray'
            )
            
            if disp_cluster:
                for i, label in enumerate(self.arche_df.columns):
                    lbl = int(label.split('_')[-1])
                    if lbl in arche_indices:
                        idxs = self.arche_df[label]
                        ax.scatter(
                            U[idxs, 0],
                            U[idxs, 1],
                            U[idxs, 2],
                            marker='o', s=3,
                            color=colors[i], label=label
                        )

            # Highlight archetype
            if disp_arche:                                  
                ax.scatter(
                    U[self.n_spots+arche_indices, 0],
                    U[self.n_spots+arche_indices, 1], 
                    U[self.n_spots+arche_indices, 2],
                    s=10, c='blue', marker='^'
                )
                for j, z in zip(arche_indices, U[self.n_spots+arche_indices]):
                    ax.text(z[0], z[1], z[2], str(j), fontsize=10, c='blue')
                    
            lgd = ax.legend(loc='right', bbox_to_anchor=(0.5, 0, 1.5, 0.5), ncol=lgd_ncol)
            
            ax.grid(False)
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ax.set_zlabel('UMAP3')
            
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            
            ax.xaxis.pane.set_edgecolor('black')
            ax.yaxis.pane.set_edgecolor('black')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            ax.view_init(20, 135)
            
        else: # 2D plot
            if self.U is None:
                self.U = self._get_umap(ndim=2)
            U = self.U

            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)

            # Color background & archetypal spots
            ax.scatter(
                U[:self.n_spots, 0],
                U[:self.n_spots, 1],
                alpha=1, s=1, color='lightgray')
            
            if disp_cluster:
                for i, label in enumerate(self.arche_df.columns):
                    lbl = int(label.split('_')[-1])
                    if lbl in arche_indices:
                        idxs = self.arche_df[label]
                        ax.scatter(
                            U[idxs, 0],
                            U[idxs, 1],
                            marker='o', s=3,
                            color=colors[i], label=label
                        )
                        
            if disp_arche:
                ax.scatter(
                    U[self.n_spots+arche_indices, 0],
                    U[self.n_spots+arche_indices, 1],
                    s=10, c='blue', marker='^'
                )
                
                for j, z in zip(arche_indices, U[self.n_spots+arche_indices]):
                    ax.text(z[0], z[1], str(j), fontsize=10, c='blue')
                lgd = ax.legend(loc='right', bbox_to_anchor=(1, 0.5), ncol=lgd_ncol)
                
            ax.grid(False)
            ax.axis('off')

        if self.savefig and self.outdir is not None:
            self._save_fig(fig, (lgd,), 'archetypes')
        return fig, ax

    def plot_anchor_archetype_clusters(
        self,
        anchor_df,
        cell_types=None,
        arche_lbls=None,
        lgd_ncol=2,
        do_3d=False
    ):
        """
        Joint display subset of anchor spots & archetypal spots (to visualize overlapping degree)
        """
        assert self.arche_df is not None, "Please compute archetypes & assign nearest-neighbors first!"

        cell_types = anchor_df.columns if cell_types is None else np.intersect1d(cell_types, anchor_df.columns)
        arche_lbls = self.arche_df.columns if arche_lbls is None else np.intersect1d(arche_lbls, self.arche_df.columns)
        u_centroids = U[self.arche_df[arche_lbls]].mean(0)        

        anchor_colors = cm.RdBu_r(np.linspace(0, 1, len(cell_types)))
        arche_colors = cm.RdBu_r(np.linspace(0, 1, len(arche_lbls)))

        if do_3d:
            if self.U_3d is None:
                self.U_3d = self._get_umap(ndim=3)
            U = self.U_3d

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=300, subplot_kw=dict(projection='3d'))

            # Display anchors
            ax1.scatter(
                U[:self.n_spots, 0], 
                U[:self.n_spots, 1],
                U[:self.n_spots, 2], 
                c='gray', marker='.', s=1, alpha=0.2
            )
            for c, label in zip(anchor_colors, cell_types):
                idxs = anchor_df[label]
                ax1.scatter(
                    U[idxs, 0], 
                    U[idxs, 1], 
                    U[idxs, 2], 
                    color=c, marker='^', s=5,
                    alpha=0.9, label=label
                )
            
            ax1.grid(False)
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_zticklabels([])
            ax1.view_init(30, 45)
            
            lgd1 = ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -1), ncol=lgd_ncol)

            # Display archetypal spots
            ax2.scatter(U[:self.n_spots, 0], U[:self.n_spots, 1], U[:self.n_spots, 2], c='gray', marker='.', s=1, alpha=0.2)
            for c, label in zip(arche_colors, arche_lbls):
                idxs = self.arche_df[label]
                ax2.scatter(U[idxs, 0], U[idxs, 1], U[idxs, 2], color=c, marker='o', s=3, alpha=0.9, label=label)

            # Highlight selected archetypes
            for label, z in zip(arche_lbls, u_centroids):
                idx = int(label.split('_')[-1])
                ax2.text(z[0], z[1], z[2], str(idx))
            
            ax2.grid(False)
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_zticklabels([])
            ax2.view_init(30, 45)
            
            lgd2 = ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -1), ncol=lgd_ncol)

        else:
            if self.U is None:
                self.U = self._get_umap(ndim=2)
            U = self.U

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), dpi=300)

            # Display anchors
            ax1.scatter(
                U[:self.n_spots, 0],
                U[:self.n_spots, 1], 
                c='gray', marker='.', s=1, alpha=0.2
            )
            
            for c, label in zip(anchor_colors, cell_types):
                idxs = anchor_df[label]
                ax1.scatter(
                    U[idxs, 0],
                    U[idxs, 1], 
                    color=c, marker='^', s=5, 
                    alpha=0.9, label=label
                )
                
            lgd1 = ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -1.75), ncol=lgd_ncol)

            # Display archetypal spots
            ax2.scatter(U[:self.n_spots, 0], U[:self.n_spots, 1], c='gray', marker='.', s=1, alpha=0.2)
            for c, label in zip(arche_colors, arche_lbls):
                idxs = self.arche_df[label]
                ax2.scatter(
                    U[idxs, 0],
                    U[idxs, 1], 
                    color=c, marker='o', s=3, 
                    alpha=0.9, label=label
                )

            # Highlight selected archetypes
            for label, z in zip(arche_lbls, u_centroids):
                idx = int(label.split('_')[-1])
                ax2.text(z[0], z[1], str(idx))
            lgd2 = ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -1.85), ncol=lgd_ncol)

        if self.savefig and self.outdir is not None:
            self._save_fig(fig, (lgd1, lgd2), 'anchor_archetypal_spots')
        return fig, (ax1, ax2)

    def plot_mapping(self, map_df, figsize=(6, 5)):
        """
        Display anchor - archetype mapping (overlapping # spot ratio)
        """
        filename = 'cluster' if self.filename is None else self.filename
        g = sns.clustermap(
            map_df, 
            method='ward',
            figsize=figsize,
            xticklabels=True, 
            yticklabels=True,
            square=True,
            annot_kws={'size': 15}
        )
        
        text = g.ax_heatmap.set_title('# Overlapped NN Spots (k={})'.format(map_df.shape[1]),
                                      fontsize=20, x=0.6, y=1.3)
        # g.ax_row_dendrogram.set_visible(False)
        # g.ax_col_dendrogram.set_visible(False)

        if self.savefig and self.outdir is not None:
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)
            g.figure.savefig(
                os.path.join(self.outdir, filename + '.eps'),
                bbox_extra_artists=(text,), bbox_inches='tight', format='eps'
            )
            
        return g
