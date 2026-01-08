import anndata
import numpy as np
import pandas as pd
import warnings

from scipy.sparse import issparse, csr_matrix

from .knn import (
    neighbors_and_weights,
    neighbors_and_weights_from_distances,
    tree_neighbors_and_weights,
    make_weights_non_redundant,
)
from .local_stats import compute_hs
from .local_stats_pairs import compute_hs_pairs, compute_hs_pairs_centered_cond

from . import modules
from .plots import local_correlation_plot, local_correlation_plot_marsilea, MARSILEA_AVAILABLE
from tqdm import tqdm


class Hotspot:
    def __init__(
        self,
        adata,
        layer_key=None,
        model="danb",
        latent_obsm_key=None,
        distances_obsp_key=None,
        tree=None,
        umi_counts_obs_key=None,
    ):
        """Initialize a Hotspot object for analysis

        Either `latent` or `tree` or `distances` is required.

        Parameters
        ----------
        adata : anndata.AnnData
            Count matrix (shape is cells by genes)
        layer_key: str
            Key in adata.layers with count data, uses adata.X if None.
        model : string, optional
            Specifies the null model to use for gene expression.
            Valid choices are:

                - 'danb': Depth-Adjusted Negative Binomial
                - 'bernoulli': Models probability of detection
                - 'normal': Depth-Adjusted Normal
                - 'none': Assumes data has been pre-standardized

        latent_obsm_key : string, optional
            Latent space encoding cell-cell similarities with euclidean
            distances.  Shape is (cells x dims). Input is key in adata.obsm
        distances_obsp_key : pandas.DataFrame, optional
            Distances encoding cell-cell similarities directly
            Shape is (cells x cells). Input is key in adata.obsp
        tree : ete3.coretype.tree.TreeNode
            Root tree node.  Can be created using ete3.Tree
        umi_counts_obs_key : str
            Total umi count per cell.  Used as a size factor.
            If omitted, the sum over genes in the counts matrix is used
        """
        counts = self._counts_from_anndata(adata, layer_key)
        distances = (
            adata.obsp[distances_obsp_key] if distances_obsp_key is not None else None
        )
        latent = adata.obsm[latent_obsm_key] if latent_obsm_key is not None else None
        umi_counts = (
            adata.obs[umi_counts_obs_key] if umi_counts_obs_key is not None else None
        )

        if latent is None and distances is None and tree is None:
            raise ValueError(
                "Neither `latent_obsm_key` or `tree` or `distances_obsp_key` arguments were supplied.  One of these is required"
            )

        if latent is not None and distances is not None:
            raise ValueError(
                "Both `latent_obsm_key` and `distances_obsp_key` provided - only one of these should be provided."
            )

        if latent is not None and tree is not None:
            raise ValueError(
                "Both `latent_obsm_key` and `tree` provided - only one of these should be provided."
            )

        if distances is not None and tree is not None:
            raise ValueError(
                "Both `distances_obsp_key` and `tree` provided - only one of these should be provided."
            )

        if latent is not None:
            latent = pd.DataFrame(latent, index=adata.obs_names)

        # because of transpose we check if its csr
        if issparse(counts) and not isinstance(counts, csr_matrix):
            warnings.warn(
                "Hotspot will work faster when counts are a csr sparse matrix."
            )

        if tree is not None:
            try:
                all_leaves = []
                for x in tree:
                    if x.is_leaf():
                        all_leaves.append(x.name)
            except:
                raise ValueError("Can't parse supplied tree")

            if len(all_leaves) != counts.shape[1] or len(
                set(all_leaves) & set(adata.obs_names)
            ) != len(all_leaves):
                raise ValueError(
                    "Tree leaf labels don't match columns in supplied counts matrix"
                )

        if umi_counts is None:
            umi_counts = counts.sum(axis=0)
            # handles sparse matrix outputs of sum
            umi_counts = np.asarray(umi_counts).ravel()
        else:
            assert umi_counts.size == counts.shape[1]

        if not isinstance(umi_counts, pd.Series):
            umi_counts = pd.Series(umi_counts, index=adata.obs_names)

        valid_models = {"danb", "bernoulli", "normal", "none"}
        if model not in valid_models:
            raise ValueError("Input `model` should be one of {}".format(valid_models))

        if issparse(counts):
            # For a sparse matrix, check if all values in each row are identical
            # A row (gene) is considered valid if it has more than one unique value.
            row_min = counts.min(axis=1).toarray().flatten()
            row_max = counts.max(axis=1).toarray().flatten()
            valid_genes = (
                row_min != row_max
            )  # Valid if min and max are not equal, indicating variation
        else:
            # For a dense matrix, check if all values in each row are identical
            valid_genes = ~(np.all(counts == counts[:, [0]], axis=1))

        # valid_genes is now a boolean array indicating which rows (genes) have non-identical values.

        n_invalid = counts.shape[0] - valid_genes.sum()
        if n_invalid > 0:
            raise ValueError(
                "\nDetected genes with zero variance. Please filter adata and reinitialize."
            )

        self.adata = adata
        self.layer_key = layer_key

        self.counts = counts
        self.latent = latent
        self.distances = distances
        self.tree = tree
        self.model = model

        self.umi_counts = umi_counts

        self.graph = None
        self.modules = None
        self.local_correlation_z = None
        self.linkage = None
        self.module_scores = None

    @classmethod
    def legacy_init(
        cls,
        counts,
        model="danb",
        latent=None,
        distances=None,
        tree=None,
        umi_counts=None,
    ):
        """
        Initialize a Hotspot object for analysis using legacy method


        Either `latent` or `tree` or `distances` is required.


        Parameters
        ----------
        counts : pandas.DataFrame
            Count matrix (shape is genes x cells)
        model : string, optional
            Specifies the null model to use for gene expression.
            Valid choices are:
                - 'danb': Depth-Adjusted Negative Binomial
                - 'bernoulli': Models probability of detection
                - 'normal': Depth-Adjusted Normal
                - 'none': Assumes data has been pre-standardized
        latent : pandas.DataFrame, optional
            Latent space encoding cell-cell similarities with euclidean
            distances.  Shape is (cells x dims)
        distances : pandas.DataFrame, optional
            Distances encoding cell-cell similarities directly
            Shape is (cells x cells)
        tree : ete3.coretype.tree.TreeNode
            Root tree node.  Can be created using ete3.Tree
        umi_counts : pandas.Series, optional
            Total umi count per cell.  Used as a size factor.
            If omitted, the sum over genes in the counts matrix is used

        Examples
        --------
        >>> gene_exp = pd.read_csv(path, index_col=0) # genes by cells
        >>> latent = pd.read_csv(latent_path, index_col=0) # cells by dims
        >>> hs = hotspot.Hotspot.legacy_init(gene_exp, model="normal", latent=latent)
        """

        if latent is None and distances is None and tree is None:
            raise ValueError(
                "Neither `latent` or `tree` or `distance` arguments were supplied.  One of these is required"
            )

        if latent is not None and distances is not None:
            raise ValueError(
                "Both `latent` and `distances` provided - only one of these should be provided."
            )

        if latent is not None and tree is not None:
            raise ValueError(
                "Both `latent` and `tree` provided - only one of these should be provided."
            )

        if distances is not None and tree is not None:
            raise ValueError(
                "Both `distances` and `tree` provided - only one of these should be provided."
            )

        if latent is not None:
            if counts.shape[1] != latent.shape[0]:
                if counts.shape[0] == latent.shape[0]:
                    raise ValueError(
                        "`counts` input should be a Genes x Cells dataframe.  Maybe needs transpose?"
                    )
                raise ValueError(
                    "Size mismatch counts/latent. Columns of `counts` should match rows of `latent`."
                )

        if distances is not None:
            assert counts.shape[1] == distances.shape[0]
            assert counts.shape[1] == distances.shape[1]

        if umi_counts is None:
            umi_counts = counts.sum(axis=0)
        else:
            assert umi_counts.size == counts.shape[1]

        if not isinstance(umi_counts, pd.Series):
            umi_counts = pd.Series(umi_counts)

        valid_genes = counts.var(axis=1) > 0
        n_invalid = counts.shape[0] - valid_genes.sum()
        if n_invalid > 0:
            counts = counts.loc[valid_genes]
            print("\nRemoving {} undetected/non-varying genes".format(n_invalid))

        input_adata = anndata.AnnData(counts)
        input_adata = input_adata.transpose()
        tc_key = "total_counts"
        input_adata.obs[tc_key] = umi_counts.values
        dkey = "distances"
        if distances is not None:
            input_adata.obsp[dkey] = distances
            dist_input = True
        else:
            dist_input = False
        lkey = "latent"
        if latent is not None:
            input_adata.obsm[lkey] = np.asarray(latent)
            latent_input = True
        else:
            latent_input = False

        return cls(
            input_adata,
            model=model,
            latent_obsm_key=lkey if latent_input else None,
            distances_obsp_key=dkey if dist_input else None,
            umi_counts_obs_key=tc_key,
            tree=tree,
        )

    @staticmethod
    def _counts_from_anndata(adata, layer_key, dense=False, pandas=False):
        counts = adata.layers[layer_key] if layer_key is not None else adata.X
        is_sparse = issparse(counts)
        # handles adata view
        # as sparse matrix in view is just a sparse matrix, while dense is ArrayView
        if not issparse(counts):
            counts = np.asarray(counts)
        counts = counts.transpose()

        if dense:
            counts = counts.toarray() if is_sparse else counts
            is_sparse = False
        if pandas and is_sparse:
            raise ValueError("Set dense=True to return pandas output")
        if pandas and not is_sparse:
            counts = pd.DataFrame(
                counts, index=adata.var_names, columns=adata.obs_names
            )

        return counts

    def create_knn_graph(
        self,
        weighted_graph=False,
        n_neighbors=30,
        neighborhood_factor=3,
        approx_neighbors=True,
    ):
        """Create's the KNN graph and graph weights

        The resulting matrices containing the neighbors and weights are
        stored in the object at `self.neighbors` and `self.weights`

        Parameters
        ----------
        weighted_graph: bool
            Whether or not to create a weighted graph
        n_neighbors: int
            Neighborhood size
        neighborhood_factor: float
            Used when creating a weighted graph.  Sets how quickly weights decay
            relative to the distances within the neighborhood.  The weight for
            a cell with a distance d will decay as exp(-d/D) where D is the distance
            to the `n_neighbors`/`neighborhood_factor`-th neighbor.
        approx_neighbors: bool
            Use approximate nearest neighbors or exact scikit-learn neighbors. Only
            when hotspot initialized with `latent`.
        """

        if self.latent is not None:
            neighbors, weights = neighbors_and_weights(
                self.latent,
                n_neighbors=n_neighbors,
                neighborhood_factor=neighborhood_factor,
                approx_neighbors=approx_neighbors,
            )
        elif self.tree is not None:
            if weighted_graph:
                raise ValueError(
                    "When using `tree` as the metric space, `weighted_graph=True` is not supported"
                )
            neighbors, weights = tree_neighbors_and_weights(
                self.tree, n_neighbors=n_neighbors, cell_labels=self.adata.obs_names
            )
        else:
            neighbors, weights = neighbors_and_weights_from_distances(
                self.distances,
                cell_index=self.adata.obs_names,
                n_neighbors=n_neighbors,
                neighborhood_factor=neighborhood_factor,
            )

        neighbors = neighbors.loc[self.adata.obs_names]
        weights = weights.loc[self.adata.obs_names]

        self.neighbors = neighbors

        if not weighted_graph:
            weights = pd.DataFrame(
                np.ones_like(weights.values),
                index=weights.index,
                columns=weights.columns,
            )

        weights = make_weights_non_redundant(neighbors.values, weights.values)

        weights = pd.DataFrame(
            weights, index=neighbors.index, columns=neighbors.columns
        )

        self.weights = weights

    def _compute_hotspot(self, jobs=1):
        """Perform feature selection using local autocorrelation

        In addition to returning output, this also stores the output
        in `self.results`.

        Alias for `self.compute_autocorrelations`

        Parameters
        ----------
        jobs: int
            Number of parallel jobs to run

        Returns
        -------
        results : pandas.DataFrame
            A dataframe with four columns:

              - C: Scaled -1:1 autocorrelation coeficients
              - Z: Z-score for autocorrelation
              - Pval:  P-values computed from Z-scores
              - FDR:  Q-values using the Benjamini-Hochberg procedure

            Gene ids are in the index

        """

        results = compute_hs(
            self.counts,
            self.neighbors,
            self.weights,
            self.umi_counts,
            self.model,
            genes=self.adata.var_names,
            centered=True,
            jobs=jobs,
        )

        self.results = results

        return self.results

    def compute_autocorrelations(self, jobs=1):
        """Perform feature selection using local autocorrelation

        In addition to returning output, this also stores the output
        in `self.results`

        Parameters
        ----------
        jobs: int
            Number of parallel jobs to run

        Returns
        -------
        results : pandas.DataFrame
            A dataframe with four columns:

              - C: Scaled -1:1 autocorrelation coeficients
              - Z: Z-score for autocorrelation
              - Pval:  P-values computed from Z-scores
              - FDR:  Q-values using the Benjamini-Hochberg procedure

            Gene ids are in the index

        """
        return self._compute_hotspot(jobs)

    def compute_local_correlations(self, genes, jobs=1):
        """Define gene-gene relationships with pair-wise local correlations

        In addition to returning output, this method stores its result
        in `self.local_correlation_z`

        Parameters
        ----------
        genes: iterable of str
            gene identifies to compute local correlations on
            should be a smaller subset of all genes
        jobs: int
            Number of parallel jobs to run

        Returns
        -------
        local_correlation_z : pd.Dataframe
                local correlation Z-scores between genes
                shape is genes x genes
        """

        print(
            "Computing pair-wise local correlation on {} features...".format(len(genes))
        )
        counts_dense = self._counts_from_anndata(
            self.adata[:, genes],
            self.layer_key,
            dense=True,
            pandas=True,
        )

        lc, lcz = compute_hs_pairs_centered_cond(
            counts_dense,
            self.neighbors,
            self.weights,
            self.umi_counts,
            self.model,
            jobs=jobs,
        )

        self.local_correlation_c = lc
        self.local_correlation_z = lcz

        return self.local_correlation_z

    def create_modules(self, min_gene_threshold=20, core_only=True, fdr_threshold=0.05):
        """Groups genes into modules

        In addition to being returned, the results of this method are retained
        in the object at `self.modules`.  Additionally, the linkage matrix
        (in the same form as that of scipy.cluster.hierarchy.linkage) is saved
        in `self.linkage` for plotting or manual clustering.

        Parameters
        ----------
        min_gene_threshold: int
            Controls how small modules can be.  Increase if there are too many
            modules being formed.  Decrease if substructre is not being captured
        core_only: bool
            Whether or not to assign ambiguous genes to a module or leave unassigned
        fdr_threshold: float
            Correlation theshold at which to stop assigning genes to modules

        Returns
        -------
        modules: pandas.Series
            Maps gene to module number.  Unassigned genes are indicated with -1


        """

        gene_modules, Z = modules.compute_modules(
            self.local_correlation_z,
            min_gene_threshold=min_gene_threshold,
            fdr_threshold=fdr_threshold,
            core_only=core_only,
        )

        self.modules = gene_modules
        self.linkage = Z

        return self.modules

    def calculate_module_scores(self):
        """Calculate Module Scores

        In addition to returning its result, this method stores
        its output in the object at `self.module_scores`

        Returns
        -------
        module_scores: pandas.DataFrame
            Scores for each module for each gene
            Dimensions are genes x modules

        """

        modules_to_compute = sorted([x for x in self.modules.unique() if x != -1])

        print("Computing scores for {} modules...".format(len(modules_to_compute)))

        module_scores = {}
        for module in tqdm(modules_to_compute):
            module_genes = self.modules.index[self.modules == module]

            counts_dense = self._counts_from_anndata(
                self.adata[:, module_genes], self.layer_key, dense=True
            )

            scores = modules.compute_scores(
                counts_dense,
                self.model,
                self.umi_counts.values,
                self.neighbors.values,
                self.weights.values,
            )

            module_scores[module] = scores

        module_scores = pd.DataFrame(module_scores)
        module_scores.index = self.adata.obs_names

        self.module_scores = module_scores

        return self.module_scores

    def plot_local_correlations(
        self, mod_cmap="tab10", vmin=-8, vmax=8, z_cmap="RdBu_r", yticklabels=False,
        use_marsilea=False, width=10, height=10, font_size=10,
        add_dendrogram=True, add_module_colors=True, add_module_labels=True,
        show_values=False, value_fontsize=6, title="Local Gene Correlations"
    ):
        """Plots a clustergrid of the local correlation values

        Parameters
        ----------
        mod_cmap: valid matplotlib colormap str or object
            discrete colormap for module assignments on the left side
        vmin: float
            minimum value for colorscale for Z-scores
        vmax: float
            maximum value for colorscale for Z-scores
        z_cmap: valid matplotlib colormap str or object
            continuous colormap for correlation Z-scores
        yticklabels: bool
            Whether or not to plot all gene labels (seaborn mode only)
            Default is false as there are too many.  However
            if using this plot interactively you may with to set
            to true so you can zoom in and read gene names
        use_marsilea: bool, optional (default=False)
            If True, uses marsilea for enhanced visualization.
            If False, uses the default seaborn clustermap.
            Requires marsilea package: pip install marsilea
        width: float, optional (default=10)
            Figure width (marsilea mode only)
        height: float, optional (default=10)
            Figure height (marsilea mode only)
        font_size: int, optional (default=10)
            Font size for labels (marsilea mode only)
        add_dendrogram: bool, optional (default=True)
            Whether to add dendrogram (marsilea mode only)
        add_module_colors: bool, optional (default=True)
            Whether to add module color bars (marsilea mode only)
        add_module_labels: bool, optional (default=True)
            Whether to add module labels (marsilea mode only)
        show_values: bool, optional (default=False)
            Whether to show correlation values in cells (marsilea mode only)
        value_fontsize: int, optional (default=6)
            Font size for cell values (marsilea mode only)
        title: str, optional (default="Local Gene Correlations")
            Plot title (marsilea mode only)

        Returns
        -------
        ClusterGrid or marsilea Heatmap object
            Seaborn ClusterGrid if use_marsilea=False
            Marsilea Heatmap object if use_marsilea=True

        Examples
        --------
        >>> # Default seaborn visualization
        >>> hs.plot_local_correlations()
        >>>
        >>> # Enhanced marsilea visualization
        >>> hs.plot_local_correlations(use_marsilea=True, width=12, height=12)
        >>>
        >>> # Marsilea with custom settings
        >>> hs.plot_local_correlations(
        ...     use_marsilea=True,
        ...     show_values=True,
        ...     add_module_labels=True,
        ...     title="Hotspot Local Correlations"
        ... )
        """

        if use_marsilea:
            if not MARSILEA_AVAILABLE:
                print("Warning: marsilea package is not available.")
                print("Install it with: pip install marsilea")
                print("Falling back to default seaborn visualization.")
                use_marsilea = False

        if use_marsilea:
            return local_correlation_plot_marsilea(
                self.local_correlation_z,
                self.modules,
                self.linkage,
                mod_cmap=mod_cmap,
                vmin=vmin,
                vmax=vmax,
                z_cmap=z_cmap,
                width=width,
                height=height,
                font_size=font_size,
                add_dendrogram=add_dendrogram,
                add_module_colors=add_module_colors,
                add_module_labels=add_module_labels,
                show_values=show_values,
                value_fontsize=value_fontsize,
                title=title
            )
        else:
            return local_correlation_plot(
                self.local_correlation_z,
                self.modules,
                self.linkage,
                mod_cmap=mod_cmap,
                vmin=vmin,
                vmax=vmax,
                z_cmap=z_cmap,
                yticklabels=yticklabels,
            )
