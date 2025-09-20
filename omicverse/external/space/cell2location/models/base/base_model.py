# -*- coding: utf-8 -*-
"""Base model class"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ...plt.plot_heatmap import clustermap


# base model class - defining shared methods but not the model itself
class BaseModel:
    r"""Base class for pymc3 and pyro models.

    :param X_data: Numpy array of gene expression (cols) in spatial locations (rows)
    :param n_fact: Number of factors
    :param n_iter: Number of training iterations
    :param learning_rate: ADAM learning rate for optimising Variational inference objective
    :param data_type: theano data type used to store parameters ('float32' for single, 'float64' for double precision)
    :param total_grad_norm_constraint: gradient constraints in optimisation
    :param verbose: print diagnostic messages?
    :param var_names: Variable names (e.g. gene identifiers)
    :param var_names_read: Readable variable names (e.g. gene symbol)
    :param obs_names: Observation names (e.g. cell or spot id)
    :param fact_names: Factor names
    :param sample_id: Sample identifiers (e.g. different experiments)
    """

    def __init__(
        self,
        X_data: np.ndarray,
        n_fact: int = 10,
        data_type: str = "float32",
        n_iter: int = 200000,
        learning_rate=0.001,
        total_grad_norm_constraint=200,
        verbose=True,
        var_names=None,
        var_names_read=None,
        obs_names=None,
        fact_names=None,
        sample_id=None,
    ):
        # Initialise parameters
        self.X_data = X_data
        self.n_fact = n_fact
        self.n_var = X_data.shape[1]
        self.n_obs = X_data.shape[0]
        self.data_type = data_type
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.total_grad_norm_constraint = total_grad_norm_constraint
        self.verbose = verbose
        self.fact_filt = None
        self.gene_loadings = None
        self.minibatch_size = None
        self.minibatch_seed = None
        self.extra_data = None  # input data
        self.extra_data_tt = None  # minibatch parameters

        # add essential annotations
        if var_names is None:
            self.var_names = pd.Series(
                ["g_" + str(i) for i in range(self.n_var)], index=["g_" + str(i) for i in range(self.n_var)]
            )
        else:
            self.var_names = pd.Series(var_names, index=var_names)

        if var_names_read is None:
            self.var_names_read = pd.Series(self.var_names, index=self.var_names)
        else:
            self.var_names_read = pd.Series(var_names_read, index=self.var_names)

        if obs_names is None:
            self.obs_names = pd.Series(
                ["c_" + str(i) for i in range(self.n_obs)], index=["c_" + str(i) for i in range(self.n_obs)]
            )
        else:
            self.obs_names = pd.Series(obs_names, index=obs_names)

        if fact_names is None:
            self.fact_names = pd.Series(["fact_" + str(i) for i in range(self.n_fact)])
        else:
            self.fact_names = pd.Series(fact_names)

        if sample_id is None:
            self.sample_id = pd.Series(["sample" for i in range(self.n_obs)], index=self.obs_names)
        else:
            self.sample_id = pd.Series(sample_id, index=self.obs_names)

    def plot_prior_vs_data(self, data_target_name="data_target", data_node="X_data", log_transform=True):
        r"""Plot data vs a single sample from the prior in a 2D histogram
        Uses self.X_data and self.prior_trace['data_target'].
        :param data_node: name of the object slot containing data
        """

        if type(data_node) is str:
            data_node = getattr(self, data_node)

        if type(data_target_name) is str:
            data_target_name = self.prior_trace[data_target_name]

        # If there are multiple prior samples, expand the data array
        if len(data_target_name.shape) > 2:
            data_node = np.array([data_node for _ in range(data_target_name.shape[0])])

        if log_transform:
            data_node = np.log10(data_node + 1)
            data_target_name = np.log10(data_target_name + 1)

        plt.hist2d(data_node.flatten(), data_target_name.flatten(), bins=50, norm=matplotlib.colors.LogNorm())
        plt.xlabel("Data, log10(nUMI)")
        plt.ylabel("Prior sample, log10(nUMI)")
        plt.title("UMI counts (all spots, all genes)")
        plt.tight_layout()

    @staticmethod
    def align_plot_stability(fac1, fac2, name1, name2, align=True, return_aligned=False):
        r"""Align columns between two np.ndarrays using scipy.optimize.linear_sum_assignment,
        then plot correlations between columns in fac1 and fac2, ordering fac2 according to alignment

        :param fac1: np.ndarray 1, factors in columns
        :param fac2: np.ndarray 2, factors in columns
        :param name1: axis x name
        :param name2: axis y name
        :param align: boolean, match columns in fac1 and fac2 using linear_sum_assignment?
        """

        corr12 = np.corrcoef(fac1, fac2, False)
        ind_top = np.arange(0, fac1.shape[1])
        ind_right = np.arange(0, fac1.shape[1]) + fac1.shape[1]
        corr12 = corr12[ind_top, :][:, ind_right]
        corr12[np.isnan(corr12)] = -1

        if align:
            img = corr12[:, linear_sum_assignment(2 - corr12)[1]]
        else:
            img = corr12

        plt.imshow(img)

        plt.title(f"Training initialisation \n {name1} vs {name2}")
        plt.xlabel(name2)
        plt.ylabel(name1)

        plt.tight_layout()

        if return_aligned:
            return linear_sum_assignment(2 - corr12)[1]

    def generate_cv_data(self, n: int = 2, discrete: bool = True, non_discrete_mean_var: float = 1):
        r"""Generate X_data for molecular cross-validation by sampling molecule counts
        with np.random.binomial

        :param n: number of cross-validation folds of equal size to generate, for now, only n=2 is implemented
        """

        if n != 2:
            raise ValueError("only n=2 is implemented for molecular cross-validation")

        self.X_data_sample = {}
        if discrete:
            self.X_data_sample[0] = np.random.binomial(self.X_data.astype("int64"), 1 / n).astype(np.float)
        else:
            shape = (self.X_data / n) ** 2 / ((self.X_data / n) * non_discrete_mean_var)
            scale = ((self.X_data / n) * non_discrete_mean_var) / (self.X_data / n)
            self.X_data_sample[0] = np.random.gamma(shape=shape, scale=scale, size=scale.shape).astype(np.float)
        self.X_data_sample[1] = np.abs(self.X_data - self.X_data_sample[0])

    def bootstrap_data(self, n=10, downsampling_p=0.8, discrete=True, non_discrete_mean_var=1):
        r"""Generate X_data for bootstrap analysis by sampling molecule counts
        with np.random.binomial

        :param n: number of bootstrap samples to generate
        :param downsampling_p: sample this proportion of values
        :param non_discrete_mean_var: low means lower variance
        """

        self.X_data_sample = {}

        for i in range(n):
            if discrete:
                self.X_data_sample[i] = np.random.binomial(self.X_data.astype("int64"), downsampling_p).astype(np.float)
            else:
                shape = (self.X_data * downsampling_p) ** 2 / (self.X_data * downsampling_p * non_discrete_mean_var)
                scale = (self.X_data * downsampling_p * non_discrete_mean_var) / (self.X_data * downsampling_p)
                self.X_data_sample[i] = np.random.gamma(shape=shape, scale=scale, size=scale.shape).astype(np.float)

    def plot_posterior_mu_vs_data(self, mu_node_name="mu", data_node="X_data"):
        r"""Plot expected value of the model (e.g. mean of poisson distribution)

        :param mu_node_name: name of the object slot containing expected value
        :param data_node: name of the object slot containing data
        """

        if type(mu_node_name) is str:
            mu = getattr(self, mu_node_name)
        else:
            mu = mu_node_name

        if type(data_node) is str:
            data_node = getattr(self, data_node)

        plt.hist2d(
            np.log10(data_node.flatten() + 1), np.log10(mu.flatten() + 1), bins=50, norm=matplotlib.colors.LogNorm()
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("Data, log10(nUMI)")
        plt.ylabel("Posterior sample, log10(nUMI)")
        plt.title("UMI counts (all cell, all genes)")
        plt.tight_layout()

    def plot_history(self, iter_start=0, iter_end=-1, mean_field_slot=None, log_y=True, ax=None):
        r"""Plot training history

        :param iter_start: omit initial iterations from the plot
        :param iter_end: omit last iterations from the plot
        """

        if ax is None:
            ax = plt
            ax.set_xlabel = plt.xlabel
            ax.set_ylabel = plt.ylabel

        if mean_field_slot is None:
            mean_field_slot = self.hist.keys()

        if isinstance(mean_field_slot, str):
            mean_field_slot = [mean_field_slot]

        for i in mean_field_slot:
            if iter_end == -1:
                iter_end = np.array(self.hist[i]).flatten().shape[0]

            y = np.array(self.hist[i]).flatten()[iter_start:iter_end]
            if log_y:
                y = np.log10(y)
            ax.plot(np.arange(iter_start, iter_end), y, label="train")
            ax.set_xlabel("Training epochs")
            ax.set_ylabel("Reconstruction accuracy (ELBO loss)")
            ax.legend()
            plt.tight_layout()

    def plot_validation_history(self, start_step=0, end_step=-1, mean_field_slot="init_1", log_y=True, ax=None):
        r"""Plot model loss (NB likelihood + penalty) using the model on training and validation data"""

        if ax is None:
            ax = plt
            ax.set_xlabel = plt.xlabel
            ax.set_ylabel = plt.ylabel

        if end_step == -1:
            end_step = np.array(self.training_hist[mean_field_slot]).flatten().shape[0]

        y = np.array(self.training_hist[mean_field_slot]).flatten()[start_step:end_step]
        if log_y:
            y = np.log10(y)
        ax.plot(np.arange(start_step, end_step), y, label="train")

        y = np.array(self.validation_hist[mean_field_slot]).flatten()[start_step:end_step]
        if log_y:
            y = np.log10(y)
        ax.plot(np.arange(start_step, end_step), y, label="validation")
        ax.set_xlabel("Training epochs")
        ax.set_ylabel("Reconstruction accuracy (log10 NB + L2 loss) / ELBO loss")
        ax.legend()
        plt.tight_layout()

    def plot_posterior_vs_data(self, gene_fact_name="gene_factors", cell_fact_name="cell_factors"):
        # extract posterior samples of scaled gene and cell factors (before the final likelihood step)
        gene_fact_s = self.samples["post_sample_means"][gene_fact_name]
        cell_factors_scaled_s = self.samples["post_sample_means"][cell_fact_name]

        # compute the poisson rate
        self.mu = np.dot(cell_factors_scaled_s, gene_fact_s.T)

        plt.hist2d(
            np.log10(self.X_data.flatten() + 1),
            np.log10(self.mu.flatten() + 1),
            bins=50,
            norm=matplotlib.colors.LogNorm(),
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("Data, log10(nUMI)")
        plt.ylabel("Posterior sample, log10(nUMI)")
        plt.title("UMI counts (all cell, all genes)")
        plt.tight_layout()

    def set_fact_filt(self, fact_filt):
        r"""Specify which factors are not relevant/ not expressed.
        It is currently used to filter results shown by .print_gene_loadings() and .plot_gene_loadings()

        :param fact_filt: logical array specifying which factors are to be retained
        """
        self.fact_filt = fact_filt

    def apply_fact_filt(self, df):
        r"""Select DataFrame columns by factor filter which was saved in the model object

        :param df: pd.DataFrame
        """
        if self.fact_filt is not None:
            df = df.iloc[:, self.fact_filt]

        return df

    def print_gene_loadings(
        self, gene_fact_name="gene_factors", loadings_attr="gene_loadings", top_n=10, gene_filt=None, fact_filt=None
    ):
        r"""Print top-10 genes for each factor in gene loadings matrix.

        :param gene_fact_name: model parameter name to extract from samples if self.gene_loadings doesn't exist
        :param loadings_attr: model object attribute name that stores loadings
        :param top_n: number of genes to plot for each factor
        :param gene_filt: boolean filter for genes (e.g. restrict printed markers to TFs)
        :param fact_filt: boolean filter for factors
        """

        if getattr(self, loadings_attr) is None:
            gene_f = self.samples["post_sample_means"][gene_fact_name]
            setattr(
                self, loadings_attr, pd.DataFrame.from_records(gene_f, index=self.var_names, columns=self.fact_names)
            )

        gene_loadings = getattr(self, loadings_attr).copy()

        if fact_filt is not None:
            gene_loadings = gene_loadings.loc[:, fact_filt]

        if gene_filt is not None:
            gene_loadings = gene_loadings.loc[gene_filt, :]

        rows = []
        index = []
        columns = [f"top-{i + 1}" for i in range(top_n)]

        for clmn_ in gene_loadings.columns:
            loading_ = gene_loadings[clmn_].sort_values(ascending=False)
            index.append(clmn_)
            row = [f"{self.var_names_read[i]}: {loading_[i]:.2}" for i in loading_.head(top_n).index]
            rows.append(row)

        return pd.DataFrame(rows, index=index, columns=columns)

    def plot_gene_loadings(
        self,
        sel_var_names,
        var_names,
        gene_fact_name="gene_factors",
        loadings_attr="gene_loadings",
        figsize=(15, 7),
        cluster_factors=False,
        cluster_genes=True,
        cmap="viridis",
        title="",
        fact_filt=None,
        fun_type="heatmap",
        return_linkage=False,
    ):
        r"""Plot gene loadings as a heatmap

        :param sel_var_names: list of variable names to select
        :param var_names: `sel_var_names` matches some names in var_names
            which identifies each gene in gene loadings
        :param gene_fact_name: model parameter name to extract from samples if self.gene_loadings doesn't exist
        :param figsize: histogram figure size
        :param cluster_factors: hierarchically cluster factors?
        :param cluster_genes: hierarchically cluster genes?
        :param cmap: matplotlib colormap
        :param title: plot title
        :param fact_filt: boolean or character filter for factors
        """

        if getattr(self, loadings_attr) is None:
            gene_f = self.samples["post_sample_means"][gene_fact_name]
            setattr(
                self, loadings_attr, pd.DataFrame.from_records(gene_f, index=self.var_names, columns=self.fact_names)
            )

        gene_loadings = getattr(self, loadings_attr).copy()

        if fact_filt is not None:
            gene_loadings = gene_loadings.loc[:, fact_filt]

        # selected variable index
        sel_index = var_names.isin(sel_var_names)
        key_markers = gene_loadings.loc[sel_index, :]
        key_markers.index = self.var_names_read[sel_index]  # use readable names
        key_markers = key_markers.loc[sel_var_names, :]  # reorder

        clustermap(
            key_markers,
            cluster_rows=cluster_genes,
            cluster_cols=cluster_factors,
            return_linkage=return_linkage,
            figure_size=figsize,
            cmap=cmap,
            title=title,
            fun_type=fun_type,
        )

    def plot_loading_distribution(self, loadings_name="gene_factors", loadings=None):
        r"""Plot histogram for each loading (column-wise)

        :param loadings_name: character name to be extracted from `self.samples['post_sample_means']`
        :param loadings: np.ndarray to be plotted column-wise. Supersedes `loadings_name`.
        """

        if loadings is None:
            loadings = np.log10(self.samples["post_sample_means"][loadings_name])

        for i in range(loadings.shape[1]):
            plt.subplot(loadings.shape[1], 1, i + 1)
            plt.hist(loadings[:, i], bins=20)
            plt.xlabel(loadings_name + " value")
            plt.title("Factor: " + str(i))
        plt.tight_layout()

    def factor_expressed_plot(
        self,
        shape_cut=4,
        rate_cut=15,
        sample_type="post_sample_means",
        shape="cell_fact_shape_hyp",
        rate="cell_fact_rate_hyp",
        shape_lab="cell_factors, Gamma shape",
        rate_lab="cell_factors, Gamma rate",
        invert_selection=False,
    ):
        r"""Show which factors are expressed on a scatterplot of their regularising priors

        :param shape_cut: Gamma shape cutoff below which factors are expressed
        :param rate_cut: Gamma rate cutoff below which factors are expressed
        :param sample_type: which posterior summary to look at, default 'post_sample_means'
        :param shape: parameter name for the Gamma shape of each factor, default 'cell_fact_mu_hyp'
        :param rate: parameter name for the Gamma rate of each factor, default 'cell_fact_sd_hyp'
        :param shape_lab: axis label for shape
        :param rate_lab: axis label for rate
        :param invert_selection: if values below cutoffs are for not expressed, set invert_selection to True.
        """

        # Expression shape and rate across cells
        shape = self.samples[sample_type][shape]
        rate = self.samples[sample_type][rate]
        plt.scatter(shape, rate)
        plt.xlabel(shape_lab)
        plt.ylabel(rate_lab)
        plt.vlines(shape_cut, 0, rate_cut)
        plt.hlines(rate_cut, 0, shape_cut)

        low_lab = high_lab = "expressed"
        if not invert_selection:
            high_lab = f"not {high_lab}"
        else:
            low_lab = f"not {low_lab}"

        plt.text(shape_cut - 0.5 * shape_cut, rate_cut - 0.5 * rate_cut, low_lab)
        plt.text(shape_cut + 0.1 * shape_cut, rate_cut + 0.1 * rate_cut, high_lab)

        return {"shape": shape, "rate": rate, "shape_cut": shape_cut, "rate_cut": rate_cut}

    def plot_reconstruction_history(self, n_type="cv", start_step=0, end_step=45):
        r"""Plot reconstruction error using the model on training and validation data"""

        # Extract RMSE from cross-validation parameter tracking
        rmse = np.array([[i["rmse_total"], i["rmse_total_cv"]] for i in self.tracking["init_1"]["rmse"]])
        rmse_cv = (self.X_data_sample[1] - self.X_data_sample[0]) * (self.X_data_sample[1] - self.X_data_sample[0])

        plt.plot(
            np.arange(start_step, end_step) * self.tracking_every,
            np.log10(rmse[start_step:end_step, 0]),
            label="model on training data",
        )
        plt.plot(
            np.arange(start_step, end_step) * self.tracking_every,
            np.log10(rmse[start_step:end_step, 1]),
            label="model on cross-validation data",
        )
        plt.hlines(
            np.log10(np.sqrt(rmse_cv.mean())),
            start_step,
            end_step * self.tracking_every,
            label="reconstructing using cross-validation data",
        )
        plt.xlabel("Training iterations")
        plt.ylabel("Reconstruction accuracy (log10 RMSE)")
        plt.legend()

    def export2adata(self, adata, slot_name="mod"):
        r"""Add posterior mean and sd for all parameters to unstructured data `adata.uns['mod']`.

        :param adata: anndata object
        """
        # add factor filter and samples of all parameters to unstructured data
        adata.uns[slot_name] = {}

        adata.uns[slot_name]["mod_name"] = str(self.__class__.__name__)
        adata.uns[slot_name]["fact_filt"] = self.fact_filt
        adata.uns[slot_name]["fact_names"] = self.fact_names.tolist()
        adata.uns[slot_name]["var_names"] = self.var_names.tolist()
        adata.uns[slot_name]["obs_names"] = self.obs_names.tolist()
        adata.uns[slot_name]["post_sample_means"] = self.samples["post_sample_means"]
        adata.uns[slot_name]["post_sample_sds"] = self.samples["post_sample_sds"]
        adata.uns[slot_name]["post_sample_q05"] = self.samples["post_sample_q05"]
        adata.uns[slot_name]["post_sample_q95"] = self.samples["post_sample_q95"]

        return adata
