# -*- coding: utf-8 -*-
r"""Archetypal tissue zones."""

import matplotlib
import matplotlib.pyplot as plt

# +
import numpy as np
import pandas as pd

from ...models.base.base_model import BaseModel


# defining the model itself
class ArchetypalAnalysis(BaseModel):
    r"""This model identified archetypal tissue zones using PCHA algorithm.

    If you would like to use this function please first run `pip install py_pcha` to install the dependency.

    This model takes the absolute cell density inferred by cell2location as input
    to archetypal analysis aimed to find a set of most distinct tissue zones,
    which can be spatially interlaced unlike standard clustering.

    To perform this analysis we initialise the model and train it several times to evaluate consitency.
    This class wraps around py_pcha package to perform training, visualisation, export of the results.

    For more details on Archetypal Analysis using Principle Convex Hull Analysis (PCHA)
    see https://github.com/ulfaslak/py_pcha.

    Note
    -----
        Archetypes are exchangeable so while you find archetypes with consistent cell type composition,
        every time you train the model you get those archetypes in a different order.

    Density :math:`w_{sf}` of each cell type `f` across locations `s` is modelled as an additive function of
    the archetype `r`. This means the density of one cell type in one location can
    be explained by 2 distinct archetypes `r`.

    Cell type density is therefore a function of the following non-negative components:

    .. math::
        w_{sf} = \sum_{r} ({i_{sr} \: k_{rf} \: m_{f}})

    Components
      * :math:`k_{rf}` represents the proportion of cells of each type `f` that correspond to each
        co-located combination `r`, normalised for total abundance of each cell type :math:`m_{f}`.
      * :math:`m_{f}` total abundance of each cell type.
      * :math:`i_{sr}` represents the contribution of each archetype `r` in each location `s`,
        constrained as follows:

      .. math::
          \sum_{r} i_{sr} = 1


    In practice :math:`q_{rf} = k_{rf} \: m_{f}` is obtained by performing archetypal analysis
    and normalised by the sum across combinations `r` to obtain :math:`k_{rf}`:

    .. math::
        k_{rf} = q_{rf} / (\sum_{r} q_{rf})

    Note
    ----
        So, the model reports the proportion of cells of each type that belong to each combination
        (parameter called 'cell_type_fractions').
        For example, 81% of Astro_2 are found in fact_28.
        This way we account for the absolute abundance of each cell type.

    Parameters
    ----------
    n_fact :
        Maximum number archetypes
    X_data :
        Numpy array of the cell abundance (cols) in locations (rows)
    n_iter :
        number of training iterations
    verbose :
        var_names, var_names_read, obs_names, fact_names, sample_id: See parent class BaseModel for details.
    init, random_state, alpha, l1_ratio:
        arguments for sklearn.decomposition.NMF with sensible defaults see help(sklearn.decomposition.NMF) for more details
    pcha_kwd_args :
        dictionary with more keyword arguments for py_pcha.PCHA

    """

    def __init__(
        self,
        n_fact: int,
        X_data: np.ndarray,
        n_iter=5000,
        verbose=True,
        var_names=None,
        var_names_read=None,
        obs_names=None,
        fact_names=None,
        sample_id=None,
        random_state=0,
        pcha_kwd_args={},
    ):
        ############# Initialise parameters ################
        super().__init__(
            X_data, n_fact, 0, n_iter, 0, 0, verbose, var_names, var_names_read, obs_names, fact_names, sample_id
        )

        self.location_factors_df = None
        self.X_data_sample = None

        self.random_state = random_state
        np.random.seed(random_state)
        self.pcha_kwd_args = pcha_kwd_args

    def fit(self, n=3, n_type="restart"):
        """Find parameters using py_pcha.PCHA, optionally restart several times,
        and export parameters to self.samples['post_sample_means']

        Parameters
        ----------
        n :
            number of independent initialisations (Default value = 3)
        n_type :
            type of repeated initialisation:

            * 'restart' to pick different initial value,
            * 'cv' for molecular cross-validation - splits counts into n datasets,
              for now, only n=2 is implemented
            * 'bootstrap' for fitting the model to multiple downsampled datasets.
              Run `mod.bootstrap_data()` to generate variants of data (Default value = 'restart')

        Returns
        -------
        None
            exported parameters in self.samples['post_sample_means']

        """

        self.models = {}
        self.results = {}
        self.samples = {}

        self.n_type = n_type

        if np.isin(n_type, ["bootstrap"]):
            if self.X_data_sample is None:
                self.bootstrap_data(n=n)
        elif np.isin(n_type, ["cv"]):
            if self.X_data_sample is None:
                self.generate_cv_data()  # cv data added to self.X_data_sample

        init_names = ["init_" + str(i + 1) for i in np.arange(n)]

        for i, name in enumerate(init_names):
            # when type is molecular cross-validation or bootstrap,
            # replace self.x_data with new data
            if np.isin(n_type, ["cv", "bootstrap"]):
                self.x_data = self.X_data_sample[i]
            else:
                self.x_data = self.X_data

            from py_pcha import PCHA

            XC, S, C, SSE, varexpl = PCHA(self.x_data.T, noc=self.n_fact, maxiter=self.n_iter, **self.pcha_kwd_args)
            self.results[name] = {
                "post_sample_means": {
                    "location_factors": np.array(S.T),
                    "cell_type_factors": np.array(XC),
                    "nUMI_factors": (np.array(S.T) * np.array(XC).sum(0)),
                    "C": np.array(C),
                    "SSE": np.array(SSE),
                    "varexpl": np.array(varexpl),
                },
                "post_sample_sds": None,
                "post_sample_q05": None,
                "post_sample_q95": None,
            }
            self.samples = self.results[name]

            # plot training history
            if self.verbose:
                print(f"{name} - variance explained: {varexpl}")

    def evaluate_stability(self, node_name, align=True, n_samples=1000):
        """Evaluate stability of the solution between training initialisations
        (correlates the values of archetypes between training initialisations)

        Parameters
        ----------
        node_name :
            name of the parameter to evaluate, see `self.samples['post_sample_means'].keys()`
            Factors should be in columns.
        align :
            boolean, match factors between training restarts using linear_sum_assignment? (Default value = True)
        n_samples:
            does nothing, added to preserve call signature consistency with bayesian models

        Returns
        -------
        None
            plots comparing all training initialisations to initialisation 1.

        """

        n_plots = len(self.results.keys()) - 1
        ncol = int(np.min((n_plots, 3)))
        nrow = int(np.ceil(n_plots / ncol))
        for i in range(n_plots):
            plt.subplot(nrow, ncol, i + 1)
            self.align_plot_stability(
                self.results["init_" + str(1)]["post_sample_means"][node_name],
                self.results["init_" + str(i + 2)]["post_sample_means"][node_name],
                str(1),
                str(i + 2),
                align=align,
            )

    def sample_posterior(
        self, node="all", n_samples=1000, save_samples=False, return_samples=True, mean_field_slot="init_1"
    ):
        """This function does nothing but added to preserve call signature with future Bayesian versions of the model."""
        pass

    def compute_expected(self):
        """Compute expected abundance of each cell type in each location."""

        # compute the poisson rate
        self.mu = np.dot(
            self.samples["post_sample_means"]["location_factors"],
            self.samples["post_sample_means"]["cell_type_factors"].T,
        )

    def compute_expected_fact(self, fact_ind=None):
        """Compute expected abundance of each cell type in each location
        that comes from a subset of archetypes.

        Parameters
        ----------
        fact_ind :
             (Default value = None)

        """

        if fact_ind is None:
            fact_ind = self.fact_filt

        # compute the poisson rate
        self.mu = np.dot(
            self.samples["post_sample_means"]["location_factors"][:, fact_ind],
            self.samples["post_sample_means"]["cell_type_factors"].T[fact_ind, :],
        )

    def plot_posterior_mu_vs_data(self, mu_node_name="mu", data_node="X_data"):
        """Plot expected value (of cell density) of the model against observed input data:
        2D histogram, where each point is each point in the input data matrix

        Parameters
        ----------
        mu_node_name :
            name of the object slot containing expected value (Default value = 'mu')
        data_node :
            name of the object slot containing data (Default value = 'X_data')

        """

        if type(mu_node_name) is str:
            mu = getattr(self, mu_node_name)
        else:
            mu = mu_node_name

        if type(data_node) is str:
            data_node = getattr(self, data_node)

        plt.hist2d(data_node.flatten(), mu.flatten(), bins=50, norm=matplotlib.colors.LogNorm())
        plt.xlabel("Data, values")
        plt.ylabel("Posterior sample, values")
        plt.title("UMI counts (all spots, all genes)")
        plt.tight_layout()

    def sample2df(self, node_name="nUMI_factors", ct_node_name="cell_type_factors"):
        """Export archetypes and their profile across locations as Pandas data frames.

        Parameters
        ----------
        node_name :
            name of the location loading model parameter to be exported (Default value = 'nUMI_factors')
        ct_node_name :
            name of the cell_type loadings model parameter to be exported (Default value = 'cell_type_factors')

        Returns
        -------
        None
            8 Pandas dataframes added to model object:
            .cell_type_loadings, .cell_factors_sd, .cell_factors_q05, .cell_factors_q95
            .gene_loadings, .gene_loadings_sd, .gene_loadings_q05, .gene_loadings_q95

        """

        # export location factors
        self.location_factors_df = pd.DataFrame.from_records(
            self.samples["post_sample_means"][node_name],
            index=self.obs_names,
            columns=["mean_" + node_name + i for i in self.fact_names],
        )

        self.cell_type_loadings = pd.DataFrame.from_records(
            self.samples["post_sample_means"][ct_node_name],
            index=self.var_names,
            columns=["mean_" + ct_node_name + i for i in self.fact_names],
        )

        self.cell_type_fractions = (self.cell_type_loadings.T / self.cell_type_loadings.sum(1)).T

        self.cell_type_loadings_sd = None
        self.cell_type_loadings_q05 = None
        self.cell_type_loadings_q95 = None

    def annotate_adata(self, adata):
        """Add location loadings to anndata.obs

        Parameters
        ----------
        adata :
            anndata object to annotate

        Returns
        -------
        anndata object
            updated anndata object

        """

        if self.location_factors_df is None:
            self.sample2df()

        # location factors
        # add location factors to adata
        adata.obs[self.location_factors_df.columns] = self.location_factors_df.loc[adata.obs.index, :]

        return adata
