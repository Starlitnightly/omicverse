from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from pyro import clear_param_store
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model.base import BaseModelClass, PyroSampleMixin, PyroSviTrainMixin
from scvi.utils import setup_anndata_dsp

from ...cluster_averages import compute_cluster_averages
from ..base._pyro_base_reference_module import RegressionBaseModule
from ..base._pyro_mixin import PltExportMixin, QuantileMixin
from ._reference_module import RegressionBackgroundDetectionTechPyroModel


class RegressionModel(QuantileMixin, PyroSampleMixin, PyroSviTrainMixin, PltExportMixin, BaseModelClass):
    """
    Model which estimates per cluster average mRNA count account for batch effects. User-end model class.

    https://github.com/BayraktarLab/cell2location

    Parameters
    ----------
    adata
        single-cell AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    use_gpu
        Use the GPU?
    **model_kwargs
        Keyword args for :class:`~scvi.external.LocationModelLinearDependentWMultiExperimentModel`

    Examples
    --------
    TODO add example
    >>>
    """

    def __init__(
        self,
        adata: AnnData,
        model_class=None,
        use_average_as_initial: bool = True,
        **model_kwargs,
    ):
        # in case any other model was created before that shares the same parameter names.
        clear_param_store()

        super().__init__(adata)

        if model_class is None:
            model_class = RegressionBackgroundDetectionTechPyroModel

        # annotations for cell types
        self.n_factors_ = self.summary_stats["n_labels"]
        self.factor_names_ = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping
        # annotations for extra categorical covariates
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            self.extra_categoricals_ = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            self.n_extra_categoricals_ = self.extra_categoricals_.n_cats_per_key
            model_kwargs["n_extra_categoricals"] = self.n_extra_categoricals_

        # use per class average as initial value
        if use_average_as_initial:
            # compute cluster average expression
            aver = self._compute_cluster_averages(key=REGISTRY_KEYS.LABELS_KEY)
            model_kwargs["init_vals"] = {"per_cluster_mu_fg": aver.values.T.astype("float32") + 0.0001}

        self.module = RegressionBaseModule(
            model=model_class,
            n_obs=self.summary_stats["n_cells"],
            n_vars=self.summary_stats["n_vars"],
            n_factors=self.n_factors_,
            n_batch=self.summary_stats["n_batch"],
            **model_kwargs,
        )
        self._model_summary_string = f'RegressionBackgroundDetectionTech model with the following params: \nn_factors: {self.n_factors_} \nn_batch: {self.summary_stats["n_batch"]} '
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs).astype("int64")
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: Optional[int] = None,
        batch_size: int = 2500,
        train_size: float = 1,
        lr: float = 0.002,
        **kwargs,
    ):
        """Train the model with useful defaults

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        train_size
            Size of training set in the range [0.0, 1.0].
        batch_size
            Minibatch size to use during training. If `None`, no minibatching occurs and all
            data is copied to device (e.g., GPU).
        lr
            Optimiser learning rate (default optimiser is :class:`~pyro.optim.ClippedAdam`).
            Specifying optimiser via plan_kwargs overrides this choice of lr.
        kwargs
            Other arguments to scvi.model.base.PyroSviTrainMixin().train() method
        """

        kwargs["max_epochs"] = max_epochs
        kwargs["batch_size"] = batch_size
        kwargs["train_size"] = train_size
        kwargs["lr"] = lr

        super().train(**kwargs)

    def _compute_cluster_averages(self, key=REGISTRY_KEYS.LABELS_KEY):
        """
        Compute average per cluster (key=REGISTRY_KEYS.LABELS_KEY) or per batch (key=REGISTRY_KEYS.BATCH_KEY).

        Returns
        -------
        pd.DataFrame with variables in rows and labels in columns
        """
        # find cell label column
        label_col = self.adata_manager.get_state_registry(key).original_key

        # find data slot
        x_dict = self.adata_manager.data_registry["X"]
        if x_dict["attr_name"] == "X":
            use_raw = False
        else:
            use_raw = True
        if x_dict["attr_name"] == "layers":
            layer = x_dict["attr_key"]
        else:
            layer = None

        # compute mean expression of each gene in each cluster/batch
        aver = compute_cluster_averages(self.adata, labels=label_col, use_raw=use_raw, layer=layer)

        return aver

    def export_posterior(
        self,
        adata,
        sample_kwargs: Optional[dict] = None,
        export_slot: str = "mod",
        add_to_varm: list = ["means", "stds", "q05", "q95"],
        scale_average_detection: bool = True,
        use_quantiles: bool = False,
    ):
        """
        Summarise posterior distribution and export results (cell abundance) to anndata object:
        1. adata.obsm: Estimated references expression signatures (average mRNA count in each cell type),
            as pd.DataFrames for each posterior distribution summary `add_to_varm`,
            posterior mean, sd, 5% and 95% quantiles (['means', 'stds', 'q05', 'q95']).
            If export to adata.varm fails with error, results are saved to adata.var instead.
        2. adata.uns: Posterior of all parameters, model name, date,
            cell type names ('factor_names'), obs and var names.

        Parameters
        ----------
        adata
            anndata object where results should be saved
        sample_kwargs
            arguments for self.sample_posterior (generating and summarising posterior samples), namely:
                num_samples - number of samples to use (Default = 1000).
                batch_size - data batch size (keep low enough to fit on GPU, default 2048).
                use_gpu - use gpu for generating samples?
        export_slot
            adata.uns slot where to export results
        add_to_varm
            posterior distribution summary to export in adata.varm (['means', 'stds', 'q05', 'q95']).
        use_quantiles
            compute quantiles directly (True, more memory efficient) or use samples (False, default).
            If True, means and stds cannot be computed so are not exported and returned.
        Returns
        -------

        """

        sample_kwargs = sample_kwargs if isinstance(sample_kwargs, dict) else dict()

        # get posterior distribution summary
        if use_quantiles:
            add_to_varm = [i for i in add_to_varm if (i not in ["means", "stds"]) and ("q" in i)]
            if len(add_to_varm) == 0:
                raise ValueError("No quantiles to export - please add add_to_obsm=['q05', 'q50', 'q95'].")
            self.samples = dict()
            for i in add_to_varm:
                q = float(f"0.{i[1:]}")
                self.samples[f"post_sample_{i}"] = self.posterior_quantile(q=q, **sample_kwargs)
        else:
            # generate samples from posterior distributions for all parameters
            # and compute mean, 5%/95% quantiles and standard deviation
            self.samples = self.sample_posterior(**sample_kwargs)

        # export posterior distribution summary for all parameters and
        # annotation (model, date, var, obs and cell type names) to anndata object
        adata.uns[export_slot] = self._export2adata(self.samples)

        # export estimated expression in each cluster
        # first convert np.arrays to pd.DataFrames with cell type and observation names
        # data frames contain mean, 5%/95% quantiles and standard deviation, denoted by a prefix
        for k in add_to_varm:
            sample_df = self.sample2df_vars(
                self.samples,
                site_name="per_cluster_mu_fg",
                summary_name=k,
                name_prefix="",
            )
            if scale_average_detection and ("detection_y_c" in list(self.samples[f"post_sample_{k}"].keys())):
                sample_df = sample_df * self.samples[f"post_sample_{k}"]["detection_y_c"].mean()
            try:
                adata.varm[f"{k}_per_cluster_mu_fg"] = sample_df.loc[adata.var.index, :]
            except ValueError:
                # Catching weird error with obsm: `ValueError: value.index does not match parent’s axis 1 names`
                adata.var[sample_df.columns] = sample_df.loc[adata.var.index, :]

        return adata

    def plot_QC(
        self,
        summary_name: str = "means",
        use_n_obs: int = 1000,
        scale_average_detection: bool = True,
    ):
        """
        Show quality control plots:
        1. Reconstruction accuracy to assess if there are any issues with model training.
            The plot should be roughly diagonal, strong deviations signal problems that need to be investigated.
            Plotting is slow because expected value of mRNA count needs to be computed from model parameters. Random
            observations are used to speed up computation.

        2. Estimated reference expression signatures (accounting for batch effect)
            compared to average expression in each cluster. We expect the signatures to be different
            from average when batch effects are present, however, when this plot is very different from
            a perfect diagonal, such as very low values on Y-axis, non-zero density everywhere)
            it indicates problems with signature estimation.

        Parameters
        ----------
        summary_name
            posterior distribution summary to use ('means', 'stds', 'q05', 'q95')

        Returns
        -------

        """

        super().plot_QC(summary_name=summary_name, use_n_obs=use_n_obs)
        plt.show()

        inf_aver = self.samples[f"post_sample_{summary_name}"]["per_cluster_mu_fg"].T
        if scale_average_detection and ("detection_y_c" in list(self.samples[f"post_sample_{summary_name}"].keys())):
            inf_aver = inf_aver * self.samples[f"post_sample_{summary_name}"]["detection_y_c"].mean()
        aver = self._compute_cluster_averages(key=REGISTRY_KEYS.LABELS_KEY)
        aver = aver[self.factor_names_]

        plt.hist2d(
            np.log10(aver.values.flatten() + 1),
            np.log10(inf_aver.flatten() + 1),
            bins=50,
            norm=matplotlib.colors.LogNorm(),
        )
        plt.xlabel("Mean expression for every gene in every cluster")
        plt.ylabel("Estimated expression for every gene in every cluster")
        plt.show()
