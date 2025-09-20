from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
from anndata import AnnData
from pyro import clear_param_store
from pyro.infer import Trace_ELBO, TraceEnum_ELBO
from pyro.nn import PyroModule
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)
from scvi.dataloaders import DataSplitter, DeviceBackedDataSplitter
from scvi.model.base import BaseModelClass, PyroSampleMixin, PyroSviTrainMixin
from scvi.model.base._pyromixin import PyroJitGuideWarmup
from scvi.train import TrainRunner
from scvi.utils import setup_anndata_dsp

from ..models._cell2location_WTA_module import (
    LocationModelWTAMultiExperimentHierarchicalGeneLevel,
)
from ..models.base._pyro_base_loc_module import Cell2locationBaseModule
from ..models.base._pyro_mixin import (
    PltExportMixin,
    PyroAggressiveConvergence,
    PyroAggressiveTrainingPlan,
    QuantileMixin,
)
from ..utils import select_slide


class Cell2location_WTA(QuantileMixin, PyroSampleMixin, PyroSviTrainMixin, PltExportMixin, BaseModelClass):
    r"""
    Cell2location model. User-end model class. See Module class for description of the model (incl. math).

    Parameters
    ----------
    adata
        spatial AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    cell_state_df
        pd.DataFrame with reference expression signatures for each gene (rows) in each cell type/population (columns).
    use_gpu
        Use the GPU?
    **model_kwargs
        Keyword args for :class:`~cell2location.models.LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel`

    Examples
    --------
    TODO add example
    >>>
    """

    def __init__(
        self,
        adata: AnnData,
        cell_state_df: pd.DataFrame,
        model_class: Optional[PyroModule] = None,
        detection_mean_per_sample: bool = False,
        detection_mean_correction: float = 1.0,
        **model_kwargs,
    ):
        # in case any other model was created before that shares the same parameter names.
        clear_param_store()

        if not np.all(adata.var_names == cell_state_df.index):
            raise ValueError("adata.var_names should match cell_state_df.index, find interecting variables/genes first")

        super().__init__(adata)

        self.mi_ = []

        if model_class is None:
            model_class = LocationModelWTAMultiExperimentHierarchicalGeneLevel

        self.cell_state_df_ = cell_state_df
        self.n_factors_ = cell_state_df.shape[1]
        self.factor_names_ = cell_state_df.columns.values

        if not detection_mean_per_sample:
            # compute expected change in sensitivity (m_g in V1 or y_s in V2)
            sc_total = cell_state_df.sum(0).mean()
            sp_total = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY).sum(1).mean()
            self.detection_mean_ = (sp_total / model_kwargs.get("N_cells_per_location", 1)) / sc_total
            self.detection_mean_ = self.detection_mean_ * detection_mean_correction
            model_kwargs["detection_mean"] = self.detection_mean_
        else:
            # compute expected change in sensitivity (m_g in V1 and y_s in V2)
            sc_total = cell_state_df.sum(0).mean()
            sp_total = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY).sum(1)
            batch = self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY).flatten()
            sp_total = np.array([sp_total[batch == b].mean() for b in range(self.summary_stats["n_batch"])])
            self.detection_mean_ = (sp_total / model_kwargs.get("N_cells_per_location", 1)) / sc_total
            self.detection_mean_ = self.detection_mean_ * detection_mean_correction
            model_kwargs["detection_mean"] = self.detection_mean_.reshape((self.summary_stats["n_batch"], 1)).astype(
                "float32"
            )

        detection_alpha = model_kwargs.get("detection_alpha", None)
        if detection_alpha is not None:
            if type(detection_alpha) is dict:
                batch_mapping = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).categorical_mapping
                self.detection_alpha_ = pd.Series(detection_alpha)[batch_mapping]
                model_kwargs["detection_alpha"] = self.detection_alpha_.values.reshape(
                    (self.summary_stats["n_batch"], 1)
                ).astype("float32")

        self.module = Cell2locationBaseModule(
            model=model_class,
            n_obs=self.summary_stats["n_cells"],
            n_vars=self.summary_stats["n_vars"],
            n_factors=self.n_factors_,
            n_neg_probes=self.summary_stats["n_neg_probes"],
            n_batch=self.summary_stats["n_batch"],
            cell_state_mat=self.cell_state_df_.values.astype("float32"),
            **model_kwargs,
        )
        self._model_summary_string = f'cell2location model with the following params: \nn_factors: {self.n_factors_} \nn_batch: {self.summary_stats["n_batch"]} '
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        nuclei_key: Optional[str] = None,
        neg_probes_key: Optional[str] = None,
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
            NumericalObsField("nuclei", nuclei_key),
            ObsmField("neg_probes", neg_probes_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int = 30000,
        batch_size: int = None,
        train_size: float = 1,
        lr: float = 0.002,
        num_particles: int = 1,
        scale_elbo: float = 1.0,
        **kwargs,
    ):
        """Train the model with useful defaults

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        train_size
            Size of training set in the range [0.0, 1.0]. Use all data points in training because
            we need to estimate cell abundance at all locations.
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

        if "plan_kwargs" not in kwargs.keys():
            kwargs["plan_kwargs"] = dict()
        if getattr(self.module.model, "discrete_variables", None) and (len(self.module.model.discrete_variables) > 0):
            kwargs["plan_kwargs"]["loss_fn"] = TraceEnum_ELBO(num_particles=num_particles)
        else:
            kwargs["plan_kwargs"]["loss_fn"] = Trace_ELBO(num_particles=num_particles)
        if scale_elbo != 1.0:
            if scale_elbo is None:
                scale_elbo = 1.0 / (self.summary_stats["n_cells"] * self.summary_stats["n_genes"])
            kwargs["plan_kwargs"]["scale_elbo"] = scale_elbo

        super().train(**kwargs)

    def train_aggressive(
        self,
        max_epochs: Optional[int] = 1000,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 1,
        validation_size: Optional[float] = None,
        batch_size: int = None,
        early_stopping: bool = False,
        lr: Optional[float] = None,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training. If `None`, no minibatching occurs and all
            data is copied to device (e.g., GPU).
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        lr
            Optimiser learning rate (default optimiser is :class:`~pyro.optim.ClippedAdam`).
            Specifying optimiser via plan_kwargs overrides this choice of lr.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if max_epochs is None:
            n_obs = self.adata_manager.adata.n_obs
            max_epochs = np.min([round((20000 / n_obs) * 1000), 1000])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        if lr is not None and "optim" not in plan_kwargs.keys():
            plan_kwargs.update({"optim_kwargs": {"lr": lr}})

        if batch_size is None:
            # use data splitter which moves data to GPU once
            data_splitter = DeviceBackedDataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
        else:
            data_splitter = DataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
        training_plan = PyroAggressiveTrainingPlan(pyro_module=self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]

        if "callbacks" not in trainer_kwargs.keys():
            trainer_kwargs["callbacks"] = []
        trainer_kwargs["callbacks"].append(PyroJitGuideWarmup())
        trainer_kwargs["callbacks"].append(PyroAggressiveConvergence())

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        res = runner()
        self.mi_ = self.mi_ + training_plan.mi
        return res

    def export_posterior(
        self,
        adata,
        sample_kwargs: Optional[dict] = None,
        export_slot: str = "mod",
        add_to_obsm: list = ["means", "stds", "q05", "q95"],
    ):
        """
        Summarise posterior distribution and export results (cell abundance) to anndata object:

        1. adata.obsm: Estimated cell abundance as pd.DataFrames for each posterior distribution summary `add_to_obsm`,
            posterior mean, sd, 5% and 95% quantiles (['means', 'stds', 'q05', 'q95']).
            If export to adata.obsm fails with error, results are saved to adata.obs instead.
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
        add_to_obsm
            posterior distribution summary to export in adata.obsm (['means', 'stds', 'q05', 'q95']).
        Returns
        -------

        """

        sample_kwargs = sample_kwargs if isinstance(sample_kwargs, dict) else dict()

        # generate samples from posterior distributions for all parameters
        # and compute mean, 5%/95% quantiles and standard deviation
        self.samples = self.sample_posterior(**sample_kwargs)
        # TODO use add_to_obsm to determine which quantiles need to be computed,
        # and if means and stds are not in the list - use quantile methods rather than sampling posterior

        # export posterior distribution summary for all parameters and
        # annotation (model, date, var, obs and cell type names) to anndata object
        adata.uns[export_slot] = self._export2adata(self.samples)

        # add estimated cell abundance as dataframe to obsm in anndata
        # first convert np.arrays to pd.DataFrames with cell type and observation names
        # data frames contain mean, 5%/95% quantiles and standard deviation, denoted by a prefix
        for k in add_to_obsm:
            sample_df = self.sample2df_obs(
                self.samples,
                site_name="w_sf",
                summary_name=k,
                name_prefix="cell_abundance",
            )
            try:
                adata.obsm[f"{k}_cell_abundance_w_sf"] = sample_df.loc[adata.obs.index, :]
            except ValueError:
                # Catching weird error with obsm: `ValueError: value.index does not match parent’s axis 1 names`
                adata.obs[sample_df.columns] = sample_df.loc[adata.obs.index, :]

        return adata

    def plot_spatial_QC_across_batches(self):
        """QC plot: compare total RNA count with estimated total cell abundance and detection sensitivity."""

        adata = self.adata

        # get batch key and the list of samples
        batch_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).original_key
        samples = adata.obs[batch_key].unique()

        # figure out plot shape
        ncol = len(samples)
        nrow = 3
        fig, axs = plt.subplots(nrow, ncol, figsize=(1 + 4 * ncol, 1 + 4 * nrow))
        if ncol == 1:
            axs = axs.reshape((nrow, 1))

        # compute total counts
        # find data slot
        x_dict = self.adata_manager.data_registry[REGISTRY_KEYS.X_KEY]
        if x_dict["attr_name"] == "X":
            use_raw = False
        else:
            use_raw = True
        if x_dict["attr_name"] == "layers":
            layer = x_dict["attr_key"]
        else:
            layer = None

        # get data
        if layer is not None:
            x = adata.layers[layer]
        else:
            if not use_raw:
                x = adata.X
            else:
                x = adata.raw.X
        # compute total counts per location
        cell_type = "total RNA counts"
        adata.obs[cell_type] = np.array(x.sum(1)).flatten()

        # figure out colour map scaling
        vmax = np.quantile(adata.obs[cell_type].values, 0.992)
        # plot, iterating across samples
        for i, s in enumerate(samples):
            sp_data_s = select_slide(adata, s, batch_key=batch_key)
            scanpy.pl.spatial(
                sp_data_s,
                cmap="magma",
                color=cell_type,
                size=1.3,
                img_key="hires",
                alpha_img=1,
                vmin=0,
                vmax=vmax,
                ax=axs[0, i],
                show=False,
            )
            axs[0, i].title.set_text(cell_type + "\n" + s)

        cell_type = "Total cell abundance (sum_f w_sf)"
        adata.obs[cell_type] = adata.uns["mod"]["post_sample_means"]["w_sf"].sum(1).flatten()
        # figure out colour map scaling
        vmax = np.quantile(adata.obs[cell_type].values, 0.992)
        # plot, iterating across samples
        for i, s in enumerate(samples):
            sp_data_s = select_slide(adata, s, batch_key=batch_key)
            scanpy.pl.spatial(
                sp_data_s,
                cmap="magma",
                color=cell_type,
                size=1.3,
                img_key="hires",
                alpha_img=1,
                vmin=0,
                vmax=vmax,
                ax=axs[1, i],
                show=False,
            )
            axs[1, i].title.set_text(cell_type + "\n" + s)

        cell_type = "RNA detection sensitivity (y_s)"
        adata.obs[cell_type] = adata.uns["mod"]["post_sample_q05"]["detection_y_s"]
        # figure out colour map scaling
        vmax = np.quantile(adata.obs[cell_type].values, 0.992)
        # plot, iterating across samples
        for i, s in enumerate(samples):
            sp_data_s = select_slide(adata, s, batch_key=batch_key)
            scanpy.pl.spatial(
                sp_data_s,
                cmap="magma",
                color=cell_type,
                size=1.3,
                img_key="hires",
                alpha_img=1,
                vmin=0,
                vmax=vmax,
                ax=axs[2, i],
                show=False,
            )
            axs[2, i].title.set_text(cell_type + "\n" + s)

        fig.tight_layout(pad=0.5)

        return fig
