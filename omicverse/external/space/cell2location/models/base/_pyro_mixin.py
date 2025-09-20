import gc
import logging
from datetime import date
from functools import partial
from typing import Optional, Union

import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import torch
from lightning.pytorch.callbacks import Callback
from pyro import poutine
from pyro.infer.autoguide import AutoNormal, init_to_feasible, init_to_mean
from scipy.sparse import issparse
from scvi import REGISTRY_KEYS
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import parse_device_args
from scvi.module.base import PyroBaseModuleClass
from scvi.train import PyroTrainingPlan as PyroTrainingPlan_scvi

from ...distributions.AutoAmortisedNormalMessenger import (
    AutoAmortisedHierarchicalNormalMessenger,
)

logger = logging.getLogger(__name__)


def init_to_value(site=None, values={}, init_fn=init_to_mean):
    if site is None:
        return partial(init_to_value, values=values)
    if site["name"] in values:
        return values[site["name"]]
    else:
        return init_fn(site)


class AutoGuideMixinModule:
    """
    This mixin class provides methods for:

    - initialising standard AutoNormal guides
    - initialising amortised guides (AutoNormalEncoder)
    - initialising amortised guides with special additional inputs

    """

    def _create_autoguide(
        self,
        model,
        amortised,
        encoder_kwargs,
        data_transform,
        encoder_mode,
        init_loc_fn=init_to_mean(fallback=init_to_feasible),
        n_cat_list: list = [],
        encoder_instance=None,
        guide_class=AutoNormal,
        guide_kwargs: Optional[dict] = None,
    ):
        if guide_kwargs is None:
            guide_kwargs = dict()

        if not amortised:
            if getattr(model, "discrete_variables", None) is not None:
                model = poutine.block(model, hide=model.discrete_variables)
            if issubclass(guide_class, poutine.messenger.Messenger):
                # messenger guides don't need create_plates function
                _guide = guide_class(
                    model,
                    init_loc_fn=init_loc_fn,
                    **guide_kwargs,
                )
            else:
                _guide = guide_class(
                    model,
                    init_loc_fn=init_loc_fn,
                    **guide_kwargs,
                    create_plates=self.model.create_plates,
                )
        else:
            encoder_kwargs = encoder_kwargs if isinstance(encoder_kwargs, dict) else dict()
            n_hidden = encoder_kwargs["n_hidden"] if "n_hidden" in encoder_kwargs.keys() else 200
            if data_transform is None:
                pass
            elif isinstance(data_transform, np.ndarray):
                # add extra info about gene clusters as input to NN
                self.register_buffer("gene_clusters", torch.tensor(data_transform.astype("float32")))
                n_in = model.n_vars + data_transform.shape[1]
                data_transform = self._data_transform_clusters()
            elif data_transform == "log1p":
                # use simple log1p transform
                data_transform = torch.log1p
                n_in = self.model.n_vars
            elif (
                isinstance(data_transform, dict)
                and "var_std" in list(data_transform.keys())
                and "var_mean" in list(data_transform.keys())
            ):
                # use data transform by scaling
                n_in = model.n_vars
                self.register_buffer(
                    "var_mean",
                    torch.tensor(data_transform["var_mean"].astype("float32").reshape((1, n_in))),
                )
                self.register_buffer(
                    "var_std",
                    torch.tensor(data_transform["var_std"].astype("float32").reshape((1, n_in))),
                )
                data_transform = self._data_transform_scale()
            else:
                # use custom data transform
                data_transform = data_transform
                n_in = model.n_vars
            amortised_vars = model.list_obs_plate_vars()
            if len(amortised_vars["input"]) >= 2:
                encoder_kwargs["n_cat_list"] = n_cat_list
            if data_transform is not None:
                amortised_vars["input_transform"][0] = data_transform
            if "n_in" in amortised_vars.keys():
                n_in = amortised_vars["n_in"]
            if getattr(model, "discrete_variables", None) is not None:
                model = poutine.block(model, hide=model.discrete_variables)
            _guide = AutoAmortisedHierarchicalNormalMessenger(
                model,
                amortised_plate_sites=amortised_vars,
                n_in=n_in,
                n_hidden=n_hidden,
                encoder_kwargs=encoder_kwargs,
                encoder_mode=encoder_mode,
                encoder_instance=encoder_instance,
                init_loc_fn=init_loc_fn,
                **guide_kwargs,
            )
        return _guide

    def _data_transform_clusters(self):
        def _data_transform(x):
            return torch.log1p(torch.cat([x, x @ self.gene_clusters], dim=1))

        return _data_transform

    def _data_transform_scale(self):
        def _data_transform(x):
            # return (x - self.var_mean) / self.var_std
            return x / self.var_std

        return _data_transform


class QuantileMixin:
    """
    This mixin class provides methods for:

    - computing median and quantiles of the posterior distribution using both direct and amortised inference

    """

    def _optim_param(
        self,
        lr: float = 0.01,
        autoencoding_lr: float = None,
        clip_norm: float = 200,
        module_names: list = ["encoder", "hidden2locs", "hidden2scales"],
    ):
        # TODO implement custom training method that can use this function.
        # create function which fetches different lr for autoencoding guide
        def optim_param(module_name, param_name):
            # detect variables in autoencoding guide
            if autoencoding_lr is not None and np.any([n in module_name + "." + param_name for n in module_names]):
                return {
                    "lr": autoencoding_lr,
                    # limit the gradient step from becoming too large
                    "clip_norm": clip_norm,
                }
            else:
                return {
                    "lr": lr,
                    # limit the gradient step from becoming too large
                    "clip_norm": clip_norm,
                }

        return optim_param

    @torch.no_grad()
    def _posterior_quantile_minibatch(
        self,
        q: float = 0.5,
        batch_size: int = 2048,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        use_median: bool = True,
        exclude_vars: list = None,
        data_loader_indices=None,
    ):
        """
        Compute median of the posterior distribution of each parameter, separating local (minibatch) variable
        and global variables, which is necessary when performing amortised inference.

        Note for developers: requires model class method which lists observation/minibatch plate
        variables (self.module.model.list_obs_plate_vars()).

        Parameters
        ----------
        q
            quantile to compute
        batch_size
            number of observations per batch
        use_gpu
            Bool, use gpu?
        use_median
            Bool, when q=0.5 use median rather than quantile method of the guide

        Returns
        -------
        dictionary {variable_name: posterior quantile}

        """

        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device="torch",
            validate_single_device=True,
        )

        self.module.eval()

        train_dl = AnnDataLoader(self.adata_manager, shuffle=False, batch_size=batch_size, indices=data_loader_indices)

        # sample local parameters
        i = 0
        for tensor_dict in train_dl:
            args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
            args = [a.to(device) for a in args]
            kwargs = {k: v.to(device) for k, v in kwargs.items()}
            self.to_device(device)

            if i == 0:
                # find plate sites
                obs_plate_sites = self._get_obs_plate_sites(args, kwargs, return_observed=True)
                if len(obs_plate_sites) == 0:
                    # if no local variables - don't sample
                    break
                # find plate dimension
                obs_plate_dim = list(obs_plate_sites.values())[0]
                if use_median and q == 0.5:
                    means = self.module.guide.median(*args, **kwargs)
                else:
                    means = self.module.guide.quantiles([q], *args, **kwargs)
                means = {
                    k: means[k].cpu().numpy()
                    for k in means.keys()
                    if (k in obs_plate_sites) and (k not in exclude_vars)
                }

            else:
                if use_median and q == 0.5:
                    means_ = self.module.guide.median(*args, **kwargs)
                else:
                    means_ = self.module.guide.quantiles([q], *args, **kwargs)
                means_ = {
                    k: means_[k].cpu().numpy()
                    for k in means_.keys()
                    if (k in obs_plate_sites) and (k not in exclude_vars)
                }
                means = {k: np.concatenate([means[k], means_[k]], axis=obs_plate_dim) for k in means.keys()}
            i += 1

        # sample global parameters
        tensor_dict = next(iter(train_dl))
        args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
        args = [a.to(device) for a in args]
        kwargs = {k: v.to(device) for k, v in kwargs.items()}
        self.to_device(device)

        if use_median and q == 0.5:
            global_means = self.module.guide.median(*args, **kwargs)
        else:
            global_means = self.module.guide.quantiles([q], *args, **kwargs)
        global_means = {
            k: global_means[k].cpu().numpy()
            for k in global_means.keys()
            if (k not in obs_plate_sites) and (k not in exclude_vars)
        }

        for k in global_means.keys():
            means[k] = global_means[k]

        # quantile returns tensors with 0th dimension = 1
        if not (use_median and q == 0.5) and (
            not isinstance(self.module.guide, AutoAmortisedHierarchicalNormalMessenger)
        ):
            means = {k: means[k].squeeze(0) for k in means.keys()}

        self.module.to(device)

        return means

    @torch.no_grad()
    def _posterior_quantile(
        self,
        q: float = 0.5,
        batch_size: int = None,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        use_median: bool = True,
        exclude_vars: list = None,
        data_loader_indices=None,
    ):
        """
        Compute median of the posterior distribution of each parameter pyro models trained without amortised inference.

        Parameters
        ----------
        q
            Quantile to compute
        use_gpu
            Bool, use gpu?
        use_median
            Bool, when q=0.5 use median rather than quantile method of the guide

        Returns
        -------
        dictionary {variable_name: posterior quantile}

        """

        self.module.eval()
        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device="torch",
            validate_single_device=True,
        )
        if batch_size is None:
            batch_size = self.adata_manager.adata.n_obs
        train_dl = AnnDataLoader(self.adata_manager, shuffle=False, batch_size=batch_size, indices=data_loader_indices)
        # sample global parameters
        tensor_dict = next(iter(train_dl))
        args, kwargs = self.module._get_fn_args_from_batch(tensor_dict)
        args = [a.to(device) for a in args]
        kwargs = {k: v.to(device) for k, v in kwargs.items()}
        self.to_device(device)

        if use_median and q == 0.5:
            means = self.module.guide.median(*args, **kwargs)
        else:
            means = self.module.guide.quantiles([q], *args, **kwargs)
        means = {k: means[k].cpu().detach().numpy() for k in means.keys() if k not in exclude_vars}

        # quantile returns tensors with 0th dimension = 1
        if not (use_median and q == 0.5) and (
            not isinstance(self.module.guide, AutoAmortisedHierarchicalNormalMessenger)
        ):
            means = {k: means[k].squeeze(0) for k in means.keys()}

        return means

    def posterior_quantile(self, exclude_vars: list = None, batch_size: int = None, **kwargs):
        """
        Compute median of the posterior distribution of each parameter.

        Parameters
        ----------
        q
            Quantile to compute
        use_gpu
            Bool, use gpu?
        use_median
            Bool, when q=0.5 use median rather than quantile method of the guide

        Returns
        -------

        """
        if exclude_vars is None:
            exclude_vars = []
        if kwargs is None:
            kwargs = dict()

        if isinstance(self.module.guide, AutoNormal):
            # median/quantiles in AutoNormal does not require minibatches
            batch_size = None

        if batch_size is not None:
            return self._posterior_quantile_minibatch(exclude_vars=exclude_vars, batch_size=batch_size, **kwargs)
        else:
            return self._posterior_quantile(exclude_vars=exclude_vars, batch_size=batch_size, **kwargs)


class PltExportMixin:
    r"""
    This mixing class provides methods for common plotting tasks and data export.
    """

    @staticmethod
    def plot_posterior_mu_vs_data(mu, data):
        r"""Plot expected value of the model (e.g. mean of NB distribution) vs observed data

        :param mu: expected value
        :param data: data value
        """

        plt.hist2d(
            np.log10(data.flatten() + 1),
            np.log10(mu.flatten() + 1),
            bins=50,
            norm=matplotlib.colors.LogNorm(),
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("Data, log10")
        plt.ylabel("Posterior expected value, log10")
        plt.title("Reconstruction accuracy")
        plt.tight_layout()

    def plot_history(self, iter_start=0, iter_end=-1, ax=None):
        r"""Plot training history
        Parameters
        ----------
        iter_start
            omit initial iterations from the plot
        iter_end
            omit last iterations from the plot
        ax
            matplotlib axis
        """
        if ax is None:
            ax = plt.gca()
        if iter_end == -1:
            iter_end = len(self.history_["elbo_train"])

        ax.plot(
            np.array(self.history_["elbo_train"].index[iter_start:iter_end]),
            np.array(self.history_["elbo_train"].values.flatten())[iter_start:iter_end],
            label="train",
        )
        ax.legend()
        ax.set_xlim(0, len(self.history_["elbo_train"]))
        ax.set_xlabel("Training epochs")
        ax.set_ylabel("-ELBO loss")
        plt.tight_layout()

    def _export2adata(self, samples):
        r"""
        Export key model variables and samples

        Parameters
        ----------
        samples
            dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``

        Returns
        -------
            Updated dictionary with additional details is saved to ``adata.uns['mod']``.
        """
        # add factor filter and samples of all parameters to unstructured data
        results = {
            "model_name": str(self.module.__class__.__name__),
            "date": str(date.today()),
            "factor_filter": list(getattr(self, "factor_filter", [])),
            "factor_names": list(self.factor_names_),
            "var_names": self.adata.var_names.tolist(),
            "obs_names": self.adata.obs_names.tolist(),
            "post_sample_means": samples["post_sample_means"] if "post_sample_means" in samples else None,
            "post_sample_stds": samples["post_sample_stds"] if "post_sample_stds" in samples else None,
        }
        # add posterior quantiles
        for k, v in samples.items():
            if k.startswith("post_sample_"):
                results[k] = v
        if type(self.factor_names_) is dict:
            results["factor_names"] = self.factor_names_

        return results

    def sample2df_obs(
        self,
        samples: dict,
        site_name: str = "w_sf",
        summary_name: str = "means",
        name_prefix: str = "cell_abundance",
        factor_names_key: str = "",
    ):
        """Export posterior distribution summary for observation-specific parameters
        (e.g. spatial cell abundance) as Pandas data frame
        (means, 5%/95% quantiles or sd of posterior distribution).

        Parameters
        ----------
        samples
            dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``
        site_name
            name of the model parameter to be exported
        summary_name
            posterior distribution summary to return ['means', 'stds', 'q05', 'q95']
        name_prefix
            prefix to add to column names (f'{summary_name}{name_prefix}_{site_name}_{self\.factor_names_}')

        Returns
        -------
        Pandas data frame corresponding to either means, 5%/95% quantiles or sd of the posterior distribution

        """
        if type(self.factor_names_) is dict:
            factor_names_ = self.factor_names_[factor_names_key]
        else:
            factor_names_ = self.factor_names_

        return pd.DataFrame(
            samples[f"post_sample_{summary_name}"].get(site_name, None),
            index=self.adata.obs_names,
            columns=[f"{summary_name}{name_prefix}_{site_name}_{i}" for i in factor_names_],
        )

    def sample2df_vars(
        self,
        samples: dict,
        site_name: str = "gene_factors",
        summary_name: str = "means",
        name_prefix: str = "",
        factor_names_key: str = "",
    ):
        r"""Export posterior distribution summary for variable-specific parameters as Pandas data frame
        (means, 5%/95% quantiles or sd of posterior distribution).

        Parameters
        ----------
        samples
            dictionary with posterior mean, 5%/95% quantiles, SD, samples, generated by ``.sample_posterior()``
        site_name
            name of the model parameter to be exported
        summary_name
            posterior distribution summary to return ('means', 'stds', 'q05', 'q95')
        name_prefix
            prefix to add to column names (f'{summary_name}{name_prefix}_{site_name}_{self\.factor_names_}')

        Returns
        -------
        Pandas data frame corresponding to either means, 5%/95% quantiles or sd of the posterior distribution

        """
        if type(self.factor_names_) is dict:
            factor_names_ = self.factor_names_[factor_names_key]
        else:
            factor_names_ = self.factor_names_
        site = samples[f"post_sample_{summary_name}"].get(site_name, None)
        return pd.DataFrame(
            site,
            columns=self.adata.var_names,
            index=[f"{summary_name}{name_prefix}_{site_name}_{i}" for i in factor_names_],
        ).T

    def plot_QC(self, summary_name: str = "means", use_n_obs: int = 1000):
        """
        Show quality control plots:

        1. Reconstruction accuracy to assess if there are any issues with model training.
           The plot should be roughly diagonal, strong deviations signal problems that need to be investigated.
           Plotting is slow because expected value of mRNA count needs to be computed from model parameters. Random
           observations are used to speed up computation.

        Parameters
        ----------
        summary_name
            posterior distribution summary to use ('means', 'stds', 'q05', 'q95')

        Returns
        -------

        """

        if getattr(self, "samples", False) is False:
            raise RuntimeError("self.samples is missing, please run self.export_posterior() first")
        if use_n_obs is not None:
            ind_x = np.random.choice(
                self.adata_manager.adata.n_obs, np.min((use_n_obs, self.adata.n_obs)), replace=False
            )
        else:
            ind_x = None

        self.expected_nb_param = self.module.model.compute_expected(
            self.samples[f"post_sample_{summary_name}"], self.adata_manager, ind_x=ind_x
        )
        x_data = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)[ind_x, :]
        if issparse(x_data):
            x_data = np.asarray(x_data.toarray())
        self.plot_posterior_mu_vs_data(self.expected_nb_param["mu"], x_data)


class PyroAggressiveConvergence(Callback):
    """
    A callback to compute/apply aggressive training convergence criteria for amortised inference.
    Motivated by this paper: https://arxiv.org/pdf/1901.05534.pdf
    """

    def __init__(self, dataloader: AnnDataLoader = None, patience: int = 10, tolerance: float = 1e-4) -> None:
        super().__init__()
        self.dataloader = dataloader
        self.patience = patience
        self.tolerance = tolerance

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ) -> None:
        """
        Compute aggressive training convergence criteria for amortised inference.
        """
        pyro_guide = pl_module.module.guide
        if hasattr(pyro_guide, "mutual_information"):
            if self.dataloader is None:
                dl = trainer.datamodule.train_dataloader()
            else:
                dl = self.dataloader
            for tensors in dl:
                tens = {k: t.to(pl_module.device) for k, t in tensors.items()}
                args, kwargs = pl_module.module._get_fn_args_from_batch(tens)
                break
            mi_ = pyro_guide.mutual_information(*args, **kwargs)
            mi_ = np.array([v for v in mi_.values()]).sum()
            pl_module.log("MI", mi_, prog_bar=True)
            if len(pl_module.mi) > 1:
                if abs(mi_ - pl_module.mi[-1]) < self.tolerance:
                    pl_module.n_epochs_patience += 1
            else:
                pl_module.n_epochs_patience = 0
            if pl_module.n_epochs_patience > self.patience:
                # stop aggressive training by setting epoch counter to max epochs
                # pl_module.aggressive_epochs_counter = pl_module.n_aggressive_epochs + 1
                logger.info('Stopped aggressive training after "{}" epochs'.format(pl_module.aggressive_epochs_counter))
            pl_module.mi.append(mi_)


class PyroTrainingPlan(PyroTrainingPlan_scvi):
    def on_train_epoch_end(self):
        """Training epoch end for Pyro training."""
        outputs = self.training_step_outputs
        elbo = 0
        n = 0
        for out in outputs:
            elbo += out["loss"]
            n += 1
        if n > 0:
            elbo /= n
        self.log("elbo_train", elbo, prog_bar=True)
        self.training_step_outputs.clear()
        gc.collect()
        torch.cuda.empty_cache()


class PyroAggressiveTrainingPlan1(PyroTrainingPlan_scvi):
    """
    Lightning module task to train Pyro scvi-tools modules.
    Parameters
    ----------
    pyro_module
        An instance of :class:`~scvi.module.base.PyroBaseModuleClass`. This object
        should have callable `model` and `guide` attributes or methods.
    loss_fn
        A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`.
        If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
    optim
        A Pyro optimizer instance, e.g., :class:`~pyro.optim.Adam`. If `None`,
        defaults to :class:`pyro.optim.Adam` optimizer with a learning rate of `1e-3`.
    optim_kwargs
        Keyword arguments for **default** optimiser :class:`pyro.optim.Adam`.
    n_aggressive_epochs
        Number of epochs in aggressive optimisation of amortised variables.
    n_aggressive_steps
        Number of steps to spend optimising amortised variables before one step optimising global variables.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    """

    def __init__(
        self,
        pyro_module: PyroBaseModuleClass,
        loss_fn: Optional[pyro.infer.ELBO] = None,
        optim: Optional[pyro.optim.PyroOptim] = None,
        optim_kwargs: Optional[dict] = None,
        n_aggressive_epochs: int = 1000,
        n_aggressive_steps: int = 20,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        aggressive_vars: Union[list, None] = None,
        invert_aggressive_selection: bool = False,
    ):
        super().__init__(
            pyro_module=pyro_module,
            loss_fn=loss_fn,
            optim=optim,
            optim_kwargs=optim_kwargs,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
        )

        self.n_aggressive_epochs = n_aggressive_epochs
        self.n_aggressive_steps = n_aggressive_steps
        self.aggressive_steps_counter = 0
        self.aggressive_epochs_counter = 0
        self.mi = []
        self.n_epochs_patience = 0

        # in list not provided use amortised variables for aggressive training
        if aggressive_vars is None:
            aggressive_vars = list(self.module.list_obs_plate_vars["sites"].keys())
            aggressive_vars = aggressive_vars + [f"{i}_initial" for i in aggressive_vars]
            aggressive_vars = aggressive_vars + [f"{i}_unconstrained" for i in aggressive_vars]

        self.aggressive_vars = aggressive_vars
        self.invert_aggressive_selection = invert_aggressive_selection
        # keep frozen variables as frozen
        self.requires_grad_false_vars = [k for k, v in self.module.guide.named_parameters() if not v.requires_grad] + [
            k for k, v in self.module.model.named_parameters() if not v.requires_grad
        ]

        self.svi = pyro.infer.SVI(
            model=pyro_module.model,
            guide=pyro_module.guide,
            optim=self.optim,
            loss=self.loss_fn,
        )

    def change_requires_grad(self, aggressive_vars_status, non_aggressive_vars_status):
        for k, v in self.module.guide.named_parameters():
            if not np.any([i in k for i in self.requires_grad_false_vars]):
                k_in_vars = np.any([i in k for i in self.aggressive_vars])
                # hide variables on the list if they are not hidden
                if k_in_vars and v.requires_grad and (aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables on the list if they are hidden
                if k_in_vars and (not v.requires_grad) and (aggressive_vars_status == "expose"):
                    v.requires_grad = True

                # hide variables not on the list if they are not hidden
                if (not k_in_vars) and v.requires_grad and (non_aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables not on the list if they are hidden
                if (not k_in_vars) and (not v.requires_grad) and (non_aggressive_vars_status == "expose"):
                    v.requires_grad = True

        for k, v in self.module.model.named_parameters():
            if not np.any([i in k for i in self.requires_grad_false_vars]):
                k_in_vars = np.any([i in k for i in self.aggressive_vars])
                # hide variables on the list if they are not hidden
                if k_in_vars and v.requires_grad and (aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables on the list if they are hidden
                if k_in_vars and (not v.requires_grad) and (aggressive_vars_status == "expose"):
                    v.requires_grad = True

                # hide variables not on the list if they are not hidden
                if (not k_in_vars) and v.requires_grad and (non_aggressive_vars_status == "hide"):
                    v.requires_grad = False
                # expose variables not on the list if they are hidden
                if (not k_in_vars) and (not v.requires_grad) and (non_aggressive_vars_status == "expose"):
                    v.requires_grad = True

    def on_train_epoch_end(self):
        self.aggressive_epochs_counter += 1

        self.change_requires_grad(
            aggressive_vars_status="expose",
            non_aggressive_vars_status="expose",
        )

        outputs = self.training_step_outputs
        elbo = 0
        n = 0
        for out in outputs:
            elbo += out["loss"]
            n += 1
        if n > 0:
            elbo /= n
        self.log("elbo_train", elbo, prog_bar=True)
        self.training_step_outputs.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        args, kwargs = self.module._get_fn_args_from_batch(batch)
        # Set KL weight if necessary.
        # Note: if applied, ELBO loss in progress bar is the effective KL annealed loss, not the true ELBO.
        if self.use_kl_weight:
            kwargs.update({"kl_weight": self.kl_weight})

        if self.aggressive_epochs_counter < self.n_aggressive_epochs:
            if self.aggressive_steps_counter < self.n_aggressive_steps:
                self.aggressive_steps_counter += 1
                # Do parameter update exclusively for amortised variables
                if self.invert_aggressive_selection:
                    self.change_requires_grad(
                        aggressive_vars_status="hide",
                        non_aggressive_vars_status="expose",
                    )
                else:
                    self.change_requires_grad(
                        aggressive_vars_status="expose",
                        non_aggressive_vars_status="hide",
                    )
                loss = torch.Tensor([self.svi.step(*args, **kwargs)])
            else:
                self.aggressive_steps_counter = 0
                # Do parameter update exclusively for non-amortised variables
                if self.invert_aggressive_selection:
                    self.change_requires_grad(
                        aggressive_vars_status="expose",
                        non_aggressive_vars_status="hide",
                    )
                else:
                    self.change_requires_grad(
                        aggressive_vars_status="hide",
                        non_aggressive_vars_status="expose",
                    )
                loss = torch.Tensor([self.svi.step(*args, **kwargs)])
        else:
            # Do parameter update for both types of variables
            self.change_requires_grad(
                aggressive_vars_status="expose",
                non_aggressive_vars_status="expose",
            )
            loss = torch.Tensor([self.svi.step(*args, **kwargs)])

        return {"loss": loss}


class PyroAggressiveTrainingPlan(PyroAggressiveTrainingPlan1):
    """
    Lightning module task to train Pyro scvi-tools modules.
    Parameters
    ----------
    pyro_module
        An instance of :class:`~scvi.module.base.PyroBaseModuleClass`. This object
        should have callable `model` and `guide` attributes or methods.
    loss_fn
        A Pyro loss. Should be a subclass of :class:`~pyro.infer.ELBO`.
        If `None`, defaults to :class:`~pyro.infer.Trace_ELBO`.
    optim
        A Pyro optimizer instance, e.g., :class:`~pyro.optim.Adam`. If `None`,
        defaults to :class:`pyro.optim.Adam` optimizer with a learning rate of `1e-3`.
    optim_kwargs
        Keyword arguments for **default** optimiser :class:`pyro.optim.Adam`.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    """

    def __init__(
        self,
        scale_elbo: Union[float, None] = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if scale_elbo != 1.0:
            self.svi = pyro.infer.SVI(
                model=poutine.scale(self.module.model, scale_elbo),
                guide=poutine.scale(self.module.guide, scale_elbo),
                optim=self.optim,
                loss=self.loss_fn,
            )
        else:
            self.svi = pyro.infer.SVI(
                model=self.module.model,
                guide=self.module.guide,
                optim=self.optim,
                loss=self.loss_fn,
            )
