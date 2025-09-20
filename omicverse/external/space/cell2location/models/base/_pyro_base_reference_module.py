from typing import Literal

from scvi.module.base import PyroBaseModuleClass

from ...models.base._pyro_mixin import AutoGuideMixinModule, init_to_value


class RegressionBaseModule(PyroBaseModuleClass, AutoGuideMixinModule):
    def __init__(
        self,
        model,
        amortised: bool = False,
        encoder_mode: Literal["single", "multiple", "single-multiple"] = "single",
        encoder_kwargs=None,
        data_transform="log1p",
        **kwargs,
    ):
        """
        Module class which defines AutoGuide given model. Supports multiple model architectures.

        Parameters
        ----------
        amortised
            boolean, use a Neural Network to approximate posterior distribution of location-specific (local) parameters?
        encoder_mode
            Use single encoder for all variables ("single"), one encoder per variable ("multiple")
            or a single encoder in the first step and multiple encoders in the second step ("single-multiple").
        encoder_kwargs
            arguments for Neural Network construction (scvi.nn.FCLayers)
        kwargs
            arguments for specific model class - e.g. number of genes, values of the prior distribution
        """
        super().__init__()
        self.hist = []

        self._model = model(**kwargs)
        self._amortised = amortised

        self._guide = self._create_autoguide(
            model=self.model,
            amortised=self.is_amortised,
            encoder_kwargs=encoder_kwargs,
            data_transform=data_transform,
            encoder_mode=encoder_mode,
            init_loc_fn=self.init_to_value,
            n_cat_list=[kwargs["n_batch"]],
        )

        self._get_fn_args_from_batch = self._model._get_fn_args_from_batch

    @property
    def model(self):
        return self._model

    @property
    def guide(self):
        return self._guide

    @property
    def is_amortised(self):
        return self._amortised

    @property
    def list_obs_plate_vars(self):
        return self.model.list_obs_plate_vars()

    def init_to_value(self, site):
        if getattr(self.model, "np_init_vals", None) is not None:
            init_vals = {k: getattr(self.model, f"init_val_{k}") for k in self.model.np_init_vals.keys()}
        else:
            init_vals = dict()
        return init_to_value(site=site, values=init_vals)
