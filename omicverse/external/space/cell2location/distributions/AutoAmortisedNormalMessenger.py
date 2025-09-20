from copy import deepcopy
from typing import Callable, Literal, Optional, Union

import pyro.distributions as dist
import torch
from pyro.distributions.distribution import Distribution
from pyro.distributions.transforms import SoftplusTransform
from pyro.infer.autoguide import AutoHierarchicalNormalMessenger
from pyro.infer.autoguide.initialization import init_to_feasible, init_to_mean
from pyro.infer.autoguide.utils import (
    deep_getattr,
    deep_setattr,
    helpful_support_errors,
)
from pyro.nn.module import PyroModule, PyroParam, to_pyro_module_
from torch.distributions import biject_to, constraints

from ..nn import FCLayers


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


class FCLayersPyro(FCLayers, PyroModule):
    pass


class AutoAmortisedHierarchicalNormalMessenger(AutoHierarchicalNormalMessenger):
    """
    EXPERIMENTAL Automatic :class:`~pyro.infer.effect_elbo.GuideMessenger` ,
    intended for use with :class:`~pyro.infer.effect_elbo.Effect_ELBO` or
    similar. Amortise specific sites

    The mean-field posterior at any site is a transformed normal distribution,
    the mean of which depends on the value of that site given its dependencies in the model:

        loc = loc + transform.inv(prior.mean) * weight

    Where the value of `prior.mean` is conditional on upstream sites in the model.
    This approach doesn't work for distributions that don't have the mean.

    loc, scales and element-specific weight are amortised for each site specified in `amortised_plate_sites`.

    Derived classes may override particular sites and use this simply as a
    default, see AutoNormalMessenger documentation for example.

    :param callable model: A Pyro model.
    :param dict amortised_plate_sites: Dictionary with amortised plate details:
        the name of observation/minibatch plate,
        indexes of model args to provide to encoder,
        variable names that belong to the observation plate
        and the number of dimensions in non-plate axis of each variable - such as:
        {
            "name": "obs_plate",
            "input": [0],  # expression data + (optional) batch index ([0, 2])
            "input_transform": [torch.log1p], # how to transform input data before passing to NN
            "sites": {
                "n_s": 1,
                "y_s": 1,
                "z_sr": R,
                "w_sf": F,
            }
        }
    :param int n_in: Number of input dimensions (for encoder_class).
    :param int n_hidden: Number of hidden nodes in each layer, one of 3 options:
        1. Integer denoting the number of hidden nodes
        2. Dictionary with {"single": 200, "multiple": 200} denoting the number of hidden nodes for each `encoder_mode` (See below)
        3. Allowing different number of hidden nodes for each model site. Dictionary with the number of hidden nodes for single encode mode and each model site:
        {
            "single": 200
            "n_s": 5,
            "y_s": 5,
            "z_sr": 128,
            "w_sf": 200,
        }
    :param float init_param_scale: How to scale/normalise initial values for weights converting hidden layers to loc and scales.
    :param float scales_offset: offset between the output of the NN and scales.
    :param Callable encoder_class: Class that defines encoder network.
    :param dict encoder_kwargs: Keyword arguments for encoder class.
    :param dict multi_encoder_kwargs: Optional separate keyword arguments for encoder_class,
        useful when encoder_mode == "single-multiple".
    :param Callable encoder_instance: Encoder network instance, overrides class input and the input instance is copied with deepcopy.
    :param str encoder_mode: Use single encoder for all variables ("single"), one encoder per variable ("multiple")
        or a single encoder in the first step and multiple encoders in the second step ("single-multiple").
    :param list hierarchical_sites: List of latent variables (model sites)
        that have hierarchical dependencies.
        If None, all sites are assumed to have hierarchical dependencies. If None, for the sites
        that don't have upstream sites, the guide is representing/learning deviation from the prior.
    """

    # 'element-wise' or 'scalar'
    weight_type = "element-wise"

    def __init__(
        self,
        model: Callable,
        *,
        amortised_plate_sites: dict,
        n_in: int,
        n_hidden: dict = None,
        init_param_scale: float = 1 / 50,
        init_scale: float = 0.1,
        init_weight: float = 1.0,
        init_loc_fn: Callable = init_to_mean(fallback=init_to_feasible),
        encoder_class=FCLayersPyro,
        encoder_kwargs=None,
        multi_encoder_kwargs=None,
        encoder_instance: torch.nn.Module = None,
        encoder_mode: Literal["single", "multiple", "single-multiple"] = "single",
        hierarchical_sites: Optional[list] = None,
        bias=True,
        use_posterior_lsw_encoders=False,
    ):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        super().__init__(model, init_loc_fn=init_loc_fn)
        self._init_scale = init_scale
        self._init_weight = init_weight
        self._hierarchical_sites = hierarchical_sites
        self.amortised_plate_sites = amortised_plate_sites
        self.encoder_mode = encoder_mode
        self.bias = bias
        self.use_posterior_lsw_encoders = use_posterior_lsw_encoders
        self._computing_median = False
        self._computing_quantiles = False
        self._quantile_values = None
        self._computing_mi = False
        self.mi = dict()
        self.samples_for_mi = None

        self.softplus = SoftplusTransform()

        # default n_hidden values and checking input
        if n_hidden is None:
            n_hidden = {"single": 200, "multiple": 200}
        else:
            if isinstance(n_hidden, int):
                n_hidden = {"single": n_hidden, "multiple": n_hidden}
            elif not isinstance(n_hidden, dict):
                raise ValueError("n_hidden must be either int or dict")
        # process encoder kwargs, add n_hidden, create argument for multiple encoders
        encoder_kwargs = deepcopy(encoder_kwargs) if isinstance(encoder_kwargs, dict) else dict()
        encoder_kwargs["n_hidden"] = n_hidden["single"]
        if multi_encoder_kwargs is None:
            multi_encoder_kwargs = deepcopy(encoder_kwargs)

        # save encoder parameters
        self.encoder_kwargs = encoder_kwargs
        self.multi_encoder_kwargs = multi_encoder_kwargs
        self.single_n_in = n_in
        self.multiple_n_in = n_in
        self.n_hidden = n_hidden
        if ("single" in encoder_mode) and ("multiple" in encoder_mode):
            # if single network precedes multiple networks
            self.multiple_n_in = self.n_hidden["single"]
        self.encoder_class = encoder_class
        self.encoder_instance = encoder_instance
        self.init_param_scale = init_param_scale

    def get_posterior(
        self,
        name: str,
        prior: Distribution,
    ) -> Union[Distribution, torch.Tensor]:
        if self._computing_median:
            return self._get_posterior_median(name, prior)
        if self._computing_quantiles:
            return self._get_posterior_quantiles(name, prior)
        if self._computing_mi:
            # the messenger autoguide needs the output to fit certain dimensions
            # this is hack which saves MI to self.mi but returns cheap to compute medians
            self.mi[name] = self._get_mutual_information(name, prior)
            return self._get_posterior_median(name, prior)

        with helpful_support_errors({"name": name, "fn": prior}):
            transform = biject_to(prior.support)
        # If hierarchical_sites not specified all sites are assumed to be hierarchical
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight  # - torch.tensor(3.0, device=prior.mean.device)
            posterior = dist.TransformedDistribution(
                dist.Normal(loc, scale).to_event(transform.domain.event_dim),
                transform.with_cache(),
            )
            return posterior
        else:
            # Fall back to mean field when hierarchical_sites list is not empty and site not in the list.
            loc, scale = self._get_params(name, prior)
            posterior = dist.TransformedDistribution(
                dist.Normal(loc, scale).to_event(transform.domain.event_dim),
                transform.with_cache(),
            )
            return posterior

    def encode(self, name: str, prior: Distribution):
        """
        Apply encoder network to input data to obtain hidden layer encoding.
        Parameters
        ----------
        args
            Pyro model args
        kwargs
            Pyro model kwargs
        -------

        """
        try:
            args, kwargs = self.args_kwargs  # stored as a tuple of (tuple, dict)
            # get the data for NN from
            in_names = self.amortised_plate_sites["input"]
            x_in = [kwargs[i] if i in kwargs.keys() else args[i] for i in in_names]
            # apply data transform before passing to NN
            site_transform = self.amortised_plate_sites.get("site_transform", None)
            if site_transform is not None and name in site_transform.keys():
                # when input data transform and input dimensions differ between variables
                in_transforms = site_transform[name]["input_transform"]
                single_n_in = site_transform[name]["n_in"]
                multiple_n_in = site_transform[name]["n_in"]
                if ("single" in self.encoder_mode) and ("multiple" in self.encoder_mode):
                    # if single network precedes multiple networks
                    multiple_n_in = self.multiple_n_in
            else:
                in_transforms = self.amortised_plate_sites["input_transform"]
                single_n_in = self.single_n_in
                multiple_n_in = self.multiple_n_in

            x_in = [in_transforms[i](x) for i, x in enumerate(x_in)]
            # apply learnable normalisation before passing to NN:
            input_normalisation = self.amortised_plate_sites.get("input_normalisation", None)
            if input_normalisation is not None:
                for i in range(len(self.amortised_plate_sites["input"])):
                    if input_normalisation[i]:
                        x_in[i] = x_in[i] * deep_getattr(self, f"input_normalisation_{i}")
            if "single" in self.encoder_mode:
                # encode with a single encoder
                res = deep_getattr(self, "one_encoder")(*x_in)
                if "multiple" in self.encoder_mode:
                    # when there is a second layer of multiple encoders fetch encoders and encode data
                    x_in[0] = res
                    res = deep_getattr(self.multiple_encoders, name)(*x_in)
            else:
                # when there are multiple encoders fetch encoders and encode data
                res = deep_getattr(self.multiple_encoders, name)(*x_in)
            return res
        except AttributeError:
            pass

        # Initialize.
        # create normalisation parameters if necessary:
        input_normalisation = self.amortised_plate_sites.get("input_normalisation", None)
        if input_normalisation is not None:
            for i in range(len(self.amortised_plate_sites["input"])):
                if input_normalisation[i]:
                    deep_setattr(
                        self,
                        f"input_normalisation_{i}",
                        PyroParam(torch.ones((1, single_n_in)).to(prior.mean.device).requires_grad_(True)),
                    )
        # create encoder neural networks
        if "single" in self.encoder_mode:
            if self.encoder_instance is not None:
                # copy provided encoder instance
                one_encoder = deepcopy(self.encoder_instance).to(prior.mean.device)
                # convert to pyro module
                to_pyro_module_(one_encoder)
                deep_setattr(self, "one_encoder", one_encoder)
            else:
                # create encoder instance from encoder class
                deep_setattr(
                    self,
                    "one_encoder",
                    self.encoder_class(n_in=single_n_in, n_out=self.n_hidden["single"], **self.encoder_kwargs).to(
                        prior.mean.device
                    ),
                )
        if "multiple" in self.encoder_mode:
            # determine the number of hidden layers
            if name in self.n_hidden.keys():
                n_hidden = self.n_hidden[name]
            else:
                n_hidden = self.n_hidden["multiple"]
            multi_encoder_kwargs = deepcopy(self.multi_encoder_kwargs)
            multi_encoder_kwargs["n_hidden"] = n_hidden

            # create multiple encoders
            if self.encoder_instance is not None:
                # copy instances
                encoder_ = deepcopy(self.encoder_instance).to(prior.mean.device)
                # convert to pyro module
                to_pyro_module_(encoder_)
                deep_setattr(
                    self,
                    "multiple_encoders." + name,
                    encoder_,
                )
            else:
                # create instances
                deep_setattr(
                    self,
                    "multiple_encoders." + name,
                    self.encoder_class(n_in=multiple_n_in, n_out=n_hidden, **multi_encoder_kwargs).to(
                        prior.mean.device
                    ),
                )
        return self.encode(name, prior)

    def _get_params(self, name: str, prior: Distribution):
        if name not in self.amortised_plate_sites["sites"].keys():
            # don't use amortisation unless requested (site in the list)
            return super()._get_params(name, prior)

        args, kwargs = self.args_kwargs  # stored as a tuple of (tuple, dict)
        hidden = self.encode(name, prior)
        try:
            linear_loc = deep_getattr(self.hidden2locs, name)
            linear_scale = deep_getattr(self.hidden2scales, name)
            if not self.use_posterior_lsw_encoders:
                loc = linear_loc(hidden)
                scale = self.softplus(linear_scale(hidden) + self._init_scale_unconstrained)
            else:
                args, kwargs = self.args_kwargs  # stored as a tuple of (tuple, dict)
                # get the data for NN from
                in_names = self.amortised_plate_sites["input"]
                x_in = [kwargs[i] if i in kwargs.keys() else args[i] for i in in_names]
                x_in[0] = hidden
                # apply data transform before passing to NN
                site_transform = self.amortised_plate_sites.get("site_transform", None)
                if site_transform is not None and name in site_transform.keys():
                    # when input data transform and input dimensions differ between variables
                    in_transforms = site_transform[name]["input_transform"]
                else:
                    in_transforms = self.amortised_plate_sites["input_transform"]
                x_in = [in_transforms[i](x) if i != 0 else x for i, x in enumerate(x_in)]
                linear_loc_encoder = deep_getattr(self.hidden2locs, f"{name}.encoder")
                linear_scale_encoder = deep_getattr(self.hidden2scales, f"{name}.encoder")
                loc = linear_loc(linear_loc_encoder(*x_in))
                scale = self.softplus(linear_scale(linear_scale_encoder(*x_in)) + self._init_scale_unconstrained)
            if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
                if self.weight_type == "element-wise":
                    # weight is element-wise
                    linear_weight = deep_getattr(self.hidden2weights, name)
                    if not self.use_posterior_lsw_encoders:
                        weight = self.softplus(linear_weight(hidden) + self._init_weight_unconstrained)
                    else:
                        linear_weight_encoder = deep_getattr(self.hidden2weights, f"{name}.encoder")
                        weight = self.softplus(
                            linear_weight(linear_weight_encoder(hidden)) + self._init_weight_unconstrained
                        )
                if self.weight_type == "scalar":
                    # weight is a single value parameter
                    weight = deep_getattr(self.weights, name)
                return loc, scale, weight
            else:
                return loc, scale
        except AttributeError:
            pass

        # Initialize.
        with torch.no_grad():
            init_scale = torch.full((), self._init_scale)
            self._init_scale_unconstrained = self.softplus.inv(init_scale)
            init_weight = torch.full((), self._init_weight)
            self._init_weight_unconstrained = self.softplus.inv(init_weight)

            # determine the number of hidden layers
            if "multiple" in self.encoder_mode:
                if name in self.n_hidden.keys():
                    n_hidden = self.n_hidden[name]
                else:
                    n_hidden = self.n_hidden["multiple"]
            elif "single" in self.encoder_mode:
                n_hidden = self.n_hidden["single"]
            # determine parameter dimensions
            out_dim = self.amortised_plate_sites["sites"][name]

        deep_setattr(
            self,
            "hidden2locs." + name,
            PyroModule[torch.nn.Linear](n_hidden, out_dim, bias=self.bias, device=prior.mean.device),
        )
        deep_setattr(
            self,
            "hidden2scales." + name,
            PyroModule[torch.nn.Linear](n_hidden, out_dim, bias=self.bias, device=prior.mean.device),
        )

        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            if self.weight_type == "scalar":
                # weight is a single value parameter
                deep_setattr(self, "weights." + name, PyroParam(init_weight, constraint=constraints.positive))
            if self.weight_type == "element-wise":
                # weight is element-wise
                deep_setattr(
                    self,
                    "hidden2weights." + name,
                    PyroModule[torch.nn.Linear](n_hidden, out_dim, bias=self.bias, device=prior.mean.device),
                )

        if self.use_posterior_lsw_encoders:
            # determine the number of hidden layers
            if name in self.n_hidden.keys():
                n_hidden = self.n_hidden[name]
            else:
                n_hidden = self.n_hidden["multiple"]
            multi_encoder_kwargs = deepcopy(self.multi_encoder_kwargs)
            multi_encoder_kwargs["n_hidden"] = n_hidden

            # create multiple encoders
            if self.encoder_instance is not None:
                # copy instances
                encoder_ = deepcopy(self.encoder_instance).to(prior.mean.device)
                # convert to pyro module
                to_pyro_module_(encoder_)
                deep_setattr(
                    self,
                    f"hidden2locs.{name}.encoder",
                    encoder_,
                )
                deep_setattr(
                    self,
                    f"hidden2scales.{name}.encoder",
                    encoder_,
                )
                if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
                    deep_setattr(
                        self,
                        f"hidden2weights.{name}.encoder",
                        encoder_,
                    )
            else:
                # create instances
                deep_setattr(
                    self,
                    f"hidden2locs.{name}.encoder",
                    self.encoder_class(n_in=n_hidden, n_out=n_hidden, **multi_encoder_kwargs).to(prior.mean.device),
                )
                deep_setattr(
                    self,
                    f"hidden2scales.{name}.encoder",
                    self.encoder_class(n_in=n_hidden, n_out=n_hidden, **multi_encoder_kwargs).to(prior.mean.device),
                )
                if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
                    deep_setattr(
                        self,
                        f"hidden2weights.{name}.encoder",
                        self.encoder_class(n_in=n_hidden, n_out=n_hidden, **multi_encoder_kwargs).to(prior.mean.device),
                    )

        return self._get_params(name, prior)

    def median(self, *args, **kwargs):
        self._computing_median = True
        try:
            return self(*args, **kwargs)
        finally:
            self._computing_median = False

    @torch.no_grad()
    def _get_posterior_median(self, name, prior):
        transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
        else:
            loc, scale = self._get_params(name, prior)
        return transform(loc)

    def quantiles(self, quantiles, *args, **kwargs):
        self._computing_quantiles = True
        self._quantile_values = quantiles
        try:
            return self(*args, **kwargs)
        finally:
            self._computing_quantiles = False

    @torch.no_grad()
    def _get_posterior_quantiles(self, name, prior):
        transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
        else:
            loc, scale = self._get_params(name, prior)

        site_quantiles = torch.tensor(self._quantile_values, dtype=loc.dtype, device=loc.device)
        site_quantiles_values = dist.Normal(loc, scale).icdf(site_quantiles)
        return transform(site_quantiles_values)

    def mutual_information(self, *args, **kwargs):
        # compute samples necessary to compute MI
        self.samples_for_mi = self(*args, **kwargs)
        self._computing_mi = True
        try:
            # compute mi (saved to self.mi)
            self(*args, **kwargs)
            return self.mi
        finally:
            self._computing_mi = False

    @torch.no_grad()
    def _get_mutual_information(self, name, prior):
        """Approximate the mutual information between data x and latent variable z

            I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        Returns: Float

        """

        #### get posterior mean and variance ####
        transform = biject_to(prior.support)
        if (self._hierarchical_sites is None) or (name in self._hierarchical_sites):
            loc, scale, weight = self._get_params(name, prior)
            loc = loc + transform.inv(prior.mean) * weight
        else:
            loc, scale = self._get_params(name, prior)

        if name not in self.amortised_plate_sites["sites"].keys():
            # if amortisation is not used for a particular site return MI=0
            return 0

        #### create tensors with useful numbers ####
        one = torch.ones((), dtype=loc.dtype, device=loc.device)
        two = torch.tensor(2, dtype=loc.dtype, device=loc.device)
        pi = torch.tensor(3.14159265359, dtype=loc.dtype, device=loc.device)
        #### get sample from posterior ####
        z_samples = self.samples_for_mi[name]

        #### compute mi ####
        x_batch, nz = loc.size()
        x_batch = torch.tensor(x_batch, dtype=loc.dtype, device=loc.device)
        nz = torch.tensor(nz, dtype=loc.dtype, device=loc.device)

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+scale.loc()).sum(-1)
        neg_entropy = (
            -nz * torch.log(pi * two) * (one / two) - ((scale**two).log() + one).sum(-1) * (one / two)
        ).mean()

        # [1, x_batch, nz]
        loc, scale = loc.unsqueeze(0), scale.unsqueeze(0)
        var = scale**two

        # (z_batch, x_batch, nz)
        dev = z_samples - loc

        # (z_batch, x_batch)
        log_density = -((dev**two) / var).sum(dim=-1) * (one / two) - (
            nz * torch.log(pi * two) + (scale**two).log().sum(-1)
        ) * (one / two)

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - torch.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()
