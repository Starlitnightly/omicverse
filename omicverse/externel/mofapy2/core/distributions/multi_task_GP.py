# import gpytorch
# from gpytorch.constraints import Positive, GreaterThan
# from gpytorch.lazy import DiagLazyTensor,BlockDiagLazyTensor,MatmulLazyTensor, InterpolatedLazyTensor, PsdSumLazyTensor, RootLazyTensor, KroneckerProductLazyTensor, lazify
# from gpytorch.utils.broadcasting import _mul_broadcast_shape
# from gpytorch.kernels.kernel import Kernel
# from gpytorch.kernels import MultitaskKernel
# from typing import Any
# from gpytorch.distributions import base_distributions, MultivariateNormal
# from gpytorch.functions import add_diag
# from gpytorch.likelihoods import Likelihood, _GaussianLikelihoodBase
# from gpytorch.utils.warnings import OldVersionWarning, GPInputWarning
# from gpytorch.likelihoods.noise_models import MultitaskHomoskedasticNoise, Noise, HomoskedasticNoise
# from gpytorch.models.exact_prediction_strategies import prediction_strategy
#
# import warnings
# import torch
# from torch import Tensor
# from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
# # from mofapy2.core.distributions.myExactGP import myExactGP
#
# from gpytorch.models import ExactGP
# from gpytorch.models.gp import GP
# from gpytorch import settings
# from copy import deepcopy
#
#
# # Here we define a multitask GP model where
# # - the noise and scale parameter are coupled --> kernels do not see the raw noise parameter, liklihood is constructed based on K + likelihood.raw_noise I,
# # K is constructed by myMultitaskKernel
# # - the index kernel is constrained to a correlation matrix --> added a scaling factor to the IndexKernel
# # - likelihood is replaced by the ELBO --> the trace(Sigma^(-1) Cov(z)) term to the multitaskmarginal likelihood which is then optimized
#
# # Most of these classes are directly based on classed in gpytorch with minor changes as described above.
#
# # make sure it accepts the new likelihood
# class myExactGP(GP):
#
#     def __init__(self, train_inputs, train_targets, likelihood):
#         if train_inputs is not None and torch.is_tensor(train_inputs):
#             train_inputs = (train_inputs,)
#         if train_inputs is not None and not all(torch.is_tensor(train_input) for train_input in train_inputs):
#             raise RuntimeError("Train inputs must be a tensor, or a list/tuple of tensors")
#         if not isinstance(likelihood, _GaussianLikelihoodBase) and not isinstance(likelihood, _myGaussianLikelihoodBase) :
#             raise RuntimeError("myExactGP can only handle Gaussian likelihoods")
#
#         super(myExactGP, self).__init__()
#         if train_inputs is not None:
#             self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
#             self.train_targets = train_targets
#         else:
#             self.train_inputs = None
#             self.train_targets = None
#         self.likelihood = likelihood
#
#         self.prediction_strategy = None
#
#     @property
#     def train_targets(self):
#         return self._train_targets
#
#     @train_targets.setter
#     def train_targets(self, value):
#         object.__setattr__(self, "_train_targets", value)
#
#     def _apply(self, fn):
#         if self.train_inputs is not None:
#             self.train_inputs = tuple(fn(train_input) for train_input in self.train_inputs)
#             self.train_targets = fn(self.train_targets)
#         return super(myExactGP, self)._apply(fn)
#
#     def local_load_samples(self, samples_dict, memo, prefix):
#         """
#         Replace the model's learned hyperparameters with samples from a posterior distribution.
#         """
#         # Pyro always puts the samples in the first batch dimension
#         num_samples = next(iter(samples_dict.values())).size(0)
#         self.train_inputs = tuple(tri.unsqueeze(0).expand(num_samples, *tri.shape) for tri in self.train_inputs)
#         self.train_targets = self.train_targets.unsqueeze(0).expand(num_samples, *self.train_targets.shape)
#         super().local_load_samples(samples_dict, memo, prefix)
#
#     def set_train_data(self, inputs=None, targets=None, strict=True):
#         """
#         Set training data (does not re-fit model hyper-parameters).
#
#         :param torch.Tensor inputs: The new training inputs.
#         :param torch.Tensor targets: The new training targets.
#         :param bool strict: (default True) If `True`, the new inputs and
#             targets must have the same shape, dtype, and device
#             as the current inputs and targets. Otherwise, any shape/dtype/device are allowed.
#         """
#         if inputs is not None:
#             if torch.is_tensor(inputs):
#                 inputs = (inputs,)
#             inputs = tuple(input_.unsqueeze(-1) if input_.ndimension() == 1 else input_ for input_ in inputs)
#             if strict:
#                 for input_, t_input in zip(inputs, self.train_inputs or (None,)):
#                     for attr in {"shape", "dtype", "device"}:
#                         expected_attr = getattr(t_input, attr, None)
#                         found_attr = getattr(input_, attr, None)
#                         if expected_attr != found_attr:
#                             msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
#                             msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
#                             raise RuntimeError(msg)
#             self.train_inputs = inputs
#         if targets is not None:
#             if strict:
#                 for attr in {"shape", "dtype", "device"}:
#                     expected_attr = getattr(self.train_targets, attr, None)
#                     found_attr = getattr(targets, attr, None)
#                     if expected_attr != found_attr:
#                         msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
#                         msg = msg.format(attr=attr, e_attr=expected_attr, f_attr=found_attr)
#                         raise RuntimeError(msg)
#             self.train_targets = targets
#         self.prediction_strategy = None
#
#     def get_fantasy_model(self, inputs, targets, **kwargs):
#         """
#         Returns a new GP model that incorporates the specified inputs and targets as new training data.
#
#         Using this method is more efficient than updating with `set_train_data` when the number of inputs is relatively
#         np.all, because any computed test-time caches will be updated in linear time rather than computed from scratch.
#
#         .. note::
#             If `targets` is a batch (e.g. `b x m`), then the GP returned from this method will be a batch mode GP.
#             If `inputs` is of the same (or lesser) dimension as `targets`, then it is assumed that the fantasy points
#             are the same for each target batch.
#
#         :param torch.Tensor inputs: (`b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`) Locations of fantasy
#             observations.
#         :param torch.Tensor targets: (`b1 x ... x bk x m` or `f x b1 x ... x bk x m`) Labels of fantasy observations.
#         :return: An `myExactGP` model with `n + m` training examples, where the `m` fantasy examples have been added
#             and all test-time caches have been updated.
#         :rtype: ~gpytorch.models.myExactGP
#         """
#         if self.prediction_strategy is None:
#             raise RuntimeError(
#                 "Fantasy observations can only be added after making predictions with a model so that "
#                 "all test independent caches exist. Call the model on some data first!"
#             )
#
#         model_batch_shape = self.train_inputs[0].shape[:-2]
#
#         if self.train_targets.dim() > len(model_batch_shape) + 1:
#             raise RuntimeError("Cannot yet add fantasy observations to multitask GPs, but this is coming soon!")
#
#         if not isinstance(inputs, list):
#             inputs = [inputs]
#
#         inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in inputs]
#
#         target_batch_shape = targets.shape[:-1]
#         input_batch_shape = inputs[0].shape[:-2]
#         tbdim, ibdim = len(target_batch_shape), len(input_batch_shape)
#
#         if not (tbdim == ibdim + 1 or tbdim == ibdim):
#             raise RuntimeError(
#                 f"Unsupported batch shapes: The target batch shape ({target_batch_shape}) must have either the "
#                 f"same dimension as or one more dimension than the input batch shape ({input_batch_shape})"
#             )
#
#         # Check whether we can properly broadcast batch dimensions
#         err_msg = (
#             f"Model batch shape ({model_batch_shape}) and target batch shape "
#             f"({target_batch_shape}) are not broadcastable."
#         )
#         _mul_broadcast_shape(model_batch_shape, target_batch_shape, error_msg=err_msg)
#
#         if len(model_batch_shape) > len(input_batch_shape):
#             input_batch_shape = model_batch_shape
#         if len(model_batch_shape) > len(target_batch_shape):
#             target_batch_shape = model_batch_shape
#
#         # If input has no fantasy batch dimension but target does, we can save memory and computation by not
#         # computing the covariance for each element of the batch. Therefore we don't expand the inputs to the
#         # size of the fantasy model here - this is done below, after the evaluation and fast fantasy update
#         train_inputs = [tin.expand(input_batch_shape + tin.shape[-2:]) for tin in self.train_inputs]
#         train_targets = self.train_targets.expand(target_batch_shape + self.train_targets.shape[-1:])
#
#         full_inputs = [
#             torch.cat([train_input, input.expand(input_batch_shape + input.shape[-2:])], dim=-2)
#             for train_input, input in zip(train_inputs, inputs)
#         ]
#         full_targets = torch.cat([train_targets, targets.expand(target_batch_shape + targets.shape[-1:])], dim=-1)
#
#         try:
#             fantasy_kwargs = {"noise": kwargs.pop("noise")}
#         except KeyError:
#             fantasy_kwargs = {}
#
#         full_output = super(myExactGP, self).__call__(*full_inputs, **kwargs)
#
#         # Copy model without copying training data or prediction strategy (since we'll overwrite those)
#         old_pred_strat = self.prediction_strategy
#         old_train_inputs = self.train_inputs
#         old_train_targets = self.train_targets
#         old_likelihood = self.likelihood
#         self.prediction_strategy = None
#         self.train_inputs = None
#         self.train_targets = None
#         self.likelihood = None
#         new_model = deepcopy(self)
#         self.prediction_strategy = old_pred_strat
#         self.train_inputs = old_train_inputs
#         self.train_targets = old_train_targets
#         self.likelihood = old_likelihood
#
#         new_model.likelihood = old_likelihood.get_fantasy_likelihood(**fantasy_kwargs)
#         new_model.prediction_strategy = old_pred_strat.get_fantasy_strategy(
#             inputs, targets, full_inputs, full_targets, full_output, **fantasy_kwargs
#         )
#
#         # if the fantasies are at the same points, we need to expand the inputs for the new model
#         if tbdim == ibdim + 1:
#             new_model.train_inputs = [fi.expand(target_batch_shape + fi.shape[-2:]) for fi in full_inputs]
#         else:
#             new_model.train_inputs = full_inputs
#         new_model.train_targets = full_targets
#
#         return new_model
#
#     def train(self, mode=True):
#         if mode:
#             self.prediction_strategy = None
#         return super(myExactGP, self).train(mode)
#
#     def _load_from_state_dict(
#         self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
#     ):
#         self.prediction_strategy = None
#         super()._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
#         )
#
#     def __call__(self, *args, **kwargs):
#         train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
#         inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]
#
#         # Training mode: optimizing
#         if self.training:
#             if self.train_inputs is None:
#                 raise RuntimeError(
#                     "train_inputs, train_targets cannot be None in training mode. "
#                     "Call .eval() for prior predictions, or call .set_train_data() to add training data."
#                 )
#             if settings.debug.on():
#                 if not all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
#                     raise RuntimeError("You must train on the training inputs!")
#             res = super().__call__(*inputs, **kwargs)
#             return res
#
#         # Prior mode
#         elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
#             full_inputs = args
#             full_output = super(myExactGP, self).__call__(*full_inputs, **kwargs)
#             if settings.debug().on():
#                 if not isinstance(full_output, MultivariateNormal):
#                     raise RuntimeError("myExactGP.forward must return a MultivariateNormal")
#             return full_output
#
#         # Posterior mode
#         else:
#             if settings.debug.on():
#                 if all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
#                     warnings.warn(
#                         "The input matches the stored training data. Did you forget to call model.train()?",
#                         GPInputWarning,
#                     )
#
#             # Get the terms that only depend on training data
#             if self.prediction_strategy is None:
#                 train_output = super().__call__(*train_inputs, **kwargs)
#
#                 # Create the prediction strategy for
#                 self.prediction_strategy = prediction_strategy(
#                     train_inputs=train_inputs,
#                     train_prior_dist=train_output,
#                     train_labels=self.train_targets,
#                     likelihood=self.likelihood,
#                 )
#
#             # Concatenate the input to the training input
#             full_inputs = []
#             batch_shape = train_inputs[0].shape[:-2]
#             for train_input, input in zip(train_inputs, inputs):
#                 # Make sure the batch shapes agree for training/test data
#                 if batch_shape != train_input.shape[:-2]:
#                     batch_shape = _mul_broadcast_shape(batch_shape, train_input.shape[:-2])
#                     train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
#                 if batch_shape != input.shape[:-2]:
#                     batch_shape = _mul_broadcast_shape(batch_shape, input.shape[:-2])
#                     train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
#                     input = input.expand(*batch_shape, *input.shape[-2:])
#                 full_inputs.append(torch.cat([train_input, input], dim=-2))
#
#             # Get the joint distribution for training/test data
#             full_output = super(myExactGP, self).__call__(*full_inputs, **kwargs)
#             if settings.debug().on():
#                 if not isinstance(full_output, MultivariateNormal):
#                     raise RuntimeError("myExactGP.forward must return a MultivariateNormal")
#             full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix
#
#             # Determine the shape of the joint distribution
#             batch_shape = full_output.batch_shape
#             joint_shape = full_output.event_shape
#             tasks_shape = joint_shape[1:]  # For multitask learning
#             test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])
#
#             # Make the prediction
#             with settings._use_eval_tolerance():
#                 predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)
#
#             # Reshape predictive mean to match the appropriate event shape
#             predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
#             return full_output.__class__(predictive_mean, predictive_covar)
#
# # default multtask model (with margina likelihood, not scaled task_cov and no coupling
# class MultitaskGPModel(ExactGP):
#     def __init__(self, train_x, train_y, likelihood, n_tasks, rank):
#         super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
#
#         # self.mean_module = gpytorch.means.MultitaskMean(          # optional: learn mean function instead of centering
#         #     gpytorch.means.ConstantMean(), num_tasks=n_tasks
#         # )
#         self.mean_module = gpytorch.means.MultitaskMean(
#             gpytorch.means.ZeroMean(), num_tasks=n_tasks
#         )
#
#         self.covar_module = MultitaskKernel(
#             gpytorch.kernels.RBFKernel(), num_tasks=n_tasks, rank=rank
#         )
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
#
# class MyMultitaskGPModel(myExactGP):
#     def __init__(self, train_x, train_y, likelihood, n_tasks, rank,
#                  var_constraint = None, covar_factor_constraint = None, lengthscale_constraint = None):
#         super(MyMultitaskGPModel, self).__init__(train_x, train_y, likelihood)
#
#         # self.mean_module = gpytorch.means.MultitaskMean(          # optional: learn mean function instead of centering
#         #     gpytorch.means.ConstantMean(), num_tasks=n_tasks
#         # )
#         self.mean_module = gpytorch.means.MultitaskMean(
#             gpytorch.means.ZeroMean(), num_tasks=n_tasks
#         )
#
#         self.covar_module = myMultitaskKernel(
#             gpytorch.kernels.RBFKernel(lengthscale_constraint = lengthscale_constraint), num_tasks=n_tasks, rank=rank,
#             var_constraint = var_constraint, covar_factor_constraint = covar_factor_constraint
#         )
#
#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
#
#
# #    Based on MultitaskKernel: Using a modified IndexKernel - no other changes
# class myMultitaskKernel(Kernel):
#     r"""
#     Kernel supporting Kronecker style multitask Gaussian processes (where every data point is evaluated at every
#     task) using :class:`gpytorch.kernels.IndexKernel` as a basic multitask kernel.
#
#     Given a base covariance module to be used for the data, :math:`K_{XX}`, this kernel computes a task kernel of
#     specified size :math:`K_{TT}` and returns :math:`K = K_{TT} \otimes K_{XX}`. as an
#     :obj:`gpytorch.lazy.KroneckerProductLazyTensor`.
#
#     :param ~gpytorch.kernels.Kernel data_covar_module: Kernel to use as the data kernel.
#     :param int num_tasks: Number of tasks
#     :param int rank: (default 1) Rank of index kernel to use for task covariance matrix.
#     :param ~gpytorch.priors.Prior task_covar_prior: (default None) Prior to use for task kernel.
#         See :class:`gpytorch.kernels.IndexKernel` for details.
#     :param dict kwargs: Additional arguments to pass to the kernel.
#     """
#
#     def __init__(self, data_covar_module, num_tasks, rank=1, task_covar_prior=None,
#                  var_constraint = None, covar_factor_constraint = None, **kwargs):
#         """
#         """
#         super(myMultitaskKernel, self).__init__(**kwargs)
#         self.task_covar_module = myIndexKernel(
#             num_tasks=num_tasks, batch_shape=self.batch_shape, rank=rank, prior=task_covar_prior,
#             var_constraint=var_constraint, covar_factor_constraint = covar_factor_constraint
#         )
#         self.data_covar_module = data_covar_module
#         self.num_tasks = num_tasks
#
#     def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
#         if last_dim_is_batch:
#             raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
#         covar_i = self.task_covar_module.covar_matrix
#         if len(x1.shape[:-2]):
#             covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
#         covar_x = lazify(self.data_covar_module.forward(x1, x2, **params))
#         res = KroneckerProductLazyTensor(covar_x, covar_i)
#         return renp.diag() if diag else res
#
#     def num_outputs_per_input(self, x1, x2):
#         """
#         Given `n` data points `x1` and `m` datapoints `x2`, this multitask
#         kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
#         """
#         return self.num_tasks
#
# # implement an index kernel B^TB +sigma I
# # based on IndexKernel with changes:
# # - a constant diagonal offset
# # - scaled to correlation matrix
# class myIndexKernel(Kernel):
#     r"""
#     A kernel for discrete indices. Kernel is defined by a lookup table.
#
#     .. math::
#
#         \begin{equation}
#             k(i, j) = \left(BB^\top + \text{diag}(\mathbf v) \right)_{i, j}
#         \end{equation}
#
#     where :math:`B` is a low-rank matrix, and :math:`\mathbf v` is a  non-negative vector.
#     These parameters are learned.
#
#     Args:
#         :attr:`num_tasks` (int):
#             Total number of indices.
#         :attr:`batch_shape` (torch.Size, optional):
#             Set if the MultitaskKernel is operating on batches of data (and you want different
#             parameters for each batch)
#         :attr:`rank` (int):
#             Rank of :math:`B` matrix.
#         :attr:`prior` (:obj:`gpytorch.priors.Prior`):
#             Prior for :math:`B` matrix.
#         :attr:`var_constraint` (Constraint, optional):
#             Constraint for added diagonal component. Default: `Positive`.
#
#     Attributes:
#         covar_factor:
#             The :math:`B` matrix.
#         lov_var:
#             The element-wise log of the :math:`\mathbf v` vector.
#     """
#
#     def __init__(self, num_tasks, rank=1, prior=None, var_constraint=None, covar_factor_constraint = None, **kwargs):
#         if rank > num_tasks:
#             raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
#         super().__init__(**kwargs)
#         self.num_tasks = num_tasks
#
#         if var_constraint is None:
#             var_constraint = Positive()
#
#         self.register_parameter(
#             name="raw_covar_factor", parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, num_tasks, rank))
#         )
#         self.register_parameter(name="raw_var",
#                                 parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, 1)))  # 1 instead of num_tasks
#         if prior is not None:
#             self.register_prior("IndexKernelPrior", prior, self._eval_covar_matrix)
#
#         self.register_constraint("raw_var", var_constraint)
#         if covar_factor_constraint is not None:
#             self.register_constraint("raw_covar_factor", covar_factor_constraint)
#             self.has_covar_factor_constraint = True
#         else:
#             self.has_covar_factor_constraint = False
#
#     @property
#     def var(self):
#         return self.raw_var_constraint.transform(self.raw_var)
#
#     @var.setter
#     def var(self, value):
#         self._set_var(value)
#
#     def _set_var(self, value):
#         self.initialize(raw_var=self.raw_var_constraint.inverse_transform(value))
#
#     @property
#     def covar_factor(self):
#         if self.has_covar_factor_constraint:
#             return self.raw_covar_factor_constraint.transform(self.raw_covar_factor)
#         else:
#             return self.raw_covar_factor
#
#     @covar_factor.setter
#     def covar_factor(self, value):
#         self._set_covar_factor(value)
#
#     def _set_covar_factor(self, value):
#         self.initialize(raw_covar_factor=self.raw_covar_factor_constraint.inverse_transform(value))
#
#
#     def _eval_covar_matrix(self):
#         cf = self.covar_factor
#         C = cf @ cf.transpose(-1, -2) + self.var *  torch.eye(self.num_tasks)   # instead of diag(var)
#         Cdiag = torch.diag(C)
#         C = torch.diag(1 / Cdiag.sqrt()) @ C @ torch.diag(1 / Cdiag.sqrt())       # scale to correlation matrix
#         return C
#
#     @property
#     def covar_matrix(self):
#         # var = self.var
#         # res = PsdSumLazyTensor(RootLazyTensor(self.covar_factor), DiagLazyTensor(var))
#         # return res
#         return self._eval_covar_matrix()
#
#     def forward(self, i1, i2, **params):
#         covar_matrix = self._eval_covar_matrix()
#         batch_shape = _mul_broadcast_shape(i1.shape[:-2], self.batch_shape)
#         index_shape = batch_shape + i1.shape[-2:]
#
#         res = InterpolatedLazyTensor(
#             base_lazy_tensor=covar_matrix,
#             left_interp_indices=i1.expand(index_shape),
#             right_interp_indices=i2.expand(index_shape),
#         )
#         return res
#
#
#
# class ELBO(MarginalLogLikelihood):
#     """
#     Class to implment ELBO as loss function - based on ExactMarginalLogLikelihood
#     """
#
#     def __init__(self, likelihood, model, ZCov):
#         if not isinstance(likelihood, _GaussianLikelihoodBase) and not isinstance(likelihood, _myGaussianLikelihoodBase):
#             raise RuntimeError("Likelihood must be Gaussian for exact inference")
#         super(ELBO, self).__init__(likelihood, model)
#         self.ZCov = ZCov
#
#     def forward(self, function_dist, target, *params):
#         r"""
#         Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.
#
#         :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
#             the outputs of the latent function (the :obj:`gpytorch.models.myExactGP`)
#         :param torch.Tensor target: :math:`\mathbf y` The target values
#         :rtype: torch.Tensor
#         :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
#         """
#         if not isinstance(function_dist, MultivariateNormal):
#             raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")
#
#         # Get the log prob of the marginal distribution
#         output = self.likelihood(function_dist, *params) # this is N(mu, K + noise_var I) should be N(mu, (1-noise_var) * K + noise_var I) TODO multiply kernel by (1- self.likelihood.raw_noise)
#
#         res = output.log_prob(target)
#         # modify to get N(mu, (1-noise_var) * K + noise_var I)
#
#         # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
#         for added_loss_term in self.model.added_loss_terms():
#             res = res.add(added_loss_term.loss(*params))
#
#         # Add uncertainty in Z
#         res.add(-0.5 * torch.trace(output.precision_matrix @ self.ZCov))
#
#         # Add log probs of priors on the (functions of) parameters
#         for _, prior, closure, _ in self.named_priors():
#             res.add_(prior.log_prob(closure()).sum())
#
#         # Scale by the amount of data we have
#         num_data = target.size(-1)
#         return res.div_(num_data)
#
#
#
# # Modify liklelihoods: is N(mu, K + noise_var I) should be N(mu, (1-noise_var) * K + noise_var I
# class _myGaussianLikelihoodBase(Likelihood):
#     """Base class for Gaussian Likelihoods, supporting general heteroskedastic noise models."""
#
#     def __init__(self, noise_covar: Noise, **kwargs: Any) -> None:
#
#         super().__init__()
#         param_transform = kwargs.get("param_transform")
#         if param_transform is not None:
#             warnings.warn(
#                 "The 'param_transform' argument is now deprecated. If you want to use a different "
#                 "transformaton, specify a different 'noise_constraint' instead.",
#                 DeprecationWarning,
#             )
#
#         self.noise_covar = noise_covar
#
#     def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
#         return self.noise_covar(*params, shape=base_shape, **kwargs)
#
#     def expected_log_prob(self, target: Tensor, input: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
#         mean, variance = input.mean, input.variance
#         num_event_dim = len(input.event_shape)
#
#         noise = self._shaped_noise_covar(mean.shape, *params, **kwargs).diag()
#         # Potentially reshape the noise to deal with the multitask case
#         noise = noise.view(*noise.shape[:-1], *input.event_shape)
#
#         res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
#         res = res.mul(-0.5)
#         if num_event_dim > 1:  # Do appropriate summation for multitask Gaussian likelihoods
#             res = res.sum(list(range(-1, -num_event_dim, -1)))
#         return res
#
#     def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> base_distributions.Normal:
#         noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diag()
#         return base_distributions.Normal(function_samples, noise.sqrt())
#
#     def log_marginal(
#         self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
#     ) -> Tensor:
#         marginal = self.marginal(function_dist, *params, **kwargs)
#         # We're making everything conditionally independent
#         indep_dist = base_distributions.Normal(marginal.mean, marginal.variance.clamp_min(1e-8).sqrt())
#         res = indep_dist.log_prob(observations)
#
#         # Do appropriate summation for multitask Gaussian likelihoods
#         num_event_dim = len(function_dist.event_shape)
#         if num_event_dim > 1:
#             res = res.sum(list(range(-1, -num_event_dim, -1)))
#         return res
#
#     def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
#         mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
#         noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
#         # print(self.raw_noise)
#         # print(self.noise)
#         # print(self.raw_noise_constraint)
#         full_covar = (1-self.noise) * covar + noise_covar
#         # full_covar = covar + noise_covar
#         return function_dist.__class__(mean, full_covar)
#
# # as rank =0 we do not model noise correlation - most things that happen here are irrelevant
# class _myMultitaskGaussianLikelihoodBase(_myGaussianLikelihoodBase):
#     """Base class for multi-task Gaussian Likelihoods, supporting general heteroskedastic noise models. """
#
#     def __init__(self, num_tasks, noise_covar, rank=0, task_correlation_prior=None, batch_shape=torch.Size()):
#         """
#         Args:
#             num_tasks (int):
#                 Number of tasks.
#             noise_covar (:obj:`gpytorch.module.Module`):
#                 A model for the noise covariance. This can be a simple homoskedastic noise model, or a GP
#                 that is to be fitted on the observed measurement errors.
#             rank (int):
#                 The rank of the task noise covariance matrix to fit. If `rank` is set to 0, then a diagonal covariance
#                 matrix is fit.
#             task_correlation_prior (:obj:`gpytorch.priors.Prior`):
#                 Prior to use over the task noise correlation matrix. Only used when `rank` > 0.
#             batch_shape (torch.Size):
#                 Number of batches.
#         """
#         super().__init__(noise_covar=noise_covar)
#         if rank != 0:
#             if rank > num_tasks:
#                 raise ValueError(f"Cannot have rank ({rank}) greater than num_tasks ({num_tasks})")
#             tidcs = torch.tril_indices(num_tasks, rank, dtype=torch.long)
#             self.tidcs = tidcs[:, 1:]  # (1, 1) must be 1.0, no need to parameterize this
#             task_noise_corr = torch.randn(*batch_shape, self.tidcs.size(-1))
#             self.register_parameter("task_noise_corr", torch.nn.Parameter(task_noise_corr))
#             if task_correlation_prior is not None:
#                 self.register_prior(
#                     "MultitaskErrorCorrelationPrior", task_correlation_prior, lambda: self._eval_corr_matrix
#                 )
#         elif task_correlation_prior is not None:
#             raise ValueError("Can only specify task_correlation_prior if rank>0")
#         self.num_tasks = num_tasks
#         self.rank = rank
#         # # Handle deprecation of parameterization - TODO: Remove in future release
#         # self._register_load_state_dict_pre_hook(deprecate_task_noise_corr)
#
#     def _eval_corr_matrix(self):
#         tnc = self.task_noise_corr
#         fac_diag = torch.ones(*tnc.shape[:-1], self.num_tasks, device=tnc.device, dtype=tnc.dtype)
#         Cfac = torch.diag_embed(fac_diag)
#         Cfac[..., self.tidcs[0], self.tidcs[1]] = self.task_noise_corr
#         # squared rows must sum to one for this to be a correlation matrix
#         C = Cfac / Cfac.pow(2).sum(dim=-1, keepdim=True).sqrt()
#         return C @ C.transpose(-1, -2)
#
#     def _shaped_noise_covar(self, base_shape, *params):
#         if len(base_shape) >= 2:
#             *batch_shape, n, _ = base_shape
#         else:
#             *batch_shape, n = base_shape
#
#         # compute the noise covariance
#         if len(params) > 0:
#             shape = None
#         else:
#             shape = base_shape if len(base_shape) == 1 else base_shape[:-1]
#         noise_covar = self.noise_covar(*params, shape=shape)
#
#         if self.rank > 0:
#             # if rank > 0, compute the task correlation matrix
#             # TODO: This is inefficient, change repeat so it can repeat LazyTensors w/ multiple batch dimensions
#             task_corr = self._eval_corr_matrix()
#             exp_shape = torch.Size([*batch_shape, n]) + task_corr.shape[-2:]
#             task_corr_exp = lazify(task_corr.unsqueeze(-3).expand(exp_shape))
#             noise_sem = noise_covar.sqrt()
#             task_covar_blocks = MatmulLazyTensor(MatmulLazyTensor(noise_sem, task_corr_exp), noise_sem)
#         else:
#             # otherwise tasks are uncorrelated
#             task_covar_blocks = noise_covar
#
#         if len(batch_shape) == 1:
#             # TODO: Properly support general batch shapes in BlockDiagLazyTensor (no shape arithmetic)
#             tcb_eval = task_covar_blocks.evaluate()
#             task_covar = BlockDiagLazyTensor(lazify(tcb_eval), block_dim=-3)
#         else:
#             task_covar = BlockDiagLazyTensor(task_covar_blocks)
#
#         return task_covar
#
#     def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> base_distributions.Normal:
#         noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diag()
#         noise = noise.view(*noise.shape[:-1], *function_samples.shape[-2:])
#         return base_distributions.Independent(base_distributions.Normal(function_samples, noise.sqrt()), 1)
#
#
# class myMultitaskGaussianLikelihood(_myMultitaskGaussianLikelihoodBase):
#     """
#     A convenient extension of the :class:`gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
#     for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
#     If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
#     allows for a different `log_noise` parameter for each task.). This likelihood assumes homoskedastic noise.
#
#     Like the Gaussian likelihood, this object can be used with exact inference.
#     """
#
#     def __init__(
#         self,
#         num_tasks,
#         rank=0,
#         task_correlation_prior=None,
#         batch_shape=torch.Size(),
#         noise_prior=None,
#         noise_constraint=None,
#     ):
#         """
#         Args:
#             num_tasks (int): Number of tasks.
#
#             rank (int): The rank of the task noise covariance matrix to fit. If `rank` is set to 0,
#             then a diagonal covariance matrix is fit.
#
#             task_correlation_prior (:obj:`gpytorch.priors.Prior`): Prior to use over the task noise correlaton matrix.
#             Only used when `rank` > 0.
#
#         """
#         if noise_constraint is None:
#             noise_constraint = GreaterThan(1e-4)
#
#         noise_covar = MultitaskHomoskedasticNoise(
#             num_tasks=num_tasks, noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
#         )
#         super().__init__(
#             num_tasks=num_tasks,
#             noise_covar=noise_covar,
#             rank=rank,
#             task_correlation_prior=task_correlation_prior,
#             batch_shape=batch_shape,
#         )
#
#         self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
#         self.register_constraint("raw_noise", noise_constraint)
#
#     @property
#     def noise(self):
#         return self.raw_noise_constraint.transform(self.raw_noise)
#
#     @noise.setter
#     def noise(self, value):
#         self._set_noise(value)
#
#     def _set_noise(self, value):
#         if not torch.is_tensor(value):
#             value = torch.as_tensor(value).to(self.raw_noise)
#         self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))
#
#     def _shaped_noise_covar(self, base_shape, *params):
#         noise_covar = super()._shaped_noise_covar(base_shape, *params)
#         noise = self.noise
#         return noise_covar.add_diag(noise)