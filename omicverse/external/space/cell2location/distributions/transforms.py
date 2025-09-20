from torch.distributions import constraints
from torch.distributions.transforms import Transform


class SoftplusTransform(Transform):
    r"""
    Transform via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    """
    domain = constraints.real_vector
    codomain = constraints.positive

    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        return (1 + x.exp()).log()

    def _inverse(self, y):
        return y.expm1().log()

    def log_abs_det_jacobian(self, x, y):
        return -(1 + (-x).exp()).log()
