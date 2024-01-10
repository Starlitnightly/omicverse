import numpy as np


class LinearOT:
    """Transformer class for linear OT problem."""

    def __init__(self, rho: float = 1.0, reg: float = 1e-6):
        self.rho = rho
        self.reg = reg

    def _fit(self, xs, xt, ws, wt):
        return barycenter(xs, xt, self.rho, ws, wt, self.reg)

    def fit(self, xs, xt, ws=None, wt=None):
        """Fit linear OT problem."""
        self.As, self.bs, self.At, self.bt = self._fit(xs, xt, ws, wt)

        return self

    def fit_transform(self, xs, xt, ws=None, wt=None):
        """Fit and transform linear OT problem."""
        self.As, self.bs, self.At, self.bt = self._fit(xs, xt, ws, wt)

        return xs @ self.As + self.bs, xt @ self.At + self.bt


def psd_sqrt(A):
    """Square root of a positive semidefinite matrix A."""
    u, s, v = np.linalg.svd(A)
    return u @ np.diag(s**0.5) @ v


def compute_stats(x, w=None, reg=1e-12):
    """Compute mean, covariance, correlation, and variances from set of data points."""
    d = x.shape[1]
    n = x.shape[0]

    # if weights not given, make them all 1
    if w is None:
        w = np.ones(n)

    # reshape weights to make them easier to multiply
    w_reshaped = w.reshape(-1, 1)

    # mean of the distribution
    m = (x.T @ w_reshaped / np.sum(w_reshaped)).T

    # center points
    x_centered = x - m

    # covariance of each distribution
    cs = (x_centered * w_reshaped).T @ x_centered / np.sum(w)
    cs += np.eye(d) * reg

    # convert covariance to correlation
    variances = np.diag(cs)

    # standarddevs
    stdevs_inverse = (np.sqrt(variances).reshape(-1, 1)) ** -1

    # compute correlation
    corr = cs * (stdevs_inverse @ stdevs_inverse.T)

    res = {"mean": m, "covariance": cs, "correlation": corr, "variance": variances}

    return res


def compute_transformation(cs, ct, ms, mt, rho=1.0, reg=1e-6):
    """Given mean and covariance, compute transport plan."""
    n, ix = ms.shape
    d = cs.shape[0]

    # add regularization to make sure invertible
    cs += np.eye(d) * reg
    ct += np.eye(d) * reg

    # square roots
    cs_sqrt = psd_sqrt(cs)
    ct_sqrt = psd_sqrt(ct)

    cst = psd_sqrt(ct_sqrt @ cs @ ct_sqrt)

    # compute mean and cov of barycenter
    mb = (1 - rho) * ms + rho * mt

    term1 = (1 - rho) * np.eye(d) + rho * ct_sqrt @ np.linalg.pinv(cst) @ ct_sqrt
    term2 = cs
    cb = term1 @ term2 @ term1

    # transport plan for source
    cs_sqrt_pinv = np.linalg.pinv(cs_sqrt)
    As = cs_sqrt_pinv @ psd_sqrt(cs_sqrt @ cb @ cs_sqrt) @ cs_sqrt_pinv
    bs = mb - (As @ ms.T).T

    # transport plan for target
    ct_sqrt_pinv = np.linalg.pinv(ct_sqrt)
    At = ct_sqrt_pinv @ psd_sqrt(ct_sqrt @ cb @ ct_sqrt) @ ct_sqrt_pinv
    bt = mb - (At @ mt.T).T

    return As.T, bs, At.T, bt


def barycenter(
    xs,
    xt,
    rho=1.0,
    ws=None,
    wt=None,
    reg: float = 1e-6,
):
    """Wasserstein barycenter between two Gaussians."""
    # number of points
    d = xs.shape[1]
    ns = xs.shape[0]
    nt = xt.shape[0]

    # if weights not given, make them all 1
    if ws is None:
        ws = np.ones(ns)
    if wt is None:
        wt = np.ones(nt)

    # reshape weights to make them easier to multiply
    ws_reshaped = ws.reshape(-1, 1)
    wt_reshaped = wt.reshape(-1, 1)

    # mean of each distribution
    ms = (xs.T @ ws_reshaped / np.sum(ws_reshaped)).T
    mt = (xt.T @ wt_reshaped / np.sum(wt_reshaped)).T
    # ms = np.mean(xs, axis=0)
    # mt = np.mean(xt, axis=0)

    # center each set of points
    xs_centered = xs - ms
    xt_centered = xt - mt

    # covariance of each distribution
    cs = (xs_centered * ws_reshaped).T @ xs_centered / np.sum(ws)
    ct = (xt_centered * wt_reshaped).T @ xt_centered / np.sum(wt)

    cs += np.eye(d) * reg
    ct += np.eye(d) * reg
    # cs = np.cov(xs.T) + np.eye(d) * reg
    # ct = np.cov(xt.T) + np.eye(d) * reg

    # square roots
    cs_sqrt = psd_sqrt(cs)
    ct_sqrt = psd_sqrt(ct)

    cst = psd_sqrt(ct_sqrt @ cs @ ct_sqrt)

    # compute mean and cov of b,,arycenter
    mb = (1 - rho) * ms + rho * mt

    term1 = (1 - rho) * np.eye(d) + rho * ct_sqrt @ np.linalg.pinv(cst) @ ct_sqrt
    term2 = cs
    cb = term1 @ term2 @ term1

    # transport plan for source
    cs_sqrt_pinv = np.linalg.pinv(cs_sqrt)
    As = cs_sqrt_pinv @ psd_sqrt(cs_sqrt @ cb @ cs_sqrt) @ cs_sqrt_pinv
    bs = mb - (As @ ms.T).T

    # transport plan for target
    ct_sqrt_pinv = np.linalg.pinv(ct_sqrt)
    At = ct_sqrt_pinv @ psd_sqrt(ct_sqrt @ cb @ ct_sqrt) @ ct_sqrt_pinv
    bt = mb - (At @ mt.T).T

    return As.T, bs, At.T, bt
