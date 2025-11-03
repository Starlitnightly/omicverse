from .prism import *
import warnings

def get_fraction(bp, which_theta, state_or_type):
    # assert isinstance(bp, BayesPrism)
    assert isinstance(which_theta, str) and which_theta in ["first", "final"]
    assert isinstance(state_or_type, str) and state_or_type in ["state", "type"]
    if which_theta == "first" and state_or_type == "state":
        return bp.posterior_initial_cellState.theta
    if which_theta == "first" and state_or_type == "type":
        return bp.posterior_initial_cellType.theta
    if which_theta == "final":
        if state_or_type == "state":
            warnings.warn("Warning: only cell type is available for updated Gibbs. Returning cell type info.")
        return bp.posterior_theta_f.theta


def get_exp(bp, state_or_type, cell_name):
    assert isinstance(bp, BayesPrism)
    assert isinstance(state_or_type, str) and state_or_type in ["state", "type"]
    if state_or_type == "state":
        return bp.posterior_initial_cellState.Z.loc[:,:,cell_name]
    if state_or_type == "type":
        return bp.posterior_initial_cellType.Z.loc[:,:,cell_name]
