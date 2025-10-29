#from pybayesprism import prism
import numpy as np
import pandas as pd


def step1(bk, sc, cell_type, cell_state):
    r_bk = pd.read_csv("./compare/step1_r_bk.csv", index_col = 0).astype('int32')
    r_sc = pd.read_csv("./compare/step1_r_sc.csv", index_col = 0).astype('int32')
    r_cell_type = list(pd.read_csv("./compare/step1_r_celltype.csv").iloc[:,0])
    r_cell_state = list(pd.read_csv("./compare/step1_r_cellstate.csv").iloc[:,0])

    # bk
    assert np.array_equal(bk.to_numpy(), r_bk.to_numpy()), "bk values"
    assert list(bk.index) == list(r_bk.index), "bk index"
    assert list(bk.columns) == list(r_bk.columns), "bk columns"

    # sc
    assert np.array_equal(sc.to_numpy(), r_sc.to_numpy()), "sc values"
    assert list(sc.index) == list(r_sc.index), "sc index"
    assert list(sc.columns) == list(r_sc.columns), "sc columns"

    # cell_type
    assert cell_type == r_cell_type, "cell_type"

    # cell_state
    assert cell_state == r_cell_state, "cell_state"


def step2(p):
    r_cellstate = pd.read_csv("./compare/step2_r_cellstate.csv", index_col = 0)
    r_celltype = pd.read_csv("./compare/step2_r_celltype.csv", index_col = 0)
    r_mixture = pd.read_csv("./compare/step2_r_mixture.csv", index_col = 0)

    # p.phi_cellState.phi
    assert np.allclose(p.phi_cellState.phi.to_numpy(), r_cellstate.to_numpy()), "phi_cellState.phi values"
    assert list(p.phi_cellState.phi.index) == list(r_cellstate.index), "phi_cellState.phi index"
    assert list(p.phi_cellState.phi.columns) == list(r_cellstate.columns), "phi_cellState.phi columns"
    assert p.phi_cellState.pseudo_min == 1e-08, "phi_cellState.pseudo_min"

    # p.phi_cellType.phi
    assert np.allclose(p.phi_cellType.phi.to_numpy(), r_celltype.to_numpy()), "phi_cellType.phi values"
    assert list(p.phi_cellType.phi.index) == list(r_celltype.index), "phi_cellType.phi index"
    assert list(p.phi_cellType.phi.columns) == list(r_celltype.columns), "phi_cellType.phi columns"
    assert p.phi_cellType.pseudo_min == 1e-08, "phi_cellType.pseudo_min"

    # p.map
    assert sum([len(v) for v in p.map.values()]) == 73, "map"

    # p.key
    assert p.key == "tumor", "key"
    
    # p.mixture
    assert np.allclose(p.mixture.to_numpy(), r_mixture.to_numpy()), "mixture values"
    assert list(p.mixture.index) == list(r_mixture.index), "mixture index"
    assert list(p.mixture.columns) == list(r_mixture.columns), "mixture columns"


def step3(bp):
    r_cs_phi = pd.read_csv("./compare/step3_r_phi.csv", index_col = 0)
    r_cs_Z_index = list(pd.read_csv("./compare/step3_r_cs_Z_index.csv", header=None).iloc[:,0])
    r_cs_Z_columns = list(pd.read_csv("./compare/step3_r_cs_Z_columns.csv", header=None).iloc[:,0])
    r_cs_Z_layer = list(pd.read_csv("./compare/step3_r_cs_Z_layer.csv", header=None).iloc[:,0])
    r_cs_theta = pd.read_csv("./compare/step3_r_cs_theta.csv", index_col = 0)
    r_cs_theta_cv = pd.read_csv("./compare/step3_r_cs_theta_cv.csv", index_col = 0)

    r_ct_Z_melt = pd.read_csv("./compare/step3_r_ct_Z.csv", index_col = 0)
    r_ct_Z = np.empty((np.max(r_ct_Z_melt.iloc[:, 0]), np.max(r_ct_Z_melt.iloc[:, 1]), np.max(r_ct_Z_melt.iloc[:, 2])))
    r_ct_Z_indices = (r_ct_Z_melt.iloc[:, 0:3].to_numpy() - 1).astype(int)
    r_ct_Z_indices_values = r_ct_Z_melt.iloc[:, 3].to_numpy()
    r_ct_Z[r_ct_Z_indices[:, 0], r_ct_Z_indices[:, 1], r_ct_Z_indices[:, 2]] = r_ct_Z_indices_values

    r_ct_Z_index = list(pd.read_csv("./compare/step3_r_ct_Z_index.csv", header=None).iloc[:,0])
    r_ct_Z_columns = list(pd.read_csv("./compare/step3_r_ct_Z_columns.csv", header=None).iloc[:,0])
    r_ct_Z_layer = list(pd.read_csv("./compare/step3_r_ct_Z_layer.csv", header=None).iloc[:,0])
    r_ct_theta = pd.read_csv("./compare/step3_r_ct_theta.csv", index_col = 0)

    # bp.posterior_initial_cellState.Z
    assert np.corrcoef(bp.prism.phi_cellState.phi.to_numpy().flatten(), r_cs_phi.to_numpy().flatten())[0, 1] > 0.999, "cs phi values"
    assert list(bp.posterior_initial_cellState.Z.coords['bulk_id'].values) == list(r_cs_Z_index), "cs Z index"
    assert list(bp.posterior_initial_cellState.Z.coords['gene_id'].values) == list(r_cs_Z_columns), "cs Z columns"
    assert list(bp.posterior_initial_cellState.Z.coords['cell_type'].values) == list(r_cs_Z_layer), "cs Z layer"

    # bp.posterior_initial_cellState.theta
    assert np.corrcoef(bp.posterior_initial_cellState.theta.to_numpy().flatten(), r_cs_theta.to_numpy().flatten())[0, 1] > 0.999, "cs theta values"
    assert list(bp.posterior_initial_cellState.theta.index) == list(r_cs_theta.index), "cs theta index"
    assert list(bp.posterior_initial_cellState.theta.columns) == list(r_cs_theta.columns), "cs theta columns"

    # bp.posterior_initial_cellState.theta_cv
    assert np.corrcoef(bp.posterior_initial_cellState.theta_cv.to_numpy().flatten(), r_cs_theta_cv.to_numpy().flatten())[0, 1] > 0.900, "cs theta_cv values"
    assert list(bp.posterior_initial_cellState.theta_cv.index) == list(r_cs_theta_cv.index), "cs theta_cv index"
    assert list(bp.posterior_initial_cellState.theta_cv.columns) == list(r_cs_theta_cv.columns), "cs theta_cv columns"

    # bp.posterior_initial_cellState.constant
    assert bp.posterior_initial_cellState.constant == 0, "cs constant"

    # bp.posterior_initial_cellType.Z
    assert np.corrcoef(bp.posterior_initial_cellType.Z.to_numpy().flatten(), r_ct_Z.flatten())[0, 1] > 0.999, "ct Z values"
    assert list(bp.posterior_initial_cellType.Z.coords['bulk_id'].values) == list(r_ct_Z_index), "ct Z index"
    assert list(bp.posterior_initial_cellType.Z.coords['gene_id'].values) == list(r_ct_Z_columns), "ct Z columns"
    assert list(bp.posterior_initial_cellType.Z.coords['cell_type_merged'].values) == list(r_ct_Z_layer), "ct Z layer"

    # bp.posterior_initial_cellType.theta
    assert np.corrcoef(bp.posterior_initial_cellType.theta.to_numpy().flatten(), r_ct_theta.to_numpy().flatten())[0, 1] > 0.999, "ct theta values"
    assert list(bp.posterior_initial_cellType.theta.index) == list(r_ct_theta.index), "ct theta index"
    assert list(bp.posterior_initial_cellType.theta.columns) == list(r_ct_theta.columns), "ct theta columns"


def step4(bp):
    r_reference_psi_mal = pd.read_csv("./compare/step4_r_reference_psi_mal.csv", index_col = 0)
    r_reference_psi_env = pd.read_csv("./compare/step4_r_reference_psi_env.csv", index_col = 0)
    r_posterior_theta = pd.read_csv("./compare/step4_r_posterior_theta.csv", index_col = 0)
    r_posterior_theta_cv = pd.read_csv("./compare/step4_r_posterior_theta_cv.csv", index_col = 0)

    # bp.reference_update.psi_mal
    assert np.corrcoef(bp.reference_update.psi_mal.to_numpy().flatten(), r_reference_psi_mal.to_numpy().flatten())[0, 1] > 0.999, "reference psi_mal values"
    assert list(bp.reference_update.psi_mal.index) == list(r_reference_psi_mal.index), "reference psi_mal index"
    assert list(bp.reference_update.psi_mal.columns) == list(r_reference_psi_mal.columns), "reference psi_mal columns"

    # bp.reference_update.psi_env
    assert np.corrcoef(bp.reference_update.psi_env.to_numpy().flatten(), r_reference_psi_env.to_numpy().flatten())[0, 1] > 0.999, "reference psi_env values"
    assert list(bp.reference_update.psi_env.index) == list(r_reference_psi_env.index), "reference psi_env index"
    assert list(bp.reference_update.psi_env.columns) == list(r_reference_psi_env.columns), "reference psi_env columns"

    # bp.reference_update.key
    assert bp.reference_update.key == "tumor", "reference key"

    # bp.posterior_theta_f.theta
    assert np.corrcoef(bp.posterior_theta_f.theta.to_numpy().flatten(), r_posterior_theta.to_numpy().flatten())[0, 1] > 0.999, "posterior theta values"
    assert list(bp.posterior_theta_f.theta.index) == list(r_posterior_theta.index), "posterior theta index"
    assert list(bp.posterior_theta_f.theta.columns) == list(r_posterior_theta.columns), "posterior theta columns"

    # bp.posterior_theta_f.theta_cv
    assert np.corrcoef(bp.posterior_theta_f.theta_cv.to_numpy().flatten(), r_posterior_theta_cv.to_numpy().flatten())[0, 1] > 0.900, "posterior theta_cv values"
    assert list(bp.posterior_theta_f.theta_cv.index) == list(r_posterior_theta_cv.index), "posterior theta_cv index"
    assert list(bp.posterior_theta_f.theta_cv.columns) == list(r_posterior_theta_cv.columns), "posterior theta_cv columns"
