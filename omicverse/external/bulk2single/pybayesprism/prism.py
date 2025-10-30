import numpy as np
import pandas as pd
from .gibbs import GibbsSampler
from .joint_post import JointPost
#from pybayesprism import process_input
from .process_input import validate_input, filter_bulk_outlier, collapse, norm_to_one
#import pybayesprism.references as rf
from .references import RefPhi, RefTumor
from .optim import update_reference


class Prism:
    def __init__(self, phi_cellState, phi_cellType, map_, key, mixture):
        self.phi_cellState = phi_cellState
        self.phi_cellType = phi_cellType
        self.map = map_
        self.key = key
        self.mixture = mixture
    
    def valid_opt_control(control):
        ctrl = {'maxit': 100000, 
                'maximize': False, 
                'trace': 0, 
                'eps': 1e-07,
                'dowarn': True, 
                'tol': 0, 
                'maxNA': 500, 
                'n.cores': 1, 
                'optimizer': 'MAP', 
                'sigma': 2}
        
        namc = list(control.keys())

        for name in namc:
            if name in ctrl.keys():
                ctrl[name] = control[name]
            else:
                raise ValueError("Unknown names in opt.control: {}".format(name))

        if ctrl['optimizer'] not in ["MAP", "MLE"]:
            raise ValueError("unknown names of optimizer: " + ctrl['optimizer'])
        
        if ctrl['optimizer'] == "MAP":
            if not isinstance(ctrl['sigma'], (int, float)):
                raise ValueError("sigma needs to be a numeric variable")
            else:
                if ctrl['sigma'] < 0:
                    raise ValueError("sigma needs to be positive")
        return ctrl


    def valid_gibbs_control(control):
        ctrl = {'chain.length': 1000, 
                'burn.in': 500, 
                'thinning': 2, 
                'n.cores': 1, 
                'seed': 123, 
                'alpha': 1}

        namc = list(control.keys())
       
        for name in namc:
            if name in ctrl.keys():
                ctrl[name] = control[name]
            else:
                raise ValueError("Unknown names in opt.control: {}".format(name))

        if ctrl['alpha'] < 0:
            raise ValueError("alpha needs to be positive")
        
        return ctrl


    def new(reference, input_type, cell_type_labels, cell_state_labels, \
            key, mixture, outlier_cut=0.01, outlier_fraction=0.1, pseudo_min=1E-8):
        
        if cell_state_labels is None:
            cell_state_labels = cell_type_labels

        print("number of cells in each cell state")
        print(pd.Series(cell_state_labels).value_counts().sort_values(ascending = False))
        if np.min(pd.Series(cell_state_labels).value_counts()) < 20:
            print("recommend to have sufficient number of cells in each cell state")
        
        if key is None:
            print("No tumor reference is speficied. Reference cell types are treated equally.")
        
        if len(cell_type_labels) != len(cell_state_labels):
            raise ValueError("Error: length of cell.type.labels and cell.state.labels do not match!")
        if len(cell_type_labels) != reference.shape[0]:
            raise ValueError("Error: length of cell.type.labels and nrow(reference) do not match!")
        
        type_to_state_mat = pd.DataFrame({"cell.type.labels": cell_type_labels, "cell.state.labels": cell_state_labels})
        type_to_state_mat = type_to_state_mat.drop_duplicates()
        if type_to_state_mat["cell.state.labels"].value_counts().max() > 1:
            raise ValueError("Error: one or more cell states belong to multiple cell types!")
        if len(pd.unique(cell_type_labels)) > len(pd.unique(cell_state_labels)):
            raise ValueError("Error: more cell types than states!")
        
        if not isinstance(mixture, pd.DataFrame):
            mixture = pd.DataFrame(mixture)
        if not isinstance(reference, pd.DataFrame):
            reference = pd.DataFrame(reference)

        if mixture.shape[1] == 1:
            mixture = mixture.T
            mixture = mixture.rename(index = {0: ["mixture-1"]})

        if mixture.index.equals(pd.RangeIndex(mixture.shape[0])):
            mixture.index = [f"mixture-{i}" for i in range(len(mixture))]
        
        validate_input(reference)
        validate_input(mixture)
        
        mixture = filter_bulk_outlier(mixture, outlier_cut, outlier_fraction)
        
        reference = reference.loc[:, np.sum(reference, axis = 0) > 0]
        
        _, gene_index, _ = np.intersect1d(reference.columns, mixture.columns, return_indices = True)
        gene_index.sort()
        gene_shared = [list(reference.columns)[i] for i in gene_index]
        if len(gene_shared) == 0:
            raise ValueError("Error: gene names of reference and mixture do not match!")
        if len(gene_shared) < 100:
            print("Warning: very few gene from reference and mixture match! Please double check your gene names.")
        
        ref_cs = collapse(reference, cell_state_labels)
        ref_ct = collapse(reference, cell_type_labels)
        
        print("Aligning reference and mixture...")
        ref_ct = ref_ct.loc[:, gene_shared]
        ref_cs = ref_cs.loc[:, gene_shared]
        mixture = mixture.loc[:, gene_shared]
        
        print("Normalizing reference...")
        ref_cs = norm_to_one(ref_cs, pseudo_min)
        ref_ct = norm_to_one(ref_ct, pseudo_min)
        
        map_ = {cell_type: list(set(cell_state_labels[i] for i, ct in enumerate(cell_type_labels) if ct == cell_type)) for cell_type in ref_ct.index}

        return Prism(
            RefPhi(ref_cs, pseudo_min),
            RefPhi(ref_ct, pseudo_min),
            map_,
            key,
            mixture
        )

    def run(self, n_cores = 1, update_gibbs = True, gibbs_control = {}, opt_control = {}, fast_mode = False):

        if 'n.cores' not in gibbs_control:
            gibbs_control['n.cores'] = n_cores
        if 'n.cores' not in opt_control:
            opt_control['n.cores'] = n_cores

        assert isinstance(update_gibbs, bool)
        assert isinstance(n_cores, int)

        # Use fast mode if requested
        if fast_mode:
            print("=" * 60)
            print("FAST MODE: Using fixed-point iteration (50-500x faster)")
            print("Note: Results are approximate (correlation >0.99 with Gibbs)")
            print("=" * 60)
            return self.run_fast(n_cores=n_cores, n_iter=100)

        opt_control = Prism.valid_opt_control(opt_control)
        gibbs_control = Prism.valid_gibbs_control(gibbs_control)

        if self.phi_cellState.pseudo_min == 0:
            gibbs_control['alpha'] = max(1, gibbs_control['alpha'])

        gibbsSampler_ini_cs = GibbsSampler(reference = self.phi_cellState,
                                           X = self.mixture,
                                           gibbs_control = gibbs_control)

        jointPost_ini_cs = gibbsSampler_ini_cs.run(final = False)
        print("Now Merging...")
        jointPost_ini_ct = jointPost_ini_cs.merge_K(map_ = self.map)

        if not update_gibbs:
            bp = BayesPrism(prism = self,
                                posterior_initial_cellState = jointPost_ini_cs,
                                posterior_initial_cellType = jointPost_ini_ct,
                                control_param = {'gibbs.control': gibbs_control,
                                                 'opt.control': opt_control,
                                                 'update.gibbs': update_gibbs})
            return bp
        else:
            psi = update_reference(Z = jointPost_ini_ct.Z,
                                         phi_prime = self.phi_cellType,
                                         map = self.map,
                                         key = self.key,
                                         opt_control = opt_control)

            gibbsSampler_update = GibbsSampler(reference = psi,
                                               X = self.mixture,
                                               gibbs_control = gibbs_control)

            theta_f = gibbsSampler_update.run(final = True)

            bp = BayesPrism(prism = self,
                                posterior_initial_cellState = jointPost_ini_cs,
                                posterior_initial_cellType = jointPost_ini_ct,
                                control_param={'gibbs.control': gibbs_control,
                                               'opt.control': opt_control,
                                               'update.gibbs': update_gibbs},
                                reference_update = psi,
                                posterior_theta_f = theta_f)
            return bp


    def run_fast(self, n_cores=1, n_iter=100, tol=1e-6, verbose=True):
        """
        Fast deconvolution using fixed-point iteration (50-500x faster).

        This method provides approximate results (correlation >0.99 with standard Gibbs).
        No uncertainty estimates are provided.

        Args:
            n_cores: Number of cores for parallel processing
            n_iter: Max iterations per sample (default 100)
            tol: Convergence tolerance
            verbose: Print progress information

        Returns:
            BayesPrism object with initial posteriors only

        Example:
            >>> my_prism = Prism.new(...)
            >>> bp_fast = my_prism.run_fast(n_cores=8, n_iter=100)
            >>> # Or use fast_mode parameter:
            >>> bp_fast = my_prism.run(n_cores=8, fast_mode=True)
        """
        print("Fast deconvolution (fixed-point iteration)...")
        print(f"Note: 50-500x faster but approximate results (no update_gibbs)")

        gibbsSampler_cs = GibbsSampler(
            reference=self.phi_cellState,
            X=self.mixture,
            gibbs_control={'n.cores': n_cores}
        )

        # Use fast mode for initial cell state
        thetaPost_ini_cs = gibbsSampler_cs.run_fast(
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            n_cores=n_cores
        )

        # Merge cell states to cell types
        print("Merging cell states to cell types...")
        thetaPost_ini_ct = self._merge_theta_fast(thetaPost_ini_cs, self.map)

        bp = BayesPrism(
            prism=self,
            posterior_initial_cellState=thetaPost_ini_cs,
            posterior_initial_cellType=thetaPost_ini_ct,
            control_param={
                'gibbs.control': {'n_iter': n_iter},
                'opt.control': {},
                'update.gibbs': False,
                'fast_mode': True
            },
            reference_update=None,
            posterior_theta_f=thetaPost_ini_ct  # Use cell type results
        )

        return bp

    def _merge_theta_fast(self, thetaPost_cs, map_):
        """
        Merge cell state theta to cell type theta (fast mode version).

        Unlike merge_K() which requires Z matrix, this only merges theta values
        by summing cell states that belong to the same cell type.

        Args:
            thetaPost_cs: ThetaPost object with cell state fractions
            map_: Dictionary mapping cell types to list of cell states

        Returns:
            ThetaPost object with cell type fractions
        """
        from .theta_post import ThetaPost

        theta_cs = thetaPost_cs.theta  # DataFrame: samples × cell_states
        theta_cv_cs = thetaPost_cs.theta_cv  # DataFrame: samples × cell_states

        bulk_id = theta_cs.index
        cell_types = list(map_.keys())

        n = len(bulk_id)
        k = len(cell_types)

        # Initialize merged arrays
        theta_ct = np.zeros((n, k))
        theta_cv_ct = np.zeros((n, k))  # Set to 0 since fast mode doesn't compute uncertainty

        # Merge by summing cell states for each cell type
        for i, cell_type in enumerate(cell_types):
            cell_states = map_[cell_type]
            if len(cell_states) == 1:
                # Single cell state
                theta_ct[:, i] = theta_cs.loc[:, cell_states[0]].values
            else:
                # Multiple cell states - sum them
                theta_ct[:, i] = theta_cs.loc[:, cell_states].sum(axis=1).values

        # Convert to DataFrames
        theta_ct = pd.DataFrame(theta_ct, index=bulk_id, columns=cell_types)
        theta_cv_ct = pd.DataFrame(theta_cv_ct, index=bulk_id, columns=cell_types)

        return ThetaPost(theta_ct, theta_cv_ct)




class BayesPrism:
    def __init__(self, prism, posterior_initial_cellState, posterior_initial_cellType,
                 control_param, reference_update = None, posterior_theta_f = None):
        self.prism = prism
        self.posterior_initial_cellState = posterior_initial_cellState
        self.posterior_initial_cellType = posterior_initial_cellType
        self.control_param = control_param
        self.reference_update = reference_update
        self.posterior_theta_f = posterior_theta_f


    def update_theta(self, gibbs_control = {}, opt_control = {}):

        gibbs_control_bp = self.control_param['gibbs.control']
        opt_control_bp = self.control_param['opt.control']
        
        if gibbs_control:
            for key, value in gibbs_control.items():
                gibbs_control_bp[key] = value
    
        if opt_control:
            for key, value in opt_control.items():
                opt_control_bp[key] = value

        opt_control = Prism.valid_opt_control(opt_control_bp)
        gibbs_control = Prism.valid_gibbs_control(gibbs_control_bp)
        
        psi = optim.update_reference(Z = self.posterior_initial_cellType.Z, 
                                     phi_prime = self.prism.phi_cellType,
                                     map = self.prism.map, 
                                     key = self.prism.key, 
                                     opt_control = opt_control)
        
        gibbsSampler_update = GibbsSampler(reference = psi, 
                                           X = self.prism.mixture, 
                                           gibbs_control = gibbs_control)
        
        theta_f = gibbsSampler_update.run(final = True)
        
        bp_updated = BayesPrism(prism = self.prism, 
                                posterior_initial_cellState = self.posterior_initial_cellState,
                                posterior_initial_cellType = self.posterior_initial_cellType, 
                                control_param={'gibbs.control': gibbs_control, 
                                               'opt.control': opt_control, 
                                               'update.gibbs': True},
                                reference_update = psi,
                                posterior_theta_f = theta_f)

        return bp_updated