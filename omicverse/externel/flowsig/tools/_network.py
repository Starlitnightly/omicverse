from typing import List, Tuple, Optional
import networkx as nx
from scipy.sparse import issparse
import numpy as np
import random as rm
#from causaldag import unknown_target_igsp, gsp
#from causaldag import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester
#from causaldag import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester
#from graphical_models import DAG
from sklearn.utils import safe_mask
from timeit import default_timer as timer
from functools import reduce
from joblib import Parallel, delayed
import anndata as ad
import warnings
warnings.filterwarnings('ignore')

# Define the sampling step functions where we input the initial list of permutations
def run_gsp(adata: ad.AnnData,
            flowsig_expr_key: str,
            flow_vars: List[str],
            use_spatial: bool = False,
            block_key: str = None,
            alpha: float = 1e-3,
            seed: int = 0):
    from causaldag import gsp
    from causaldag import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester
    # Reseed the random number generator
    np.random.seed(seed) # Set the seed for reproducibility reasons

    samples = adata.obsm[flowsig_expr_key].copy()

    # Get the number of samples for each dataframe  
    num_samples = samples.shape[0]

    # Subsample WITHIN blocks replacement
    resampled = samples.copy()

    # If we want to do block bootstrapping (for spatial data),
    # we divide the data by spatially-seprated clusters and then resample within
    # the clusters
    if use_spatial:

        # Define the blocks for spatial block bootstrapping
        #block_clusters = sorted(adata.obs[block_key].unique().tolist())
        adata.obs[block_key]=adata.obs[block_key].astype('category')
        block_clusters = adata.obs[block_key].cat.categories.tolist()
        

        for block in block_clusters:
            block_indices = np.where(adata.obs[block_key] == block)[0] # Sample only those cells within the block
            
            block_subsamples = np.random.choice(block_indices, len(block_indices))
            resampled[block_indices, :] = samples[safe_mask(samples, block_subsamples), :]

    else:
        # Subsample with replacement
        subsamples = np.random.choice(num_samples, num_samples)

        resampled = samples[safe_mask(samples, subsamples), :]

    # We need to subset the gene expression matrices for ligands with non-zero standard deviation in BOTH cases
    resampled_std = resampled.std(0)

    nonzero_flow_vars_indices = resampled_std.nonzero()[0]

    # Subset based on the ligands with zero std in both cases
    considered_flow_vars = list([flow_vars[ind] for ind in nonzero_flow_vars_indices])
    
    nodes = set(considered_flow_vars)

    resampled = resampled[:, nonzero_flow_vars_indices]

    ### RunGSP using partial correlation  

    # Form sufficient statistics using partial correlation (assumes linear Gaussian model)
    obs_suffstat = partial_correlation_suffstat(resampled, invert=True)

    # Create conditional independence tester and invariance tester
    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha)

    ## Run UT-IGSP by considering all possible initial permutations
    est_dag = gsp(nodes,
                    ci_tester,
                    nruns=20)
    
    # Convert to CPDAG, which contains directed arcs and undirected edgse
    est_cpdag = est_dag.cpdag()
    adjacency_cpdag = est_cpdag.to_amat()[0]

    return {'nonzero_flow_vars_indices':nonzero_flow_vars_indices,
            'adjacency_cpdag':adjacency_cpdag}

def run_utigsp(adata: ad.AnnData,
                condition_key: str,
                control_key: str,
                flowsig_expr_key: str,
                flow_vars: List[str],
                use_spatial: bool = False,
                block_key: str = None,
                alpha: float = 1e-3,
                alpha_inv: float = 1e-3,
                seed: int = 0):

    # Reseed the random number generator
    from causaldag import unknown_target_igsp
    from causaldag import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester
    from causaldag import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester
    np.random.seed(seed) # Set the seed for reproducibility reasons

    adata_control = adata[adata.obs[condition_key] == control_key]
    control_samples = adata_control.obsm[flowsig_expr_key] # Define the control data
    
    adata_perturbed = adata[adata.obs[condition_key] != control_key]
    perturbed_keys = [cond for cond in adata.obs[condition_key].unique() if cond != control_key]
    perturbed_samples = [adata_perturbed[adata_perturbed.obs[condition_key] == cond].obsm[flowsig_expr_key] for cond in perturbed_keys] # Get the perturbed data
    perturbed_resampled = []

    if use_spatial:

        control_resampled = control_samples.copy()
        
        # Define the blocks for bootstrapping
        adata_control.obs[block_key]=adata_control.obs[block_key].astype('category')
        block_clusters_control = adata_control.obs[block_key].cat.categories.tolist()

        for block in block_clusters_control:

            block_indices = np.where(adata_control.obs[block_key] == block)[0] # Sample only those cells within the block
            
            block_subsamples = np.random.choice(block_indices, len(block_indices))
            control_resampled[block_indices, :] = control_samples[safe_mask(control_samples, block_subsamples), :]

        for i, pert in enumerate(perturbed_keys):

            pert_resampled = perturbed_samples[i].copy()

            adata_pert = adata_perturbed[adata_perturbed.obs[condition_key] == pert]

            adata_pert.obs[block_key]=adata_pert.obs[block_key].astype('category')
            block_clusters_pert = adata_pert.obs[block_key].cat.categories.tolist()

            #block_clusters_pert = sorted(adata_pert.obs[block_key].unique().tolist())

            for block in block_clusters_pert:

                block_indices = np.where(adata_pert.obs[block_key] == block)[0] # Sample only those cells within the block
                
                block_subsamples = np.random.choice(block_indices, len(block_indices))
                pert_resampled[block_indices, :] = pert_resampled[safe_mask(pert_resampled, block_subsamples), :]

            perturbed_resampled.append(pert_resampled)

    else:

        # Just sub-sample across all cells per condition
        num_samples_control = control_samples.shape[0]
        num_samples_perturbed = [sample.shape[0] for sample in perturbed_samples]

        # Subsample with replacement
        subsamples_control = np.random.choice(num_samples_control, num_samples_control)
        subsamples_perturbed = [np.random.choice(num_samples, num_samples) for num_samples in num_samples_perturbed]
        
        control_resampled = control_samples[safe_mask(control_samples, subsamples_control), :]

        for i in range(len(perturbed_samples)):
            num_subsamples = subsamples_perturbed[i]
            perturbed_sample = perturbed_samples[i]

            resampled = perturbed_sample[safe_mask(num_subsamples, num_subsamples), :]
            perturbed_resampled.append(resampled)

    # We need to subset the gene expression matrices for ligands with non-zero standard deviation in BOTH cases
    control_resampled_std = control_resampled.std(0)
    perturbed_resampled_std = [sample.std(0) for sample in perturbed_resampled]

    nonzero_flow_vars_indices_control = control_resampled_std.nonzero()[0]
    nonzero_flow_vars_indices_perturbed = [resampled_std.nonzero()[0] for resampled_std in perturbed_resampled_std]

    nonzero_flow_vars_indices = reduce(np.intersect1d, (nonzero_flow_vars_indices_control, *nonzero_flow_vars_indices_perturbed))

    # Subset based on the ligands with zero std in both cases
    considered_flow_vars = list([flow_vars[ind] for ind in nonzero_flow_vars_indices])
    
    nodes = set(considered_flow_vars)

    control_resampled = control_resampled[:, nonzero_flow_vars_indices]

    for i, resampled in enumerate(perturbed_resampled):

        perturbed_resampled[i] = resampled[:, nonzero_flow_vars_indices]

    ### Run UT-IGSP using partial correlation  

    # Form sufficient statistics using partial correlation (assumes linear Gaussian model)
    obs_suffstat = partial_correlation_suffstat(control_resampled, invert=True)
    invariance_suffstat = gauss_invariance_suffstat(control_resampled, perturbed_resampled)

    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)

    # Assume unknown interventions for UT-IGSP
    setting_list = [dict(known_interventions=[]) for _ in perturbed_resampled]

    # Run UT-IGSP by considering all possible initial permutations
    est_dag, est_targets_list = unknown_target_igsp(setting_list,
                                                        nodes,
                                                        ci_tester,
                                                        invariance_tester,
                                                        nruns=20)

    est_icpdag = est_dag.interventional_cpdag(est_targets_list, cpdag=est_dag.cpdag())

    adjacency_cpdag = est_icpdag.to_amat()[0]
    
    perturbed_targets_list = []
    
    for i in range(len(est_targets_list)):
        targets_list = list(est_targets_list[i])
        targets_ligand_indices = [flow_vars.index(considered_flow_vars[target]) for target in targets_list]
        perturbed_targets_list.append(targets_ligand_indices)

    return {'nonzero_flow_vars_indices':nonzero_flow_vars_indices,
            'adjacency_cpdag':adjacency_cpdag,
            'perturbed_targets_indices':perturbed_targets_list}

def learn_intercellular_flows(adata: ad.AnnData,
                        condition_key: str = None,
                        control_key: str = None, 
                        flowsig_key: str = 'flowsig_network',
                        flow_expr_key: str = 'X_flow',
                        use_spatial: Optional[bool] = False,
                        block_key: Optional[bool] = None,
                        n_jobs: int = 1,
                        n_bootstraps: int = 100,
                        alpha_ci: float = 1e-3,
                        alpha_inv: float = 1e-3):
    """
    Learn the causal signaling network from cell-type-ligand expression constructed
    from scRNA-seq and a base network derived from cell-cell communication inference.

    This method splits the cell-type-ligand expression into control and perturbed
    samples (one sample for each perturbed condition). We then use UT-IGSP [Squires2020]
    and partial correlation testing to learn the causal signaling DAG and the list of 
    perturbed (intervention) targets.
    
    The base network is also used as a list of initial node permutations for DAG learning.
    To overcome the DAG assumption, as cell-cell communication networks are not necessarily
    DAGS, we use bootstrap aggregation to cover as many possible causal edges and the list
    of node permutations is constructed from all possible DAG subsets of the base network.
    Each boostrap sample is generated by sampling with replacement.

    Parameters
    ----------
    adata
        The annotated dataframe (typically from Scanpy) of the single-cell data.
        Must contain constructed flow expression matrices and knowledge of
        possible cellular flow variables.

    condition_key 
        The label in adata.obs which we use to partition the data.

    control_key
        The category in adata.obs[condition_key] that specifies which cells belong 
        to the control condition, which is known in causal inference as the observational 
        data.

    flowsig_key
        The label for which output will be stored in adata.uns

    flow_expr_key
        The label for which the augmente dflow expression expression is stored in adata.obsm

    use_spatial
        Boolean for whether or not we are analysing spatial data, and thus need to use
        block bootstrapping rather than normal bootstrapping, where we resample across all
        cells.

    block_key
        The label that specfies from which observation key we use to construct (hopefully)
        spatially correlated blocks used for block bootstrapping to learn spatially resolved
        cellular flows. These blocks can be simply just dividing the tissue into rougly
        equally spaced tissue regions, or can be based on tissue annotation (e.g. organ, cell type).
    
    n_jobs
        Number of CPU cores that are used during bootstrap aggregation. If n_jobs > 1, jobs
        are submitted in parallel using multiprocessing

    n_boostraps
        Number of bootstrap samples to generate for causal DAG learning.

    alpha_ci
        The significance level used to test for conditional independence
    
    alpha_inv
        The significance level used to test for conditional invariance.

    Returns
    -------
    flow_vars
        The list of cell-type-ligand pairs used during causal structure learning,
        stored in adata.uns[flowsig_key]['flow_vars'].

    adjacency
        The weighted adjacency matrix encoding a bagged CPDAG,
        where weights are determined from bootstrap aggregation. Stored in 
        adata.uns[flowsig_key]['adjacency']

    perturbed_targets
        The list of inferred perturbed targets, as determined by conditional invariance
        testing and their bootstrapped probability of perturbations. Stored in
        adata.uns[flowsig_key]['perturbed_targets']

    References
    ----------

    .. [Squires2020] Squires, C., Wang, Y., & Uhler, C. (2020, August). Permutation-based
     causal structure learning with unknown intervention targets. In Conference on
     Uncertainty in Artificial Intelligence (pp. 1039-1048). PMLR.

    """

    # Extract the control and perturbed samples

    # Initialise the results
    flowsig_network_results = {}

    if condition_key is not None: # If there is more than one condition, then we use UT-IGSP with a control vs perturbed condition

        conditions = adata.obs[condition_key].unique().tolist()        
        perturbed_keys = [cond for cond in conditions if cond != control_key]

        flow_vars = list(adata.uns[flowsig_key]['flow_var_info'].index)

        # Randomly shuffle to edges to generate initial permutations for initial DAGs
        bagged_adjacency = np.zeros((len(flow_vars), len(flow_vars)))
        bagged_perturbed_targets = [np.zeros(len(flow_vars)) for key in perturbed_keys]

        start = timer()

        print(f'starting computations on {n_jobs} cores')


        args = [(adata,
                condition_key,
                control_key,
                flow_expr_key,
                flow_vars,
                use_spatial,
                block_key,
                alpha_ci,
                alpha_inv,
                boot) for boot in range(n_bootstraps)]
                                
        bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(run_utigsp)(*arg) for arg in args)

        end = timer()

        print(f'elapsed time: {end - start}')

        # Sum the results for UT-IGSP with initial permutations
        for res in bootstrap_results:

            nz_indices = res['nonzero_flow_vars_indices']
            adjacency = res['adjacency_cpdag']
            pert_indices = res['perturbed_targets_indices']
            
            # Update the bagged adjacency
            bagged_adjacency[np.ix_(nz_indices, nz_indices)] += adjacency

            # Update the intervention targets
            for i in range(len(pert_indices)):

                nonzero_pert_indices = pert_indices[i]
                perturbed_targets = bagged_perturbed_targets[i]
                perturbed_targets[nonzero_pert_indices] += 1
                bagged_perturbed_targets[i] = perturbed_targets

        # Average the adjacencies
        bagged_adjacency /= float(n_bootstraps)

        # Average the intervened targets
        for i in range(len(pert_indices)):

            perturbed_targets = bagged_perturbed_targets[i]
            perturbed_targets /= float(n_bootstraps) # Average the results
            bagged_perturbed_targets[i] = perturbed_targets

        flowsig_network_results =  {'flow_vars': flow_vars,
                'adjacency': bagged_adjacency,
                'perturbed_targets': bagged_perturbed_targets}

    else: # Else we have no perturbation and we will use GSP

        flow_vars = list(adata.uns[flowsig_key]['flow_var_info'].index)

        # Randomly shuffle to edges to generate initial permutations for initial DAGs
        bagged_adjacency = np.zeros((len(flow_vars), len(flow_vars)))

        start = timer()

        print(f'starting computations on {n_jobs} cores')

        args = [(adata, 
                flow_expr_key,
                flow_vars,
                use_spatial,
                block_key,
                alpha_ci,
                boot) for boot in range(n_bootstraps)]
                            
        bootstrap_results = Parallel(n_jobs=n_jobs)(delayed(run_gsp)(*arg) for arg in args)

        end = timer()

        print(f'elapsed time: {end - start}')

        # Sum the results for UT-IGSP with initial permutations
        for res in bootstrap_results:

            nz_indices = res['nonzero_flow_vars_indices']
            adjacency = res['adjacency_cpdag']

            # Update the bagged adjacency
            bagged_adjacency[np.ix_(nz_indices, nz_indices)] += adjacency

        # Average the adjacencies
        bagged_adjacency /= float(n_bootstraps)

        flowsig_network_results = {'flow_vars': flow_vars,
                                    'adjacency': bagged_adjacency}

    # Store the results
    adata.uns[flowsig_key]['network'] = flowsig_network_results
