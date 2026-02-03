from .._settings import Colors, EMOJI
from .._monitor import monitor


class Velo:
    def __init__(self, adata):
        self.adata = adata
        print(f"{Colors.WARNING}In Velo module, you should keep all genes' expression not normalized.{Colors.ENDC}")

    def run(self):
        print(f"{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} Vela Analysis Initialization:{Colors.ENDC}")
        print(f"   {Colors.CYAN}Input data shape: {Colors.BOLD}{self.adata.shape[0]} cells × {self.adata.shape[1]} genes{Colors.ENDC}")
        print(f"   {Colors.BLUE}Total UMI counts: {Colors.BOLD}{self.adata.X.sum():,.0f}{Colors.ENDC}")
        print(f"   {Colors.BLUE}Mean genes per cell: {Colors.BOLD}{self.adata.X.mean():,.1f}{Colors.ENDC}")
        print(f"   {Colors.GREEN}Vela Analysis Completed:{Colors.ENDC}")

    def filter_genes(self,min_shared_counts=20,**kwargs):
        from scvelo.pp import filter_genes
        filter_genes(self.adata,min_shared_counts=min_shared_counts,**kwargs)

    def preprocess(self, recipe='monocle',
                   n_neighbors=30,
                   n_pcs=30,
                   **kwargs):
        import dynamo as dyn 
        preprocessor = dyn.pp.Preprocessor(cell_cycle_score_enable=True,**kwargs)
        preprocessor.preprocess_adata(self.adata, recipe=recipe)
        from ..pp import neighbors
        neighbors(self.adata,n_neighbors=n_neighbors,n_pcs=n_pcs,use_rep='X_pca')

    def moments(self,backend='dynamo',n_pcs=30,n_neighbors=30,**kwargs):
        if backend == 'dynamo':
            import dynamo as dyn 
            dyn.tl.moments(self.adata,n_pca_components=n_pcs,n_neighbors=n_neighbors,**kwargs)
            self.adata.layers['Ms'] = self.adata.layers['M_s']
            self.adata.layers['Mu'] = self.adata.layers['M_u']
        elif backend == 'scvelo':
            import scvelo as scv 
            scv.pp.moments(self.adata, n_pcs=n_pcs, n_neighbors=n_neighbors,**kwargs)
        else:
            raise ValueError(f"Backend {backend} not supported")
    
    def dynamics(self,backend='dynamo',**kwargs):
        if backend == 'dynamo':
            import dynamo as dyn 
            dyn.tl.dynamics(self.adata,**kwargs)
        elif backend == 'scvelo':
            import scvelo as scv 
            scv.tl.recover_dynamics(self.adata,**kwargs)
        else:
            raise ValueError(f"Backend {backend} not supported")
    
    
    def cal_velocity(
        self,
        method='dynamo',
        batch_key=None,
        celltype_key=None,
        velocity_key='velocity_S',
        n_jobs=1,
        n_top_genes=2000,
        param_name_key='tmp/latentvelo_params',
        latentvelo_VAE_kwargs={},
        **kwargs
    ):
        
        if method == 'dynamo':
            import dynamo as dyn 
            dyn.tl.cell_velocities(self.adata,**kwargs)
            self.adata.var[f'{velocity_key}_genes']=self.adata.var['use_for_transition']

        elif method == 'scvelo':
            import scvelo as scv 
            scv.tl.velocity(self.adata,**kwargs)
            self.adata.layers[velocity_key] = self.adata.layers['velocity']
            self.adata.var[f'{velocity_key}_genes']=self.adata.var['velocity_genes']

        elif method == 'latentvelo':
            self._latentvelo_cal(
                velocity_key=velocity_key,
                celltype_key=celltype_key,
                batch_key=batch_key,
                latentvelo_VAE_kwargs=latentvelo_VAE_kwargs,
                param_name_key=param_name_key,
                **kwargs
            )
        elif method == 'graphvelo':
            dynamo_flag = False
            try:
                import dynamo as dyn 
                dyn.tl.neighbors(self.adata)
                dyn.tl.cell_velocities(self.adata,**kwargs)
                #self.adata.var[f'{velocity_key}_genes']=self.adata.var['use_for_transition']
                self._graphvelo_cal(backend='dynamo',xkey='Ms',vkey='velocity_S',n_jobs=n_jobs,**kwargs)
                dynamo_flag = True
            except:
                print(f"{Colors.WARNING}dynamo run failed.{Colors.ENDC}")
            if dynamo_flag==False:
                try:
                    import scvelo as scv
                    scv.tl.velocity(self.adata,**kwargs)
                    #self.adata.layers[velocity_key] = self.adata.layers['velocity']
                    #self.adata.var[f'{velocity_key}_genes']=self.adata.var['velocity_genes']

                    self._graphvelo_cal(backend='scvelo',xkey='Ms',vkey='velocity',
                    n_jobs=n_jobs,**kwargs)
                except:
                    print(f"{Colors.WARNING}scvelo run failed.{Colors.ENDC}")
                    raise ValueError("scvelo also run failed.")
            
            


        else:
            raise ValueError(f"Method {method} not supported")

    def graphvelo(
        self,xkey='Ms',vkey='velocity_S',
        n_jobs=1,
        basis_keys=['X_umap','X_pca'],
        gene_subset=None,
        **kwargs
    ):
        from ..external.graphvelo.graph_velocity import GraphVelo
        from ..external.graphvelo.utils import adj_to_knn
        indices, _ = adj_to_knn(self.adata.obsp['connectivities'])
        self.adata.uns['neighbors']['indices'] = indices
        gv=GraphVelo(self.adata, xkey=xkey, vkey=vkey,gene_subset=gene_subset,**kwargs)
        gv.train(n_jobs=n_jobs)
        self.adata.layers['velocity_gv'] = gv.project_velocity(self.adata.layers[xkey])

        self.adata.var['velocity_gv_genes']=False
        self.adata.var['velocity_gv_genes']=self.adata.var.loc[gene_subset,'velocity_gv_genes']=True
        if issparse(self.adata.layers['velocity_gv']):
            self.adata.layers['velocity_gv'] = self.adata.layers['velocity_gv'].toarray()
        for basis_key in basis_keys:
            self.adata.obsm[f'gv_{basis_key}'] = gv.project_velocity(self.adata.obsm[basis_key])



    def velocity_graph(self,basis='umap',vkey='velocity_S',**kwargs):
        import scvelo as scv
        scv.tl.velocity_graph(self.adata, vkey=vkey, **kwargs)
    
    def velocity_embedding(self,basis='umap',vkey='velocity_S',**kwargs):   
        import scvelo as scv
        scv.tl.velocity_embedding(self.adata, basis=basis, vkey=vkey, **kwargs)
        #return self.adata


    def _graphvelo_cal(self,backend='dynamo',xkey='Ms',vkey='velocity_S',n_jobs=1,**kwargs):
        from ..external.graphvelo.graph_velocity import GraphVelo
        from ..external.graphvelo.utils import mack_score, adj_to_knn
        if backend == 'dynamo':
            gv=GraphVelo(self.adata, xkey=xkey, vkey=vkey,**kwargs)
            gv.train(n_jobs=n_jobs)
        elif backend == 'scvelo':
            indices, _ = adj_to_knn(self.adata.obsp['connectivities'])
            self.adata.uns['neighbors']['indices'] = indices
            gene_subset=self.adata.var.loc[self.adata.var['velocity_genes']].index.tolist()
            gv=GraphVelo(self.adata, xkey=xkey, vkey=vkey,gene_subset=gene_subset,**kwargs)
            gv.train(n_jobs=n_jobs)
        else:
            raise ValueError(f"Backend {backend} not supported")
        
        self.adata.layers[vkey] = gv.project_velocity(self.adata.layers['M_s'])
        self.adata.obsm['gv_pca'] = gv.project_velocity(self.adata.obsm['X_pca'])
        self.adata.obsm['gv_umap'] = gv.project_velocity(self.adata.obsm['X_umap'])


    def _latentvelo_cal(
        self,param_name_key='tmp/latentvelo_params',
        velocity_key='velocity_S',
        celltype_key=None,
        batch_key=None,
        latentvelo_VAE_kwargs={},
        use_rep=None,
        **kwargs):
        try:
            import torchdiffeq
        except:
            print(f"{Colors.WARNING}torchdiffeq not installed, please install it with 'pip install torchdiffeq'.{Colors.ENDC}")
            raise ValueError("torchdiffeq not installed")
        import os
        os.makedirs(param_name_key, exist_ok=True)
        # latentvelo
        from ..external.latentvelo.models.vae_model import VAE
        from ..external.latentvelo.models.annot_vae_model import AnnotVAE
        from ..external.latentvelo.train import train
        from ..external.latentvelo.utils import standard_clean_recipe, anvi_clean_recipe
        # Optional device override for latentvelo stack
        device_override = kwargs.pop('device', None)
        if device_override is not None:
            from ..external.latentvelo import trainer as lv_trainer
            from ..external.latentvelo import trainer_anvi as lv_trainer_anvi
            from ..external.latentvelo import trainer_atac as lv_trainer_atac
            from ..external.latentvelo import output_results as lv_out_mod
            from ..external.latentvelo import utils as lv_utils
            for m in (lv_trainer, lv_trainer_anvi, lv_trainer_atac, lv_out_mod, lv_utils):
                if hasattr(m, 'set_device'):
                    m.set_device(device_override)

        # Shared preprocessing
        if celltype_key == None:
            self.adata = standard_clean_recipe(self.adata, batch_key=batch_key, 
                        celltype_key=celltype_key, r2_adjust=True,use_rep=use_rep)

            self.model = VAE(**latentvelo_VAE_kwargs)
            epochs, vae, val_traj = train(self.model,self.adata,name=param_name_key,**kwargs)
        else:
            self.adata=anvi_clean_recipe(self.adata, celltype_key=celltype_key,
                        batch_key=batch_key,r2_adjust=True,use_rep=use_rep)
            # Get required parameters from adata
            observed = self.adata.n_vars
            celltypes = len(self.adata.obs[celltype_key].unique())
            self.model = AnnotVAE(observed=observed, celltypes=celltypes, **latentvelo_VAE_kwargs)
            epochs, vae, val_traj = train(self.model,self.adata,name=param_name_key,**kwargs)
        self.adata.uns['latentvelo_train_params'] = {
                    'epochs': epochs,
                    'vae': vae,
                    'val_traj': val_traj
                }
        from ..external.latentvelo.output_results import output_results as lv_output
        latent_data, adta = lv_output(self.model,self.adata,gene_velocity=True,)
        self.adata.var[f'{velocity_key}_genes'] = self.adata.var['velocity_genes']
        #covert to csr
        import scipy as sp
        if not issparse(adta.layers['velo_s']):
            self.adata.layers['velocity_S'] = sp.sparse.csr_matrix(adta.layers['velo_s'])
        else:
            self.adata.layers['velocity_S'] = adta.layers['velo_s']
        if not issparse(adta.layers['velo_u']):
            self.adata.layers['velocity_U'] = sp.sparse.csr_matrix(adta.layers['velo_u'])
        else:
            self.adata.layers['velocity_U'] = adta.layers['velo_u']
        self.adata.obsm['X_latentvelo'] = latent_data.X
        self.adata.obsm['X_latentvelo_velo_s'] = latent_data.layers['spliced_velocity']
        self.adata.obsm['X_latentvelo_velo_u'] = latent_data.layers['unspliced_velocity']

    



import warnings

import numpy as np
from scipy.sparse import issparse




# TODO: Addd docstrings
def quiver_autoscale(X_emb, V_emb):
    """TODO."""
    import matplotlib.pyplot as pl

    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = pl.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.clf()
    pl.close(fig)
    return Q.scale / scale_factor


def velocity_embedding(
    data,
    basis=None,
    vkey="velocity",
    scale=10,
    self_transitions=True,
    use_negative_cosines=True,
    direct_pca_projection=None,
    retain_scale=False,
    autoscale=True,
    all_comps=True,
    T=None,
    copy=False,
):
    r"""Projects the single cell velocities into any embedding.

    Given normalized difference of the embedding positions

    .. math::
        \tilde \delta_{ij} = \frac{x_j-x_i}{\left\lVert x_j-x_i \right\rVert},

    the projections are obtained as expected displacements with respect to the
    transition matrix :math:`\tilde \pi_{ij}` as

    .. math::
        \tilde \nu_i = E_{\tilde \pi_{i\cdot}} [\tilde \delta_{i \cdot}]
        = \sum_{j \neq i} \left( \tilde \pi_{ij} - \frac1n \right) \tilde
        \delta_{ij}.


    Parameters
    ----------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    basis: `str` (default: `'tsne'`)
        Which embedding to use.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    scale: `int` (default: 10)
        Scale parameter of gaussian kernel for transition matrix.
    self_transitions: `bool` (default: `True`)
        Whether to allow self transitions, based on the confidences of transitioning to
        neighboring cells.
    use_negative_cosines: `bool` (default: `True`)
        Whether to project cell-to-cell transitions with negative cosines into
        negative/opposite direction.
    direct_pca_projection: `bool` (default: `None`)
        Whether to directly project the velocities into PCA space,
        thus skipping the velocity graph.
    retain_scale: `bool` (default: `False`)
        Whether to retain scale from high dimensional space in embedding.
    autoscale: `bool` (default: `True`)
        Whether to scale the embedded velocities by a scalar multiplier,
        which simply ensures that the arrows in the embedding are properly scaled.
    all_comps: `bool` (default: `True`)
        Whether to compute the velocities on all embedding components.
    T: `csr_matrix` (default: `None`)
        Allows the user to directly pass a transition matrix.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to `adata`.

    Returns
    -------
    velocity_umap: `.obsm`
        coordinates of velocity projection on embedding (e.g., basis='umap')
    """
    from scvelo import logging as logg
    from scvelo import settings
    from scvelo.core import l2_norm
    from scvelo.tools.transition_matrix import transition_matrix
    adata = data.copy() if copy else data

    if basis is None:
        keys = [
            key for key in ["pca", "tsne", "umap"] if f"X_{key}" in adata.obsm.keys()
        ]
        if len(keys) > 0:
            basis = "pca" if direct_pca_projection else keys[-1]
        else:
            raise ValueError("No basis specified")

    if f"X_{basis}" not in adata.obsm_keys():
        raise ValueError("You need to compute the embedding first.")

    if direct_pca_projection and "pca" in basis:
        logg.warn(
            "Directly projecting velocities into PCA space is for exploratory analysis "
            "on principal components.\n"
            "         It does not reflect the actual velocity field from high "
            "dimensional gene expression space.\n"
            "         To visualize velocities, consider applying "
            "`direct_pca_projection=False`.\n"
        )

    logg.info("computing velocity embedding", r=True)

    if issparse(adata.layers[vkey]):
        V=adata.layers[vkey].toarray()
    else:
        V = adata.layers[vkey]
    vgenes = np.ones(adata.n_vars, dtype=bool)
    if f"{vkey}_genes" in adata.var.keys():
        vgenes &= np.array(adata.var[f"{vkey}_genes"], dtype=bool)
    vgenes &= ~np.isnan(V.sum(0))
    V = V[:, vgenes]

    if direct_pca_projection and "pca" in basis:
        PCs = adata.varm["PCs"] if all_comps else adata.varm["PCs"][:, :2]
        PCs = PCs[vgenes]

        X_emb = adata.obsm[f"X_{basis}"]
        V_emb = (V - V.mean(0)).dot(PCs)

    else:
        X_emb = (
            adata.obsm[f"X_{basis}"] if all_comps else adata.obsm[f"X_{basis}"][:, :2]
        )
        V_emb = np.zeros(X_emb.shape)

        T = (
            transition_matrix(
                adata,
                vkey=vkey,
                scale=scale,
                self_transitions=self_transitions,
                use_negative_cosines=use_negative_cosines,
            )
            if T is None
            else T
        )
        T.setdiag(0)
        T.eliminate_zeros()

        densify = adata.n_obs < 1e4
        TA = T.toarray() if densify else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(adata.n_obs):
                indices = T[i].indices
                dX = X_emb[indices] - X_emb[i, None]  # shape (n_neighbors, 2)
                if not retain_scale:
                    dX /= l2_norm(dX)[:, None]
                dX[np.isnan(dX)] = 0  # zero diff in a steady-state
                probs = TA[i, indices] if densify else T[i].data
                V_emb[i] = probs.dot(dX) - probs.mean() * dX.sum(0)

        if retain_scale:
            X = (
                adata.layers["Ms"]
                if "Ms" in adata.layers.keys()
                else adata.layers["spliced"]
            )
            delta = T.dot(X[:, vgenes]) - X[:, vgenes]
            if issparse(delta):
                delta = delta.toarray()
            cos_proj = (V * delta).sum(1) / l2_norm(delta)
            V_emb *= np.clip(cos_proj[:, None] * 10, 0, 1)

    if autoscale:
        V_emb /= 3 * quiver_autoscale(X_emb, V_emb)

    if f"{vkey}_params" in adata.uns.keys():
        adata.uns[f"{vkey}_params"]["embeddings"] = (
            []
            if "embeddings" not in adata.uns[f"{vkey}_params"]
            else list(adata.uns[f"{vkey}_params"]["embeddings"])
        )
        adata.uns[f"{vkey}_params"]["embeddings"].extend([basis])

    vkey += f"_{basis}"
    adata.obsm[vkey] = V_emb

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added\n" f"    '{vkey}', embedded velocity vectors (adata.obsm)")

    return adata if copy else None
