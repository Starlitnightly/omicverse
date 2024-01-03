from ..SEACells import SEACells, summarize_by_SEACell,summarize_by_soft_SEACell
from ..SEACells import compute_celltype_purity,separation,compactness

import pandas as pd

class MetaCell(object):

    def __init__(self,adata,use_rep,
                 n_metacells=None,
                 use_gpu: bool = False,
                verbose: bool = True,
                n_waypoint_eigs: int = 10,
                n_neighbors: int = 15,
                convergence_epsilon: float = 1e-3,
                l2_penalty: float = 0,
                max_franke_wolfe_iters: int = 50,
                use_sparse: bool = False,) -> None:
        
        if n_metacells is None:
            n_metacells=adata.shape[0]//75


        self.model=SEACells(adata, 
                  build_kernel_on=use_rep, 
                  n_SEACells=n_metacells, 
                  use_gpu=use_gpu,
                  verbose=verbose,
                  n_waypoint_eigs=n_waypoint_eigs,
                    n_neighbors=n_neighbors,
                    convergence_epsilon=convergence_epsilon,
                    l2_penalty=l2_penalty,
                    max_franke_wolfe_iters=max_franke_wolfe_iters,
                    use_sparse=use_sparse)
        self.adata=adata

        self.model.construct_kernel_matrix()
        self.M = self.model.kernel_matrix
        self.metacells_ad=None

    def initialize_archetypes(self,**kwargs):
        self.model.initialize_archetypes(**kwargs)
    
    def train(self,min_iter=10, max_iter=50,**kwargs):
        self.model.fit(min_iter=min_iter, max_iter=max_iter,**kwargs)

    def predicted(self,method='soft',celltype_label='celltype',
                  summarize_layer='raw',minimum_weight=0.05):
        if method=='soft':
            ad=summarize_by_soft_SEACell(self.adata, self.model.A_, 
                                        celltype_label=celltype_label,
                                        summarize_layer=summarize_layer, 
                                        minimum_weight=minimum_weight
                                    )
        elif method=='hard':
            ad=summarize_by_SEACell(self.adata, 
                                    SEACells_label='SEACell', 
                                    summarize_layer=summarize_layer,
                                    celltype_label=celltype_label
                                )
        self.metacells_ad=ad
        return ad
    
    def save(self,model_path='seacells/model.pkl'):
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        return None

    def load(self,model_path='seacells/model.pkl'):
        import pickle
        with open(model_path, "wb") as f:
            self.model=pickle.load(f)

    def step(self,n_steps=5):
        # You can force the model to run additional iterations step-wise using the .step() function
        print(f'Ran for {len(self.model.RSS_iters)} iterations')
        for _ in range(n_steps):
            self.model.step()
        print(f'Ran for {len(self.model.RSS_iters)} iterations')

    def compute_celltype_purity(self,celltype_label='celltype',):
        if self.metacells_ad is None:
            raise ValueError('Please run .predicted() first')
        else:
            return compute_celltype_purity(self.metacells_ad,
                                           celltype_label)
    
    def separation(self,use_rep='X_pca',nth_nbr=1,**kwargs):
        if self.metacells_ad is None:
            raise ValueError('Please run .predicted() first')
        else:
            return separation(self.metacells_ad,
                                           use_rep,nth_nbr=nth_nbr,**kwargs)
        
    def compactness(self,use_rep='X_pca',**kwargs):
        if self.metacells_ad is None:
            raise ValueError('Please run .predicted() first')
        else:
            return compactness(self.metacells_ad,
                                           use_rep,**kwargs)


def plot_metacells(ax,metacells_ad,use_rep='X_umap',color='#1f77b4',
                   size=15,
                   edgecolors='b',linewidths=0.6,alpha=1,**kwargs,):
    umap = pd.DataFrame(metacells_ad.obsm[use_rep]).set_index(metacells_ad.obs_names).join(metacells_ad.obs["SEACell"])
    umap["SEACell"] = umap["SEACell"].astype("category")
    mcs = umap.groupby("SEACell").mean().reset_index()

    ax.scatter(mcs[0],mcs[1],s=size,c=color,
           edgecolors=edgecolors,linewidths=linewidths,
          alpha=alpha,**kwargs)
    return ax
