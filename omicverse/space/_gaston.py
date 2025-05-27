import scanpy as sc
import os
import torch
import numpy as np
from ..externel.gaston import neural_net,process_NN_output,dp_related,cluster_plotting
from scipy.sparse import issparse, csr_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from .._settings import add_reference

class GASTON(object):

    def __init__(self,adata) -> None:
        self.adata=adata

    def get_gaston_input(self,get_rgb=False, spot_umi_threshold=50):
        import squidpy as sq
        adata=self.adata
        adata.obsm['spatial']=adata.obsm['spatial'].astype(float)
        sc.pp.filter_cells(adata, min_counts=spot_umi_threshold)

        gene_labels=adata.var.index.to_numpy()

        counts_mat=adata.X
        coords_mat=np.array(adata.obsm['spatial'])

        if not get_rgb:
            return counts_mat, coords_mat, gene_labels

        library_id = list(adata.uns['spatial'].keys())[0] # adata2.uns['spatial'] should have only one key
        scale=adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
        img = sq.im.ImageContainer(adata.uns['spatial'][library_id]['images']['hires'],
                                scale=scale,
                                layer="img1")
        print('calculating RGB')
        sq.im.calculate_image_features(adata, img, features="summary", key_added="features")
        columns = ['summary_ch-0_mean', 'summary_ch-1_mean', 'summary_ch-2_mean']
        RGB_mean = adata.obsm["features"][columns]
        RGB_mean = RGB_mean / 255
        self.RGB_mean=RGB_mean
        
        # RGB_mean = RGB_mean[RGB_mean.index.isin(df_pos['barcode'])]

        adata=self.adata
        return counts_mat, coords_mat, gene_labels, RGB_mean.to_numpy()
    
    def get_top_pearson_residuals(self,num_dims=5,clip=0.01,n_top_genes=5000,
                                  use_RGB=False):
        adata=self.adata
        sc.experimental.pp.highly_variable_genes(
            adata, flavor="pearson_residuals", n_top_genes=n_top_genes
        )


        adata = adata[:, adata.var["highly_variable"]]
        adata.layers["raw"] = adata.X.copy()
        adata.layers["sqrt_norm"] = np.sqrt(
            sc.pp.normalize_total(adata, inplace=False)["X"]
        )

        theta=np.Inf
        sc.experimental.pp.normalize_pearson_residuals(adata, clip=clip, theta=theta)
        sc.pp.pca(adata, n_comps=num_dims)
        A=adata.obsm['X_pca']
        if use_RGB:
            A=np.hstack((A,self.RGB_mean)) # attach to RGB mean
        return A
    
    def load_rescale(self,A):
        S=self.adata.obsm['spatial']
        from ..externel.gaston.neural_net import load_rescale_input_data
        S_torch, A_torch = load_rescale_input_data(S,A)
        self.S_torch=S_torch
        self.A_torch=A_torch
        #return S_torch, A_torch
    
    def train(self,isodepth_arch=[20,20],expression_fn_arch=[20,20],
              num_epochs=10000,checkpoint=500,out_dir='result/test_outputs',
              optimizer="adam",num_restarts=30):
        ######################################
        # NEURAL NET PARAMETERS (USER CAN CHANGE)
        # architectures are encoded as list, eg [20,20] means two hidden layers of size 20 hidden neurons
        #isodepth_arch=[20,20] # architecture for isodepth neural network d(x,y) : R^2 -> R 
        #expression_fn_arch=[20,20] # architecture for 1-D expression function h(w) : R -> R^G

       # num_epochs = 10000 # number of epochs to train NN (NOTE: it is sometimes beneficial to train longer)
        #checkpoint = 500 # save model after number of epochs = multiple of checkpoint
        #out_dir='colorectal_tumor_tutorial_outputs' # folder to save model runs
        #optimizer = "adam"
        #num_restarts=30

        ######################################
        #mkdir out_dir
        os.makedirs(out_dir, exist_ok=True)

        seed_list=range(num_restarts)
        for seed in tqdm(seed_list):
            #print(f'training neural network for seed {seed}')
            out_dir_seed=f"{out_dir}/rep{seed}"
            os.makedirs(out_dir_seed, exist_ok=True)
            mod, loss_list = neural_net.train(self.S_torch, self.A_torch,
                                S_hidden_list=isodepth_arch, A_hidden_list=expression_fn_arch, 
                                epochs=num_epochs, checkpoint=checkpoint, 
                                save_dir=out_dir_seed, optim=optimizer, seed=seed, save_final=True)
            
        add_reference(self.adata,'GASTON','spatial depth estimation with GASTON')

    def get_best_model(self,out_dir='result/test_outputs',
                       max_domain_num=8,start_from=2):
        gaston_model, A, S= process_NN_output.process_files(out_dir) # MATCH PAPER FIGURES
        self.model=gaston_model
        self.A=A
        self.S=S
        from ..externel.gaston import model_selection
        model_selection.plot_ll_curve(gaston_model, A, S, max_domain_num=max_domain_num, start_from=start_from)
        return gaston_model, A, S
    
    def cal_iso_depth(self,num_domains=10):
        #num_domains=5 # CHANGE FOR YOUR APPLICATION: use number of layers from above!
        gaston_isodepth, gaston_labels=dp_related.get_isodepth_labels(self.model,
                                                                      self.A,
                                                                      self.S,
                                                                      num_domains)


        # DATASET-SPECIFIC: so domains are ordered with tumor being last
        gaston_isodepth= np.max(gaston_isodepth) -1 * gaston_isodepth
        gaston_labels=(num_domains-1)-gaston_labels
        self.gaston_isodepth=gaston_isodepth
        self.gaston_labels=gaston_labels
        return gaston_isodepth, gaston_labels
    
    def plot_isodepth(self,show_streamlines=True,
                      rotate_angle=-90,arrowsize=2,
                      figsize=(7,6),**kwargs):
        
        rotate = np.radians(rotate_angle) # rotate coordinates by -90

        cluster_plotting.plot_isodepth(self.gaston_isodepth, 
                                       self.S, 
                                       self.model, figsize=figsize, streamlines=show_streamlines, 
                                    rotate=rotate,arrowsize=arrowsize, 
                                    **kwargs) # since we did isodepth -> -1*isodepth above, we also need to do gradient -> -1*gradient

    def plot_clusters(self,domain_colors,figsize=(6,6),
                      s=20,lgd=False,show_boundary=True,
                      rotate_angle=-90,boundary_lw=5,**kwargs):
        rotate = np.radians(rotate_angle) # rotate coordinates by -90
        cluster_plotting.plot_clusters(self.gaston_labels, self.S, figsize=figsize, 
                               colors=domain_colors, s=s, lgd=lgd, 
                               show_boundary=show_boundary, 
                               gaston_isodepth=self.gaston_isodepth, 
                               boundary_lw=boundary_lw, rotate=rotate,
                               **kwargs)
        
    def plot_clusters_restrict(self,domain_colors,isodepth_min=4.5,isodepth_max=6.8,
                               rotate_angle=-90,s=20,lgd=False,figsize=(6,6), **kwargs):
        # This is the range we used for reproducing figure papers. We found these bounds manually.
        rotate = np.radians(rotate_angle) # rotate coordinates by -90
        cluster_plotting.plot_clusters_restrict(self.gaston_labels, self.S, self.gaston_isodepth, 
                                                isodepth_min=isodepth_min, isodepth_max=isodepth_max, figsize=figsize, 
                                                colors=domain_colors, s=s, lgd=lgd, rotate=rotate, **kwargs)
        
    def restrict_spot(self,isodepth_min=4.5,isodepth_max=6.8,
                      adjust_physical=True,scale_factor=100,
                      plotisodepth=True,show_streamlines=True,
                      rotate_angle=-90,arrowsize=1, figsize=(6,3), 
                      neg_gradient=True,
                      **kwargs):
        # Optional: adjust isodepth for physical distance
        rotate = np.radians(rotate_angle) # rotate coordinates by -90

        from ..externel.gaston.restrict_spots import restrict_spots
        counts_mat_restrict, coords_mat_restrict, gaston_isodepth_restrict, gaston_labels_restrict, S_restrict=restrict_spots(
                                                                    self.adata.X, 
                                                                    self.adata.obsm['spatial'], 
                                                                    self.S, self.gaston_isodepth, self.gaston_labels, 
                                                                    isodepth_min=isodepth_min, isodepth_max=isodepth_max, 
                                                                    adjust_physical=adjust_physical, scale_factor=scale_factor,
                                                                    plotisodepth=plotisodepth, show_streamlines=show_streamlines, 
                                                                    gaston_model=self.model, rotate=rotate, figsize=figsize, 
                                                                    arrowsize=arrowsize, 
                                                                    neg_gradient=neg_gradient,**kwargs) # since we reversed gradient direction earlier, we need to reverse it back
        self.counts_mat_restrict=counts_mat_restrict
        self.coords_mat_restrict=coords_mat_restrict
        self.gaston_isodepth_restrict=gaston_isodepth_restrict
        self.gaston_labels_restrict=gaston_labels_restrict
        self.S_restrict=S_restrict

        return counts_mat_restrict, coords_mat_restrict, gaston_isodepth_restrict, gaston_labels_restrict, S_restrict
    
    def filter_genes(self,umi_thresh = 1000,exclude_prefix=['Mt-', 'Rpl', 'Rps']):

        #umi_thresh = 1000 # only analyze genes with at least 1000 total UMIs
        self.umi_thresh=umi_thresh
        counts_mat=self.adata.X
        if issparse(counts_mat):
            counts_mat=counts_mat.toarray()
        from ..externel.gaston.filter_genes import filter_genes
        idx_kept, gene_labels_idx=filter_genes(counts_mat, self.adata.var.index.to_numpy(), 
                                       umi_threshold=umi_thresh, 
                                       exclude_prefix=exclude_prefix)
        self.idx_kept=idx_kept
        self.gene_labels_idx=gene_labels_idx

    def pw_linear_fit(self,cell_type_df=None,
                        ct_list=[],**kwargs):
        from ..externel.gaston.segmented_fit import pw_linear_fit
        pw_fit_dict=pw_linear_fit(self.counts_mat_restrict, 
                                  self.gaston_labels_restrict, 
                                  self.gaston_isodepth_restrict,
                                        cell_type_df, ct_list,  idx_kept=self.idx_kept, 
                                        umi_threshold=self.umi_thresh, isodepth_mult_factor=0.01,
                                        **kwargs)
        self.pw_fit_dict=pw_fit_dict
        return pw_fit_dict
    
    def bin_data(self,cell_type_df=None,
                 num_bins=15,q_discont=0.95,q_cont=0.8,**kwargs):
        # for plotting, 
        from ..externel.gaston.binning_and_plotting import bin_data
        from ..externel.gaston.spatial_gene_classification import get_discont_genes, get_cont_genes
        binning_output=bin_data(self.counts_mat_restrict, 
                                self.gaston_labels_restrict, 
                                self.gaston_isodepth_restrict, 
                                cell_type_df, self.adata.var.index.to_numpy(), 
                                idx_kept=self.idx_kept, num_bins=num_bins, umi_threshold=self.umi_thresh,
                                **kwargs)
        self.binning_output=binning_output
        self.discont_genes_layer=get_discont_genes(self.pw_fit_dict, binning_output,q=q_discont)
        self.cont_genes_layer=get_cont_genes(self.pw_fit_dict, binning_output,q=q_cont)  
        return binning_output
    
    def get_restricted_adata(self,offset=10**6,):
        adata=self.adata
        adata2=adata[:,self.adata.var_names[self.idx_kept]]
        #adata2.obsm['spatial']=self.coords_mat_restrict
        adata2=adata2[:,self.gene_labels_idx]
        adata2.uns['gaston']={}
        adata2.uns['gaston']['isodepth']=self.gaston_isodepth_restrict
        adata2.uns['gaston']['labels']=self.gaston_labels_restrict

        slope_mat, intercept_mat, _, _ = self.pw_fit_dict['all_cell_types']

        gene_list = list(self.binning_output['gene_labels_idx']) # 获取基因列表
        adata2=adata2[:,gene_list]
        all_gene_outputs = []

        for gene_name in tqdm(gene_list):
            if gene_name in self.binning_output['gene_labels_idx']:
                gene_index = np.where(self.gene_labels_idx == gene_name)[0]

                outputs = np.zeros(self.gaston_isodepth_restrict.shape[0])
                for i in range(self.gaston_isodepth_restrict.shape[0]):
                    dom = int(self.gaston_labels_restrict[i])
                    slope = slope_mat[gene_index, dom]
                    intercept = intercept_mat[gene_index, dom]
                    outputs[i] = np.log(offset) + intercept + slope * self.gaston_isodepth_restrict[i]

                all_gene_outputs.append(outputs)

        sparse_output_matrix = csr_matrix(all_gene_outputs)
        adata2.layers['GASTON_ReX']=sparse_output_matrix.T
        return adata2
    
    def plot_gene_pwlinear(self,gene,domain_colors,offset=10**6,
                           cell_type_list=None,pt_size=50,linear_fit=True,
                           ticksize=15, figsize=(4,2.5),
                           lw=3,domain_boundary_plotting=True):
        gene_name=gene
        print(f'gene {gene_name}: discontinuous jump after domain(s) {self.discont_genes_layer[gene_name]}') 
        print(f'gene {gene_name}: continuous gradient in domain(s) {self.cont_genes_layer[gene_name]}')

        # display log CPM (if you want to do CP500, set offset=500)
        #offset=10**6
        from ..externel.gaston.binning_and_plotting import plot_gene_pwlinear
        plot_gene_pwlinear(gene_name, self.pw_fit_dict, 
                           self.gaston_labels_restrict, 
                           self.gaston_isodepth_restrict, 
                           self.binning_output, cell_type_list=cell_type_list, pt_size=pt_size, colors=domain_colors, 
                                                linear_fit=linear_fit, ticksize=ticksize, figsize=figsize, offset=offset, lw=lw,
                                            domain_boundary_plotting=domain_boundary_plotting)
        
    def plot_gene_raw(self,gene_name,rotate_angle=-90,
                      vmin=5,figsize=(6,3),s=10,**kwargs):
        rotate=np.radians(rotate_angle)
        from ..externel.gaston.binning_and_plotting import plot_gene_raw
        if issparse(self.counts_mat_restrict):
            counts_mat_restrict=self.counts_mat_restrict.toarray()
        else:
            counts_mat_restrict=self.counts_mat_restrict
        plot_gene_raw(gene_name, self.gene_labels_idx, counts_mat_restrict[:,self.idx_kept], 
                      self.S_restrict, vmin=vmin, figsize=figsize,s=s,rotate=rotate,**kwargs)
        plt.title(f'{gene_name} Raw Expression')

    def plot_gene_gastonrex(self,gene_name,rotate_angle=-90,
                            figsize=(6,3),s=10,**kwargs):
        rotate=np.radians(rotate_angle)
        from ..externel.gaston.binning_and_plotting import plot_gene_function

        plot_gene_function(gene_name, self.S_restrict, self.pw_fit_dict, 
                           self.gaston_labels_restrict, self.gaston_isodepth_restrict, 
                                        self.binning_output, figsize=figsize, s=s, rotate=rotate, **kwargs)
        plt.title(f'{gene_name} GASTON ReX')
