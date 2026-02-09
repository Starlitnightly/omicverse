from .._settings import add_reference, EMOJI, Colors
import numpy as np
import scanpy as sc
import pandas as pd

from .._registry import register_function

@register_function(
    aliases=["空间解卷积", "deconvolution", "spatial_deconvolution", "空间解卷积分析", "空间映射"],
    category="space",
    description="Spatial deconvolution for mapping scRNA-seq cell types to spatial locations",
    prerequisites={
        'functions': []  # Requires scRNA-seq with annotations and spatial data
    },
    requires={
        'obs': ['cell_type'],  # Requires clusters column in scRNA-seq (user-specified)
    },
    produces={
        'obsm': []  # Cell type predictions in spatial data
    },
    auto_fix='escalate',
    examples=[
        "# Basic Deconvolution analysis",
        "deconvolution = ov.space.Deconvolution(adata_sp=spatial_adata, adata_sc=sc_adata)",
        "decov_obj.deconvolution(method='Tangram', celltype_key_sc='cell_type')",
        "decov_obj.deconvolution(method='cell2location', celltype_key_sc='cell_type')",
        "decov_obj.deconvolution(method='FlashDeconv', celltype_key_sc='cell_type')",
        "decov_obj.deconvolution(method='starfysh', celltype_key_sc='cell_type')",
    ],
)
class Deconvolution(object):
    def __init__(self, adata_sp,adata_sc=None, ):
        r"""Initialize Deconvolution object.
        Parameters
        ----------
        adata_sp : AnnData
            Spatial transcriptomics data.
        adata_sc : AnnData, optional
            Single-cell RNA-seq data, by default None.
        """
        self.adata_sc = adata_sc
        self.adata_sp = adata_sp
        self.mod_sc=None
        self.mod_sp=None
        self.adata_cell2location=None
        self.adata_impute=None
        self.tangram_model=None
        self.flashdeconv_params=None
        self.method=None
        if adata_sc is not None:
            self._check_layer()


    def _check_layer(self):

        #Check the raw counts layer
        if 'counts' in self.adata_sc.layers.keys():
            print(f"{Colors.GREEN}✓ Existing 'counts' layer in scRNA-seq data{Colors.ENDC}")
        if 'counts' in self.adata_sp.layers.keys():
            print(f"{Colors.GREEN}✓ Existing 'counts' layer in spatial transcriptomics data{Colors.ENDC}")

        # Check the normalized expression layer 
        log1p_1e4_warning = False
        if self.adata_sc.X.max()<np.log1p(1e4):
            print(f"{Colors.GREEN}✓ scRNA-seq data is log-normalized by 1e4{Colors.ENDC}")
            log1p_1e4_warning = True
        if self.adata_sp.X.max()<np.log1p(1e4):
            print(f"{Colors.GREEN}✓ spatial transcriptomics data is log-normalized by 1e4{Colors.ENDC}")
            log1p_1e4_warning = True
        
        if log1p_1e4_warning:   
            print(f"{Colors.WARNING}⚠️ 1e4 is the standardized target sum for `scanpy`{Colors.ENDC}")

        log1p_50_warning = False
        if self.adata_sc.X.max()<np.log1p(50*1e4):
            print(f"{Colors.GREEN}✓ scRNA-seq data is log-normalized by 50*1e4{Colors.ENDC}")
            log1p_50_warning = True
        if self.adata_sp.X.max()<np.log1p(50*1e4):
            print(f"{Colors.GREEN}✓ spatial transcriptomics data is log-normalized by 50*1e4{Colors.ENDC}")
            log1p_50_warning = True

        if log1p_50_warning:   
            print(f"{Colors.WARNING}⚠️ 50*1e4 is the standardized target sum for `omicverse`{Colors.ENDC}")

    def preprocess_sc(self,mode='shiftlog|pearson',n_HVGs=3000,target_sum=1e4,**kwargs):
        r"""Preprocess the scRNA-seq data.
        Parameters
        ----------
        mode : str, optional
            Preprocessing mode, by default 'shiftlog|pearson'.
        n_HVGs : int, optional
            Number of highly variable genes, by default 3000.
        target_sum : int, optional
            Target sum for preprocessing, by default 1e4.
        **kwargs : dict, optional
            Additional keyword arguments for preprocessing.
        """
        from ..pp import preprocess
        self.adata_sc=preprocess(self.adata_sc,mode=mode,n_HVGs=n_HVGs,target_sum=target_sum,**kwargs)
        self.adata_sc.raw = self.adata_sc
        self.adata_sc = self.adata_sc[:, self.adata_sc.var.highly_variable_features]
        print(f"{Colors.GREEN}✓ scRNA-seq data is preprocessed{Colors.ENDC}")
        #return self.adata_sc

    def preprocess_sp(self,
        mode='pearsonr',n_svgs=3000,target_sum=50*1e4,
        platform="visium",mt_startwith='MT-',
        subset_genes=True,
        **kwargs):
        r"""Preprocess the spatial transcriptomics data.
        Parameters
        ----------
        mode : str, optional
            Preprocessing mode, by default 'pearsonr'.
        n_svgs : int, optional
            Number of space variable genes, by default 3000.
        target_sum : int, optional
            Target sum for preprocessing, by default 50*1e4.    
        platform : str, optional
            Platform of the spatial transcriptomics data, by default 'visium'.
        mt_startwith : str, optional
            Start with of the mitochondrial genes, by default 'MT-'.
        **kwargs : dict, optional
            Additional keyword arguments for preprocessing.
        """

        from ._svg import svg
        self.adata_sp_raw=self.adata_sp.copy()
        self.adata_sp=svg(self.adata_sp,mode=mode,n_svgs=n_svgs,
                    target_sum=target_sum,platform=platform,
                    mt_startwith=mt_startwith,**kwargs)
        both_genes=np.intersect1d(self.adata_sp_raw.var_names, self.adata_sp.var_names)
        self.adata_sp_raw=self.adata_sp_raw[:, both_genes]
        self.adata_sp=self.adata_sp[:, both_genes]
        self.adata_sp_raw.var['space_variable_features']=self.adata_sp.var['space_variable_features']
        self.adata_sp_raw.var['highly_variable']=self.adata_sp.var['highly_variable']
        self.adata_sp.raw = self.adata_sp
        self.adata_sp.var['highly_variable']=self.adata_sp.var['space_variable_features']
        if subset_genes:
            self.adata_sp = self.adata_sp[:, self.adata_sp.var.space_variable_features]
        else:
            self.adata_sp = self.adata_sp
        print(f"{Colors.GREEN}✓ spatial transcriptomics data is preprocessed{Colors.ENDC}")
        #return self.adata_sp

    def deconvolution(
        self,
        method='Tangram',
        celltype_key_sc='cell_type',
        batch_key_sc=None,
        batch_key_sp=None,
        tangram_kwargs=None,
        cell2location_scrna_kwargs=None,
        cell2location_spatial_kwargs=None,
        N_cells_per_location=30,
        detection_alpha=200,
        sample_kwargs=None,
        # FlashDeconv parameters
        flashdeconv_kwargs=None,
        starfysh_kwargs=None,
        spatial_type='visium',
        gene_sig=None,
    ):
        r"""Deconvolution the spatial transcriptomics data.
        Parameters
        ----------
        method : str, optional
            Deconvolution method, by default 'Tangram'.
        celltype_key_sc : str, optional
            Cell type key in scRNA-seq data, by default 'cell_type'.
        batch_key_sc : str, optional
            Batch key in scRNA-seq data, by default None.
        batch_key_sp : str, optional
            Batch key in spatial transcriptomics data, by default None.
        tangram_kwargs : dict, optional
            Keyword arguments for Tangram, by default None.
        cell2location_scrna_kwargs : dict, optional
            Keyword arguments for cell2location, by default None.
        cell2location_spatial_kwargs : dict, optional
            Keyword arguments for cell2location, by default None.
        N_cells_per_location : int, optional
            Number of cells per location, by default 30.
        detection_alpha : int, optional
            Detection alpha, by default 200.
        sample_kwargs : dict, optional
            Keyword arguments for sampling, by default None.
        flashdeconv_kwargs : dict, optional
            Keyword arguments for FlashDeconv, by default None.
        starfysh_kwargs : dict, optional
            Keyword arguments for starfysh, by default None.
        spatial_type : str, optional
            Spatial type, by default 'visium'.
        gene_sig : dict, optional
            Gene signature, by default None.
        """
        if method=='Tangram':
            self.method='Tangram'
            from ._tangram import Tangram
            if tangram_kwargs is None:
                tangram_kwargs={'mode':'clusters','num_epochs':500,'device':'cuda:0'}
            tangram=Tangram(self.adata_sc,self.adata_sp,clusters=celltype_key_sc)
            tangram.train(**tangram_kwargs)
            self.adata_cell2location=tangram.cell2location()
            print(f"{Colors.GREEN}✓ Tangram cell2location is done{Colors.ENDC}")
            print(f"The cell2location result is saved in self.adata_cell2location")
            #self.adata_impute=tangram.impute()
            self.tangram_model=tangram
            #print(f"{Colors.GREEN}✓ Tangram impute is done{Colors.ENDC}")
            #print(f"The impute result is saved in self.adata_impute")
            return tangram
        elif method=='cell2location':
            self.method='cell2location'
            from ..external.space.cell2location.models import Cell2location, RegressionModel
            from ..external.space.cell2location.plt import plot_spatial
            from ..external.space.cell2location.utils import select_slide
            from ..external.space.cell2location.utils.filtering import filter_genes

            selected = filter_genes(
                self.adata_sc, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12
            )

            # filter the object
            self.adata_sc = self.adata_sc[:, selected].copy()

            # prepare anndata for the regression model
            RegressionModel.setup_anndata(
                adata=self.adata_sc,
                # 10X reaction / sample / batch
                batch_key=batch_key_sc,
                # cell type, covariate used for constructing signatures
                labels_key=celltype_key_sc,
                # multiplicative technical effects (platform, 3' vs 5', donor effect)
                categorical_covariate_keys=["Method"],
            )
            self.mod_sc = RegressionModel(self.adata_sc)

            # Use all data for training (validation not implemented yet, train_size=1)
            if cell2location_scrna_kwargs is None:
                cell2location_scrna_kwargs={'max_epochs':250,'batch_size':2500,'train_size':1,'lr':0.002}
            self.mod_sc.train(
                **cell2location_scrna_kwargs
            )

            if sample_kwargs is None:
                sample_kwargs={"num_samples": 1000, "batch_size": 2500}

            self.adata_sc = self.mod_sc.export_posterior(
                self.adata_sc,
                sample_kwargs=sample_kwargs,
            )


            # export estimated expression in each cluster
            if "means_per_cluster_mu_fg" in self.adata_sc.varm.keys():
                inf_aver = self.adata_sc.varm["means_per_cluster_mu_fg"][
                    [f"means_per_cluster_mu_fg_{i}" for i in self.adata_sc.uns["mod"]["factor_names"]]
                ].copy()
            else:
                inf_aver = self.adata_sc.var[
                    [f"means_per_cluster_mu_fg_{i}" for i in self.adata_sc.uns["mod"]["factor_names"]]
                ].copy()
            inf_aver.columns = self.adata_sc.uns["mod"]["factor_names"]

            self.inf_aver=inf_aver

            intersect = np.intersect1d(self.adata_sp.var_names, inf_aver.index)
            self.adata_sp = self.adata_sp[:, intersect].copy()
            self.inf_aver = self.inf_aver.loc[intersect, :].copy()

            print(f"Total number of genes both in the scRNA-seq data and the spatial transcriptomics data: {len(intersect)}")

            Cell2location.setup_anndata(adata=self.adata_sp, batch_key=batch_key_sp)


            self.mod_sp = Cell2location(
                self.adata_sp,
                cell_state_df=self.inf_aver,
                # the expected average cell abundance: tissue-dependent
                # hyper-prior which can be estimated from paired histology:
                N_cells_per_location=N_cells_per_location,
                # hyperparameter controlling normalisation of
                # within-experiment variation in RNA detection (using default here):
                detection_alpha=detection_alpha,
            )

            if cell2location_spatial_kwargs is None:
                cell2location_spatial_kwargs={'max_epochs':30000,'batch_size':None,'train_size':1}

            self.mod_sp.train(
                **cell2location_spatial_kwargs
            )

            self.adata_sp = self.mod_sp.export_posterior(
                self.adata_sp,
                sample_kwargs=sample_kwargs,
            )

            # 3) 绝对丰度 → 比例
            abund = self.adata_sp.obsm['q05_cell_abundance_w_sf']
            self.adata_sp.obsm['prop_celltypes'] = abund.div(abund.sum(axis=1).clip(lower=1e-9), axis=0)

            adata_cell2location=sc.AnnData(self.adata_sp.obsm['prop_celltypes'])
            adata_cell2location.var.index=self.adata_sp.uns["mod"]["factor_names"]
            adata_cell2location.var_names=self.adata_sp.uns["mod"]["factor_names"]
            
            adata_cell2location.obsm=self.adata_sp.obsm.copy()
            adata_cell2location.obs=self.adata_sp.obs.copy()
            adata_cell2location.obsp=self.adata_sp.obsp.copy()
            adata_cell2location.uns=self.adata_sp.uns.copy()
            self.adata_cell2location=adata_cell2location

            print(f"{Colors.GREEN}✓ cell2location is done{Colors.ENDC}")
            print(f"The cell2location result is saved in self.adata_cell2location")

        elif method=='FlashDeconv':
            self.method='FlashDeconv'
            try:
                import flashdeconv as fd
            except ImportError:
                raise ImportError(
                    "FlashDeconv is not installed. Install it with: pip install flashdeconv"
                )

            # Set default kwargs for FlashDeconv
            if flashdeconv_kwargs is None:
                flashdeconv_kwargs = {
                    'sketch_dim': 512,
                    'lambda_spatial': 5000.0,
                    'n_hvg': 2000,
                    'n_markers_per_type': 50,
                }

            # Determine spatial coordinate key
            spatial_key = 'spatial'
            if spatial_key not in self.adata_sp.obsm:
                # Try alternative keys
                for alt_key in ['X_spatial', 'spatial_coords']:
                    if alt_key in self.adata_sp.obsm:
                        spatial_key = alt_key
                        break

            # Run FlashDeconv deconvolution
            print(f"Running FlashDeconv with parameters: {flashdeconv_kwargs}")
            fd.tl.deconvolve(
                self.adata_sp,
                self.adata_sc,
                cell_type_key=celltype_key_sc,
                spatial_key=spatial_key,
                key_added='flashdeconv',
                **flashdeconv_kwargs
            )

            # Convert results to match omicverse API format
            # Create adata_cell2location from the deconvolution results
            proportions_df = self.adata_sp.obsm['flashdeconv']
            adata_cell2location = sc.AnnData(proportions_df)
            adata_cell2location.var_names = proportions_df.columns.tolist()
            adata_cell2location.obs_names = proportions_df.index.tolist()

            # Copy spatial information
            adata_cell2location.obsm = self.adata_sp.obsm.copy()
            adata_cell2location.obs = self.adata_sp.obs.copy()
            if hasattr(self.adata_sp, 'obsp') and self.adata_sp.obsp:
                adata_cell2location.obsp = self.adata_sp.obsp.copy()
            adata_cell2location.uns = self.adata_sp.uns.copy()

            self.adata_cell2location = adata_cell2location
            self.flashdeconv_params = self.adata_sp.uns.get('flashdeconv_params', flashdeconv_kwargs)

            print(f"{Colors.GREEN}✓ FlashDeconv deconvolution is done{Colors.ENDC}")
            print(f"The deconvolution result is saved in self.adata_cell2location")
            print(f"Cell type proportions are also stored in self.adata_sp.obsm['flashdeconv']")

        elif method=='starfysh':
            from ..external.starfysh import (AA, utils, plot_utils, post_analysis)
            from ..external.starfysh import _starfysh as sf_model
            self.method='starfysh'

            import torch
            import scanpy as sc
            sc.settings.verbosity = 0
                
            starfysh_default_kwargs={
                'n_repeats':3,
                'epochs':200,
                'patience':50,
                'device':None,
                'batch_size':32,
                'alpha_mul':50,
                'lr':1e-4,
                'poe':False,
                'verbose':True,
                'n_anchors':60,
                'window_size':3,
            }

            #check the starfysh_kwargs if ignore the None values
            if starfysh_kwargs is None:
                starfysh_kwargs=starfysh_default_kwargs
            else:
                # Update default kwargs with user provided kwargs
                # This ensures all keys exist, using defaults for missing ones
                # and user values for provided ones
                full_kwargs = starfysh_default_kwargs.copy()
                full_kwargs.update(starfysh_kwargs)
                starfysh_kwargs = full_kwargs
                
            if spatial_type=='visium':
                sample_id=list(self.adata_sp.uns['spatial'].keys())[0]
                tissue_position_list = pd.DataFrame(self.adata_sp.obsm['spatial'],index=self.adata_sp.obs.index,)
                map_info = tissue_position_list#.iloc[:, -4:-2]
                map_info.columns = ['array_row', 'array_col']
                map_info.loc[:, 'imagerow'] = tissue_position_list.iloc[:, -2]
                map_info.loc[:, 'imagecol'] = tissue_position_list.iloc[:, -1]
                map_info.loc[:, 'sample'] = sample_id
                map_info['array_row']=self.adata_sp.obs.loc[map_info.index,'array_row']
                map_info['array_col']=self.adata_sp.obs.loc[map_info.index,'array_col']

                img_metadata={
                    'img':self.adata_sp.uns['spatial'][sample_id]['images']['hires'],
                    'scalefactor':self.adata_sp.uns['spatial'][sample_id]['scalefactors'],
                    'map_info':map_info
                }
                if gene_sig is None:
                    print(f"you need to provide the gene signature for starfysh")
                    return

                # Parameters for training
                visium_args = utils.VisiumArguments(self.adata_sp,
                                                    self.adata_sp,
                                                    gene_sig,
                                                    img_metadata,
                                                    n_anchors=starfysh_kwargs['n_anchors'],
                                                    window_size=starfysh_kwargs['window_size'],
                                                    sample_id=sample_id
                                                )

                adata, adata_normed = visium_args.get_adata()
                anchors_df = visium_args.get_anchors()
                adata.obs['log library size']=visium_args.log_lib
                adata.obs['windowed log library size']=visium_args.win_loglib

                aa_model = AA.ArchetypalAnalysis(adata_orig=adata_normed)
                archetype, arche_dict, major_idx, evs = aa_model.compute_archetypes(cn=40)
                # (1). Find archetypal spots & archetypal clusters
                arche_df = aa_model.find_archetypal_spots(major=True)

                # (2). Find marker genes associated with each archetypal cluster
                markers_df = aa_model.find_markers(n_markers=30, display=False)

                # (3). Map archetypes to closest anchors (1-1 per cell type)
                map_df, map_dict = aa_model.assign_archetypes(anchors_df)

                # (4). Optional: Find the most distant archetypes that are not assigned to any annotated cell types
                distant_arches = aa_model.find_distant_archetypes(anchors_df, n=3)

                visium_args = utils.refine_anchors(
                    visium_args,
                    aa_model,
                    #thld=0.7,  # alignment threshold
                    n_genes=5,
                    #n_iters=1
                )

                # Get updated adata & signatures
                adata, adata_normed = visium_args.get_adata()
                gene_sig = visium_args.gene_sig
                cell_types = gene_sig.columns

                

                n_repeats = starfysh_kwargs['n_repeats']
                epochs = starfysh_kwargs['epochs']
                patience = starfysh_kwargs['patience']
                device = starfysh_kwargs['device']
                if device is None:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch_size = starfysh_kwargs['batch_size']
                alpha_mul = starfysh_kwargs['alpha_mul']
                lr = starfysh_kwargs['lr']
                poe = starfysh_kwargs['poe']
                verbose = starfysh_kwargs['verbose']
                # Run models
                model, loss = utils.run_starfysh(visium_args,
                                                n_repeats=n_repeats,
                                                epochs=epochs,
                                                #patience=patience,
                                                device=device,
                                                batch_size=batch_size,
                                                alpha_mul=alpha_mul,
                                                lr=lr,
                                                poe=poe,
                                                verbose=verbose
                                                )

                adata, adata_normed = visium_args.get_adata()
                inference_outputs, generative_outputs,adata_ = sf_model.model_eval(model,
                                                                            adata,
                                                                            visium_args,
                                                                            poe=False,
                                                                            device=device)

                def cell2proportion(adata):
                    adata_plot=sc.AnnData(adata.X)
                    adata_plot.obs=utils.extract_feature(adata, 'qc_m').obs.copy()
                    adata_plot.var=adata.var.copy()
                    adata_plot.obsm=adata.obsm.copy()
                    adata_plot.obsp=adata.obsp.copy()
                    adata_plot.uns=adata.uns.copy()
                    return adata_plot
                adata_plot=cell2proportion(adata_)
                self.adata_cell2location=adata_plot
                self.adata_sp=adata_normed
                self.starfysh_model=model 
                print(f"{Colors.GREEN}✓ starfysh is done{Colors.ENDC}")
                print(f"The starfysh result is saved in self.adata_cell2location")
                print(f"The starfysh model is saved in self.starfysh_model")
            

        else:
            raise ValueError(f"Method {method} is not supported. Choose from: 'Tangram', 'cell2location', 'FlashDeconv'")

    def impute(self,method='Tangram'):
        if method=='Tangram':
            self.method='Tangram'
            from ._tangram import Tangram
            self.adata_impute=self.tangram_model.impute()
            print(f"{Colors.GREEN}✓ Tangram impute is done{Colors.ENDC}")
            print(f"The impute result is saved in self.adata_impute")
        elif method=='cell2location':
            self.method='cell2location'
            from ..external.space.cell2location.models import Cell2location
            expected_dict = self.mod_sp.module.model.compute_expected_per_cell_type(
                self.mod_sp.samples["post_sample_q05"], self.mod_sp.adata_manager
            )
            # Add to anndata layers
            for i, n in enumerate(self.mod_sp.factor_names_):
                self.adata_sp.layers[n] = expected_dict["mu"][i]
            print(f"{Colors.GREEN}✓ cell2location impute is done{Colors.ENDC}")
            print(f"Compare with the tangram impute result, cell2location's impute stores in self.adata_sp.layers")
    
    def load_cell2location_model(self,mod_sp_path):
        self.method='cell2location'
        from ..utils import load
        #self.mod_sc=load(mod_sc_path)
        self.mod_sp=load(mod_sp_path)
        print(f"{Colors.GREEN}✓ cell2location model is loaded{Colors.ENDC}")
        print(f"The cell2location model is saved in self.mod_sc and self.mod_sp")

    def cell2location_inference(self,sample_kwargs=None):
        if sample_kwargs is None:
            sample_kwargs={"num_samples": 1000, "batch_size": 2500}
        
        self.adata_sp = self.mod_sp.export_posterior(
            self.adata_sp,
            sample_kwargs=sample_kwargs,
        )
        # 3) 绝对丰度 → 比例
        abund = self.adata_sp.obsm['q05_cell_abundance_w_sf']
        self.adata_sp.obsm['prop_celltypes'] = abund.div(abund.sum(axis=1).clip(lower=1e-9), axis=0)

        adata_cell2location=sc.AnnData(self.adata_sp.obsm['prop_celltypes'])
        adata_cell2location.var.index=self.adata_sp.uns["mod"]["factor_names"]
        adata_cell2location.var_names=self.adata_sp.uns["mod"]["factor_names"]
        
        adata_cell2location.obsm=self.adata_sp.obsm.copy()
        adata_cell2location.obs=self.adata_sp.obs.copy()
        adata_cell2location.obsp=self.adata_sp.obsp.copy()
        adata_cell2location.uns=self.adata_sp.uns.copy()
        self.adata_cell2location=adata_cell2location

        print(f"{Colors.GREEN}✓ cell2location is done{Colors.ENDC}")
        print(f"The cell2location result is saved in self.adata_cell2location")

    def load_tangram_model(self,model_path):
        self.method='Tangram'
        #from ._tangram import Tangram
        from ..utils import load
        
        self.tangram_model=load(model_path)
        print(f"{Colors.GREEN}✓ Tangram model is loaded{Colors.ENDC}")
        print(f"The Tangram model is saved in self.tangram")

    def tangram_inference(self,sample_kwargs=None):
        self.adata_cell2location=self.tangram_model.cell2location()
        print(f"{Colors.GREEN}✓ Tangram is done{Colors.ENDC}")
        print(f"The Tangram result is saved in self.adata_cell2location")


def calculate_gene_signature(
    adata_sc,clustertype,
    rank=True,
    key='rank_genes_groups',
    foldchange=2,
    topgenenumber=20 
):
    r"""Calculate the gene signature.
    Parameters
    ----------
    adata_sc : AnnData
        Single-cell RNA-seq data.
    clustertype : str
        Cell type key in scRNA-seq data.
    rank : bool, optional
        Whether to rank the genes, by default True.
    key : str, optional
        Key in scRNA-seq data, by default 'rank_genes_groups'.
    foldchange : int, optional
        Fold change, by default 2.
    topgenenumber : int, optional
        Number of top genes, by default 20.
    """ 
    # 1) 用“已注释细胞类型”作为分组
    from ..single import get_celltype_marker
    all_markers = get_celltype_marker(
        adata_sc,
        clustertype=clustertype,
        rank=rank,
        key=key,
        foldchange=foldchange,
        topgenenumber=topgenenumber
    )
    # 2) dict -> 表格（列=celltype，行=genes）
    max_len = max(len(v) for v in all_markers.values())
    gene_sig = pd.DataFrame({
        k: list(v) + [pd.NA] * (max_len - len(v))
        for k, v in all_markers.items()
    })

    return gene_sig


