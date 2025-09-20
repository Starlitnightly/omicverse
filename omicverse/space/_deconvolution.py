from .._settings import add_reference, EMOJI, Colors
import numpy as np
import scanpy as sc


class Deconvolution(object):
    def __init__(self, adata_sc, adata_sp):
        self.adata_sc = adata_sc
        self.adata_sp = adata_sp
        self.mod_sc=None
        self.mod_sp=None
        self.adata_cell2location=None
        self.adata_impute=None
        self.tangram_model=None
        self.method=None

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
        from ..pp import preprocess
        self.adata_sc=preprocess(self.adata_sc,mode=mode,n_HVGs=n_HVGs,target_sum=target_sum,**kwargs)
        self.adata_sc.raw = self.adata_sc
        self.adata_sc = self.adata_sc[:, self.adata_sc.var.highly_variable_features]
        print(f"{Colors.GREEN}✓ scRNA-seq data is preprocessed{Colors.ENDC}")
        #return self.adata_sc

    def preprocess_sp(self,mode='pearsonr',n_svgs=3000,target_sum=50*1e4,platform="visium",mt_startwith='MT-',**kwargs):
        from ._svg import svg
        self.adata_sp=svg(self.adata_sp,mode=mode,n_svgs=n_svgs,
                    target_sum=target_sum,platform=platform,
                    mt_startwith=mt_startwith,**kwargs)
        self.adata_sp.raw = self.adata_sp
        self.adata_sp = self.adata_sp[:, self.adata_sp.var.space_variable_features]
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

    ):
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

            
        else:
            raise ValueError(f"Method {method} is not supported")

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