

# Differential celltype abundance analysis using pertpy
import scanpy as sc
from anndata import AnnData

class DEGCT:
    def __init__(self, 
                 adata: AnnData, 
                 condition: str,
                 ctrl_group: str,
                 test_group: str,
                 cell_type_key: str,
                 method: str='sccoda',
                 sample_key=None,
                 use_rep=None,
    ):
        import pertpy as pt
        # filter adata for condition and test group
        self.adata = adata
        self.condition = condition
        self.ctrl_group = ctrl_group
        self.test_group = test_group
        self.adata.obs[condition] = self.adata.obs[condition].astype('category')
        self.adata=self.adata[self.adata.obs[condition].isin([ctrl_group, test_group])]
        self.adata.obs[condition] = self.adata.obs[condition].cat.reorder_categories([ctrl_group, test_group])

        if method == 'sccoda':
            self.model = pt.tl.Sccoda()
            self.sccoda_data = self.model.load(
                adata,
                type="cell_level",
                #generate_sample_level=True,
                cell_type_identifier=cell_type_key,
                sample_identifier=sample_key,
                covariate_obs=[condition],
            )
            self.sccoda_data = self.model.prepare(
                self.sccoda_data,
                modality_key="coda",
                formula="condition",
                #reference_cell_type="Goblet",
            )
        elif method == 'milo':
            #check if use_rep is provided
            if use_rep is None:
                raise ValueError("use_rep must be provided for milo")
            elif use_rep not in adata.obsm.keys():
                raise ValueError("use_rep must be a valid embedding in adata.obsm")
            
            self.model = pt.tl.Milo()
            self.mdata = self.model.load(adata)
            sc.pp.neighbors(self.mdata[use_rep], use_rep=use_rep, n_neighbors=150)
            self.model.make_nhoods(self.mdata[use_rep], prop=0.1)
            self.mdata = self.model.count_nhoods(self.mdata, sample_col=sample_key)


    def run(self,**kwargs):

        if self.method == 'sccoda':
            self.model.run_nuts(self.sccoda_data, modality_key="coda", **kwargs)
            self.model.credible_effects(self.sccoda_data, modality_key="coda")
        elif self.method == 'milo':
            self.model.da_nhoods(self.mdata, design=f"~{self.condition}", model_contrasts=f"{self.condition}{self.test_group}-{self.condition}{self.ctrl_group}")
            self.model.build_nhood_graph(self.mdata)
            self.model.annotate_nhoods(self.mdata, anno_col=self.cell_type_key)

            



    def get_results(self,mix_threshold: float=0.6):
        if self.method == 'sccoda':
            return self.model.get_effect_df(self.sccoda_data, modality_key="coda")
        elif self.method == 'milo':
            self.mdata["milo"].var[ "nhood_annotation"]=self.mdata["milo"].var[ "nhood_annotation"].astype(str)
            self.mdata["milo"].var.loc[self.mdata["milo"].var["nhood_annotation_frac"] < mix_threshold, "nhood_annotation"] = "Mixed"
            return self.mdata["milo"].var
        
            