# Differential celltype abundance analysis using pertpy
import scanpy as sc
from anndata import AnnData
import numpy as np
from scipy.sparse import issparse
import pandas as pd

from .._settings import EMOJI,add_reference

class DCT:
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
        """
        Init the differential cell type abundance analysis

        Arguments:
            adata: AnnData object containing the single-cell data
            condition: The column name in adata.obs containing condition information
            ctrl_group: The control group name in the condition column
            test_group: The test group name in the condition column
            cell_type_key: The column name in adata.obs containing cell type information
            method: Method for differential abundance analysis, either 'sccoda' or 'milo'
            sample_key: The column name in adata.obs containing sample information
            use_rep: The representation in adata.obsm to use for Milo analysis

        Returns:
            None
        """
        import pertpy as pt
        # filter adata for condition and test group
        self.adata = adata
        self.condition = condition
        self.ctrl_group = ctrl_group
        self.test_group = test_group
        self.adata.obs[condition] = self.adata.obs[condition].astype('category')
        self.adata=self.adata[self.adata.obs[condition].isin([ctrl_group, test_group])]
        self.adata.obs[condition] = self.adata.obs[condition].cat.reorder_categories([ctrl_group, test_group])
        self.method = method
        self.cell_type_key = cell_type_key

        # Print EMOJI
        print(f"{EMOJI['check_mark']} Differential cell type abundance analysis initialized")
        print(f"{EMOJI['bar']} DCT analysis using {self.method} method")
        print(f"{EMOJI['bar']} Condition: {self.condition}, Control group: {self.ctrl_group}, Test group: {self.test_group}")

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
            sc.pp.neighbors(self.mdata["rna"], use_rep=use_rep, n_neighbors=150)
            self.model.make_nhoods(self.mdata["rna"], prop=0.1)
            self.mdata = self.model.count_nhoods(self.mdata, sample_col=sample_key)
        else:
            raise ValueError(f"Method {self.method} not supported")


    def run(self,**kwargs):
        """
        Run the differential cell type abundance analysis

        Arguments:
            **kwargs: Additional arguments to pass to the differential abundance method

        Returns:
            None
        """
        if self.method == 'sccoda':
            self.model.run_nuts(self.sccoda_data, modality_key="coda", **kwargs)
            self.model.credible_effects(self.sccoda_data, modality_key="coda")
            print(f"{EMOJI['check_mark']} {self.method} DCT analysis completed")
            add_reference(self.adata,'Sccoda','differential cell type abundance analysis with Sccoda')
            add_reference(self.adata,'pertpy','Sccoda is a part of pertpy')
        elif self.method == 'milo':
            self.model.da_nhoods(self.mdata, design=f"~{self.condition}", model_contrasts=f"{self.condition}{self.test_group}-{self.condition}{self.ctrl_group}")
            self.model.build_nhood_graph(self.mdata)
            self.model.annotate_nhoods(self.mdata, anno_col=self.cell_type_key)
            print(f"{EMOJI['check_mark']} {self.method} DCT analysis completed")
            add_reference(self.adata,'Milo','differential cell type abundance analysis with Milo')
            add_reference(self.adata,'pertpy','Milo is a part of pertpy')
        else:
            raise ValueError(f"Method {self.method} not supported")

    def get_results(self,mix_threshold: float=0.6):
        """
        Get the results of the differential cell type abundance analysis

        Arguments:
            mix_threshold: The threshold for determining mixed neighborhoods in Milo analysis

        Returns:
            DataFrame: Results of the differential abundance analysis
        """
        if self.method == 'sccoda':
            return self.model.get_effect_df(self.sccoda_data, modality_key="coda")
        elif self.method == 'milo':
            self.mdata["milo"].var[ "nhood_annotation"]=self.mdata["milo"].var[ "nhood_annotation"].astype(str)
            self.mdata["milo"].var.loc[self.mdata["milo"].var["nhood_annotation_frac"] < mix_threshold, "nhood_annotation"] = "Mixed"
            return self.mdata["milo"].var
        else:
            raise ValueError(f"Method {self.method} not supported")
        
            

class DEG:
    def __init__(self,
                 adata: AnnData,
                 condition: str,
                 ctrl_group: str,
                 test_group: str,
                 method: str='wilcoxon',
                 ):
        """
        Init the differential expression gene analysis

        Arguments:
            adata: AnnData object containing the single-cell data
            condition: The column name in adata.obs containing condition information
            ctrl_group: The control group name in the condition column
            test_group: The test group name in the condition column
            method: Method for differential expression analysis, either 'wilcoxon', 't-test', or 'memento-de'

        Returns:
            None
        """
        self.adata=adata
        self.condition=condition
        self.ctrl_group=ctrl_group
        self.test_group=test_group
        self.method=method
        
        from scipy.sparse import csr_matrix
        try:
            assert type(self.adata.X) == csr_matrix
        except:
            self.adata.X = csr_matrix(self.adata.X)

        #print EMOJI
        print(f"{EMOJI['check_mark']} Differential expression analysis initialized")
        print(f"{EMOJI['bar']} DEG analysis using {self.method} method")
        #print the condition and ctrl_group and test_group
        print(f"{EMOJI['bar']} Condition: {self.condition}, Control group: {self.ctrl_group}, Test group: {self.test_group}")
        
        
    def run(self,
            celltype_key: str,
            celltype_group=None,
            max_cells: int=100000,
            **kwargs):
        """
        Run the differential expression analysis

        Arguments:
            celltype_key: The column name in adata.obs containing cell type information
            celltype_group: List of cell types to analyze, if None, all cell types will be analyzed
            **kwargs: Additional arguments for the differential expression method
                capture_rate: float, default=0.07
                    The capture rate for the DE analysis
                treatment_col: str, default='stim'
                    The column name of the treatment variable
                num_cpus: int, default=12
                    The number of CPUs to use for the DE analysis
                num_boot: int, default=5000
                    The number of bootstraps to use for the DE analysis

        Returns:
            None
        """
        if celltype_group is None:
            celltype_group = self.adata.obs[celltype_key].unique()
        else:
            celltype_group = celltype_group

        self.adata_test=self.adata[self.adata.obs[celltype_key].isin(celltype_group)]

        if max_cells is None:
            max_cells = self.adata_test.shape[0]

        print(f"{EMOJI['bar']} Celltype key: {celltype_key}, Celltype group: {celltype_group}")
        print(f"Total cells: {self.adata_test.shape[0]} will be used for DEG analysis")
        if self.adata_test.shape[0] == 0:
            raise ValueError(f"No cells found for DEG analysis")
        elif self.adata_test.shape[0] > max_cells:
            EMOJI['warning']="⚠️"
            print(f"{EMOJI['warning']} Total cells: {self.adata_test.shape[0]} is too large, will be downsampled to {max_cells}")
            print(f"If you want to keep all cells, please set max_cells to None")
            try:
                sc.pp.subsample(self.adata_test, n_obs=max_cells)
            except:
                try:
                    sc.pp.sample(self.adata_test, n_obs=max_cells)
                except:
                    raise ValueError(f"Failed to downsample the data, please check the data")
                

  

        if self.method == 'wilcoxon' or self.method == 't-test':
            self.adata_test=self.adata_test[self.adata_test.obs[self.condition].isin([self.ctrl_group, self.test_group])]

            if is_all_integer(self.adata_test.X):
                sc.pp.normalize_total(self.adata_test, inplace=True)
                sc.pp.log1p(self.adata_test)

            sc.tl.rank_genes_groups( 
                self.adata_test,  
                groupby=self.condition, 
                groups=[self.test_group, self.ctrl_group], 
                reference=self.ctrl_group, 
                n_genes=self.adata_test.shape[1], 
                method=self.method 
            ) 
            print(f"{EMOJI['check_mark']} {self.method} DEG analysis completed")
            add_reference(self.adata,'Wilcoxon','differential expression analysis with Wilcoxon')
            add_reference(self.adata,'T-test','differential expression analysis with T-test')
            add_reference(self.adata,'scanpy','differential expression analysis with SCANPY')
        elif self.method == 'memento-de':
            import memento
            self.adata_test=self.adata_test[self.adata_test.obs[self.condition].isin([self.ctrl_group, self.test_group])]
            self.adata_test.obs['stim'] = self.adata_test.obs[self.condition].apply(lambda x: 0 if x == self.ctrl_group else 1)

            # check counts in adata_test.layers['counts']
            if is_all_integer(self.adata_test.X):
                pass
            elif 'counts' in self.adata_test.layers:
                self.adata_test.X = self.adata_test.layers['counts']
            else:
                from ..pp import recover_counts
                from scipy.sparse import issparse, csr_matrix

                if issparse(self.adata_test.X):
                    pass 
                else:
                    self.adata_test.X=csr_matrix(self.adata_test.X)
                #detect the lognorm 10e4 or other value use max value
                if self.adata_test.X.max() < np.log1p(1e4):
                    lognorm = 1e4
                elif self.adata_test.X.max() < np.log1p(10*1e4):
                    lognorm = 10*1e4
                elif self.adata_test.X.max() < np.log1p(50*1e4):
                    lognorm = 50*1e4
                elif self.adata_test.X.max() < np.log1p(100*1e4):
                    lognorm = 100*1e4
                else:
                    lognorm = self.adata_test.X.max()
                print('Recover the counts matrix from log-normalized data.')
                X_counts_recovered, size_factors_sub=recover_counts(self.adata_test.X, lognorm, 
                                                                    lognorm*10,  log_base=None, 
                                                          chunk_size=10000)
                self.adata_test.X=X_counts_recovered

            result_1d = memento.binary_test_1d(
                adata=self.adata_test, 
                treatment_col='stim', 
                **kwargs
            )
            self.result_1d = result_1d
            print(f"{EMOJI['check_mark']} {self.method} DEG analysis completed")
            add_reference(self.adata,'memento','differential expression analysis with memento-de')
        else:
            raise ValueError(f"Method {self.method} not supported")
 
            
        
    def get_results(self):
        """
        Get the results of the differential expression analysis

        Arguments:
            None

        Returns:
            DataFrame: Results of the differential expression analysis
        """
        if self.method == 'wilcoxon' or self.method == 't-test':
            md_d = ( 
                sc.get.rank_genes_groups_df(self.adata_test, group=self.test_group) 
                .set_index("names", drop=False) 
            ) 
            res=pd.DataFrame(index=md_d['names'].tolist())
            res['log2FC']=md_d['logfoldchanges'].tolist()
            res['pvalue']=md_d['pvals'].tolist()
            res['padj']=md_d['pvals_adj'].tolist()
            res['qvalue']=md_d['pvals_adj'].tolist()
            res['size']=np.abs(res['log2FC'])/10
            res['sig']='normal'
            res.loc[res['padj']<0.05,'sig']='sig'
            res['-log(pvalue)'] = -np.log10(res['pvalue'])
            res['-log(qvalue)'] = -np.log10(res['qvalue'])
            #calculate the Mean of the self.adata_test's genes(var)
            res['baseMean']=np.mean(self.adata_test.to_df(),axis=0).loc[res.index]
            self.result=res
            return res
        elif self.method == 'memento-de':
            self.result=self.result_1d
            self.result['baseMean']=np.mean(self.adata_test.to_df(),axis=0).loc[self.result.index].to_list()
            return self.result
        else:
            raise ValueError(f"Method {self.method} not supported")




def is_all_integer(data, allow_negative=True, tol=0):
    """
    判断矩阵 data 中所有元素是否为整数。

    参数
    ----
    data : np.ndarray 或 scipy.sparse 矩阵
        待检测的数据矩阵。
    allow_negative : bool, default True
        如果为 False，则遇到任何负值直接返回 False。
    tol : float, default 0
        允许的浮点数误差容限。如果你的数据理论上是整数但由于浮点运算会出现 1e-16 之类的小偏差，
        可以把 tol 设为 1e-8 之类的值。

    返回
    ----
    bool
        如果所有值都是整数（且符合 allow_negative 要求），返回 True；否则 False。
    """
    # 1) 取出实际的 ndarray
    if issparse(data):
        arr = data.data
    else:
        arr = np.asarray(data)

    # 2) 可选：是否允许出现负数
    if not allow_negative and np.signbit(arr).any():
        return False

    # 3) 判断"小数部分"是否都在 tol 以内
    #    np.modf(arr) 会返回 (frac_part, int_part)
    frac_part, _ = np.modf(arr)
    return np.all(np.abs(frac_part) <= tol)