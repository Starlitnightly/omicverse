import scanpy as sc
import os
import anndata
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union

class pyTCGA(object):
    r"""
    TCGA analysis module 
    """
    def __init__(self,gdc_sample_sheep:str,gdc_download_files:str,clinical_cart:str):
        r"""
        Init the TCGA module

        Arguments:
            gdc_sample_sheep: Path of the Sample Sheet button of TCGA, and we can get tsv file from it
            gdc_download_files: Path of the Download/Cart button of TCGA, and we get tar.gz included all file you selected/
            clinical_cart: Path of the Clinical button of TCGA, and we can get tar.gz included all clinical of your files

        """
        self.gdc_sample_sheep=gdc_sample_sheep
        self.gdc_download_files=gdc_download_files
        self.clinical_cart=clinical_cart
        exist_files=[i for i in os.listdir(gdc_download_files) if 'txt' not in i]
        
        self.sample_sheet=pd.read_csv(self.gdc_sample_sheep,sep='\t',index_col=0)
        exist_files=list(set(exist_files) & set(self.sample_sheet.index))
        self.sample_sheet=self.sample_sheet.loc[exist_files]
        self.clinical_sheet=pd.read_csv('{}/clinical.tsv'.format(self.clinical_cart),sep='\t',index_col=0)
        #self.clinical_sheet=self.clinical_sheet.loc[exist_files]
        

        sample_index=self.sample_sheet.index[0]
        sample_id=self.sample_sheet.loc[sample_index,'Sample ID']
        sample_file_id=sample_index
        sample_file_name=self.sample_sheet.loc[sample_index,'File Name']
        self.data_test=pd.read_csv('{}/{}/{}'.format(self.gdc_download_files,sample_file_id,sample_file_name),
                             sep='\t',index_col=0,skiprows=1)
        print('tcga module init success')
        
        
    def adata_read(self,path:str):
        r"""
        Read the anndata file

        Arguments:
            path: Path of the anndata file
        """
        print('... anndata reading')
        self.adata=sc.read(path)
        
    def adata_init(self):
        self.index_init()
        self.expression_init()
        self.matrix_construct()
        
    def adata_meta_init(self,var_names:list=['gene_name','gene_type'],
                  obs_names:list=['Case ID','Sample Type'])->anndata.AnnData:
        r"""
        Init the anndata meta data

        Arguments:
            var_names: The column name of the var meta data
            obs_names: The column name of the obs meta data

        Returns:
            adata: The anndata object with meta data

        """
        print('...anndata meta init',var_names,obs_names)
        adata=self.adata
        #var_pd=pd.DataFrame(index=self.adata.var.index)
        var_pd=self.data_test.loc[adata.var.index,var_names]
        var_pd['gene_id']=var_pd.index.tolist()
        var_pd.index=var_pd['gene_name'].values
        #obs_pd=pd.DataFrame(index=data_pd_count.columns)
        sample_sheet_tmp=self.sample_sheet.copy()
        sample_sheet_tmp.index=sample_sheet_tmp['Sample ID']
        obs_pd=sample_sheet_tmp.loc[adata.obs.index,obs_names]
        obs_pd=obs_pd[~obs_pd.index.duplicated(keep='first')]
        adata.obs=obs_pd.loc[adata.obs.index]
        adata.var=var_pd
        adata.var.index=adata.var['gene_name'].astype('str').values
        adata.var_names_make_unique()
        self.adata=adata
        return adata
        
    def survial_init(self):
        r"""
        Init the survial data
        """
        day_li=[]
        pd_c=self.clinical_sheet
        for i in pd_c.index:
            if pd_c.loc[i,'vital_status'].iloc[0]=='Alive':
                day_li.append(pd_c.loc[i,'days_to_last_follow_up'].iloc[0])
            elif pd_c.loc[i,'vital_status'].iloc[0]=='Dead':
                day_li.append(pd_c.loc[i,'days_to_death'].iloc[0])
            else:
                day_li.append(pd_c.loc[i,'days_to_last_follow_up'].iloc[0])
        pd_c['days']=day_li
        
        s_pd=pd_c[["case_submitter_id",
              "vital_status",
              "days_to_last_follow_up",
                "days_to_death",
                "age_at_index",
                "tumor_grade","days"]].copy()
        s_pd=s_pd.drop_duplicates(subset='case_submitter_id')
        s_pd.set_index(s_pd.columns[0],inplace=True)
        self.s_pd=s_pd
        
        
        self.adata.obs['vital_status']='Not Reported'
        self.adata.obs['days']=np.nan
        for i in self.adata.obs.index:
            if self.adata.obs.loc[i,'Case ID'] not in s_pd.index:
                self.adata=self.adata[self.adata.obs.index!=i]
                continue
            self.adata.obs.loc[i,'vital_status']=s_pd.loc[self.adata.obs.loc[i,'Case ID'],'vital_status']
            self.adata.obs.loc[i,'days']=s_pd.loc[self.adata.obs.loc[i,'Case ID'],'days']
        
        
        
    def index_init(self)->list:
        r"""
        Init the index of the anndata object

        Returns:
            all_lncRNA_index: The index of the anndata object
        """
        print('...index init')
        all_lncRNA_index=[]
        for sample_index in self.sample_sheet.index:
            sample_id=self.sample_sheet.loc[sample_index,'Sample ID']
            sample_file_id=sample_index
            sample_file_name=self.sample_sheet.loc[sample_index,'File Name']
            data_test=pd.read_csv('{}/{}/{}'.format(self.gdc_download_files,sample_file_id,sample_file_name),
                             sep='\t',index_col=0,skiprows=1)
            #data_test=data_test.loc[data_test['gene_type']=='lncRNA']
            data_c_s=data_test['tpm_unstranded'].sort_values(ascending=False)
            data_c_s=data_c_s[~data_c_s.index.duplicated(keep='first')]
            all_lncRNA_index=list(set(all_lncRNA_index) | set(data_c_s.index.tolist()))
        self.tcga_index=all_lncRNA_index
        return all_lncRNA_index
    
    def expression_init(self):
        r"""
        Init the expression matrix of the anndata object
        """
        print('... expression matrix init')
        data_pd_count=pd.DataFrame(index=self.tcga_index)
        data_pd_tpm=pd.DataFrame(index=self.tcga_index)
        data_pd_fpkm=pd.DataFrame(index=self.tcga_index)

        for sample_index in self.sample_sheet.index:
            sample_id=self.sample_sheet.loc[sample_index,'Sample ID']
            sample_file_id=sample_index
            sample_file_name=self.sample_sheet.loc[sample_index,'File Name']
            #print(sample_id)
            data_test=pd.read_csv('{}/{}/{}'.format(self.gdc_download_files,sample_file_id,sample_file_name),
                             sep='\t',index_col=0,skiprows=1)
            #data_test=data_test.loc[data_test['gene_type']=='lncRNA']
            data_c_s=data_test['unstranded'].sort_values(ascending=False)
            data_c_s=data_c_s[~data_c_s.index.duplicated(keep='first')]
            data_pd_count[sample_id]=0
            data_pd_count.loc[data_c_s.index,sample_id]=data_c_s.values

            data_c_s=data_test['tpm_unstranded'].sort_values(ascending=False)
            data_c_s=data_c_s[~data_c_s.index.duplicated(keep='first')]
            data_pd_tpm[sample_id]=0
            data_pd_tpm.loc[data_c_s.index,sample_id]=data_c_s.values

            data_c_s=data_test['fpkm_unstranded'].sort_values(ascending=False)
            data_c_s=data_c_s[~data_c_s.index.duplicated(keep='first')]
            data_pd_fpkm[sample_id]=0
            data_pd_fpkm.loc[data_c_s.index,sample_id]=data_c_s.values
            
        self.data_pd_count=data_pd_count
        self.data_pd_tpm=data_pd_tpm
        self.data_pd_fpkm=data_pd_fpkm
        self.data_test=data_test
    
    def matrix_construct(self):
        r"""
        Construct the anndata object
        """
        print('...anndata construct')
        var_pd=pd.DataFrame(index=self.data_pd_count.index)
        obs_pd=pd.DataFrame(index=self.data_pd_count.columns)
        adata=anndata.AnnData(self.data_pd_count.T,var=var_pd,obs=obs_pd)
        adata.layers['tpm']=self.data_pd_tpm.T.values
        adata.layers['fpkm']=self.data_pd_fpkm.T.values
        adata.layers['deseq_normalize']=self.matrix_normalize(self.data_pd_count).T.values
        self.adata=adata
        return adata
    
    def matrix_normalize(self,data:pd.DataFrame)->pd.DataFrame:
        r"""
        normalize the matrix by Deseq2 methods

        Arguments:
            data: The matrix to be normalized

        Returns:
            data: The normalized matrix
        """
        avg1=data.apply(np.log,axis=1).mean(axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        data1=data.loc[avg1.index]
        data_log=data1.apply(np.log,axis=1)
        scale=data_log.sub(avg1.values,axis=0).median(axis=0).apply(np.exp)
        return data/scale

    
    
    def survival_analysis(self,gene:str,layer:str='raw',plot:bool=False,gene_threshold:str='median')->Tuple[float,float]:
        r"""
        Analysis the survival of the gene

        Arguments:
            gene: The gene name
            layer: The layer of the anndata object
            plot: Whether to plot the survival curve
            gene_threshold: The threshold of the gene expression, can be 'median' or 'mean'

        Returns:
            test_statistic: The test statistic
            pvalue: The survival pvalue

        """
        goal_gene=gene
        
        s_pd=self.s_pd
        s_pd=s_pd.loc[self.adata.obs['Case ID']]
        if layer!='raw':
            if layer not in self.adata.layers.keys():
                s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].X.mean(axis=1).toarray()
            else:
                s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].layers[layer].mean(axis=1).toarray()
            
        else:
            s_pd[goal_gene]=self.adata[self.adata.obs.index,self.adata.var['gene_name']==goal_gene].X.mean(axis=1).toarray()
        if gene_threshold=='median':
            s_pd['{}_status'.format(goal_gene)]=['High' if i>s_pd[goal_gene].median() else 'Low' for i in s_pd[goal_gene] ]
        elif gene_threshold=='mean':
            s_pd['{}_status'.format(goal_gene)]=['High' if i>s_pd[goal_gene].mean() else 'Low' for i in s_pd[goal_gene] ]
        else:
            s_pd['{}_status'.format(goal_gene)]=['High' if i>gene_threshold else 'Low' for i in s_pd[goal_gene] ]
        s_pd=s_pd.loc[s_pd['days']!="'--"]
        s_pd['fustat'] = [0 if 'Alive'==i else 1 for i in s_pd['vital_status']]
        s_pd['gene_fustat'] = [0 if 'High'==i else 1 for i in s_pd['{}_status'.format(goal_gene)]]

        km = KaplanMeierFitter()
        T = s_pd['days'].astype(float) / 365
        E=s_pd['fustat']

        gender = (s_pd['{}_status'.format(goal_gene)] == 'High')
        lr = logrank_test(T[gender], T[~gender], E[gender], E[~gender], alpha=.95)
        if plot==True:
            fig, ax = plt.subplots(figsize=(3,3))
            km.fit(T[gender], event_observed=E[gender], label="High")
            km.plot(ax=ax,color='#941456')
            km.fit(T[~gender], event_observed=E[~gender], label="Low")
            km.plot(ax=ax,color='#368650')
            lr = logrank_test(T[gender], T[~gender], E[gender], E[~gender], alpha=.95)
            lr.p_value

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

            plt.xlabel('Years')
            plt.ylabel('Pecent Survial')
            plt.title('Survial: {}\np-value: {}'.format(goal_gene,round(lr.p_value,3)))
            plt.grid(False)
            
        return lr.test_statistic,lr.p_value
    
    def survial_analysis_all(self):
        r"""
        analysis the survival of all the genes
        """
        res_l_lnc=[]
        res_l_tt=[]
        for i in self.adata.var.index:
            res_l_tt.append(self.survival_analysis(i)[0])
            res_l_lnc.append(self.survival_analysis(i)[1])
        self.adata.var['survial_test_statistic']=res_l_tt
        self.adata.var['survial_p']=res_l_lnc
        
    
    