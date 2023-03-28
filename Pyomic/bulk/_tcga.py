import scanpy as sc
import os
import anndata
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd
import matplotlib.pyplot as plt

class TCGA(object):
    
    def __init__(self,gdc_sample_sheep,gdc_download_files,clinical_cart):
        self.gdc_sample_sheep=gdc_sample_sheep
        self.gdc_download_files=gdc_download_files
        self.clinical_cart=clinical_cart
        exist_files=[i for i in os.listdir(gdc_download_files) if 'txt' not in i]
        
        self.sample_sheet=pd.read_csv(self.gdc_sample_sheep,sep='\t',index_col=0)
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
        
        
    def adata_read(self,path):
        print('... anndata reading')
        self.adata=sc.read(path)
        
    def adata_init(self):
        self.index_init()
        self.expression_init()
        self.matrix_construct()
        
    def adata_meta_init(self,var_names=['gene_name','gene_type'],
                  obs_names=['Case ID','Sample Type']):
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
        return adata
        
    def survial_init(self):
        day_li=[]
        pd_c=self.clinical_sheet
        for i in pd_c.index:
            if pd_c.loc[i,'vital_status'].iloc[0]=='Alive':
                day_li.append(pd_c.loc[i,'days_to_last_follow_up'].iloc[0])
            elif pd_c.loc[i,'vital_status'].iloc[0]=='Dead':
                day_li.append(pd_c.loc[i,'days_to_death'].iloc[0])
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
            self.adata.obs.loc[i,'vital_status']=s_pd.loc[self.adata.obs.loc[i,'Case ID'],'vital_status']
            self.adata.obs.loc[i,'days']=s_pd.loc[self.adata.obs.loc[i,'Case ID'],'days']
        
        
        
    def index_init(self):
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
       
        print('...anndata construct')
        var_pd=pd.DataFrame(index=self.data_pd_count.index)
        obs_pd=pd.DataFrame(index=self.data_pd_count.columns)
        adata=anndata.AnnData(self.data_pd_count.T,var=var_pd,obs=obs_pd)
        adata.layers['tpm']=self.data_pd_tpm.T.values
        adata.layers['fpkm']=self.data_pd_fpkm.T.values
        adata.layers['deseq_normalize']=self.matrix_normalize(self.data_pd_count).T.values
        self.adata=adata
        return adata
    
    def matrix_normalize(self,data):
        avg1=data.apply(np.log,axis=1).mean(axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        data1=data.loc[avg1.index]
        data_log=data1.apply(np.log,axis=1)
        scale=data_log.sub(avg1.values,axis=0).median(axis=0).apply(np.exp)
        return data/scale

    
    
    def survival_analysis(self,gene,layer='raw',plot=False):
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
        
        s_pd['{}_status'.format(goal_gene)]=['High' if i>s_pd[goal_gene].median() else 'Low' for i in s_pd[goal_gene] ]
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
            
        return lr.p_value
    
    def survial_analysis_all(self):
        res_l_lnc=[]
        for i in self.adata.var.index:
            res_l_lnc.append(self.survival_analysis(i))
        self.adata.var['survial_p']=res_l_lnc
        
    
    