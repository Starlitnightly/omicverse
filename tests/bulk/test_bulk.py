import pytest 
import omicverse as ov

from omicverse import bulk
import pandas as pd

data=pd.read_csv('https://github.com/Starlitnightly/omicverse/raw/master/sample/counts.txt',
                            index_col=0,header=1,sep='\t')
#replace the columns `.bam` to `` 
data.columns=[i.split('/')[-1].replace('.bam','') for i in data.columns]

def test_deg():
    
    ov.utils.download_geneid_annotation_pair()
    test_data=data.copy()
    test_data=ov.bulk.Matrix_ID_mapping(test_data,'genesets/pair_GRCm39.tsv')

    
    dds=bulk.pyDEG(test_data)
    dds.drop_duplicates_index()

    treatment_groups=['4-3','4-4']
    control_groups=['1--1','1--2']
    result=dds.deg_analysis(treatment_groups,control_groups,method='DEseq2')
    # -1 means automatically calculates
    dds.foldchange_set(fc_threshold=-1,
                    pval_threshold=0.05,
                    logp_max=10)
    if result is not None:
        assert isinstance(result,pd.DataFrame)

    deg_genes=dds.result.loc[dds.result['sig']!='normal'].index.tolist()
    ov.utils.download_pathway_database()
    pathway_dict=ov.utils.geneset_prepare('genesets/WikiPathways_2019_Mouse.txt',organism='Mouse')
    enr=ov.bulk.geneset_enrichment(gene_list=deg_genes,
                                    pathways_dict=pathway_dict,
                                    pvalue_type='auto',
                                    organism='mouse')
    if len(enr)>0:
        assert isinstance(enr,pd.DataFrame)


    

