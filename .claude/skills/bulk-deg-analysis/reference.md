# Bulk DEG quick commands

```python
import omicverse as ov
import scanpy as sc
import matplotlib.pyplot as plt

ov.plot_set()
ov.utils.download_geneid_annotation_pair()
counts = ov.pd.read_csv('counts.txt', sep='\t', header=1, index_col=0)
counts.columns = [c.split('/')[-1].replace('.bam', '') for c in counts.columns]
counts = ov.bulk.Matrix_ID_mapping(counts, 'genesets/pair_GRCm39.tsv')

dds = ov.bulk.pyDEG(counts)
dds.drop_duplicates_index()
dds.normalize()

trt = ['4-3', '4-4']
ctl = ['1--1', '1--2']
res = dds.deg_analysis(trt, ctl, method='ttest')

dds.result = dds.result.loc[dds.result['log2(BaseMean)'] > 1]
dds.foldchange_set(fc_threshold=-1, pval_threshold=0.05, logp_max=6)

dds.plot_volcano(title='DEG Analysis', figsize=(4, 4), plot_genes_num=8)
dds.plot_boxplot(genes=['Ckap2'], treatment_groups=trt, control_groups=ctl)
```

For enrichment:

```python
ov.utils.download_pathway_database()
pathways = ov.utils.geneset_prepare('genesets/WikiPathways_2019_Mouse.txt', organism='Mouse')

deg_genes = dds.result.loc[dds.result['sig'] != 'normal'].index.tolist()
enr = ov.bulk.geneset_enrichment(gene_list=deg_genes,
                                pathways_dict=pathways,
                                pvalue_type='auto',
                                organism='mouse')

ov.bulk.geneset_plot(enr, figsize=(2, 5), fig_title='Wiki Pathway enrichment',
                     cax_loc=[2, 0.45, 0.5, 0.02],
                     bbox_to_anchor_used=(-0.25, -13),
                     node_diameter=10,
                     custom_ticks=[5, 7],
                     text_knock=3,
                     cmap='Reds')
```
