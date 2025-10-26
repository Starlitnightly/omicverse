# PyDESeq2 workflow quick commands

```python
import omicverse as ov

ov.utils.ov_plot_set()

data = ov.utils.read('sample/counts.txt', index_col=0, header=1)
data.columns = [c.split('/')[-1].replace('.bam', '') for c in data.columns]

ov.utils.download_geneid_annotation_pair()
data = ov.bulk.Matrix_ID_mapping(data, 'genesets/pair_GRCm39.tsv')

dds = ov.bulk.pyDEG(data)
dds.drop_duplicates_index()

treatment_groups = ['4-3', '4-4']
control_groups = ['1--1', '1--2']
res = dds.deg_analysis(treatment_groups, control_groups, method='DEseq2')

res = res.loc[res['log2(BaseMean)'] > 1]
dds.result = res

dds.foldchange_set(fc_threshold=-1, pval_threshold=0.05, logp_max=6)
dds.plot_volcano(title='DESeq2 DEG', figsize=(4, 4), plot_genes_num=10)
dds.plot_boxplot(genes=['Ckap2'], treatment_groups=treatment_groups,
                control_groups=control_groups, figsize=(2, 3), legend_bbox=(2, 0.55))
```

```python
# Enrichment setup
ov.utils.download_pathway_database()
pathway_dict = ov.utils.geneset_prepare('genesets/WikiPathways_2019_Mouse.txt', organism='Mouse')

rnk = dds.ranking2gsea()
gsea_obj = ov.bulk.pyGSEA(rnk, pathway_dict)
enrich_res = gsea_obj.enrichment()

gsea_obj.plot_enrichment(num=10, node_size=[10, 20, 30],
                        fig_title='Wiki Pathway Enrichment',
                        fig_xlabel='Fractions of genes',
                        figsize=(2, 4), cmap='YlGnBu',
                        text_knock=2, text_maxsize=30,
                        cax_loc=[2.5, 0.45, 0.5, 0.02],
                        bbox_to_anchor_used=(-0.25, -13), node_diameter=10)

fig = gsea_obj.plot_gsea(term_num=1,
                         gene_set_title='Matrix Metalloproteinases',
                         figsize=(3, 4), cmap='RdBu_r',
                         title_fontsize=14, title_y=0.95)
```
