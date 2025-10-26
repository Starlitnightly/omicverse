# STRING PPI quick commands

```python
import omicverse as ov

ov.utils.ov_plot_set()

gene_list = ['FAA4', 'POX1', 'FAT1', 'FAS2', 'FAS1', 'FAA1', 'OLE1', 'YJU3', 'TGL3', 'INA1', 'TGL5']

gene_type_dict = dict(zip(gene_list, ['Type1'] * 5 + ['Type2'] * 6))
gene_color_dict = dict(zip(gene_list, ['#F7828A'] * 5 + ['#9CCCA4'] * 6))

G_res = ov.bulk.string_interaction(gene_list, 4932)
print(G_res.head())

ppi = ov.bulk.pyPPI(gene=gene_list,
                    gene_type_dict=gene_type_dict,
                    gene_color_dict=gene_color_dict,
                    species=4932)
ppi.interaction_analysis()
ppi.plot_network()
```
