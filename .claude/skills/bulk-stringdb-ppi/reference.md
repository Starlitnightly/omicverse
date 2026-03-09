# STRING PPI Quick Commands

```python
import omicverse as ov

ov.utils.ov_plot_set()

# --- Gene list and metadata ---
gene_list = ['FAA4', 'POX1', 'FAT1', 'FAS2', 'FAS1', 'FAA1', 'OLE1', 'YJU3', 'TGL3', 'INA1', 'TGL5']
gene_list = list(dict.fromkeys(gene_list))  # Deduplicate while preserving order

# Map genes to functional groups and colours for styled network plot
gene_type_dict = dict(zip(gene_list, ['Synthesis'] * 5 + ['Transport'] * 6))
gene_color_dict = dict(zip(gene_list, ['#F7828A'] * 5 + ['#9CCCA4'] * 6))

# --- Query STRING ---
# species_id: 9606=Human, 10090=Mouse, 4932=Yeast, 10116=Rat, 7955=Zebrafish
species_id = 4932  # Yeast (S. cerevisiae)
G_res = ov.bulk.string_interaction(gene_list, species_id)
print(G_res.head())  # Check combined_score and evidence channels

# --- Build network ---
ppi = ov.bulk.pyPPI(
    gene=gene_list,
    gene_type_dict=gene_type_dict,
    gene_color_dict=gene_color_dict,
    species=species_id,
)

# Default: only edges between input genes
ppi.interaction_analysis()

# For sparse networks (<10 genes): expand with STRING-predicted partners
# ppi.interaction_analysis(add_nodes=5)  # Adds top 5 predicted interaction partners

# --- Visualize ---
ppi.plot_network()
```