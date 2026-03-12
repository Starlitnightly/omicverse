---
name: string-protein-interaction-analysis-with-omicverse
title: STRING protein interaction analysis with omicverse
description: "STRING protein-protein interaction network analysis with pyPPI: query STRING database, build PPI graphs, expand with add_nodes, and visualize styled networks for bulk gene lists."
---

# STRING Protein Interaction Analysis with OmicVerse

## Overview
Use this skill when the user has a gene list and wants to explore protein-protein interactions via the STRING database. The workflow covers species selection, STRING API queries, network construction, and styled visualization through `ov.bulk.pyPPI`.

## Instructions

### 1. Set up libraries
```python
import omicverse as ov
ov.style()  # or ov.plot_set()
```

### 2. Collect and validate gene inputs
```python
gene_list = ['FAA4', 'POX1', 'FAT1', 'FAS2', 'FAS1', 'FAA1', 'OLE1', 'YJU3', 'TGL3', 'INA1', 'TGL5']

# Remove duplicates and validate
gene_list = list(dict.fromkeys(gene_list))  # preserves order
assert len(gene_list) >= 2, "Need at least 2 genes for PPI analysis"
```

### 3. Assign metadata for plotting
```python
# Map genes to types and colours for the network figure
gene_type_dict = dict(zip(gene_list, ['Lipid_synthesis'] * 5 + ['Lipid_transport'] * 6))
gene_color_dict = dict(zip(gene_list, ['#F7828A'] * 5 + ['#9CCCA4'] * 6))
```
Consistent group labels and colours improve legend readability. Every gene in `gene_list` must appear in both dictionaries.

### 4. Query STRING interactions
```python
G_res = ov.bulk.string_interaction(gene_list, species_id)
print(G_res.head())
```
Inspect the DataFrame for `combined_score` and evidence channels to verify coverage before building the network.

### 5. Construct and visualise the network
```python
ppi = ov.bulk.pyPPI(
    gene=gene_list,
    gene_type_dict=gene_type_dict,
    gene_color_dict=gene_color_dict,
    species=species_id,
)
ppi.interaction_analysis()
ppi.plot_network()
```

## Species ID Reference

STRING requires NCBI taxonomy integer IDs, not species names. The agent must map the user's species to the correct ID.

| Species | Taxonomy ID | Gene Symbol Format |
|---------|------------|-------------------|
| Human | 9606 | Official HGNC symbols (e.g., TP53, BRCA1) |
| Mouse | 10090 | Official MGI symbols (e.g., Trp53, Brca1) |
| Rat | 10116 | Official RGD symbols |
| Yeast (S. cerevisiae) | 4932 | Systematic names (e.g., YOR317W) or standard names (e.g., FAA4) |
| Zebrafish | 7955 | ZFIN symbols |
| Drosophila | 7227 | FlyBase symbols |
| C. elegans | 6239 | WormBase symbols |
| Arabidopsis | 3702 | TAIR symbols |

## Critical API Reference

### Expanding sparse networks with `add_nodes`

Small gene lists (<10 genes) often produce disconnected networks because the query genes may not directly interact. The `add_nodes` parameter asks STRING to include its top predicted interaction partners.

```python
# For sparse networks: expand by adding STRING's top predicted partners
ppi.interaction_analysis(add_nodes=5)  # adds up to 5 STRING-predicted partners

# For focused networks: no expansion (default)
ppi.interaction_analysis()  # only edges between input genes
```

Use `add_nodes` when the initial network is disconnected or sparse. The added nodes are real proteins from STRING's database, but they may not be biologically relevant to your specific study—verify them before including in publications.

### Gene symbol format must match the species

```python
# CORRECT for human — official HGNC symbols
gene_list = ['TP53', 'BRCA1', 'MDM2']
G_res = ov.bulk.string_interaction(gene_list, 9606)

# WRONG — Ensembl IDs won't match STRING's symbol index
# gene_list = ['ENSG00000141510', 'ENSG00000012048']  # No interactions returned!
```

If genes are in Ensembl format, map them to symbols first (e.g., via `ov.bulk.Gene_mapping()`).

## Defensive Validation Patterns

```python
# Validate gene list
assert gene_list and len(gene_list) >= 2, "Need at least 2 genes for PPI"
gene_list = list(dict.fromkeys(gene_list))  # deduplicate

# Verify all genes appear in metadata dicts
for g in gene_list:
    assert g in gene_type_dict, f"Gene '{g}' missing from gene_type_dict"
    assert g in gene_color_dict, f"Gene '{g}' missing from gene_color_dict"

# Verify species_id is a valid integer
assert isinstance(species_id, int) and species_id > 0, f"species_id must be a positive integer, got {species_id}"

# After query: check if interactions were found
G_res = ov.bulk.string_interaction(gene_list, species_id)
if G_res is None or len(G_res) == 0:
    print("WARNING: No STRING interactions found. Check species_id and gene symbol format.")
```

## Troubleshooting

- **No interactions returned (empty DataFrame)**: Check that `species_id` matches the gene symbol format. Yeast uses systematic names or standard gene names, not human-style symbols. Verify at string-db.org manually.
- **`HTTPError 429` (rate-limited)**: STRING limits API requests. Wait 60 seconds between queries, or provide a cached interaction table from a previous run.
- **Gene not found in STRING**: The gene symbol may not exist in STRING's database for that species. Map Ensembl IDs to gene symbols first using `ov.bulk.Gene_mapping()`.
- **Network plot has disconnected nodes**: Use `add_nodes=5` (or higher) in `interaction_analysis()` to expand the network with STRING-predicted partners. Alternatively, lower the `combined_score` threshold.
- **`KeyError` in gene_color_dict during plotting**: Every gene in `gene_list` must have an entry in both `gene_type_dict` and `gene_color_dict`. After adding nodes with `add_nodes`, the expanded gene list may include new genes—update the dictionaries accordingly.
- **Network plot too dense/cluttered**: For large gene lists (>50 genes), consider filtering to a subset of top DEGs or hub genes before building the PPI network.

## Examples
- "Retrieve STRING interactions for my yeast fatty acid genes and plot the network with two colour-coded groups."
- "Build a human PPI network for my top 20 DEGs, expand with 5 predicted partners, and highlight up/down-regulated genes."
- "Download the STRING edge table for my mouse gene panel and colour nodes by WGCNA module."

## References
- Tutorial notebook: `t_network.ipynb`
- Quick copy/paste commands: [`reference.md`](reference.md)
