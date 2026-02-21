---
name: biocontext-mcp
title: BioContext External Database Queries via MCP
description: >
  Query external biomedical databases (STRING, UniProt, KEGG, Reactome,
  PanglaoDB, Open Targets, EuropePMC) in real-time using MCP tools.
  Use when analysis requires protein interactions, pathway data,
  cell type markers, or literature search.
---

# BioContext MCP Integration

## When to Use

Use MCP database queries when the user's request involves:
- **Protein interactions**: "find interaction partners for TP53", "PPI network for BRCA1"
- **Pathway lookup**: "what pathways involve EGFR", "KEGG pathway hsa04110"
- **Cell-type markers**: "marker genes for T cells", "PanglaoDB markers for macrophages"
- **Protein annotation**: "UniProt info for P53_HUMAN", "protein function of ENSG00000141510"
- **Literature search**: "recent papers on single-cell CRISPR screens"
- **Drug targets**: "Open Targets for BRAF", "disease associations for JAK2"
- **Reactome pathways**: "Reactome pathway for apoptosis"

Do **NOT** use MCP tools for:
- Standard OmicVerse analysis (QC, clustering, DEG, etc.)
- Operations that only need the local `adata` object
- Visualization or plotting tasks
- File I/O or data loading

## Available Tools

The `mcp_call(tool_name, arguments_dict)` function is available in the
execution environment when BioContext MCP is enabled.

### Core Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `string_interaction_partners` | Protein-protein interactions from STRING | `identifiers`, `species` (9606=human), `limit` |
| `string_functional_enrichment` | Functional enrichment via STRING | `identifiers`, `species` |
| `string_network_image` | STRING network visualization URL | `identifiers`, `species` |
| `uniprot_protein_lookup` | Protein details from UniProt | `accession` |
| `kegg_pathway_info` | KEGG pathway details | `pathway_id` |
| `kegg_gene_info` | KEGG gene annotation | `gene_id` |
| `reactome_pathway_info` | Reactome pathway | `pathway_id` |
| `panglao_cell_markers` | Cell-type markers from PanglaoDB | `cell_type`, `species` |
| `europepmc_search` | Literature search | `query`, `limit` |
| `opentargets_target_info` | Drug target info | `target_id` |
| `opentargets_disease_associations` | Disease-target links | `target_id` |

### Parameters Reference

**STRING tools** (`species` codes):
- `9606` = Human (default)
- `10090` = Mouse
- `10116` = Rat
- `7955` = Zebrafish

**PanglaoDB** (`species`):
- `"Hs"` = Human (default)
- `"Mm"` = Mouse

## Code Patterns

### Basic query
```python
# Query STRING for TP53 interactions
result = mcp_call("string_interaction_partners", {
    "identifiers": "TP53",
    "species": 9606,
    "limit": 25
})
print("Found " + str(len(result)) + " interaction partners")
```

### Multiple genes
```python
# Query interactions for a gene list
genes = ["TP53", "BRCA1", "EGFR"]
for gene in genes:
    partners = mcp_call("string_interaction_partners", {
        "identifiers": gene,
        "species": 9606,
        "limit": 10
    })
    print(gene + ": " + str(len(partners)) + " partners")
```

### Integrate with adata
```python
# Get marker genes for a cell type, then check overlap with adata
markers_result = mcp_call("panglao_cell_markers", {
    "cell_type": "T cells",
    "species": "Hs"
})
# Extract gene names from result
marker_genes = [m["official gene symbol"] for m in markers_result
                if "official gene symbol" in m]
# Check which markers are in our dataset
found = [g for g in marker_genes if g in adata.var_names]
print("Found " + str(len(found)) + "/" + str(len(marker_genes)) + " markers in dataset")
```

### Pathway enrichment context
```python
# Get KEGG pathway info to provide biological context
pathway_info = mcp_call("kegg_pathway_info", {"pathway_id": "hsa04110"})
print("Pathway: " + str(pathway_info.get("name", "unknown")))

# Combine with OmicVerse enrichment
import omicverse as ov
pathway_dict = ov.utils.geneset_prepare("your_geneset.gmt", organism="Human")
```

## Result Processing

MCP tool results are returned as parsed JSON (dict or list). Always:

1. **Check the result type**: `isinstance(result, dict)` or `isinstance(result, list)`
2. **Handle empty results**: `if not result: print("No results found")`
3. **Extract relevant fields**: Results vary by tool; inspect keys first
4. **Convert to DataFrame when useful**:
   ```python
   import pandas as pd
   if isinstance(result, list):
       df = pd.DataFrame(result)
   ```

## Caching

Results are automatically cached by the FilesystemContextManager.
- Repeated queries with identical parameters return cached results instantly
- Cache expires after 1 hour (configurable via `cache_ttl`)
- No special code needed; caching is transparent

## Common Workflows

### Gene-to-Network-to-Enrichment
```python
import omicverse as ov
import pandas as pd

# 1. Get interaction network from STRING
interactions = mcp_call("string_interaction_partners", {
    "identifiers": "TP53",
    "species": 9606,
    "limit": 50
})

# 2. Extract partner gene names
if isinstance(interactions, list):
    partner_genes = [p.get("preferredName", p.get("preferredName_B", ""))
                     for p in interactions]
elif isinstance(interactions, dict):
    partner_genes = list(interactions.keys())
partner_genes = [g for g in partner_genes if g]  # filter empty

# 3. Subset adata to network genes
network_genes = [g for g in partner_genes if g in adata.var_names]
if network_genes:
    adata_sub = adata[:, network_genes].copy()
    print("Subset to " + str(len(network_genes)) + " network genes")

# 4. Run downstream analysis on the subsetted data
ov.pp.scale(adata_sub)
ov.pp.pca(adata_sub)
```

### Cell-Type Annotation Validation
```python
# Validate cluster annotations against PanglaoDB markers
cell_types = ["T cells", "B cells", "Monocytes", "NK cells"]
marker_dict = {}
for ct in cell_types:
    result = mcp_call("panglao_cell_markers", {
        "cell_type": ct, "species": "Hs"
    })
    if isinstance(result, list):
        genes = [m.get("official gene symbol", "") for m in result]
        marker_dict[ct] = [g for g in genes if g in adata.var_names]
        print(ct + ": " + str(len(marker_dict[ct])) + " markers found in data")
```
