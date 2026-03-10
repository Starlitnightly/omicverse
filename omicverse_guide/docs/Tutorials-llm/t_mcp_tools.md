---
title: OmicVerse MCP Tool Catalog
---

# OmicVerse MCP Tool Catalog

The current MCP server exposes up to 58 tools:

- `P0`: 15 analysis tools
- `P0.5`: 13 additional analysis tools
- `P2`: 5 advanced class-backed tools
- Meta tools: 25 always-on tools

## Phase System Overview

| Phase | Tools | Description |
|-------|-------|-------------|
| **P0** | 15 | Core single-cell pipeline, built-in datasets, and essential preprocessing |
| **P0.5** | 13 | Marker genes, pathway analysis, and visualization |
| **P2** | 5 | Advanced class-backed tools |
| **Meta** | 25 | Discovery, AnnData inspection, session, observability, artifacts, safety |
| **Total** | **58** | |

## P0: Core Pipeline & Data Access

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.utils.read` | Load a file and return an `adata_id` | `path` |
| `ov.datasets.pbmc3k` | Load built-in PBMC3k | — |
| `ov.datasets.pbmc8k` | Load built-in PBMC8k | — |
| `ov.datasets.seqfish` | Load built-in seqFISH | — |
| `ov.pp.qc` | Compute QC metrics | `adata_id` |
| `ov.pp.filter_cells` | Filter cells | `adata_id`, thresholds |
| `ov.pp.filter_genes` | Filter genes | `adata_id`, thresholds |
| `ov.pp.log1p` | Log-transform expression | `adata_id` |
| `ov.pp.highly_variable_genes` | Identify HVGs | `adata_id` |
| `ov.pp.scale` | Scale expression and create `scaled` layer | `adata_id` |
| `ov.pp.pca` | Compute PCA | `adata_id`, `n_pcs` |
| `ov.pp.neighbors` | Build neighbor graph | `adata_id` |
| `ov.pp.umap` | Compute UMAP | `adata_id` |
| `ov.pp.leiden` | Leiden clustering | `adata_id`, `resolution` |
| `ov.pp.louvain` | Louvain clustering | `adata_id`, `resolution` |

**Pipeline order**: `read` or `ov.datasets.*` -> `qc` / filtering -> `log1p` / HVG -> `scale` -> `pca` -> `neighbors` -> `umap` / clustering

Typical prompts:

- `Load the built-in seqfish dataset`
- `Run QC and identify highly variable genes`
- `Run PCA with 50 components`

## P0.5: Analysis & Visualization

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.single.find_markers` | Differential marker discovery | `adata_id` |
| `ov.single.get_markers` | Extract marker table | `adata_id`, `n_markers` |
| `ov.single.cosg` | Rank markers with COSG | `adata_id` |
| `ov.single.pathway_enrichment` | Pathway enrichment | marker set or result handle |
| `ov.single.pathway_enrichment_plot` | Plot enrichment results | enrichment result |
| `ov.pl.embedding` | Generic embedding plot | `adata_id`, `color`, `save` |
| `ov.pl.violin` | Violin plot | `adata_id`, `keys`, `groupby` |
| `ov.pl.dotplot` | Dotplot | `adata_id`, `var_names`, `groupby` |
| `ov.pl.markers_dotplot` | Marker-gene dotplot | `adata_id` |
| `ov.pl.umap` | UMAP plotting helper | `adata_id`, `color` |
| `ov.pl.tsne` | tSNE plotting helper | `adata_id`, `color` |
| `ov.pl.rank_genes_groups_dotplot` | Ranked marker dotplot | `adata_id` |
| `ov.pl.marker_heatmap` | Marker heatmap | `adata_id` |

Typical prompts:

- `Find marker genes for each Leiden cluster`
- `Run COSG to rank marker genes`
- `Perform pathway enrichment on the markers`
- `Plot a marker heatmap`

## P2: Advanced Class-backed Tools

These tools expose a lifecycle such as `create -> run -> results -> destroy`.

| Tool | Description |
|------|-------------|
| `ov.bulk.pydeg` | Differential expression |
| `ov.single.pyscsa` | Automated cell annotation |
| `ov.single.metacell` | Metacell construction |
| `ov.single.dct` | Differential cell composition |
| `ov.utils.lda_topic` | Topic modeling |

These tools may appear in `ov.list_tools` but still return `tool_unavailable` if optional dependencies are missing.

P2 tools use a multi-action lifecycle such as:

1. `create`
2. `run` / `annotate` / `train`
3. `results` / `predict`
4. `destroy`

### Runtime environment status

OmicVerse validates tool availability against tested dependency stacks:

- **core-runtime**: verified (`anndata`, `scanpy`, `scipy`)
- **scientific-runtime**: verified (`+ scvelo`, `squidpy`)
- **extended-runtime**: constrained (`SEACells` works; `mira-multiome` is currently blocked)

Use `ov.describe_tool` to check whether a specific tool can run in your environment.

## AnnData Inspection Tools

These tools are important because they let the client inspect the current dataset instead of guessing.

| Tool | Use |
|------|-----|
| `ov.adata.describe` | High-level summary of shape, names, layers, embeddings, metadata |
| `ov.adata.peek` | Preview a slot such as `obs`, `var`, `obsm`, `layers`, or `uns` |
| `ov.adata.find_var` | Search `var_names` for genes such as `CD3D` |
| `ov.adata.value_counts` | Count values in an `obs` column |
| `ov.adata.inspect` | Inspect nested entries in `obsm`, `obsp`, `layers`, `varm`, or `uns` |

Typical prompts:

- `Describe the current adata`
- `What is the first gene in var?`
- `Does CD3D exist in var_names?`
- `Show value counts for leiden`
- `Inspect adata.obsm X_pca`
- `Inspect adata.uns`

## Meta Tools

Meta tools are always available regardless of phase selection.

### Discovery

- `ov.list_tools`
- `ov.search_tools`
- `ov.describe_tool`

### Session & Persistence

- `ov.get_session`
- `ov.list_handles`
- `ov.persist_adata`
- `ov.restore_adata`

### Observability

- `ov.get_metrics`
- `ov.list_events`
- `ov.get_trace`
- `ov.list_traces`

### Artifacts

- `ov.list_artifacts`
- `ov.describe_artifact`
- `ov.register_artifact`
- `ov.delete_artifact`
- `ov.cleanup_artifacts`
- `ov.export_artifacts_manifest`

Artifact types: `file`, `image`, `table`, `json`, `plot`, `report`, `export`

### Runtime Safety

- `ov.get_limits`
- `ov.cleanup_runtime`
- `ov.get_health`

## Dependency Tiers

| Tier | Packages | Tools Unlocked | Install |
|------|----------|----------------|---------|
| **Core** | `anndata`, `scanpy`, `numpy`, `scipy`, `matplotlib` | P0 + P0.5 | `pip install omicverse[mcp]` |
| **Scientific** | + `scvelo`, `squidpy` | velocity & spatial analysis | `pip install -r requirements/mcp-scientific-runtime.txt` |
| **Extended** | + `SEACells`, `pertpy` | P2 class tools | `pip install -r requirements/mcp-extended-runtime.txt` |

### Checking Your Environment

- Ask the client to run `ov.get_health` for a lightweight runtime summary.
- Ask it to run `ov.get_limits` for quota and handle usage.
- Ask it to run `ov.describe_tool` for one specific tool if you want availability and dependency details.

## Choosing Between Similar Tools

- Prefer `ov.datasets.*` over file loading when a built-in tutorial dataset is enough.
- Prefer `ov.adata.*` before asking the model to infer what is inside `obs`, `var`, `obsm`, or `uns`.
- Prefer `ov.pl.embedding` when you want one generic embedding plotting entrypoint.
- Use `ov.pl.umap` or `ov.pl.tsne` when you want the plot type to be explicit.

## Related Pages

- Setup: [Quick Start](t_mcp_quickstart.md)
- Deployment: [Clients & Deployment](t_mcp_clients.md)
- Runtime behavior: [Runtime & Troubleshooting](t_mcp_runtime.md)
