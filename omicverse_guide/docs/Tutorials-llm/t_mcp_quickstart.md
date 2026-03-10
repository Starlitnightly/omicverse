---
title: OmicVerse MCP Quick Start
---

# OmicVerse MCP Quick Start

This is the shortest path to a working OmicVerse MCP setup.

## Prerequisites

Before you start, make sure you have:

- Claude Code or another MCP client
- Python `>=3.10`
- either a local `.h5ad` file or willingness to use a built-in dataset

## Install

```bash
pip install omicverse[mcp]
python -m omicverse.mcp --version
```

## Start the Server

### Common phase selections

```bash
# Core only
python -m omicverse.mcp --phase P0

# Default: core + analysis/visualization
python -m omicverse.mcp --phase P0+P0.5

# Full rollout including P2 class-backed tools
python -m omicverse.mcp --phase P0+P0.5+P2
```

### Default `stdio` mode

```bash
python -m omicverse.mcp --phase P0+P0.5
```

### Local HTTP mode

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl \
python -m omicverse.mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8765 \
  --http-path /mcp \
  --phase P0+P0.5
```

Use `stdio` by default. Use local HTTP when you want a separately managed MCP process or easier server-side debugging.

If you need advanced class-backed tools, use the same startup pattern with `--phase P0+P0.5+P2`.

## Claude Code Setup

### Option A: Claude launches OmicVerse directly

```json
{
  "mcpServers": {
    "omicverse": {
      "command": "python",
      "args": ["-m", "omicverse.mcp", "--phase", "P0+P0.5"]
    }
  }
}
```

For P2:

```json
{
  "mcpServers": {
    "omicverse": {
      "command": "python",
      "args": ["-m", "omicverse.mcp", "--phase", "P0+P0.5+P2"]
    }
  }
}
```

### Option B: Claude connects to your local HTTP MCP

```json
{
  "mcpServers": {
    "omicverse": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

More deployment patterns are in [Clients & Deployment](t_mcp_clients.md).

## First Commands to Try

### Load a built-in dataset

```text
Load the built-in pbmc3k dataset
```

Or load a file:

```text
Load the pbmc3k.h5ad file
```

### Inspect the dataset

```text
Describe the current adata
What is the first gene in var?
Does CD3D exist in var_names?
Inspect adata.uns
Inspect adata.obsm
```

### Run a standard preprocessing workflow

```text
Run QC, scale, PCA with 50 components, build neighbors, compute UMAP, and run Leiden clustering at resolution 1.0
```

### Plot and summarize

```text
Plot the UMAP colored by leiden
Find marker genes for each Leiden cluster
Plot a marker gene dotplot
```

## What the Server Is Actually Doing

Under the hood, OmicVerse MCP is calling tools such as:

1. `ov.datasets.pbmc3k` or `ov.utils.read`
2. `ov.pp.qc`
3. `ov.pp.scale`
4. `ov.pp.pca`
5. `ov.pp.neighbors`
6. `ov.pp.umap`
7. `ov.pp.leiden`

The dataset stays server-side and is referenced by `adata_id`.

## Minimal JSON Shape You Will See

Loading data typically returns an object reference like:

```json
{
  "ok": true,
  "tool_name": "ov.utils.read",
  "outputs": [
    {
      "type": "object_ref",
      "ref_type": "adata",
      "ref_id": "adata_a1b2c3d4e5f6"
    }
  ]
}
```

You normally do not need to manage the `adata_id` manually because Claude tracks it across turns.

## One Short Walkthrough

1. `Load the built-in pbmc3k dataset`
2. `Describe the current adata`
3. `Run QC, scale, PCA with 50 components, build neighbors, compute UMAP, and run Leiden clustering`
4. `Plot the UMAP colored by leiden`
5. `Find marker genes for each Leiden cluster and show the top 5`

## Next Pages

- Full onboarding path: [Full Start](t_mcp_full_start.md)
- Full tool list: [Tool Catalog](t_mcp_tools.md)
- Claude and deployment details: [Clients & Deployment](t_mcp_clients.md)
- Runtime behavior and troubleshooting: [Runtime & Troubleshooting](t_mcp_runtime.md)
