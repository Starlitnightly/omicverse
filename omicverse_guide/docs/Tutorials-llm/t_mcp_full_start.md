---
title: OmicVerse MCP Full Start
---

# OmicVerse MCP Full Start

This page is the complete getting-started path for OmicVerse MCP. It sits between the short [Quick Start](t_mcp_quickstart.md) and the more specialized reference pages.

## Who This Is For

Read this page if:

- the quick start feels too compressed
- you want one continuous setup-to-analysis walkthrough
- you want both the commands and the reasoning behind the default choices

## What You Will End Up With

By the end of this page, you will have:

- an OmicVerse MCP server running
- a client connected through `stdio` or local HTTP
- a dataset loaded as an `adata_id`
- a standard preprocessing workflow completed
- a plot and marker analysis generated
- a persisted `.h5ad` you can restore later

## Step 1: Install

```bash
pip install omicverse[mcp]
python -m omicverse.mcp --version
```

If you are working from a clone:

```bash
pip install -e ".[mcp]"
```

## Step 2: Choose a Transport

OmicVerse MCP supports two local transports.

### Common phase selections

```bash
# Core only
python -m omicverse.mcp --phase P0

# Default
python -m omicverse.mcp --phase P0+P0.5

# Include advanced P2 tools
python -m omicverse.mcp --phase P0+P0.5+P2
```

### Option A: `stdio`

Use this when you want the simplest setup and want Claude to own the MCP process lifecycle.

```bash
python -m omicverse.mcp --phase P0+P0.5
```

### Option B: `streamable-http`

Use this when you want a separately managed MCP process, clearer logs, or easier reconnect behavior.

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl \
python -m omicverse.mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8765 \
  --http-path /mcp \
  --phase P0+P0.5
```

### Default Recommendation

Start with `stdio`. Move to local HTTP when debugging, using larger datasets, or keeping one MCP process alive across reconnects.

## Step 3: Connect a Client

### Claude Code with `stdio`

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

If you want Claude Code to launch the full P2 rollout directly:

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

### Claude Code with local HTTP

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

If you use Claude Code, you can also read the [Claude Code walkthrough](t_mcp_claude_code.md).

## Step 4: Understand What Crosses the Boundary

`AnnData` objects stay on the server side. The client does not receive the full in-memory object. Instead, it receives lightweight handles such as:

- `adata_id` for datasets
- `artifact_id` for plots and files
- `instance_id` for P2 class-backed tools

That means a typical successful load looks like:

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

## Step 5: Load Data

You can start from a built-in dataset:

```text
Load the built-in pbmc3k dataset
```

or from a local file:

```text
Load the pbmc3k.h5ad file
```

Built-in loaders currently include:

- `ov.datasets.pbmc3k`
- `ov.datasets.pbmc8k`
- `ov.datasets.seqfish`

## Step 6: Inspect Before You Analyze

Before asking for preprocessing, inspect the current dataset. This is important because it lets the model see what is actually present in `obs`, `var`, `obsm`, and `uns`.

Useful prompts:

```text
Describe the current adata
What is the first gene in var?
Does CD3D exist in var_names?
Inspect adata.obsm
Inspect adata.uns
Show value counts for leiden
```

These route to:

- `ov.adata.describe`
- `ov.adata.peek`
- `ov.adata.find_var`
- `ov.adata.inspect`
- `ov.adata.value_counts`

## Step 7: Run the Standard Workflow

The default analysis chain is:

1. `ov.pp.qc`
2. `ov.pp.scale`
3. `ov.pp.pca`
4. `ov.pp.neighbors`
5. `ov.pp.umap`
6. `ov.pp.leiden`

Natural-language request:

```text
Run QC, scale, PCA with 50 components, build neighbors, compute UMAP, and run Leiden clustering at resolution 1.0
```

The server enforces prerequisites. For example, PCA requires the `scaled` layer, and neighbors requires `X_pca`.

## Step 8: Plot and Interpret

After the embedding and clustering are ready, ask for plots:

```text
Plot the UMAP colored by leiden
Show me a violin plot of n_genes grouped by leiden cluster
Create a dot plot for genes CD3D, CD79A, LYZ, NKG7 grouped by leiden
```

This typically uses:

- `ov.pl.embedding`
- `ov.pl.violin`
- `ov.pl.dotplot`

Plot outputs are registered as artifacts.

## Step 9: Marker Analysis and Enrichment

Once clusters exist, ask for marker analysis:

```text
Find marker genes for each Leiden cluster and show me the top 5 per cluster
Plot a marker gene dotplot
Run COSG to rank marker genes
Perform pathway enrichment on the marker genes
Plot the pathway enrichment results
```

This can use:

- `ov.single.find_markers`
- `ov.single.get_markers`
- `ov.pl.markers_dotplot`
- `ov.single.cosg`
- `ov.single.pathway_enrichment`
- `ov.single.pathway_enrichment_plot`

## Step 10: Persist and Restore

The analysis state is in memory until you persist it.

Save:

```text
Save the current dataset to disk
```

Restore later:

```text
Restore the dataset from /path/to/file.h5ad
```

Under the hood this uses:

- `ov.persist_adata`
- `ov.restore_adata`

## Optional Step 11: Move to P2

If you need advanced class-backed tools such as DEG, annotation, metacells, DCT, or topic modeling, start the server with:

```bash
python -m omicverse.mcp --phase P0+P0.5+P2
```

P2 tools may appear in tool listings but still be unavailable if their optional dependencies are not installed.

## Troubleshooting Checklist

### Tools are missing

- verify the `--phase`
- ask the client to run `ov.list_tools`

### A tool is unavailable

- ask the client to run `ov.describe_tool`
- check missing optional dependencies

### The dataset handle is gone

- the server may have restarted
- reload the data or use `ov.restore_adata`

### A long-running task is confusing to debug

- prefer local HTTP mode
- inspect `ov.list_traces`, `ov.get_trace`, `ov.list_events`, and `ov.get_health`

## Tips and Best Practices

1. Start with `P0+P0.5`. Only add `+P2` when you need advanced class-backed tools.
2. Use `--persist-dir` if you want saved datasets to survive across sessions.
3. Let the client track `adata_id` for you instead of managing handles manually.
4. Use `ov.adata.*` inspection tools before asking for interpretation of `obs`, `var`, `obsm`, or `uns`.
5. Use `ov.describe_tool` when you want prerequisite or availability details.
6. Use remote deployment when the dataset or dependency stack outgrows your local machine.
7. Persist before stopping if you plan to continue later.

## Where to Go Next

- Shortest path: [Quick Start](t_mcp_quickstart.md)
- Full tool inventory: [Tool Catalog](t_mcp_tools.md)
- Deployment patterns: [Clients & Deployment](t_mcp_clients.md)
- Runtime details: [Runtime & Troubleshooting](t_mcp_runtime.md)
- Exact flags and envelopes: [Reference](t_mcp_reference.md)
