---
title: Using OmicVerse MCP with Claude Code
---

# Using OmicVerse MCP with Claude Code — Step by Step

This page focuses on the Claude Code workflow: how to connect OmicVerse MCP, verify the tools, run a standard analysis, and inspect the results. Broader deployment details, the full tool catalog, and runtime internals now live in the dedicated MCP pages.

## Related Pages

- Complete onboarding: [Full Start](t_mcp_full_start.md)
- Deployment patterns: [Clients and Deployment](t_mcp_clients.md)
- Full tool inventory: [Tool Catalog](t_mcp_tools.md)
- Runtime behavior and troubleshooting: [Runtime and Troubleshooting](t_mcp_runtime.md)
- Flags and JSON-RPC reference: [Reference](t_mcp_reference.md)

## Minimal Claude Code Setup

### Project-level `.mcp.json` with `stdio`

```json
{
  "mcpServers": {
    "omicverse-local": {
      "type": "stdio",
      "command": "python",
      "args": [
        "-m", "omicverse.mcp",
        "--phase", "P0+P0.5",
        "--persist-dir", "/tmp/ov_persist_local"
      ],
      "env": {}
    }
  }
}
```

### Global shortcut

```bash
claude mcp add omicverse -- python -m omicverse.mcp --phase P0+P0.5
```

### Local HTTP option

Start OmicVerse MCP:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl \
python -m omicverse.mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8765 \
  --http-path /mcp \
  --phase P0+P0.5
```

Then point Claude Code at it:

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

## Step-by-Step Walkthrough

### Step 1: Launch Claude Code

Navigate to your project directory and start Claude Code:

```bash
cd /path/to/your/project
claude
```

### Step 2: Verify tools are loaded

Ask:

```text
List all available OmicVerse MCP tools
```

![step1_img](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260309173629775.png)

### Step 3: Load your data

Ask:

```text
Load the pbmc3k.h5ad file
```

Or use a built-in dataset:

```text
Load the built-in seqfish dataset
```

![step2_img1](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260309174104707.png)

The `adata_id` is a server-side reference. Claude tracks it automatically.

![step2_img2](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260309174238355.png)

### Step 4: Quality control

Ask:

```text
Run quality control on the data
```

Claude calls `ov.pp.qc`, which computes per-cell metrics such as:

- `n_genes`
- `n_counts`
- `pct_counts_mt`

![step3_img1](https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/20260309174643057.png)

### Step 5: Run preprocessing

Ask:

```text
Run the standard preprocessing: scale, PCA with 50 components, build neighbors, compute UMAP, and do Leiden clustering at resolution 1.0
```

Claude runs the tools in sequence:

1. `ov.pp.scale`
2. `ov.pp.pca`
3. `ov.pp.neighbors`
4. `ov.pp.umap`
5. `ov.pp.leiden`

### Automatic prerequisite checking

The MCP server enforces the correct ordering. If Claude tries to skip a step, the server returns an error such as:

```json
{"ok": false, "error_code": "missing_data_requirements", "suggested_next_tools": ["ov.pp.scale"]}
```

Claude can use `suggested_next_tools` to self-correct.

### Step 6: Visualize

Ask:

```text
Plot the UMAP colored by Leiden clusters
Show me a violin plot of n_genes grouped by leiden cluster
Create a dot plot for genes CD3D, CD79A, LYZ, NKG7 grouped by leiden
```

### Step 7: Marker analysis

Ask:

```text
Find marker genes for each Leiden cluster and show me the top 5 per cluster
Plot a marker gene dotplot
Run COSG to rank marker genes
Perform pathway enrichment on the marker genes
Plot the pathway enrichment results
```

### Step 8: Save and restore

Ask:

```text
Save the current dataset to disk
```

Later:

```text
Restore the dataset from /path/to/file.h5ad
```

## Complete Conversation Example

```text
You: Load the file pbmc3k.h5ad

Claude: Loaded AnnData with 2700 cells x 32738 genes.

You: Run QC, preprocessing, and cluster the cells

Claude: I'll run the full pipeline step by step.
  [calls ov.pp.qc]
  [calls ov.pp.scale]
  [calls ov.pp.pca]
  [calls ov.pp.neighbors]
  [calls ov.pp.umap]
  [calls ov.pp.leiden]

You: Show me the UMAP

Claude: [calls ov.pl.embedding]

You: Find marker genes and show top 3 per cluster

Claude: [calls ov.single.find_markers]
  [calls ov.single.get_markers]

You: Save the results

Claude: [calls ov.persist_adata]
```

## P2 Workflows

If you need advanced class-backed tools, start OmicVerse with:

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

This enables workflows such as:

- `ov.bulk.pydeg`
- `ov.single.pyscsa`
- `ov.single.metacell`
- `ov.single.dct`
- `ov.utils.lda_topic`

### pyDEG

Typical lifecycle:

1. `ov.bulk.pydeg` with `create`
2. `ov.bulk.pydeg` with `run`
3. `ov.bulk.pydeg` with `results`
4. `ov.bulk.pydeg` with `destroy`

### pySCSA

Typical lifecycle:

1. `ov.single.pyscsa` with `create`
2. `ov.single.pyscsa` with `annotate`
3. `ov.single.pyscsa` with `destroy`

### MetaCell

Typical lifecycle:

1. `ov.single.metacell` with `create`
2. `ov.single.metacell` with `train`
3. `ov.single.metacell` with `predict`
4. `ov.single.metacell` with `destroy`

## Short Troubleshooting List

- Missing tools: ask Claude to run `ov.list_tools`
- Missing dependencies: ask Claude to run `ov.describe_tool`
- Lost session state: reload or `ov.restore_adata`
- Long-running task confusion: inspect [Runtime and Troubleshooting](t_mcp_runtime.md)
