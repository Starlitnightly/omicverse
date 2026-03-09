---
title: Using OmicVerse MCP with Claude Code
---

# Using OmicVerse MCP with Claude Code — Step by Step

This tutorial walks you through setting up and using OmicVerse MCP tools inside [Claude Code](https://docs.anthropic.com/en/docs/claude-code), Anthropic's CLI for Claude. By the end, you'll be able to load single-cell data, run preprocessing, clustering, marker detection, and visualization — all through natural language conversation with Claude.

---

## What is OmicVerse MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard that lets AI assistants call external tools. OmicVerse exposes its analysis functions as MCP tools, so Claude Code can run bioinformatics pipelines on your behalf — no Python coding required.

### Architecture

```
┌─────────────────┐     stdio (JSON-RPC)     ┌──────────────────────┐
│                 │◄────────────────────────►│                      │
│   Claude Code   │   adata_id / results     │  OmicVerse MCP Server│
│   (your laptop) │                          │  (local or remote)   │
│                 │  Only handle IDs and     │                      │
│                 │  summaries cross the     │  ┌────────────────┐  │
│                 │  wire — raw data stays   │  │ AnnData objects│  │
│                 │  on the server side      │  │ (in memory)    │  │
│                 │                          │  └────────────────┘  │
└─────────────────┘                          │  ┌────────────────┐  │
                                             │  │ Artifacts      │  │
                                             │  │ (plots, tables)│  │
                                             │  └────────────────┘  │
                                             └──────────────────────┘
```

**Key design**: AnnData objects never cross the protocol boundary. The server holds them in memory and returns lightweight `adata_id` handles. All tool calls reference these handles, keeping communication fast regardless of dataset size.

---

## Prerequisites

Before you start, make sure you have:

- **Claude Code** installed ([installation guide](https://docs.anthropic.com/en/docs/claude-code/getting-started))
- **Python >= 3.10**
- A single-cell dataset (e.g. `pbmc3k.h5ad`)

### Install OmicVerse with MCP support

```bash
# Production install
pip install omicverse[mcp]

# Or development install (if you cloned the repo)
pip install -e ".[mcp]"

# Verify the MCP server works
python -m omicverse.mcp --version
```

---

## Deployment: Local vs Remote

The OmicVerse MCP server supports two deployment modes. You can use one or both simultaneously.

### Local Deployment

Run the MCP server on your own machine. Best for:

- Quick exploration with local datasets
- Fast iteration (no network latency)
- Standard P0+P0.5 workflows (preprocessing, clustering, visualization)

#### Project-level config (recommended)

Create a `.mcp.json` file in your **project root** (the directory where you run `claude`):

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

#### Global config

To make OmicVerse tools available across all projects:

```bash
claude mcp add omicverse -- python -m omicverse.mcp --phase P0+P0.5
```

### Remote Deployment (SSH)

Run the MCP server on a remote machine (GPU server, HPC cluster, cloud instance). Claude Code connects via SSH and pipes the MCP protocol over the SSH tunnel. Best for:

- Large datasets that live on remote storage
- GPU-accelerated tools or high-memory analysis
- Centralized persistence shared across team members
- Full P2 class tools that require extended dependencies

```json
{
  "mcpServers": {
    "omicverse-remote": {
      "type": "stdio",
      "command": "ssh",
      "args": [
        "-i", "/path/to/ssh/key",
        "-p", "22",
        "-o", "StrictHostKeyChecking=no",
        "user@remote-host",
        "cd /path/to/project && python -m omicverse.mcp --phase P0+P0.5+P2 --persist-dir /data/ov_persist"
      ],
      "env": {}
    }
  }
}
```

**How it works**:

1. Claude Code opens an SSH connection to the remote host
2. Launches the Python MCP server process on the remote machine
3. Pipes stdin/stdout over the SSH tunnel for JSON-RPC communication
4. All computation and data stay on the remote server
5. Only handle IDs (`adata_id`) and result summaries travel over the wire

!!! tip "Remote environment tips"

    - Use `conda`/`micromamba` paths if Python is in a virtual environment: `~/micromamba/bin/python -m omicverse.mcp`
    - Use `-p <port>` for non-standard SSH ports
    - Use `-i /path/to/key` for SSH key authentication (passwordless login required)
    - The remote machine must have `omicverse[mcp]` installed

### Local + Remote Side-by-Side

You can define **both** servers in a single `.mcp.json`. Claude Code will have access to tools from both servers simultaneously:

```json
{
  "mcpServers": {
    "omicverse-local": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "omicverse.mcp", "--phase", "P0+P0.5", "--persist-dir", "/tmp/ov_persist_local"],
      "env": {}
    },
    "omicverse-remote": {
      "type": "stdio",
      "command": "ssh",
      "args": [
        "-i", "/path/to/ssh/key",
        "-p", "22",
        "user@gpu-server",
        "cd /data/project && python -m omicverse.mcp --phase P0+P0.5+P2 --persist-dir /data/ov_persist"
      ],
      "env": {}
    }
  }
}
```

In this setup, the local server handles quick preprocessing tasks, while the remote server handles heavy P2 analysis with extended dependencies.

!!! warning "Handle isolation"

    Each server has its own session. An `adata_id` from the local server cannot be used with the remote server, and vice versa. Use `ov.persist_adata` to save data on one server and `ov.restore_adata` to reload it on the other (if the file paths are accessible).

### Deployment Comparison

| | Local | Remote (SSH) |
|---|---|---|
| **Latency** | Instant | SSH overhead (~50-200ms per call) |
| **Data location** | Must be on your machine | Stays on remote filesystem |
| **Compute** | Limited by laptop resources | GPU, high RAM, many cores |
| **Typical phase** | `P0+P0.5` (15 tools) | `P0+P0.5+P2` (20+ tools) |
| **Dependencies** | Core only | Can install extended deps |
| **Persistence** | Local disk | Shared/durable remote storage |
| **Setup** | `pip install omicverse[mcp]` | SSH key + remote install |

### Configuration Reference

All CLI flags for the MCP server:

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | `P0+P0.5` | Rollout phase(s) to expose (e.g. `P0`, `P0+P0.5`, `P0+P0.5+P2`) |
| `--session-id` | `default` | Logical session identifier for handle isolation |
| `--persist-dir` | (temp dir) | Directory for persisting AnnData via `ov.persist_adata` |
| `--max-adata` | `50` | Maximum AnnData handles per session |
| `--max-artifacts` | `200` | Maximum artifact handles per session |
| `--version` | — | Show OmicVerse version and exit |

---

## Tool Catalog: What's Inside

The OmicVerse MCP server exposes up to **40 tools**, organized into phases and meta-tool categories.

### Phase System Overview

| Phase | Tools | Description |
|-------|-------|-------------|
| **P0** | 9 | Core single-cell pipeline: load, QC, preprocess, cluster |
| **P0.5** | 6 | Marker genes + visualization |
| **P2** | 5 | Advanced class-backed tools (DEG, annotation, metacells) |
| **Meta** | 20 | Always-on: discovery, session, observability, artifacts, safety |
| **Total** | **40** | |

Use `--phase` to control which analysis tools are exposed. Meta tools are always available regardless of phase.

### P0 — Core Pipeline (9 tools)

The minimal single-cell preprocessing chain. Available by default.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.utils.read` | Load data file (`.h5ad`, `.csv`, `.tsv`, `.txt`), returns `adata_id` | `path` (required) |
| `ov.utils.store_layers` | Snapshot the current X matrix into `adata.uns` for later retrieval | `adata_id` |
| `ov.utils.retrieve_layers` | Restore X matrix from a stored snapshot | `adata_id` |
| `ov.pp.qc` | Quality control: gene counts, UMI counts, mitochondrial % | `adata_id` |
| `ov.pp.scale` | Scale to unit variance; adds `scaled` layer | `adata_id` |
| `ov.pp.pca` | PCA dimensionality reduction (requires `scaled` layer) | `adata_id`, `n_pcs` (default: 50) |
| `ov.pp.neighbors` | Build k-nearest-neighbor graph (requires `X_pca`) | `adata_id` |
| `ov.pp.umap` | Compute UMAP embedding (requires neighbors) | `adata_id` |
| `ov.pp.leiden` | Leiden community detection clustering (requires neighbors) | `adata_id`, `resolution` (default: 1.0) |

**Pipeline order**: `read` → `qc` → `scale` → `pca` → `neighbors` → `umap` / `leiden`

The server enforces this ordering via prerequisite checks. If you skip a step, the error response includes `suggested_next_tools`.

### P0.5 — Analysis & Visualization (6 tools)

Marker gene detection and plotting. Included by default with `P0+P0.5`.

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.single.find_markers` | Find marker genes per cluster (differential expression) | `adata_id` |
| `ov.single.get_markers` | Extract top markers as a table | `adata_id`, `n_markers` |
| `ov.pl.embedding` | Plot UMAP/tSNE embedding | `adata_id`, `color`, `save` |
| `ov.pl.violin` | Violin plot for gene expression or QC metrics | `adata_id`, `keys`, `groupby` |
| `ov.pl.dotplot` | Dot plot for gene expression across groups | `adata_id`, `var_names`, `groupby` |
| `ov.pl.markers_dotplot` | Marker genes dot plot (uses `find_markers` results) | `adata_id` |

Visualization tools automatically register output images as **artifacts** in the session.

### P2 — Class-backed Tools (5 tools)

Advanced analysis tools with a **multi-action lifecycle**. Enable with `--phase P0+P0.5+P2`.

| Tool | Description | Actions | Extra Dependencies |
|------|-------------|---------|-------------------|
| `ov.bulk.pydeg` | Differential expression analysis (pyDEG) | `create` → `run` → `results` → `destroy` | — |
| `ov.single.pyscsa` | Automated cell type annotation (pySCSA) | `create` → `annotate` → `destroy` | — |
| `ov.single.metacell` | Metacell construction (SEACells) | `create` → `train` → `predict` → `destroy` | `SEACells` package |
| `ov.single.dct` | Differential cell type composition | `create` → `run` → `results` → `destroy` | `pertpy` package |
| `ov.utils.lda_topic` | LDA topic modeling | `create` → `run` → `results` → `destroy` | `mira` package |

**Availability gating**: P2 tools appear in `ov.list_tools` but may return `tool_unavailable` if their dependencies are missing. Use `ov.describe_tool` to check.

**Multi-action lifecycle**: Unlike P0/P0.5 tools (single function call), P2 tools create a server-side instance that you interact with through multiple actions:

```
1. create  → instantiate the analyzer, get instance_id
2. run     → execute the analysis
3. results → retrieve output tables
4. destroy → free memory
```

Claude handles this lifecycle automatically — you just say "run DEG analysis" and Claude manages the create/run/results/destroy sequence.

### Meta Tools (20 tools, always available)

Meta tools are available regardless of `--phase`. They manage the server itself.

#### Discovery (3 tools)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.list_tools` | List available tools | `category`, `execution_class` (optional) |
| `ov.search_tools` | Search tools by keyword across names and descriptions | `query` (required), `max_results` |
| `ov.describe_tool` | Full description, parameters, prerequisites, availability | `tool_name` (required) |

#### Session Management (4 tools)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.get_session` | Current session info: ID, handle counts, persist directory | — |
| `ov.list_handles` | All handles (adata, artifact, instance) in current session | `type` (optional filter) |
| `ov.persist_adata` | Save AnnData to `.h5ad` with `.meta.json` sidecar | `adata_id` (required), `path` |
| `ov.restore_adata` | Restore AnnData from a `.h5ad` file | `path` (required), `adata_id` |

#### Observability (4 tools)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.get_metrics` | Aggregated session metrics: handle counts, tool call stats | `scope` (`session` or `tools`) |
| `ov.list_events` | Recent session events (handle lifecycle, tool calls) | `limit`, `event_type`, `tool_name` |
| `ov.get_trace` | Details of a single tool call trace (timing, handles) | `trace_id` (required) |
| `ov.list_traces` | Recent tool call traces with duration and status | `limit`, `tool_name`, `ok` |

#### Artifact Management (6 tools)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.list_artifacts` | List artifacts with optional filters | `artifact_type`, `content_type`, `source_tool`, `limit` |
| `ov.describe_artifact` | Full metadata including file existence and size | `artifact_id` (required) |
| `ov.register_artifact` | Manually register an existing file as a session artifact | `path` (required), `artifact_type`, `content_type` |
| `ov.delete_artifact` | Delete artifact handle, optionally delete file from disk | `artifact_id` (required), `delete_file` |
| `ov.cleanup_artifacts` | Batch cleanup by type/age (dry-run by default) | `artifact_type`, `older_than_seconds`, `dry_run` |
| `ov.export_artifacts_manifest` | Export all session artifacts as a JSON manifest | — |

Artifact types: `file`, `image`, `table`, `json`, `plot`, `report`, `export`

#### Runtime Safety (3 tools)

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.get_limits` | Current quota configuration and usage counts | — |
| `ov.cleanup_runtime` | Manual cleanup of expired events, traces, artifacts | `target`, `dry_run`, `delete_files` |
| `ov.get_health` | Lightweight health summary with quota proximity warnings | — |

### Dependency Tiers

Different deployment environments may have different packages installed, which affects tool availability:

| Tier | Packages | Tools Unlocked | Install |
|------|----------|----------------|---------|
| **Core** | `anndata`, `scanpy`, `numpy`, `scipy`, `matplotlib` | P0 + P0.5 (15 tools) | `pip install omicverse[mcp]` |
| **Scientific** | + `scvelo`, `squidpy` | velocity & spatial analysis | `pip install -r requirements/mcp-scientific-runtime.txt` |
| **Extended** | + `SEACells`, `pertpy` | P2 class tools (metacell, DCT) | `pip install -r requirements/mcp-extended-runtime.txt` |

!!! note "Checking your environment"

    Ask Claude: **"Check the health of the MCP server"** — Claude calls `ov.get_health` and `ov.get_limits` to show quota usage and any warnings.

    Ask Claude: **"Is the metacell tool available?"** — Claude calls `ov.describe_tool` to check runtime availability and tells you which package is missing.

---

## Step-by-Step Walkthrough

### Step 1: Launch Claude Code

Navigate to your project directory (where the data and `.mcp.json` live) and start Claude Code:

```bash
cd /path/to/your/project
claude
```

Claude Code automatically detects the `.mcp.json` and starts the OmicVerse MCP server.

!!! note "Verify tools are loaded"

    Ask Claude: **"List all available OmicVerse MCP tools"** — Claude will call `ov.list_tools` and show you everything that's available.

### Step 2: Load Your Data

> **You:** Load the pbmc3k.h5ad file

Claude calls `ov.utils.read` and returns something like:

```
Loaded AnnData (2700 x 32738). adata_id: adata_a1b2c3d4e5f6
```

The `adata_id` is a server-side reference. Claude tracks it automatically — you don't need to remember it.

### Step 3: Quality Control

> **You:** Run quality control on the data

Claude calls `ov.pp.qc`, which computes per-cell metrics:

- `n_genes` — number of genes detected
- `n_counts` — total UMI counts
- `pct_counts_mt` — mitochondrial gene percentage

### Step 4: Preprocessing Pipeline

> **You:** Run the standard preprocessing: scale, PCA with 50 components, build neighbors, compute UMAP, and do Leiden clustering at resolution 1.0

Claude runs these tools **in sequence** (each depends on the previous):

1. `ov.pp.scale` — normalize to unit variance, creates `scaled` layer
2. `ov.pp.pca` — PCA dimensionality reduction (50 components)
3. `ov.pp.neighbors` — build k-nearest-neighbor graph
4. `ov.pp.umap` — compute UMAP embedding
5. `ov.pp.leiden` — Leiden community detection clustering

!!! info "Automatic prerequisite checking"

    The MCP server enforces the correct ordering. If Claude tries to skip a step (e.g. running PCA without scaling first), the server returns:
    ```json
    {"ok": false, "error_code": "missing_data_requirements", "suggested_next_tools": ["ov.pp.scale"]}
    ```
    Claude reads `suggested_next_tools` and self-corrects.

### Step 5: Visualization

> **You:** Plot the UMAP colored by Leiden clusters

Claude calls `ov.pl.embedding` and generates a plot image. The image is automatically registered as an artifact.

Other plot types:

> **You:** Show me a violin plot of n_genes grouped by leiden cluster

> **You:** Create a dot plot for genes CD3D, CD79A, LYZ, NKG7 grouped by leiden

### Step 6: Marker Genes

> **You:** Find marker genes for each Leiden cluster and show me the top 5 per cluster

Claude calls `ov.single.find_markers` then `ov.single.get_markers`, and presents the results as a table.

> **You:** Plot a marker gene dotplot

### Step 7: Save Your Results

> **You:** Save the current dataset to disk

Claude calls `ov.persist_adata`, which writes:

- `adata_a1b2c3d4e5f6.h5ad` — the full AnnData object
- `adata_a1b2c3d4e5f6.meta.json` — sidecar with session ID, timestamps, and metadata

To restore in a later session:

> **You:** Restore the dataset from /tmp/ov_persist_local/adata_a1b2c3d4e5f6.h5ad

### Complete Conversation Example

```
You: Load the file pbmc3k.h5ad

Claude: [calls ov.utils.read]
  Loaded AnnData with 2700 cells x 32738 genes.

You: Run QC, preprocessing, and cluster the cells

Claude: I'll run the full pipeline step by step.
  [calls ov.pp.qc]        -> QC metrics computed
  [calls ov.pp.scale]     -> Scaled layer created
  [calls ov.pp.pca]       -> PCA computed (50 components)
  [calls ov.pp.neighbors] -> Neighbor graph built
  [calls ov.pp.umap]      -> UMAP embedding computed
  [calls ov.pp.leiden]    -> 8 Leiden clusters identified

You: Show me the UMAP

Claude: [calls ov.pl.embedding]
  Here's the UMAP plot colored by Leiden clusters.
  [displays image]

You: Find marker genes and show top 3 per cluster

Claude: [calls ov.single.find_markers]
  [calls ov.single.get_markers]
  Here are the top 3 marker genes per cluster:
  | Cluster | Gene 1 | Gene 2 | Gene 3 |
  |---------|--------|--------|--------|
  | 0       | LYZ    | S100A9 | S100A8 |
  | 1       | IL7R   | MAL    | LTB    |
  | ...     | ...    | ...    | ...    |

You: Save the results

Claude: [calls ov.persist_adata]
  Dataset saved to /tmp/ov_persist_local/adata_a1b2c3d4e5f6.h5ad
```

---

## Advanced: P2 Class-backed Tools

For differential expression, cell annotation, and metacell analysis, enable P2 tools:

```json
"args": ["-m", "omicverse.mcp", "--phase", "P0+P0.5+P2"]
```

### pyDEG — Differential Expression

> **You:** Run differential expression analysis comparing clusters 0 and 1

Claude manages the full lifecycle:

```
1. ov.bulk.pydeg  action:"create"  -> instance_id: inst_abc123
2. ov.bulk.pydeg  action:"run"     -> DE test executed
3. ov.bulk.pydeg  action:"results" -> table of DE genes returned
4. ov.bulk.pydeg  action:"destroy" -> memory freed
```

### pySCSA — Cell Type Annotation

> **You:** Annotate cell types for the Leiden clusters

Claude calls:

```
1. ov.single.pyscsa  action:"create"   -> configure annotation (species, tissue)
2. ov.single.pyscsa  action:"annotate" -> run annotation, returns cell type labels
3. ov.single.pyscsa  action:"destroy"  -> clean up
```

### MetaCell — Metacell Construction

Requires the `SEACells` package. Claude will check availability before attempting:

```
1. ov.single.metacell  action:"create"  -> initialize SEACells model
2. ov.single.metacell  action:"train"   -> train metacell model
3. ov.single.metacell  action:"predict" -> assign cells to metacells
4. ov.single.metacell  action:"destroy" -> clean up
```

!!! warning "Instance handles are ephemeral"

    P2 class instances (`instance_id`) are **memory-only** — they are lost when the server restarts. If the server crashes mid-analysis, you need to recreate the instance. AnnData handles can be recovered via `ov.persist_adata` / `ov.restore_adata`.

---

## Session Management & Observability

### Handle Types

The MCP server manages three types of handles:

| Handle Type | ID Prefix | Persistable | Max per Session | Example |
|-------------|-----------|-------------|-----------------|---------|
| `adata` | `adata_` | Yes (`.h5ad`) | 50 | Loaded datasets |
| `artifact` | `artifact_` | Yes (file path) | 200 | Plot images, CSV exports |
| `instance` | `inst_` | No (memory-only) | 50 | P2 class instances |

### Check Session Status

> **You:** What's the current session status?

Claude calls `ov.get_session` and `ov.list_handles`:

```json
{
  "session_id": "default",
  "persist_dir": "/tmp/ov_persist_local",
  "adata_count": 1,
  "artifact_count": 3,
  "instance_count": 0
}
```

### View Execution History

> **You:** Show me the recent tool call traces

Claude calls `ov.list_traces`:

```json
[
  {"trace_id": "abc...", "tool_name": "ov.pp.pca", "duration_ms": 245.3, "ok": true},
  {"trace_id": "def...", "tool_name": "ov.pp.qc",  "duration_ms": 102.1, "ok": true}
]
```

For detailed per-call debugging: **"Show me the trace for abc..."** → Claude calls `ov.get_trace`.

### Aggregate Metrics

> **You:** Show me session metrics

Claude calls `ov.get_metrics`:

```json
{
  "adata_count": 1,
  "artifact_count": 3,
  "tool_calls_total": 8,
  "tool_calls_failed": 0,
  "artifacts_registered_total": 3
}
```

### Artifact Management

Visualization tools automatically register images as artifacts. You can also manage artifacts manually:

> **You:** List all image artifacts from this session

> **You:** Clean up artifacts older than 1 hour (show me what would be deleted first)

Claude calls `ov.cleanup_artifacts` with `dry_run=true` first, then asks for confirmation.

> **You:** Export the full artifact manifest as JSON

---

## Tips & Best Practices

1. **Start with `P0+P0.5`**. Only add `+P2` when you need DEG, annotation, or metacell tools.

2. **Use `--persist-dir`** pointing to a stable directory. This ensures saved datasets survive across sessions. Default is a temp dir that may be cleaned up by the OS.

3. **Let Claude handle the `adata_id`**. You don't need to track handle IDs — Claude remembers them from previous tool responses.

4. **Use `ov.search_tools`** when you don't know the exact tool name. Ask Claude: *"Search for tools related to clustering"* and it will find them.

5. **Check prerequisites with `ov.describe_tool`**. Ask: *"What do I need to run before PCA?"* — Claude will look up the dependency chain.

6. **Use remote deployment for large datasets**. If your data is too large for your laptop, set up an SSH-based remote server.

7. **Persist before stopping**. Always call `ov.persist_adata` before ending a session if you want to continue later. AnnData objects are in-memory and lost on server restart.

---

## Troubleshooting

### MCP server not found / tools not available

- Verify `.mcp.json` is in your project root (the directory where you run `claude`)
- Check that `python -m omicverse.mcp --version` works in your terminal
- Ensure `mcp>=1.0` is installed: `pip install "mcp>=1.0"`

### Tools seem to be missing

- The default phase (`P0+P0.5`) exposes 15 analysis tools + 20 meta tools = 35 total
- For P2 tools, add `--phase P0+P0.5+P2` to your config
- Ask Claude to run `ov.list_tools` to see what's currently available

### P2 tool returns "tool_unavailable"

- The tool exists but its dependencies are not installed
- Ask Claude to run `ov.describe_tool` to see the specific requirement
- Install the missing package and restart Claude Code

### "adata_id not found"

- The server restarted and in-memory handles were lost
- Load data again with `ov.utils.read`, or restore from disk with `ov.restore_adata`

### Preprocessing step fails with "missing data requirements"

- The MCP server enforces pipeline ordering
- The error includes `suggested_next_tools` — Claude follows these automatically

### Remote server connection fails

- Verify SSH connectivity: `ssh -i /path/to/key -p <port> user@host echo ok`
- Ensure `omicverse[mcp]` is installed on the remote machine
- Check that the remote Python path in `.mcp.json` is correct
- Look at Claude Code's stderr output for SSH errors

### Server crashes or connection lost

- Check `stderr` output for Python tracebacks
- Restart Claude Code — it reconnects to the MCP server automatically
- Restore previously persisted data with `ov.restore_adata`

---

## What's Next

- **[MCP Server Reference](t_mcp_guide.md)** — Complete technical reference for all tools, JSON-RPC protocol details, response envelope format, and error codes
- **[Single-cell preprocessing tutorial](../Tutorials-single/t_preprocess_cpu.ipynb)** — The same pipeline using Python code directly
- **[OmicVerse documentation](https://omicverse.readthedocs.io/)** — Full library documentation
