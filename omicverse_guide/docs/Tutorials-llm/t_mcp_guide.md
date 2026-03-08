---
title: OmicVerse MCP Server Guide
---

# OmicVerse MCP Server

The OmicVerse MCP server exposes registered analysis functions as tools via the [Model Context Protocol](https://modelcontextprotocol.io/). Any MCP-compatible client — Claude Code, Claude Desktop, or a custom script — can discover and call these tools to run single-cell analysis pipelines without writing Python code directly.

!!! tip "When to use OmicVerse MCP"

    You want an AI assistant (e.g. Claude) to run OmicVerse analysis steps on your behalf, passing data and parameters through natural language instead of manual scripting.

---

## Installation & Startup

### Install

```bash
# Production install with MCP dependencies
pip install omicverse[mcp]

# Development install
pip install -e ".[mcp]"

# Verify
python -m omicverse.mcp --version
```

### Start the Server

```bash
# Method 1: python module
python -m omicverse.mcp

# Method 2: CLI entrypoint (after pip install)
omicverse-mcp

# Core pipeline only (9 tools)
python -m omicverse.mcp --phase P0

# Default: P0 + P0.5 (15 tools)
python -m omicverse.mcp --phase P0+P0.5

# All tools including class-backed P2
python -m omicverse.mcp --phase P0+P0.5+P2

# Custom session and persistence
python -m omicverse.mcp --session-id my_analysis --persist-dir /data/ov_persist
```

The server uses **stdio transport** (JSON-RPC over stdin/stdout). All log output goes to stderr.

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | `P0+P0.5` | Rollout phase(s) to expose |
| `--session-id` | `default` | Logical session identifier for handle isolation |
| `--persist-dir` | (tempdir) | Directory for persisting AnnData via `ov.persist_adata` |
| `--max-adata` | `50` | Max AnnData handles per session |
| `--max-artifacts` | `200` | Max artifact handles per session |
| `--version` | — | Show version and exit |

---

## Phase System

Tools are organized into three rollout phases. You select which phases to expose at server startup with `--phase`.

### P0 — Core Pipeline (9 tools)

The minimal single-cell preprocessing chain.

| Tool | Description |
|------|-------------|
| `ov.utils.read` | Load a data file (`.h5ad`, etc.), returns an `adata_id` |
| `ov.utils.store_layers` | Snapshot the current X matrix into `adata.uns` |
| `ov.utils.retrieve_layers` | Restore X matrix from a stored snapshot |
| `ov.pp.qc` | Quality control metrics (gene counts, UMI counts, mito %) |
| `ov.pp.scale` | Scale to unit variance; adds a `scaled` layer |
| `ov.pp.pca` | PCA (requires `scaled` layer) |
| `ov.pp.neighbors` | Build neighborhood graph (requires `X_pca`) |
| `ov.pp.umap` | Compute UMAP embedding (requires neighbors) |
| `ov.pp.leiden` | Leiden clustering (requires neighbors) |

### P0.5 — Analysis & Visualization (+6 tools)

Marker gene detection and plotting. Included by default.

| Tool | Description |
|------|-------------|
| `ov.single.find_markers` | Find marker genes per cluster |
| `ov.single.get_markers` | Extract top markers as a table |
| `ov.pl.embedding` | Plot UMAP/tSNE embedding |
| `ov.pl.violin` | Violin plot |
| `ov.pl.dotplot` | Dot plot |
| `ov.pl.markers_dotplot` | Marker genes dot plot |

### P2 — Class-backed Tools (+5 tools)

Advanced analysis tools that use a multi-action interface. These are **availability-gated** — they appear in tool listings but may not run if their dependencies are missing.

| Tool | Description | Requires |
|------|-------------|----------|
| `ov.bulk.pydeg` | Differential expression (pyDEG / DESeq2) | — |
| `ov.single.pyscsa` | Automated cell type annotation (pySCSA) | — |
| `ov.single.metacell` | Metacell construction | `SEACells` package |
| `ov.single.dct` | Differential cell composition | `pertpy` package |
| `ov.utils.lda_topic` | LDA topic modeling | `mira` package |

!!! note "Runtime environment status"

    OmicVerse validates tool availability against tested dependency stacks:

    - **core-runtime**: verified (anndata, scanpy, scipy)
    - **scientific-runtime**: verified (+ scvelo, squidpy)
    - **extended-runtime**: constrained (SEACells works; mira-multiome is currently blocked)

    Use `ov.describe_tool` to check whether a specific tool can run in your environment.

---

## Minimal Usage Walkthrough

This walkthrough runs the full P0 single-cell pipeline: load data, QC, scale, PCA, neighbors, UMAP, Leiden.

### Step 1: Load Data

```json
{"tool_name": "ov.utils.read", "arguments": {"path": "pbmc3k.h5ad"}}
```

Response:

```json
{
  "ok": true,
  "tool_name": "ov.utils.read",
  "summary": "Loaded AnnData (2700x32738)",
  "outputs": [
    {"type": "object_ref", "ref_type": "adata", "ref_id": "adata_a1b2c3d4e5f6"}
  ]
}
```

Save the `ref_id` value — you'll pass it as `adata_id` to every subsequent tool.

### Step 2: Quality Control

```json
{"tool_name": "ov.pp.qc", "arguments": {"adata_id": "adata_a1b2c3d4e5f6"}}
```

Response includes `state_updates` showing what was added:

```json
{
  "ok": true,
  "state_updates": {
    "produced": {"obs": ["n_genes", "n_counts", "pct_counts_mt"], "var": ["mt"]}
  }
}
```

### Step 3: Scale

```json
{"tool_name": "ov.pp.scale", "arguments": {"adata_id": "adata_a1b2c3d4e5f6"}}
```

Produces: `{"layers": ["scaled"]}`.

### Step 4: PCA

```json
{"tool_name": "ov.pp.pca", "arguments": {"adata_id": "adata_a1b2c3d4e5f6", "n_pcs": 50}}
```

Produces: `{"obsm": ["X_pca"], "varm": ["PCs"], "uns": ["pca"]}`.

!!! warning "Prerequisite enforcement"

    If you skip Step 3 (scale), PCA returns an error with guidance:
    ```json
    {
      "ok": false,
      "error_code": "missing_data_requirements",
      "message": "Missing required data: layers=['scaled']",
      "suggested_next_tools": ["ov.pp.scale"]
    }
    ```
    Follow the `suggested_next_tools` to fix the issue.

### Step 5: Neighbors

```json
{"tool_name": "ov.pp.neighbors", "arguments": {"adata_id": "adata_a1b2c3d4e5f6"}}
```

Produces: `{"obsp": ["distances", "connectivities"], "uns": ["neighbors"]}`.

### Step 6: UMAP

```json
{"tool_name": "ov.pp.umap", "arguments": {"adata_id": "adata_a1b2c3d4e5f6"}}
```

Produces: `{"obsm": ["X_umap"]}`.

### Step 7: Leiden Clustering

```json
{"tool_name": "ov.pp.leiden", "arguments": {"adata_id": "adata_a1b2c3d4e5f6", "resolution": 1.0}}
```

Produces: `{"obs": ["leiden"]}`.

### JSON-RPC Protocol Example

At the protocol level, MCP uses JSON-RPC. Here's what `tools/list` and `tools/call` look like:

**List available tools:**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

**Call a tool:**

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "ov.utils.read",
    "arguments": {"path": "data.h5ad"}
  }
}
```

!!! tip

    When using Claude Code or Claude Desktop, you don't need to write JSON-RPC manually — the client handles the protocol for you. The examples above are for custom MCP client implementations.

---

## Sessions, Handles & Persistence

The MCP server manages your data through three types of **handles**.

### adata_id

When you load a dataset with `ov.utils.read`, the data lives server-side in memory. You receive an `adata_id` string (e.g. `"adata_a1b2c3d4e5f6"`) that references it. All analysis tools accept `adata_id` as a parameter — the actual AnnData object never crosses the MCP protocol boundary.

- **Obtain**: call `ov.utils.read`, get `ref_id` from the response
- **Use**: pass as `adata_id` to any tool that needs data
- **Persist**: call `ov.persist_adata` to save to disk as `.h5ad` + `.meta.json`
- **Restore**: call `ov.restore_adata` to reload from a `.h5ad` file

### instance_id

Class-backed tools (P2) create server-side instances. For example, `ov.bulk.pydeg` with `action: "create"` returns an `instance_id`. You then pass this ID to subsequent actions (`run`, `results`, `destroy`).

!!! warning

    Instance handles are **memory-only** — they cannot be persisted and are lost when the server restarts.

### artifact_id

Artifacts are references to files produced during analysis — plots, tables, exports. Plotting tools automatically register image artifacts. You can also manually register files with `ov.register_artifact`.

### Persistence Example

```json
// Save to disk
{"tool_name": "ov.persist_adata", "arguments": {"adata_id": "adata_a1b2c3d4e5f6"}}
// Response: {"path": "/data/ov_persist/adata_a1b2c3d4e5f6.h5ad", "metadata_path": "...meta.json"}

// Restore in a new session
{"tool_name": "ov.restore_adata", "arguments": {"path": "/data/ov_persist/adata_a1b2c3d4e5f6.h5ad"}}
// Response: {"ref_id": "adata_new_handle_id"}
```

| Handle Type | Persistable | How to Recover |
|-------------|-------------|----------------|
| `adata` | Yes | `ov.persist_adata` / `ov.restore_adata` |
| `artifact` | Yes (file on disk) | File path remains valid |
| `instance` | No | Must recreate after restart |

---

## Meta Tools

20 built-in meta tools are always available, regardless of phase selection.

### Discovery

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.list_tools` | List available tools | `category` (optional), `execution_class` (optional) |
| `ov.search_tools` | Search tools by keyword | `query` (required), `max_results` (optional) |
| `ov.describe_tool` | Full description, parameters, availability | `tool_name` (required) |

### Session Management

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.get_session` | Current session info (ID, handle counts, persist dir) | — |
| `ov.list_handles` | All handles in current session | `type` (optional: `adata`, `artifact`, `instance`) |
| `ov.persist_adata` | Save AnnData to `.h5ad` with JSON sidecar | `adata_id` (required), `path` (optional) |
| `ov.restore_adata` | Restore AnnData from `.h5ad` file | `path` (required), `adata_id` (optional) |

### Observability

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.get_metrics` | Aggregated session metrics (handle counts, tool stats) | `scope` (optional) |
| `ov.list_events` | Recent session events | `limit`, `event_type`, `tool_name` (all optional) |
| `ov.get_trace` | Details of a single tool call trace | `trace_id` (required) |
| `ov.list_traces` | Recent tool call traces with timing | `limit`, `tool_name`, `ok` (all optional) |

### Runtime Safety

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.get_limits` | Quota and TTL configuration with usage counts | — |
| `ov.cleanup_runtime` | Manual cleanup of expired data | `target`, `dry_run`, `delete_files` (all optional) |
| `ov.get_health` | Lightweight health summary with quota warnings | — |

### Artifact Management

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `ov.list_artifacts` | List artifacts with optional filters | `artifact_type`, `content_type`, `source_tool`, `limit` |
| `ov.describe_artifact` | Full artifact metadata including file status | `artifact_id` (required) |
| `ov.register_artifact` | Register an existing file as a session artifact | `path` (required), `artifact_type`, `content_type` |
| `ov.delete_artifact` | Delete artifact handle, optionally delete file | `artifact_id` (required), `delete_file` (optional) |
| `ov.cleanup_artifacts` | Batch cleanup (dry-run by default) | `artifact_type`, `older_than_seconds`, `dry_run` |
| `ov.export_artifacts_manifest` | Export all session artifacts as JSON | — |

---

## Client Configuration

### Claude Code / Claude Desktop

Add the following to your MCP server configuration (`claude_desktop_config.json` or Claude Code settings):

```json
{
  "mcpServers": {
    "omicverse": {
      "command": "python",
      "args": ["-m", "omicverse.mcp", "--phase", "P0+P0.5"],
      "env": {}
    }
  }
}
```

Or using the CLI entrypoint:

```json
{
  "mcpServers": {
    "omicverse": {
      "command": "omicverse-mcp",
      "args": ["--phase", "P0+P0.5"],
      "env": {}
    }
  }
}
```

!!! tip "Adding persistence"

    Append `"--persist-dir", "/path/to/persist"` to the `args` array to enable durable AnnData storage across sessions.

### Generic stdio MCP Client

Any MCP client that speaks JSON-RPC over stdio can connect:

```python
import subprocess
import json

proc = subprocess.Popen(
    ["python", "-m", "omicverse.mcp"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

# List available tools
request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
proc.stdin.write(json.dumps(request) + "\n")
proc.stdin.flush()

response = json.loads(proc.stdout.readline())
print(f"Available tools: {len(response['result']['tools'])}")

# Call a tool
call_request = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "ov.utils.read",
        "arguments": {"path": "pbmc3k.h5ad"},
    },
}
proc.stdin.write(json.dumps(call_request) + "\n")
proc.stdin.flush()

result = json.loads(proc.stdout.readline())
print(result)
```

---

## Limitations & Current Status

- **P2 tools are availability-gated**: They appear in tool listings but may return `tool_unavailable` if runtime dependencies are missing. Always check with `ov.describe_tool` before calling a P2 tool.
- **extended-runtime is constrained**: SEACells tools work, but `mira-multiome` is currently blocked (heavy dependencies: torch, pyro-ppl, MOODS-python). Not all P2 tools can run in every environment.
- **Instance handles are ephemeral**: Class instances (P2) are lost on server restart and cannot be persisted.
- **stdio transport only**: The server supports stdio (JSON-RPC) transport. HTTP/SSE transport is not available.
- **Single-process**: The server runs single-threaded. Concurrent tool calls are processed sequentially.
- **No result streaming**: Large outputs (e.g. marker tables) are returned as complete JSON.

---

## Troubleshooting

**Tools not visible**

- Check the `--phase` flag. `P0` exposes 9 tools; `P0+P0.5` (default) exposes 15; `P0+P0.5+P2` exposes 20.
- Use `ov.list_tools` to see what's available in the current session.

**Missing adata_id**

```json
{"ok": false, "error_code": "missing_session_object", "message": "adata_id 'adata_xxx' not found"}
```

- Call `ov.utils.read` first to load data and obtain an `adata_id`.
- Use `ov.list_handles` to see all handles in your current session.

**Prerequisites not met**

```json
{"ok": false, "error_code": "missing_data_requirements", "suggested_next_tools": ["ov.pp.scale"]}
```

The server checks that required data exists before running a tool:

- `ov.pp.pca` requires `layers["scaled"]` — run `ov.pp.scale` first
- `ov.pp.neighbors` requires `obsm["X_pca"]` — run `ov.pp.pca` first
- `ov.pp.umap` requires `uns["neighbors"]` — run `ov.pp.neighbors` first

Follow the `suggested_next_tools` in the error response.

**Tool unavailable**

```json
{"ok": false, "error_code": "tool_unavailable"}
```

- The tool exists but its dependencies are not installed.
- Run `ov.describe_tool` with the tool name to see the specific requirement.
- Install the missing package and restart the server.

**Server exits immediately**

- Check stderr for import errors.
- Ensure `mcp>=1.0` is installed: `pip install "mcp>=1.0"`.
