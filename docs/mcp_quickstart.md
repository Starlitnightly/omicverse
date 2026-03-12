# OmicVerse MCP Server — Quickstart Guide

## Overview

The OmicVerse MCP server exposes registered analysis functions as tools via the [Model Context Protocol](https://modelcontextprotocol.io/). AI assistants (Claude Code, etc.) can discover and call these tools to run single-cell analysis pipelines.

- **Transport**: stdio (JSON-RPC over stdin/stdout)
- **Tool source**: auto-generated from the `@register_function` registry — no hand-maintained tool list
- **Session model**: AnnData objects live server-side, referenced by `adata_id` strings

## Installation

```bash
# Install OmicVerse with MCP dependencies
pip install -e ".[mcp]"

# Verify
python -m omicverse.mcp --help
```

## Starting the Server

```bash
# Method 1: python -m
python -m omicverse.mcp

# Method 2: CLI entrypoint (after pip install)
omicverse-mcp

# Options
python -m omicverse.mcp --phase P0         # P0 tools only (9 tools)
python -m omicverse.mcp --phase P0+P0.5    # default (15 tools)
```

The server starts on stdio transport. All log output goes to stderr; stdout is reserved for MCP protocol messages.

### Claude Code Integration

Add to your Claude Code MCP server configuration:

```json
{
  "mcpServers": {
    "omicverse": {
      "command": "python",
      "args": ["-m", "omicverse.mcp"]
    }
  }
}
```

## Available Tools

### P0 — Core Pipeline (9 tools)

| Tool Name | Description | Execution Class |
|-----------|-------------|-----------------|
| `ov.utils.read` | Load data file, returns `adata_id` | stateless |
| `ov.utils.store_layers` | Snapshot X matrix into `adata.uns` | adata |
| `ov.utils.retrieve_layers` | Restore X matrix from snapshot | adata |
| `ov.pp.qc` | Quality control metrics | adata |
| `ov.pp.scale` | Scale to unit variance, adds `scaled` layer | adata |
| `ov.pp.pca` | PCA (requires `scaled` layer) | adata |
| `ov.pp.neighbors` | Neighborhood graph (requires `X_pca`) | adata |
| `ov.pp.umap` | UMAP embedding (requires `neighbors`) | adata |
| `ov.pp.leiden` | Leiden clustering (requires `neighbors`) | adata |

### P0.5 — Analysis & Visualization (6 tools)

| Tool Name | Description | Execution Class |
|-----------|-------------|-----------------|
| `ov.single.find_markers` | Find marker genes per cluster | adata |
| `ov.single.get_markers` | Extract top markers as table | adata |
| `ov.pl.embedding` | Plot UMAP/tSNE embedding | adata |
| `ov.pl.violin` | Violin plot | adata |
| `ov.pl.dotplot` | Dot plot | adata |
| `ov.pl.markers_dotplot` | Marker genes dot plot | adata |

### P2 — Class-backed Tools (5 tools)

Class-backed tools use an **action-based interface**: a single tool exposes multiple actions via a required `action` parameter. Availability depends on runtime environment.

| Tool Name | Description | Status |
|-----------|-------------|--------|
| `ov.bulk.pydeg` | Differential expression analysis (pyDEG) | Available |
| `ov.single.pyscsa` | Automated cell type annotation (pySCSA) | Available |
| `ov.single.metacell` | Metacell construction (SEACells) | Requires `SEACells` package |
| `ov.single.dct` | Differential cell composition (DCT) | Deferred (requires `pertpy`) |
| `ov.utils.lda_topic` | LDA topic modeling | Deferred (requires `mira`) |

Use `--phase P0+P0.5+P2` to include these tools, or `ov.describe_tool` to check actions and availability.

### Meta Tools (always available)

Representative meta tools include:

- `ov.list_tools`
- `ov.describe_tool`
- `ov.get_session`
- `ov.get_health`
- `ov.get_limits`
- `ov.list_handles`
- `ov.list_artifacts`
- `ov.export_artifacts_manifest`

AnnData inspection meta tools are also available:

- `ov.adata.describe`
- `ov.adata.peek`
- `ov.adata.find_var`
- `ov.adata.value_counts`
- `ov.adata.inspect`

#### Discovery

| Tool Name | Description | Required Params |
|-----------|-------------|-----------------|
| `ov.list_tools` | List available tools | — (optional: `category`, `execution_class`) |
| `ov.search_tools` | Search tools by keyword | `query` (optional: `max_results`) |
| `ov.describe_tool` | Get full tool description | `tool_name` |

#### Session Management

| Tool Name | Description | Required Params |
|-----------|-------------|-----------------|
| `ov.get_session` | Current session info (ID, handle counts, persist dir) | — |
| `ov.list_handles` | All handles in current session | — (optional: `type` filter) |
| `ov.persist_adata` | Save adata to `.h5ad` with JSON sidecar | `adata_id` (optional: `path`) |
| `ov.restore_adata` | Restore adata from `.h5ad` file | `path` (optional: `adata_id`) |

#### Observability

| Tool Name | Description | Required Params |
|-----------|-------------|-----------------|
| `ov.get_metrics` | Aggregated session metrics (handle counts, tool call stats) | — (optional: `scope`) |
| `ov.list_events` | Recent session events (handle lifecycle, tool calls) | — (optional: `limit`, `event_type`, `tool_name`) |
| `ov.get_trace` | Single tool call trace details | `trace_id` |
| `ov.list_traces` | Recent tool call traces with timing and status | — (optional: `limit`, `tool_name`, `ok`) |

#### Artifact Management

| Tool Name | Description | Required Params |
|-----------|-------------|-----------------|
| `ov.list_artifacts` | List artifacts with optional filters | — (optional: `artifact_type`, `content_type`, `source_tool`, `limit`) |
| `ov.describe_artifact` | Full artifact metadata including file status | `artifact_id` |
| `ov.register_artifact` | Manually register a file as a session artifact | `path` (optional: `content_type`, `artifact_type`, `source_tool`, `metadata`) |
| `ov.delete_artifact` | Delete artifact handle, optionally the file | `artifact_id` (optional: `delete_file`) |
| `ov.cleanup_artifacts` | Batch cleanup with dry-run preview | — (optional: `artifact_type`, `older_than_seconds`, `delete_files`, `dry_run`) |
| `ov.export_artifacts_manifest` | Export all session artifacts as JSON manifest | — |

#### Runtime Safety

| Tool Name | Description | Required Params |
|-----------|-------------|-----------------|
| `ov.get_limits` | Get quota/TTL configuration and current usage | — |
| `ov.cleanup_runtime` | Manual cleanup of expired runtime data | — (optional: `target`, `dry_run`, `delete_files`) |
| `ov.get_health` | Lightweight health summary with quota warnings | — |

## Core Concept: Sessions

All handles belong to a **session**. Sessions provide isolation — handles from one session cannot be accessed by another.

- **Default session**: `SessionStore()` with no arguments creates session `"default"`. Existing workflows are unaffected.
- **Custom session**: Pass `--session-id my-session` at server start, or `SessionStore(session_id="my-session")` programmatically.
- **Cross-session access**: Attempting to access a handle from another session raises a structured error with `error_code: "cross_session_access"`.

Use `ov.get_session` to inspect the current session and `ov.list_handles` to enumerate all handles.

## Core Concept: Handle Lifecycle

The server manages three types of handles. Each has different persistence characteristics:

| Handle Type | ID Prefix | Persistable | Recovery Method |
|-------------|-----------|-------------|-----------------|
| `adata` | `adata_` | Yes | `ov.persist_adata` / `ov.restore_adata` |
| `artifact` | `artifact_` | Yes (path-based) | Rehydrate from existing file path |
| `instance` | `inst_` | No (memory-only) | Not recoverable after process restart |

- **adata**: In-memory AnnData objects. Can be explicitly persisted to `.h5ad` files with JSON sidecar metadata. Restored handles get new IDs.
- **artifact**: File references with metadata (path, content_type). The artifact metadata is tracked; the file itself exists on disk.
- **instance**: Ephemeral class instances (P2 tools). Lost on server restart. Attempting to persist returns `unsupported_persistence` error.

## Core Concept: `adata_id`

AnnData objects never cross the MCP protocol boundary. Instead:

1. `ov.utils.read` loads a file server-side and returns an `adata_id` (e.g. `"adata_a1b2c3d4e5f6"`)
2. All subsequent tools accept `adata_id` as a required parameter
3. Tools mutate the AnnData in-place and return state updates
4. The `adata_id` remains valid for the server's lifetime (or until explicitly persisted and the server restarts)

### Persisting and Restoring adata

Persistence is **explicit** — not automatic. Use `ov.persist_adata` to save and `ov.restore_adata` to reload:

```json
// 1. Persist to disk
{"tool_name": "ov.persist_adata", "arguments": {"adata_id": "adata_a1b2c3d4e5f6"}}
// Response: {"ok": true, "outputs": [{"data": {"adata_id": "adata_a1b2c3d4e5f6", "path": "/tmp/ov_persist_.../adata_a1b2c3d4e5f6.h5ad", "metadata_path": "/tmp/ov_persist_.../adata_a1b2c3d4e5f6.meta.json"}}]}

// 2. Restore from disk (new session or after restart)
{"tool_name": "ov.restore_adata", "arguments": {"path": "/tmp/ov_persist_.../adata_a1b2c3d4e5f6.h5ad"}}
// Response: {"ok": true, "outputs": [{"type": "object_ref", "ref_type": "adata", "ref_id": "adata_new_handle_id"}]}
```

The `.meta.json` sidecar file contains: `adata_id`, `session_id`, `file_path`, `content_type`, `created_at`, `persisted_at`, and `original_metadata`.

## Core Concept: `instance_id`

Class-backed tools (P2) create server-side instances. Each instance gets an `instance_id` (e.g. `"inst_abc123def456"`).

1. `action: "create"` instantiates the class and returns an `instance_id`
2. Subsequent actions (`run`, `results`, `annotate`, etc.) require the `instance_id`
3. `action: "destroy"` deletes the instance and frees memory
4. Multiple independent instances of the same tool can coexist
5. **Instance handles are memory-only** — they cannot be persisted or restored

### Example: pyDEG Lifecycle

```json
// 1. Create instance (requires adata_id)
{"tool_name": "ov.bulk.pydeg", "arguments": {"action": "create", "adata_id": "adata_a1b2c3d4e5f6"}}
// Response: {"outputs": [{"ref_type": "instance", "ref_id": "inst_abc123"}]}

// 2. Run analysis
{"tool_name": "ov.bulk.pydeg", "arguments": {"action": "run", "instance_id": "inst_abc123", "treatment_groups": ["T1","T2"], "control_groups": ["C1","C2"]}}

// 3. Get results
{"tool_name": "ov.bulk.pydeg", "arguments": {"action": "results", "instance_id": "inst_abc123"}}

// 4. Cleanup
{"tool_name": "ov.bulk.pydeg", "arguments": {"action": "destroy", "instance_id": "inst_abc123"}}
```

Use `ov.describe_tool` to see available actions for any class tool:

```json
{"tool_name": "ov.describe_tool", "arguments": {"tool_name": "ov.bulk.pydeg"}}
```

The response includes `class_actions` (list of available actions with parameters) and `availability` (whether the tool can run in the current environment).

## Walkthrough: P0 Pipeline

### Step 1: Load Data

```json
{"tool_name": "ov.utils.read", "arguments": {"path": "pbmc3k.h5ad"}}
```

Response:

```json
{
  "ok": true,
  "tool_name": "ov.utils.read",
  "summary": "Loaded AnnData (100x500)",
  "outputs": [
    {"type": "object_ref", "ref_type": "adata", "ref_id": "adata_a1b2c3d4e5f6"}
  ],
  "state_updates": {},
  "warnings": []
}
```

Save the `ref_id` — you'll pass it as `adata_id` to every subsequent tool.

### Step 2: Quality Control

```json
{"tool_name": "ov.pp.qc", "arguments": {"adata_id": "adata_a1b2c3d4e5f6"}}
```

Response:

```json
{
  "ok": true,
  "tool_name": "ov.pp.qc",
  "summary": "QC completed",
  "outputs": [
    {"type": "object_ref", "ref_type": "adata", "ref_id": "adata_a1b2c3d4e5f6"}
  ],
  "state_updates": {
    "produced": {"obs": ["n_genes", "n_counts", "pct_counts_mt"], "var": ["mt"]}
  },
  "warnings": []
}
```

### Step 3: Scale

```json
{"tool_name": "ov.pp.scale", "arguments": {"adata_id": "adata_a1b2c3d4e5f6"}}
```

Response reports `"produced": {"layers": ["scaled"]}`.

### Step 4: PCA

```json
{
  "tool_name": "ov.pp.pca",
  "arguments": {"adata_id": "adata_a1b2c3d4e5f6", "n_pcs": 50}
}
```

Response reports `"produced": {"obsm": ["X_pca"], "varm": ["PCs"], "uns": ["pca"]}`.

> If you skip Step 3 (scale), this call returns an error:
> ```json
> {
>   "ok": false,
>   "error_code": "missing_data_requirements",
>   "message": "Missing required data: layers=['scaled']",
>   "details": {"missing": {"layers": ["scaled"]}},
>   "suggested_next_tools": ["ov.pp.scale"]
> }
> ```

### Step 5: Neighbors

```json
{"tool_name": "ov.pp.neighbors", "arguments": {"adata_id": "adata_a1b2c3d4e5f6"}}
```

Response reports `"produced": {"obsp": ["distances", "connectivities"], "uns": ["neighbors"]}`.

### Step 6: UMAP

```json
{"tool_name": "ov.pp.umap", "arguments": {"adata_id": "adata_a1b2c3d4e5f6"}}
```

Response reports `"produced": {"obsm": ["X_umap"]}`.

### Step 7: Leiden Clustering

```json
{
  "tool_name": "ov.pp.leiden",
  "arguments": {"adata_id": "adata_a1b2c3d4e5f6", "resolution": 1.0}
}
```

Response reports `"produced": {"obs": ["leiden"]}`.

## MCP Protocol Examples

### `tools/list` Request

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

Response (abbreviated):

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "ov.list_tools",
        "description": "List available OmicVerse tools, optionally filtered by category",
        "inputSchema": {
          "type": "object",
          "properties": {
            "category": {"type": "string", "description": "Filter by category (e.g. preprocessing, pl, single)"},
            "execution_class": {"type": "string", "enum": ["stateless", "adata", "class"], "description": "Filter by execution class"}
          }
        }
      },
      {
        "name": "ov.pp.pca",
        "description": "Principal Component Analysis",
        "inputSchema": {
          "type": "object",
          "properties": {
            "adata_id": {"type": "string", "description": "Session dataset reference"},
            "n_pcs": {"type": "integer", "default": 50},
            "layer": {"type": "string", "default": "scaled"}
          },
          "required": ["adata_id"]
        }
      }
    ]
  }
}
```

### `tools/call` Request

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

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"ok\": true, \"tool_name\": \"ov.utils.read\", \"summary\": \"Loaded AnnData (2700x32738)\", \"outputs\": [{\"type\": \"object_ref\", \"ref_type\": \"adata\", \"ref_id\": \"adata_a1b2c3d4e5f6\"}], \"state_updates\": {}, \"warnings\": []}"
      }
    ]
  }
}
```

## Response Envelope

### Success

```json
{
  "ok": true,
  "tool_name": "ov.pp.pca",
  "summary": "PCA completed (50 components)",
  "outputs": [
    {"type": "object_ref", "ref_type": "adata", "ref_id": "adata_..."}
  ],
  "state_updates": {
    "produced": {"obsm": ["X_pca"], "varm": ["PCs"], "uns": ["pca"]}
  },
  "warnings": []
}
```

| Field | Type | Description |
|-------|------|-------------|
| `ok` | bool | `true` on success |
| `tool_name` | string | Canonical tool name |
| `summary` | string | Human-readable result summary |
| `outputs` | array | Output items (type: `json`, `object_ref`, `image`, `table`, `text`) |
| `state_updates` | object | What was added/changed on the AnnData |
| `warnings` | array | Non-fatal warnings |

### Error

```json
{
  "ok": false,
  "error_code": "missing_data_requirements",
  "message": "Missing required data: layers=['scaled']",
  "details": {"missing": {"layers": ["scaled"]}},
  "suggested_next_tools": ["ov.pp.scale"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `ok` | bool | `false` on error |
| `error_code` | string | Machine-readable error code (see below) |
| `message` | string | Human-readable error description |
| `details` | object | Structured error details |
| `suggested_next_tools` | array | Tools that would fix the issue |

### Error Codes

| Code | Meaning |
|------|---------|
| `tool_not_found` | Tool name not recognized |
| `tool_unavailable` | Tool exists but cannot run (e.g. class-backed stub) |
| `invalid_arguments` | Missing or wrong-type arguments |
| `missing_session_object` | `adata_id` not found in session |
| `missing_prerequisites` | Required predecessor tools not run |
| `missing_data_requirements` | Required data keys missing on AnnData (e.g. `scaled` layer, `X_pca`) |
| `execution_failed` | Tool raised an exception during execution |
| `cross_session_access` | Handle belongs to a different session |
| `handle_not_found` | Handle ID not found in any session |
| `persistence_failed` | I/O error during persist/restore (file not found, write failure, missing anndata) |
| `unsupported_persistence` | Attempted to persist a non-persistable handle (e.g. instance) |

## Troubleshooting

### Tools not visible

If `ov.list_tools` returns fewer tools than expected:
- Check the `--phase` flag. Default is `P0+P0.5` (15 tools). Using `--phase P0` exposes only the 9 core pipeline tools.
- P0.5 tools (find_markers, get_markers, embedding, violin, dotplot, markers_dotplot) require `--phase P0+P0.5`.

### Missing `adata_id`

```json
{"ok": false, "error_code": "missing_session_object", "message": "adata_id 'adata_xxx' not found"}
```

- You must call `ov.utils.read` first to load data and obtain an `adata_id`.
- Use the `ref_id` value from the `ov.utils.read` response.
- Each `adata_id` is only valid within the current server session.

### Cross-session access

```json
{"ok": false, "error_code": "cross_session_access", "message": "adata_id 'adata_xxx' belongs to session 'other', not 'default'"}
```

- Each handle belongs to the session that created it. You cannot access handles from a different session.
- Use `ov.list_handles` to see handles available in your current session.
- Use `ov.get_session` to check your current session ID.

### Prerequisites not met

```json
{"ok": false, "error_code": "missing_data_requirements", "suggested_next_tools": ["ov.pp.scale"]}
```

The server checks that required data keys exist on the AnnData before running a tool. For example:
- `ov.pp.pca` requires `layers["scaled"]` — run `ov.pp.scale` first
- `ov.pp.neighbors` requires `obsm["X_pca"]` — run `ov.pp.pca` first
- `ov.pp.umap` requires `uns["neighbors"]` — run `ov.pp.neighbors` first

Follow the `suggested_next_tools` in the error response.

### Tool unavailable

```json
{"ok": false, "error_code": "tool_unavailable", "message": "... is a class-backed tool not yet available"}
```

Class-backed tools (P2) are availability-gated: they appear in `ov.list_tools` / `ov.search_tools` but return `tool_unavailable` with a diagnosable reason if their runtime dependencies are missing. Use `ov.describe_tool` to check a class tool's actions and runtime availability before calling it.

## Starting the Server with Session Options

```bash
# Default session (session_id="default", no persist dir)
python -m omicverse.mcp

# Custom session ID
python -m omicverse.mcp --session-id my-analysis-session

# With persistence directory
python -m omicverse.mcp --session-id my-session --persist-dir /data/mcp_persist

# Persist dir is auto-created on first ov.persist_adata call if not specified
```

## Observability

The MCP server tracks three types of observability data per session:

- **Metrics**: Aggregated counters — handle counts, tool call totals, per-tool call/fail stats
- **Events**: Structured lifecycle records — adata created/persisted/restored, artifact registered/deleted/cleanup, tool called/failed, instance created/destroyed
- **Traces**: Per-call timing records — duration, handle refs in/out, success/failure, tool type classification

All observability data is in-memory and session-scoped. It does not persist across server restarts.

### Example: Check session health

```json
// 1. Run a tool
{"tool_name": "ov.pp.qc", "arguments": {"adata_id": "adata_abc123"}}

// 2. Check metrics
{"tool_name": "ov.get_metrics", "arguments": {}}
// Response: {"ok": true, "outputs": [{"data": {"session_id": "default", "adata_count": 1, "tool_calls_total": 2, ...}}]}

// 3. List recent traces
{"tool_name": "ov.list_traces", "arguments": {"limit": 5}}
// Response: {"ok": true, "outputs": [{"data": [{"trace_id": "abc...", "tool_name": "ov.pp.qc", "duration_ms": 12.3, "ok": true}, ...]}]}

// 4. Get a specific trace
{"tool_name": "ov.get_trace", "arguments": {"trace_id": "abc..."}}
// Response includes handle_refs_in, handle_refs_out, started_at, finished_at
```

## Artifact Management

Artifacts are references to files generated during an MCP session — plots, tables, exports, reports. Unlike adata handles (which hold live objects in memory), artifacts are lightweight metadata records pointing to files on disk.

### Artifact Types

| Type | Description |
|------|-------------|
| `file` | Generic file (default) |
| `image` | Plot or figure (e.g. PNG from visualization tools) |
| `table` | Tabular export (CSV, TSV) |
| `json` | JSON data file |
| `plot` | Named plot output |
| `report` | Analysis report |
| `export` | Data export (h5ad, xlsx, etc.) |

### Automatic Registration

Plotting tools (e.g. `ov.pl.umap`) automatically register image artifacts with `artifact_type="image"` and `source_tool` set to the tool name. No manual action needed.

### Manual Registration

Use `ov.register_artifact` to bring external files under session management:

```json
{"tool_name": "ov.register_artifact", "arguments": {
    "path": "/results/my_report.pdf",
    "artifact_type": "report",
    "content_type": "application/pdf",
    "metadata": {"description": "Final analysis report"}
}}
```

### Cleanup Strategy

`ov.cleanup_artifacts` defaults to **dry-run mode** (`dry_run=true`) — it previews what would be deleted without actually deleting anything:

```json
// Preview what would be cleaned up
{"tool_name": "ov.cleanup_artifacts", "arguments": {"artifact_type": "image", "older_than_seconds": 3600}}

// Actually clean up (handle only, files kept)
{"tool_name": "ov.cleanup_artifacts", "arguments": {"artifact_type": "image", "older_than_seconds": 3600, "dry_run": false}}

// Clean up and delete files from disk (use with caution)
{"tool_name": "ov.cleanup_artifacts", "arguments": {"dry_run": false, "delete_files": true}}
```

**Warning**: `delete_file=true` (on `ov.delete_artifact`) and `delete_files=true` (on `ov.cleanup_artifacts`) permanently remove files from disk. This cannot be undone.

### Manifest Export

Use `ov.export_artifacts_manifest` to get a JSON snapshot of all session artifacts for auditing or reproducibility. The manifest includes `file_exists` status for each artifact.

## Current Limitations

- **Class-backed tools (P2)**: Available only when runtime dependencies are present. pyDEG and pySCSA work in standard OmicVerse installs; MetaCell requires the `SEACells` package; DCT and LDA_topic are deferred (require `pertpy` and `mira` respectively). Use `ov.describe_tool` to check availability.
- **Adata persistence is explicit**: AnnData objects are in-memory by default. Use `ov.persist_adata` to save to disk and `ov.restore_adata` to reload. Persistence is not automatic.
- **Instance handles are ephemeral**: Class instances (P2 tools) cannot be persisted. They are lost on server restart.
- **Stdio transport only**: The server currently supports only stdio transport. HTTP/SSE transport is not implemented.
- **No result streaming**: Large outputs (e.g. marker tables) are returned as complete JSON, not streamed.
- **Single-process**: The server runs single-threaded. Concurrent tool calls are processed sequentially.
