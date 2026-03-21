# OmicVerse MCP — Integration Guide

## What is the OmicVerse MCP Server

The OmicVerse MCP server exposes the omicverse analysis registry as tools via the [Model Context Protocol](https://modelcontextprotocol.io/) stdio transport. Any MCP-compatible client (Claude Code, Claude Desktop, custom scripts) can call these tools to run single-cell, bulk, and spatial transcriptomics analyses.

## Prerequisites

- Python ≥ 3.10
- `pip install omicverse[mcp]` (or `pip install -e ".[mcp]"` for development)
- For full pipeline support: `anndata`, `scanpy` (installed with omicverse)
- For P2 class tools: optional deps like `pertpy`, `SEACells`, `mira` (availability-gated)

## Starting the Server

```bash
# Default: P0+P0.5 tools, auto-generated session ID
python -m omicverse.mcp

# CLI entrypoint (equivalent)
omicverse-mcp

# Core pipeline tools only
python -m omicverse.mcp --phase P0

# All tools including class-backed P2
python -m omicverse.mcp --phase P0+P0.5+P2

# Custom session and persistence
python -m omicverse.mcp --session-id my_analysis --persist-dir /data/ov_persist

# Check version
python -m omicverse.mcp --version
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | `P0+P0.5` | Rollout phase(s) to expose |
| `--session-id` | `default` | Logical session identifier for handle isolation |
| `--persist-dir` | (tempdir) | Directory for persisting AnnData via `ov.persist_adata` |
| `--max-adata` | `50` | Max AnnData handles per session |
| `--max-artifacts` | `200` | Max artifact handles per session |
| `--version` | — | Show version and exit |

## Phase Selection Guide

| Phase | Tools | Audience |
|-------|-------|----------|
| `P0` | Core pipeline: read, qc, scale, pca, neighbors, umap, leiden, layers | Quick exploration, minimal surface |
| `P0+P0.5` | + marker genes, plotting, visualization | Standard analysis workflows (default) |
| `P0+P0.5+P2` | + class tools: pyDEG, pySCSA, MetaCell, DCT, LDA | Advanced users with all deps installed |

**Recommendation**: Start with the default (`P0+P0.5`). Add `+P2` only when you need bulk DEG, cell annotation, or metacell tools and have the required packages installed.

## Session & Persistence Configuration

### Sessions

Each server instance runs in a single session. All handles (adata, artifacts, class instances) are scoped to that session. Use `--session-id` to assign a meaningful name:

```bash
python -m omicverse.mcp --session-id patient_cohort_A
```

### Persistence

By default, `ov.persist_adata` saves to a temporary directory. For durable storage:

```bash
python -m omicverse.mcp --persist-dir /data/ov_sessions
```

Files are saved as `.h5ad` with a JSON sidecar containing metadata (session ID, timestamp, obs/var summaries).

## Meta Tools Overview

20 built-in meta tools are always available regardless of phase selection:

### Discovery (3 tools)
- `ov.list_tools` — List available tools with optional category/execution_class filtering
- `ov.search_tools` — Search tools by keyword with ranked results
- `ov.describe_tool` — Get full description, parameters, and availability for a tool

### Session Management (4 tools)
- `ov.get_session` — Current session info (ID, handle counts, persist directory)
- `ov.list_handles` — All handles in current session with optional type filter
- `ov.persist_adata` — Save AnnData to `.h5ad` with JSON sidecar metadata
- `ov.restore_adata` — Restore AnnData from `.h5ad` file into current session

### Observability (4 tools)
- `ov.get_metrics` — Aggregated session metrics (handle counts, tool call stats)
- `ov.list_events` — Recent session events (handle lifecycle, tool calls)
- `ov.get_trace` — Details of a single tool call trace by trace_id
- `ov.list_traces` — Recent tool call traces with timing and status

### Artifact Management (6 tools)
- `ov.list_artifacts` — List artifacts with optional type/content_type/source_tool filtering
- `ov.describe_artifact` — Full artifact metadata including file status
- `ov.register_artifact` — Manually register existing file as session artifact
- `ov.delete_artifact` — Delete artifact handle, optionally delete underlying file
- `ov.cleanup_artifacts` — Batch cleanup artifacts by filters (dry-run default)
- `ov.export_artifacts_manifest` — Export all session artifacts as JSON manifest

### Runtime Safety (3 tools)
- `ov.get_limits` — Get current quota and TTL configuration with usage counts
- `ov.cleanup_runtime` — Manual cleanup of expired events, traces, artifacts (dry-run default)
- `ov.get_health` — Lightweight health summary with quota proximity warnings

## Client Integration: Claude Code / Claude Desktop

Add to your `claude_desktop_config.json` (or Claude Code MCP config):

```json
{
  "mcpServers": {
    "omicverse": {
      "command": "python",
      "args": ["-m", "omicverse.mcp", "--phase", "P0+P0.5", "--persist-dir", "/tmp/ov_persist"],
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

## Client Integration: Generic stdio MCP Client

Any MCP client that speaks JSON-RPC over stdio can connect:

```python
import subprocess, json

proc = subprocess.Popen(
    ["python", "-m", "omicverse.mcp"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True,
)

# Send JSON-RPC request
request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
proc.stdin.write(json.dumps(request) + "\n")
proc.stdin.flush()

# Read response
response = json.loads(proc.stdout.readline())
print(f"Available tools: {len(response['result']['tools'])}")
```

## Minimal Call Sequence

A typical analysis session follows this pattern:

```
1. tools/list                                    → see all available tools
2. tools/call: ov.utils.read {path: "data.h5ad"} → get adata_id
3. tools/call: ov.pp.qc {adata_id: "adata_xxx"}  → run QC
4. tools/call: ov.get_session {}                  → check session state
```

Each tool call returns a structured response with `ok`, `outputs`, and `state_updates` fields. The `state_updates` section shows what changed (e.g., new `.obs` columns, layer modifications).

## Dependency & Availability

P2 class tools are **availability-gated**: they appear in `ov.list_tools` but may show `available: false` if their runtime dependencies are missing. Use `ov.describe_tool` to check:

```
tools/call: ov.describe_tool {tool_name: "ov.single.metacell"}
→ available: false, reason: "requires SEACells package"
```

Install the missing package and restart the server to enable the tool.

## Environment Verification

The MCP test suite uses a 4-tier environment model with pytest markers:

| Tier | Marker | Requires |
|------|--------|----------|
| Mock | — | Nothing (default, CI-safe) |
| Core | `core` | anndata, scanpy |
| Scientific | `scientific` | + scvelo, squidpy |
| Extended | `extended` | + SEACells, pertpy, mira |

```bash
# All tests (mock pass, real skipped if deps missing)
pytest tests/mcp/ -v

# Real-runtime tests only
pytest tests/mcp/ -m "real_runtime" -v

# Mock tests only (no deps needed)
pytest tests/mcp/ -m "not real_runtime" -v
```

See [mcp_runtime_matrix.md](mcp_runtime_matrix.md) for full details and adding new tests.

### Choosing Local Verification Depth

| Scenario | Command |
|----------|---------|
| Quick check (no heavy deps) | `bash scripts/ci/mcp-fast-mock.sh` |
| After modifying MCP core | `bash scripts/ci/mcp-core-runtime.sh` |
| Before release (full stack) | `bash scripts/ci/mcp-scientific-runtime.sh` |

See [CI Profiles](mcp_ci_profiles.md) for full profile documentation.

## Troubleshooting

- **Startup timeout / handshake stuck**: On Windows (or first launch on any OS), the server may take >10 seconds to initialize due to numba JIT compilation and matplotlib font cache generation. Fix by setting `NUMBA_DISABLE_JIT=1` and pointing `NUMBA_CACHE_DIR` / `MPLCONFIGDIR` to writable directories in the `env` block of your MCP config. See [mcp_quickstart.md § Startup timeout](mcp_quickstart.md#startup-timeout--handshake-stuck-especially-on-windows) for the full config example.
- **Config changes not taking effect**: MCP server config is read once at process startup. You must start a new session after modifying `env`, `args`, or other settings — there is no hot-reload.
- **Server exits immediately**: Check stderr output for import errors. Ensure `mcp>=1.0` is installed.
- **No tools listed**: Verify phase flag. `P0` exposes ~9 tools; `P0+P0.5` exposes ~15+.
- **Tool returns `ok: false`**: Read the `error` field. Common causes: missing adata_id handle, wrong parameter types.
- **Persistence fails**: Ensure `--persist-dir` points to a writable directory.

For detailed request/response examples, see [mcp_quickstart.md](mcp_quickstart.md).
