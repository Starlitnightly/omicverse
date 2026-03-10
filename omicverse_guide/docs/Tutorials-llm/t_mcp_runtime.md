---
title: OmicVerse MCP Runtime and Troubleshooting
---

# OmicVerse MCP Runtime and Troubleshooting

This page explains how the server manages data, persistence, cancellation, and logs.

## Handles

### `adata_id`

`adata_id` is the stable MCP handle for a dataset. The dataset itself stays server-side.

- load with `ov.utils.read` or `ov.datasets.*`
- pass the returned handle to downstream tools
- persist with `ov.persist_adata`
- restore with `ov.restore_adata`

### `artifact_id`

Plotting and export tools can register files as artifacts. These are referenced by `artifact_id`.

### `instance_id`

P2 class-backed tools use `instance_id` for multi-step analyzers such as DEG or metacell workflows.

### Handle Types at a Glance

| Handle Type | Prefix | Persistable | Example |
|-------------|--------|-------------|---------|
| `adata` | `adata_` | Yes | loaded datasets |
| `artifact` | `artifact_` | Yes if file exists on disk | plots, exports |
| `instance` | `inst_` | No | P2 analyzers |

## Persistence

Use `ov.persist_adata` to write an `.h5ad` plus metadata sidecar. Use `ov.restore_adata` to recover it in a later session.

This is the recommended recovery path across client reconnects or process restarts.

### Persistence Example

```json
{"tool_name": "ov.persist_adata", "arguments": {"adata_id": "adata_a1b2c3d4e5f6"}}
```

```json
{"tool_name": "ov.restore_adata", "arguments": {"path": "/data/ov_persist/adata_a1b2c3d4e5f6.h5ad"}}
```

## Session and Observability

Useful tools:

- `ov.get_session`
- `ov.list_handles`
- `ov.get_metrics`
- `ov.list_events`
- `ov.get_trace`
- `ov.list_traces`
- `ov.get_health`
- `ov.get_limits`

Example questions:

- `What's the current session status?`
- `Show me the recent tool call traces`
- `Show me session metrics`
- `List all image artifacts from this session`
- `Clean up artifacts older than 1 hour, but show me what would be deleted first`
- `Export the full artifact manifest as JSON`

Example responses often look like:

```json
{
  "session_id": "default",
  "adata_count": 1,
  "artifact_count": 3,
  "instance_count": 0
}
```

```json
[
  {"trace_id": "abc...", "tool_name": "ov.pp.pca", "duration_ms": 245.3, "ok": true}
]
```

```json
{
  "adata_count": 1,
  "artifact_count": 3,
  "tool_calls_total": 8,
  "tool_calls_failed": 0,
  "artifacts_registered_total": 3
}
```

## AnnData Inspection

Before asking the model to analyze data, let it inspect the current state:

- `ov.adata.describe`
- `ov.adata.peek`
- `ov.adata.find_var`
- `ov.adata.value_counts`
- `ov.adata.inspect`

These tools reduce hallucination and make the workflow auditable.

## Cancellation and `Esc`

For `adata_id` tools, OmicVerse MCP now runs them through a persistent kernel-backed runtime.

That means:

- the client still sees a normal `Running...` state
- the runtime can be interrupted on cancellation
- the kernel can often be reused after the interruption
- the goal is to preserve state when possible instead of rebuilding a worker for every tool call

In practice, cancellation depends on the client and the underlying numerical code:

- if the client sends a proper cancel or disconnect signal, the server can interrupt the current runtime
- some deep numerical calls may respond slowly to interrupt
- if the runtime becomes unhealthy, the server may need to recover it before the next request

## Logging

### `stdio`

- protocol traffic uses `stdout`
- server logs go to `stderr`
- tool call start/end/failure summaries are written to `stderr`

### `streamable-http`

- run the server yourself
- inspect uvicorn and server logs directly
- this is usually the easiest way to debug connection issues

## P2 Lifecycle

P2 tools use a multi-step lifecycle:

1. `create`
2. `run` or task-specific action such as `annotate` / `train`
3. `results` or `predict`
4. `destroy`

Example:

```text
ov.bulk.pydeg create -> run -> results -> destroy
```

`instance_id` values are memory-only and are lost on server restart.

## Limitations

- P2 tools can appear in listings but still be unavailable at runtime
- extended-runtime is constrained in some environments
- local HTTP auth is intended only for localhost development
- the built-in localhost OAuth flow is memory-only and should not be exposed to untrusted networks
- the server is effectively single-process for tool execution
- some long numerical steps may respond slowly to interrupt
- no result streaming: large outputs are returned as complete results
- class instances are memory-only and are lost on server restart
- not every extended dependency stack is available in every environment

## Common Problems

### Tool is missing

- verify the `--phase` you started with
- ask the client to run `ov.list_tools`
- for P2 tools, check `ov.describe_tool` for dependency availability

### `adata_id` not found

The handle was never created, expired, or belongs to another session. Use `ov.list_handles` and `ov.get_session`.

### Plot exists but is hard to find

Use:

- `ov.list_artifacts`
- `ov.describe_artifact`
- `ov.export_artifacts_manifest`

### Long-running analysis blocks progress

- prefer local HTTP mode for observability
- persist the dataset before especially heavy steps
- use `ov.get_trace`, `ov.list_traces`, and `ov.list_events` to understand what ran

### Server crashes or connection is lost

- inspect `stderr` or HTTP server logs
- check `stderr` for Python tracebacks or import errors
- restart the MCP server
- use `ov.restore_adata` if you had persisted the dataset

### Tool is unavailable

- ask for `ov.describe_tool`
- check missing optional dependencies
- restart after installing missing packages

### Remote server connection fails

- verify SSH connectivity separately, for example `ssh -i /path/to/key -p <port> user@host echo ok`
- ensure `omicverse[mcp]` is installed on the remote machine
- check the remote Python path in the client config

### Preprocessing step fails with missing data requirements

- the server enforces pipeline ordering
- follow `suggested_next_tools` in the error response
- typical prerequisites are:
  - `ov.pp.pca` requires `layers["scaled"]`
  - `ov.pp.neighbors` requires `obsm["X_pca"]`
  - `ov.pp.umap` requires `uns["neighbors"]`

### Server exits immediately

- ensure `mcp>=1.0` is installed
- if using `streamable-http`, ensure `uvicorn` and `starlette` are installed

### Claude Code HTTP connection shows auth/discovery errors

- make sure the server was started with `--transport streamable-http`
- verify the configured URL matches exactly, for example `http://127.0.0.1:8765/mcp`
- restart the local MCP server after auth or transport changes because localhost OAuth registrations are memory-only

These observability tools are the first things to use when a workflow behaves unexpectedly.

## Related Pages

- Setup: [Quick Start](t_mcp_quickstart.md)
- Tool inventory: [Tool Catalog](t_mcp_tools.md)
- Exact CLI and JSON formats: [Reference](t_mcp_reference.md)
