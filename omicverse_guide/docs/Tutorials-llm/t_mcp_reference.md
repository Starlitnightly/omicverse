---
title: OmicVerse MCP Reference
---

# OmicVerse MCP Reference

This page is the compact technical reference for server flags, tool counts, return envelopes, and common error patterns.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--phase` | `P0+P0.5` | Rollout phase(s) to expose |
| `--transport` | `stdio` | `stdio` or `streamable-http` |
| `--session-id` | `default` | Session identifier |
| `--persist-dir` | tempdir | Persistence directory for `ov.persist_adata` |
| `--max-adata` | `50` | Max AnnData handles |
| `--max-artifacts` | `200` | Max artifact handles |
| `--host` | `127.0.0.1` | HTTP bind host |
| `--port` | `8765` | HTTP bind port |
| `--http-path` | `/mcp` | HTTP route path |
| `--version` | — | Show version |

## Current Tool Counts

Actual counts from the current server:

| Phase Selection | Total Tools |
|----------------|-------------|
| `P0` | 40 |
| `P0+P0.5` | 53 |
| `P0+P0.5+P2` | 58 |

Breakdown:

- `P0`: 15 analysis tools
- `P0.5`: 13 additional analysis tools
- `P2`: 5 advanced analysis tools
- Meta tools: 25

## Response Envelope

Most tool calls return a structure shaped like:

```json
{
  "ok": true,
  "tool_name": "ov.utils.read",
  "summary": "Loaded AnnData",
  "outputs": [],
  "state_updates": {},
  "warnings": []
}
```

Failures use:

```json
{
  "ok": false,
  "error_code": "missing_data_requirements",
  "message": "Missing required data",
  "details": {},
  "suggested_next_tools": []
}
```

## JSON-RPC Examples

### `tools/list`

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

### `tools/call`

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "ov.utils.read",
    "arguments": {
      "path": "pbmc3k.h5ad"
    }
  }
}
```

You do not need to send raw JSON-RPC manually when using Claude Code or Claude Desktop.

## Common Output Types

- `object_ref`: usually `adata_id` or `instance_id`
- `json`: structured table or metadata
- `image`: plotting result

## Typical Error Codes

| Error Code | Meaning |
|-----------|---------|
| `missing_session_object` | `adata_id` or another handle was not found |
| `missing_data_requirements` | prerequisite data such as `scaled` or `X_pca` is missing |
| `tool_unavailable` | tool exists but its dependencies or rollout state block execution |
| `execution_failed` | the underlying tool raised an error |

## Generic stdio Client Example

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

request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
proc.stdin.write(json.dumps(request) + "\n")
proc.stdin.flush()
response = json.loads(proc.stdout.readline())
print(f"Available tools: {len(response['result']['tools'])}")

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
print(json.loads(proc.stdout.readline()))
```

## Useful Meta Tools

- `ov.list_tools`
- `ov.describe_tool`
- `ov.get_session`
- `ov.list_handles`
- `ov.get_trace`
- `ov.list_traces`
- `ov.get_health`

## Related Pages

- Overview: [Guide](t_mcp_guide.md)
- Quick setup: [Quick Start](t_mcp_quickstart.md)
- Clients: [Clients & Deployment](t_mcp_clients.md)
- Runtime details: [Runtime & Troubleshooting](t_mcp_runtime.md)
