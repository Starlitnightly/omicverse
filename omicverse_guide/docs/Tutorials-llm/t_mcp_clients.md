---
title: OmicVerse MCP Clients and Deployment
---

# OmicVerse MCP Clients and Deployment

This page covers how to connect clients to OmicVerse MCP and when to use each transport.

## Architecture

```
┌─────────────────┐     MCP transport      ┌──────────────────────┐
│   Claude Code   │◄──────────────────────►│  OmicVerse MCP Server│
│   or another    │   adata_id / results   │  (local or remote)   │
│   MCP client    │                        │                      │
└─────────────────┘                        │  AnnData in memory   │
                                           │  Artifacts on disk   │
                                           └──────────────────────┘
```

Raw `AnnData` does not cross the protocol boundary. The server stores it and returns lightweight handles such as `adata_id`.

## Transport Modes

## Common launch commands

```bash
# Core only
python -m omicverse.mcp --phase P0

# Default local workflow
python -m omicverse.mcp --phase P0+P0.5

# Enable advanced class-backed tools
python -m omicverse.mcp --phase P0+P0.5+P2
```

### `stdio`

Claude or another MCP client launches `python -m omicverse.mcp` as a subprocess and speaks JSON-RPC over stdin/stdout.

Use this when:

- you want the simplest setup
- the client should own the MCP process lifecycle
- you are doing local, short-to-medium analyses

### `streamable-http`

You start OmicVerse MCP yourself and the client connects to a local URL such as `http://127.0.0.1:8765/mcp`.

Use this when:

- you want to keep the MCP process running across reconnects
- you want direct access to server-side logs
- you want cleaner fault isolation for larger or longer-running jobs

## Claude Code

### Option A: `stdio`

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

To expose P2 tools from Claude Code:

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

### Option B: Local HTTP

Start the server:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mpl \
python -m omicverse.mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8765 \
  --http-path /mcp \
  --phase P0+P0.5
```

Then configure Claude Code:

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

Local HTTP mode includes a minimal localhost OAuth flow intended for local clients only.

## Claude Desktop

Claude Desktop uses the same two patterns:

- `stdio` when Claude should launch OmicVerse directly
- local HTTP when OmicVerse runs as a separate localhost MCP server

The exact configuration shape is the same as Claude Code.

## Remote / SSH Deployment

You can also run OmicVerse MCP on a remote machine and let Claude launch it over SSH:

```json
{
  "mcpServers": {
    "omicverse-remote": {
      "type": "stdio",
      "command": "ssh",
      "args": [
        "user@remote-host",
        "cd /path/to/project && python -m omicverse.mcp --phase P0+P0.5+P2 --persist-dir /data/ov_persist"
      ]
    }
  }
}
```

You can include explicit SSH options when needed:

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
      ]
    }
  }
}
```

Use this when:

- data already lives on remote storage
- RAM or GPU requirements exceed your laptop
- you want a durable shared persistence directory

## Local and Remote Side-by-Side

You can define both a local and a remote server in one configuration:

```json
{
  "mcpServers": {
    "omicverse-local": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "omicverse.mcp", "--phase", "P0+P0.5", "--persist-dir", "/tmp/ov_persist_local"]
    },
    "omicverse-remote": {
      "type": "stdio",
      "command": "ssh",
      "args": [
        "user@gpu-server",
        "cd /data/project && python -m omicverse.mcp --phase P0+P0.5+P2 --persist-dir /data/ov_persist"
      ]
    }
  }
}
```

This is useful when:

- local server handles quick preprocessing and inspection
- remote server handles bigger datasets or P2 tools with optional dependencies

### Handle isolation

Each server has its own session namespace. An `adata_id` from one server cannot be used on the other. Persist and restore if you need to move data across servers.

## Which Mode Is Better

### For most users

Use `stdio`.

It is shorter, simpler, and usually the least fragile configuration.

### For large datasets or longer jobs

Use local `streamable-http`.

The transport is not faster, but the process boundary is easier to observe and manage. This matters when a long PCA, neighbors, or UMAP step needs debugging or reconnect handling.

## Recommended Patterns

### Small to medium local analysis

- `stdio`
- `P0+P0.5`

### Large local analysis

- local `streamable-http`
- persistent logs
- optional `--persist-dir`

### Remote compute

- SSH-launched `stdio`
- remote persistence
- `P0+P0.5+P2` if extended dependencies are installed

## Deployment Comparison

| | Local | Remote (SSH) |
|---|---|---|
| **Latency** | Instant | SSH overhead |
| **Data location** | Must be local | Stays on remote filesystem |
| **Compute** | Laptop-bound | High RAM / GPU / many cores |
| **Typical phase** | `P0+P0.5` | `P0+P0.5+P2` |
| **Persistence** | Local disk | Remote shared storage |

## Remote Environment Tips

- use explicit `conda` or `micromamba` python paths if needed
- verify SSH connectivity before adding the MCP config
- make sure `omicverse[mcp]` is installed on the remote machine
- use `-p <port>` for non-standard SSH ports
- use `-i /path/to/key` for SSH key authentication
- passwordless login is strongly recommended

## Related Pages

- Fast setup: [Quick Start](t_mcp_quickstart.md)
- Claude-specific walkthrough: [Using OmicVerse MCP with Claude Code](t_mcp_claude_code.md)
- Runtime behavior: [Runtime & Troubleshooting](t_mcp_runtime.md)
