---
title: OmicVerse MCP Server Guide
---

# OmicVerse MCP Server

The OmicVerse MCP server exposes registered analysis functions as tools via the [Model Context Protocol](https://modelcontextprotocol.io/). Claude Code, Claude Desktop, and other MCP clients can discover and call these tools while the underlying `AnnData` objects stay server-side.

!!! tip "How to read the MCP docs now"

    This page is now the overview entrypoint. The original long guide has been split by topic so you can go directly to the part you need without reading one very large page.

## Reading Map

- **Fastest path**: [Quick Start](t_mcp_quickstart.md)
- **Complete onboarding path**: [Full Start](t_mcp_full_start.md)
- **Full tool inventory**: [Tool Catalog](t_mcp_tools.md)
- **Claude / HTTP / SSH setup**: [Clients and Deployment](t_mcp_clients.md)
- **`adata_id`, persistence, cancel, logs, troubleshooting**: [Runtime and Troubleshooting](t_mcp_runtime.md)
- **Flags, JSON-RPC, error codes, exact counts**: [Reference](t_mcp_reference.md)
- **Claude Code walkthrough with screenshots**: [Using OmicVerse MCP with Claude Code](t_mcp_claude_code.md)

## What OmicVerse MCP Provides

- OmicVerse analysis tools exposed as MCP tools
- Stable handles such as `adata_id`, `artifact_id`, and `instance_id`
- Built-in datasets such as `ov.datasets.pbmc3k`, `ov.datasets.pbmc8k`, and `ov.datasets.seqfish`
- AnnData inspection tools such as `ov.adata.describe`, `ov.adata.peek`, and `ov.adata.inspect`
- Two local transports: `stdio` and `streamable-http`

## Current Scope

With the current implementation:

- `P0`: 15 analysis tools
- `P0.5`: 13 additional analysis tools
- `P2`: 5 advanced class-backed tools
- Meta tools: 25 always-on tools
- Total at `P0+P0.5+P2`: 58 tools

## What Moved Where

### Installation, startup, and first analysis

Moved to:

- [Quick Start](t_mcp_quickstart.md)
- [Full Start](t_mcp_full_start.md)

### Phase system, tool catalog, and dependency tiers

Moved to:

- [Tool Catalog](t_mcp_tools.md)

### Client configuration

Moved to:

- [Clients and Deployment](t_mcp_clients.md)

This includes:

- `stdio`
- local `streamable-http`
- Claude Code / Claude Desktop
- remote SSH launch
- local and remote side-by-side deployment

### Sessions, handles, persistence, observability, and troubleshooting

Moved to:

- [Runtime and Troubleshooting](t_mcp_runtime.md)

### JSON-RPC and raw client examples

Moved to:

- [Reference](t_mcp_reference.md)

## Minimal Example

If you only want the shortest useful flow:

```text
Load the built-in pbmc3k dataset
Describe the current adata
Run QC, scale, PCA with 50 components, build neighbors, compute UMAP, and run Leiden clustering
Plot the UMAP colored by leiden
```

Use [Quick Start](t_mcp_quickstart.md) for the command and config snippets behind this flow.
