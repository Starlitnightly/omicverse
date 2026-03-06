# Tool Catalog

The OVAgent runtime currently exposes a static tool catalog from
`omicverse/utils/smart_agent.py`.

## Current Catalog Shape

- Every tool entry should have a unique `name`.
- Every tool entry should carry a short `description`.
- Every tool entry should expose an object-shaped `parameters` schema.

## Current Runtime Substrates

These existing tools are the closest server-backed building blocks for a
Claude-style catalog:

- `delegate`: current sub-agent substrate for future `Agent`
- `search_skills`: current skill-discovery substrate for future `Skill`
- `web_fetch`: current `WebFetch` substrate
- `web_search`: current `WebSearch` substrate
- `execute_code` and `run_snippet`: current Python execution substrates
- `finish`: explicit loop termination tool

## Gaps

The current catalog is not yet a Claude-style catalog:

- no `ToolSearch`
- no deferred loading contract
- no first-class `Bash`, `Read`, `Edit`, `Write`, `Glob`, or `Grep`
- no first-class task/worktree/MCP tools

## Test Policy

Catalog tests are harness tests and must stay server-only behind
`OV_AGENT_RUN_HARNESS_TESTS=1`.

