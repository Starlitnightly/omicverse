# OmicVerse MCP — Release Checklist

## Pre-Release Verification

### 1. CLI
- [ ] `python -m omicverse.mcp --help` exits 0 with correct arg descriptions
- [ ] `python -m omicverse.mcp --version` shows correct version
- [ ] `omicverse-mcp --help` works identically

### 2. Tool Discovery
- [ ] `ov.list_tools` returns P0 tools (≥9)
- [ ] `ov.list_tools` with `category` filter works
- [ ] `ov.search_tools` returns ranked results
- [ ] `ov.describe_tool` returns full schema for any tool

### 3. P0 Pipeline Smoke Test
- [ ] ov.utils.read → ov.pp.qc → ov.pp.scale → ov.pp.pca → ov.pp.neighbors → ov.pp.umap → ov.pp.leiden
- [ ] Each step returns ok: true with expected state_updates

### 4. Session Management
- [ ] ov.get_session returns session_id
- [ ] ov.list_handles returns empty, then shows handles after tool calls
- [ ] ov.persist_adata saves to disk
- [ ] ov.restore_adata loads from disk

### 5. Observability
- [ ] ov.get_metrics returns counters
- [ ] ov.list_events shows lifecycle events
- [ ] ov.list_traces shows tool call traces
- [ ] ov.get_trace returns full trace details

### 6. Artifact Management
- [ ] ov.register_artifact registers a file
- [ ] ov.list_artifacts lists registered artifacts
- [ ] ov.describe_artifact shows full metadata
- [ ] ov.delete_artifact removes handle
- [ ] ov.cleanup_artifacts dry_run shows preview
- [ ] ov.export_artifacts_manifest exports JSON

### 7. Tests
- [ ] `pytest tests/mcp/ -v` — all tests pass, 0 failures
- [ ] No new warnings in test output

### 8. Documentation
- [ ] README.md MCP section matches current capabilities
- [ ] docs/mcp_quickstart.md tool tables match META_TOOLS
- [ ] docs/mcp_integration.md client examples use correct commands
- [ ] Meta tool count in docs matches len(META_TOOLS)

### 9. Packaging
- [ ] `pyproject.toml` version updated
- [ ] `omicverse-mcp` script entrypoint defined
- [ ] `mcp` optional dependency group defined
- [ ] `pip install -e ".[mcp]"` installs cleanly

## Test Command

```bash
pytest tests/mcp/ -v --tb=short
```
