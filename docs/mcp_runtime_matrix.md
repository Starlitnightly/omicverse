# MCP Testing: Environment Verification Matrix

## 4-Tier Environment Model

The MCP test suite uses a tiered environment model. Tests are tagged with pytest markers and gated by `skipif` decorators — missing dependencies cause skips, not failures.

| Tier | Marker | Packages Required | CI Profile | Description |
|------|--------|-------------------|------------|-------------|
| Mock | — | None | `fast-mock` (PR default) | Protocol, session, meta tools, adapters — runs everywhere |
| Core | `core` | anndata, scanpy | `core-runtime` (push/weekly) | Real registry hydration, P0 pipeline, meta tools with real backend |
| Scientific | `scientific` | + scvelo, squidpy | `scientific-runtime` (weekly/manual) | Velocity tools, extended pipeline verification |
| Extended | `extended` | + SEACells, pertpy, mira | `extended-runtime` (manual only) | P2 class tool runtime verification |

The `real_runtime` marker is a superset that selects all core + scientific + extended tests.

## Running Tests by Tier

```bash
# Default: all tests (mock run, real skipped if deps missing)
pytest tests/mcp/ -v

# Mock tests only (CI-safe, no deps)
pytest tests/mcp/ -m "not real_runtime" -v

# Core: requires anndata + scanpy
pytest tests/mcp/ -m "core" -v

# All real-runtime tests
pytest tests/mcp/ -m "real_runtime" -v

# Extended: P2 class tool verification
pytest tests/mcp/ -m "extended" -v

# Skip extended (slow) tests
pytest tests/mcp/ -m "not extended" -v
```

## Dependency Detection

The `tests/mcp/_env.py` module provides zero-cost import probes using `importlib.util.find_spec()`:

```python
from tests.mcp._env import skip_no_core, skip_no_seacells, has_anndata

@skip_no_core
@pytest.mark.core
@pytest.mark.real_runtime
class TestMyRealFeature:
    def test_something(self):
        ...
```

Available helpers:
- `has_anndata()`, `has_scanpy()`, `core_available()`
- `has_scvelo()`, `has_squidpy()`, `scientific_stack_available()`
- `has_seacells()`, `has_pertpy()`, `has_mira()`
- `skip_no_core`, `skip_no_scientific`, `skip_no_seacells`, `skip_no_pertpy`, `skip_no_mira`

## Adding New Real-Runtime Tests

1. Import helpers from `tests/mcp/_env.py`
2. Add appropriate `@pytest.mark.*` and `@skip_no_*` decorators
3. Use `RegistryMcpServer(phase=..., session_id="test")` for real server fixtures
4. Always reset hydration state in `setup_method` if testing manifest building:
   ```python
   def setup_method(self):
       import omicverse.mcp.manifest as m
       m._HYDRATED = False
   ```

## Test File Reference

| File | Tier | Tests |
|------|------|-------|
| `test_real_registry_matrix.py` | core + scientific | Registry hydration across P0, P0.5, P2 phases; scientific stack import verification |
| `test_real_p0_pipeline.py` | core | End-to-end P0 pipeline (read→qc→scale→pca→neighbors→umap→leiden) |
| `test_real_p2_availability.py` | core + extended | P2 class tool availability probing |
| `test_real_meta_tools.py` | core | Meta tools with real registry backend |
| `test_startup.py::TestRealRegistryHydration` | core | Basic hydration smoke tests |
| `test_ci_profile_docs.py` | — (mock) | CI profile documentation and script consistency |

## CI Profiles

Each tier maps to a named CI profile with a shared local/CI entry point. See [mcp_ci_profiles.md](mcp_ci_profiles.md) for full details.

```bash
bash scripts/ci/mcp-fast-mock.sh             # PR default
bash scripts/ci/mcp-core-runtime.sh           # push to master / weekly
bash scripts/ci/mcp-scientific-runtime.sh     # weekly / manual
bash scripts/ci/mcp-extended-runtime.sh       # manual only
```
