# MCP CI Profiles

## Profile Summary

| Profile | Marker | Install Command | PR Default | Push to master | Schedule | Manual |
|---------|--------|----------------|------------|----------------|----------|--------|
| `fast-mock` | `not real_runtime` | `pip install -e ".[tests,mcp]"` | Yes | Yes | Yes | Yes |
| `core-runtime` | `core` | `pip install -e ".[tests,mcp]"` | No | Yes | Weekly | Yes |
| `scientific-runtime` | `scientific` | `pip install -e ".[tests,mcp]"` + `scvelo squidpy` | No | No | Weekly | Yes |
| `extended-runtime` | `extended` | `pip install -e ".[tests,mcp]"` + `SEACells pertpy mira` | No | No | No | Yes |

## Local Execution

Each profile has a shell script in `scripts/ci/`:

```bash
bash scripts/ci/mcp-fast-mock.sh             # ~1s, no heavy deps
bash scripts/ci/mcp-core-runtime.sh           # ~5s, needs anndata+scanpy
bash scripts/ci/mcp-scientific-runtime.sh     # ~10s, needs scvelo+squidpy
bash scripts/ci/mcp-extended-runtime.sh       # ~15s, needs SEACells+pertpy+mira
```

All scripts accept extra pytest arguments:

```bash
bash scripts/ci/mcp-fast-mock.sh -q --no-header    # quiet output
bash scripts/ci/mcp-core-runtime.sh -x              # stop on first failure
```

## Profile Details

### fast-mock

- **Goal**: Fast PR regression gate
- **What it verifies**: Protocol handling, session management, meta tools, adapters, quota enforcement, TTL, cleanup — all via mock registry
- **Dependencies**: `pytest`, `pytest-asyncio` (from `[tests]` extra)
- **Expected tests**: ~402
- **Expected skips**: 0
- **Expected duration**: <2 seconds
- **When it runs**: Every PR and push to master (path-filtered to MCP files)
- **Failure triage**: Check test output directly — all tests use mock data, failures indicate code bugs

### core-runtime

- **Goal**: Verify real registry hydration and P0 pipeline with actual anndata/scanpy
- **What it verifies**: Real function registration, P0 tool chain (read→qc→scale→pca→neighbors→umap→leiden), meta tools with real backend, manifest building
- **Dependencies**: `anndata`, `scanpy` (from core package deps)
- **Expected tests**: ~31
- **Expected skips**: 0 (all deps should be available via `pip install -e .`)
- **Expected duration**: ~5 seconds
- **When it runs**: Push to master, weekly schedule, manual dispatch
- **Failure triage**: Usually import errors or API changes in anndata/scanpy. Check `tests/mcp/test_real_p0_pipeline.py` and `test_real_registry_matrix.py`

### scientific-runtime

- **Goal**: Verify scientific stack tools are importable and don't break manifest building
- **What it verifies**: scvelo and squidpy imports, manifest builds correctly with scientific stack present
- **Dependencies**: `scvelo`, `squidpy` (installed separately)
- **Expected tests**: 3
- **Expected skips**: 0 (when deps installed)
- **Expected duration**: ~10 seconds (mostly import time)
- **When it runs**: Weekly schedule, manual dispatch
- **Failure triage**: Usually version incompatibilities between scvelo/squidpy and core stack

### extended-runtime

- **Goal**: Verify P2 class tool availability gating with optional heavy packages
- **What it verifies**: `check_class_availability()` returns correct results when SEACells/pertpy/mira are present, class spec actions are valid
- **Dependencies**: `SEACells`, `pertpy`, `mira` (installed separately)
- **Expected tests**: ~2
- **Expected skips**: May skip if only a subset of deps installed
- **Expected duration**: ~15 seconds
- **When it runs**: Manual dispatch only (heavy deps, may fail to install)
- **Failure triage**: Check `tests/mcp/test_real_p2_availability.py`. Extended deps have complex transitive requirements

## Skip Semantics

| Profile | Expected Skips | Skip Meaning |
|---------|---------------|--------------|
| fast-mock | 0 | All tests should pass — any skip indicates a test miscategorization |
| core-runtime | 0 | All core tests should pass when deps installed |
| scientific-runtime | 0 | All scientific tests should pass when deps installed |
| extended-runtime | 0-2 | Some tests may skip if only partial extended deps are installed |

A skip is **not** a substitute for coverage. If a profile consistently skips tests, either the deps need installing or the test needs reclassifying.

## CI Workflow

The GitHub Actions workflow is at `.github/workflows/mcp-tests.yml`.

### Automatic Triggers

- **Push/PR to master** (path-filtered): runs `fast-mock`
- **Push to master**: also runs `core-runtime`
- **Weekly schedule** (Monday 6 AM UTC): runs `fast-mock`, `core-runtime`, `scientific-runtime`

### Manual Dispatch

```bash
# Via GitHub CLI
gh workflow run mcp-tests.yml -f profile=core-runtime
gh workflow run mcp-tests.yml -f profile=all

# Via GitHub UI
# Go to Actions → MCP Tests → Run workflow → select profile
```

## Dependency Installation

| Profile | Install Steps |
|---------|--------------|
| fast-mock | `pip install -e ".[tests,mcp]"` |
| core-runtime | `pip install -e ".[tests,mcp]"` |
| scientific-runtime | `pip install -e ".[tests,mcp]"` then `pip install -r requirements/mcp-scientific-runtime.txt` |
| extended-runtime | `pip install -e ".[tests,mcp]"` then `pip install -r requirements/mcp-extended-runtime.txt` |

Or use the unified script entry: `bash scripts/ci/mcp-<profile>.sh --install`

Note: `anndata` and `scanpy` are core dependencies in `pyproject.toml` — they install automatically with `pip install -e .`.

## Dependency Policy

| Profile | Strategy | Constraints File | Stability |
|---------|----------|-----------------|-----------|
| `fast-mock` | No extra deps | — | Stable |
| `core-runtime` | Core package deps only | — | Stable |
| `scientific-runtime` | Version-constrained | `requirements/mcp-scientific-runtime.txt` | Mostly stable |
| `extended-runtime` | Best effort | `requirements/mcp-extended-runtime.txt` | Best effort |

- **Stable**: failures indicate MCP code bugs, not dep drift
- **Mostly stable**: version ranges reduce drift, but upstream breaks are possible
- **Best effort**: packages may fail to install on some platforms; periodic manual verification

For detailed verification status and recorded version snapshots, see
[Verified Dependency Stacks](mcp_verified_stacks.md).  Each CI run generates
a version snapshot artifact (`scripts/ci/mcp-report-versions.py`) that records
the exact installed versions of all tier-relevant packages.

For the upgrade and rollback process (constrained to verified and back), see
[Verified Process](mcp_verified_process.md).

### Updating Version Constraints

When a scientific/extended profile breaks due to upstream changes:

1. Check if the failure is an import error or a test assertion failure
2. If import error: update the version range in the requirements file
3. If assertion failure: determine if MCP code needs updating or the constraint needs tightening
4. Test locally: `bash scripts/ci/mcp-<profile>.sh --install`
5. Commit the updated requirements file

## Troubleshooting CI Failures

### Installation failures
- Check the pip output for version conflicts
- Compare against `requirements/mcp-<profile>.txt` version ranges
- Try `pip install --dry-run` locally to preview resolution
- If upstream broke compatibility, tighten the upper bound in the requirements file

### Import errors
- If the package installs but fails to import: likely a transitive dep conflict
- Run `python -c "import <package>"` locally after installing with the requirements file
- Check if `pyproject.toml` core deps (scipy, numpy, pandas) have version ceilings that conflict

### Test assertion failures
- If tests fail but imports succeed: likely an API change in the upstream package
- Check the package changelog for breaking changes
- Fix the test or update MCP code as needed — don't weaken the assertion

### When to downgrade a profile
- If a profile fails persistently due to upstream ecosystem instability:
  1. File an issue documenting the failure
  2. Temporarily change the profile trigger from `schedule` to `workflow_dispatch` only
  3. Add a comment in `mcp-tests.yml` explaining why
  4. Re-enable once the upstream issue is resolved
