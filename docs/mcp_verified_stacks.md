# MCP Verified Dependency Stacks

## Status Definitions

| Status | Meaning |
|--------|---------|
| **verified** | All tests pass with a recorded version snapshot; exact versions documented below |
| **constrained** | Version ranges are pinned in requirements files or pyproject.toml; CI runs regularly but no snapshot artifact has been recorded yet |
| **best_effort** | Some dependencies may fail to install; CI runs only on manual dispatch; no version guarantee |

## Profile Status Summary

| Profile | Status | Constraint Source | CI Cadence |
|---------|--------|-------------------|------------|
| `fast-mock` | constrained | pyproject.toml (core deps only) | Every PR/push |
| `core-runtime` | **verified** | pyproject.toml (`anndata<0.12.0`, `scanpy>=1.9`, `scipy>=1.8,<1.12`) | Push + weekly |
| `scientific-runtime` | **verified** | `requirements/mcp-scientific-runtime.txt` (`scvelo>=0.3.0,<0.4`, `squidpy>=1.3.0,<1.7`) | Weekly |
| `extended-runtime` | constrained | `requirements/mcp-extended-runtime.txt` (`pertpy>=0.7.0`; SEACells/mira commented out) | Manual only |

> **Note**: `core-runtime` and `scientific-runtime` were both promoted to
> **verified** on 2026-03-08 via real GitHub Actions CI runs on
> `HendricksJudy/omicverse` (Python 3.10.19, Ubuntu).  See "Recorded
> Snapshots" below for exact version snapshots.

## Pending Promotions

### core-runtime — PROMOTED (2026-03-08)

core-runtime has been promoted to **verified**.  All 32 core-marked tests
passed on GitHub Actions (run 22828299495, `HendricksJudy/omicverse` fork,
Python 3.10.19, Linux 6.14.0-1017-azure) and a version snapshot artifact
was recorded.

### scientific-runtime — PROMOTED (2026-03-08)

scientific-runtime has been promoted to **verified**.  All 3 scientific-marked
tests passed on GitHub Actions (run 22829008242, `HendricksJudy/omicverse`
fork, Python 3.10.19, Linux 6.14.0-1017-azure) and a version snapshot
artifact was recorded.

### extended-runtime — EVALUATED (2026-03-08)

extended-runtime was evaluated via GitHub Actions run 22829256629.  The
workflow completed successfully, but SEACells and mira remain **commented out**
in `requirements/mcp-extended-runtime.txt` and are not installed.  Only
`pertpy>=0.7.0` is active.  The 2 extended-marked tests (both requiring
SEACells) were **skipped**, not passed.

**Result**: Promoted from `best_effort` to `constrained`.  Cannot be
`verified` until SEACells and mira are uncommented, installable, and all
extended tests pass.

**Current blockers for verified**:
- `SEACells`: commented out in requirements (platform availability issues)
- `mira`: commented out in requirements (platform availability issues)
- 0/2 extended tests executed (all skipped due to missing SEACells)

## Artifact Naming Convention

CI artifacts follow the pattern `mcp-versions-<profile>-py<python-version>`.
The JSON file inside each artifact is `.ci-artifacts/mcp-<profile>-versions.json`.

See the [Verified Process](mcp_verified_process.md#artifact-naming-convention)
document for the full field reference.

## Recorded Snapshots

### core-runtime — 2026-03-08

- Python: 3.10.19
- omicverse: 1.7.10rc1
- anndata: 0.11.4
- scanpy: 1.11.5
- numpy: 1.26.4
- pandas: 2.3.3
- scipy: 1.11.4
- matplotlib: 3.10.8

Status: **verified** (artifact: `mcp-versions-core-runtime-py3.10`, JSON: `.ci-artifacts/mcp-core-runtime-versions.json`)
Source: ci (GitHub Actions run 22828299495, `HendricksJudy/omicverse`, 32/32 core tests passed)

### scientific-runtime — 2026-03-08

- Python: 3.10.19
- omicverse: 1.7.10rc1
- anndata: 0.11.4
- scanpy: 1.11.5
- scvelo: 0.3.4
- squidpy: 1.6.5
- numpy: 1.26.4
- pandas: 2.3.3
- scipy: 1.11.4
- matplotlib: 3.10.8

Status: **verified** (artifact: `mcp-versions-scientific-runtime-py3.10`, JSON: `.ci-artifacts/mcp-scientific-runtime-versions.json`)
Source: ci (GitHub Actions run 22829008242, `HendricksJudy/omicverse`, 3/3 scientific tests passed)

### extended-runtime — 2026-03-08 (partial)

- Python: 3.10.19
- omicverse: 1.7.10rc1
- anndata: 0.11.4
- scanpy: 1.11.5
- pertpy: 0.10.0
- SEACells: **null** (commented out, not installed)
- mira: **null** (commented out, not installed)
- scvelo: null (not in extended requirements)
- squidpy: null (not in extended requirements)
- numpy: 1.26.4
- pandas: 2.3.3
- scipy: 1.11.4
- matplotlib: 3.10.8

Status: **constrained** (artifact: `mcp-versions-extended-runtime-py3.10`, JSON: `.ci-artifacts/mcp-extended-runtime-versions.json`)
Source: ci (GitHub Actions run 22829256629, `HendricksJudy/omicverse`, 0 passed / 2 skipped — SEACells not available)
Note: This is NOT a verified record. pertpy installs and CI runs cleanly, but SEACells/mira are not yet part of the install path.

### How to Record a Snapshot

1. Download the version snapshot artifact from the GitHub Actions run
2. Extract the JSON and verify all required package versions are non-null
3. Add an entry below using this template:

#### Template

```
### <profile> — <YYYY-MM-DD>

- Python: <version>
- omicverse: <version>
- anndata: <version>
- scanpy: <version>
- numpy: <version>
- pandas: <version>
- scipy: <version>
- matplotlib: <version>

Status: **verified** (artifact: `mcp-versions-<profile>-py<python-version>`)
Source: ci
```

## Maintenance Workflow

1. After a CI run completes, download the version snapshot artifact from the
   GitHub Actions run page.
2. If all tests passed, update the profile's status to **verified** and add the
   snapshot to the "Recorded Snapshots" section above.
3. If a profile's status is **verified** and a subsequent run fails, investigate
   whether the failure is a version drift issue.  If so, tighten the constraint
   in the requirements file and reset status to **constrained** until the next
   passing run.
4. Review snapshots quarterly.  Remove entries older than 6 months.

### Recommended Update Flow

1. Modify `requirements/mcp-*.txt` constraints as needed
2. Run the profile locally: `bash scripts/ci/mcp-<profile>.sh --install`
3. Check generated snapshot: `cat .ci-artifacts/mcp-<profile>-versions.json`
4. If profile passes, update this document with the snapshot
5. If profile fails, only update constraints — do not update verified records

## Cross-References

- [CI Profiles](mcp_ci_profiles.md) — profile details, triggers, troubleshooting
- [Runtime Matrix](mcp_runtime_matrix.md) — 4-tier model, test file reference
- [Verified Process](mcp_verified_process.md) — upgrade/rollback checklists, re-verification rules
- Version snapshot script: `scripts/ci/mcp-report-versions.py`
