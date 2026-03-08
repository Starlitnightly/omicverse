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
| `core-runtime` | constrained | pyproject.toml (`anndata<0.12.0`, `scanpy>=1.9`, `scipy>=1.8,<1.12`) | Push + weekly |
| `scientific-runtime` | constrained | `requirements/mcp-scientific-runtime.txt` (`scvelo>=0.3.0,<0.4`, `squidpy>=1.3.0,<1.7`) | Weekly |
| `extended-runtime` | best_effort | `requirements/mcp-extended-runtime.txt` (SEACells/mira commented out) | Manual only |

> **Note**: No profile is currently marked **verified** because no CI run has
> yet produced a version snapshot artifact.  Once the version audit pipeline is
> live and a CI run completes successfully, update the relevant row to
> **verified** and record the snapshot below.

## Pending Promotions

### core-runtime

core-runtime is the primary candidate for the first verified profile. It meets
all structural prerequisites:

- Dependencies sourced from `pyproject.toml` (stable, no separate requirements file)
- CI job runs on every push to master and weekly schedule
- Version snapshot artifact: `mcp-versions-core-runtime-py3.10`
- Test suite: 32 tests covering real P0 pipeline

**Remaining step**: A successful CI run must produce the artifact. Once that
happens, follow the [upgrade checklist](mcp_verified_process.md#upgrade-checklist-constrained---verified)
to promote.

## Artifact Naming Convention

CI artifacts follow the pattern `mcp-versions-<profile>-py<python-version>`.
The JSON file inside each artifact is `.ci-artifacts/mcp-<profile>-versions.json`.

See the [Verified Process](mcp_verified_process.md#artifact-naming-convention)
document for the full field reference.

## Recorded Snapshots

_No snapshots recorded yet.  After the first successful CI run with version
reporting enabled, add entries below following the template._

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
