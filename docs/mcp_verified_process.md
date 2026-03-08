# MCP Verified Stack Process

## What "Verified" Means

A profile reaches **verified** status when all three conditions are met:

1. The profile's CI job completes successfully (all pytest tests pass)
2. The version snapshot artifact is uploaded to GitHub Actions
3. The exact package versions are recorded in `docs/mcp_verified_stacks.md`

Verified means: "these exact versions, on this platform, passed all tests on this date."
It is not "this should theoretically work" — it requires evidence.

## Why core-runtime Is the First Candidate

- Its dependencies (`anndata`, `scanpy`, `numpy`, `pandas`, `scipy`, `matplotlib`) come from `pyproject.toml` — no separate requirements file needed
- It runs on every push to master and weekly, giving frequent verification opportunities
- Its test suite (~31 tests) covers the real P0 pipeline end-to-end
- Its constraint source (`pyproject.toml`) is the most stable of all profiles

Other profiles follow later:
- `scientific-runtime` is currently **constrained** (has requirements file, awaits verified run)
- `extended-runtime` is currently **best_effort** (SEACells/mira ecosystem instability)

## Artifact Naming Convention

CI artifacts follow the pattern:

    mcp-versions-<profile>-py<python-version>

Examples:
- `mcp-versions-core-runtime-py3.10`
- `mcp-versions-fast-mock-py3.11`
- `mcp-versions-scientific-runtime-py3.10`

Artifacts are retained for 90 days. The JSON file inside each artifact is:

    .ci-artifacts/mcp-<profile>-versions.json

## Version Snapshot Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Always `1` for current format |
| `profile` | string | Profile name (e.g., `core-runtime`) |
| `timestamp` | string | ISO 8601 UTC timestamp |
| `platform` | string | `platform.platform()` output |
| `python` | string | Python version (e.g., `3.10.14`) |
| `packages` | object | Map of package name to installed version (or `null` if missing) |
| `source` | string | `ci` if generated in CI, `local` if generated locally |

## Upgrade Checklist (Constrained -> Verified)

Before promoting a profile to verified:

- [ ] CI job for the profile completed successfully (green check in GitHub Actions)
- [ ] All tests in the profile passed (no failures; skips acceptable only for extended-runtime)
- [ ] Version snapshot artifact was uploaded (check the "Artifacts" section of the run)
- [ ] Download the artifact and verify the JSON contains no `null` values for required packages
- [ ] No open issues tagged with the profile name that indicate known failures

To promote:

1. Download the version snapshot JSON from the GitHub Actions run
2. Open `docs/mcp_verified_stacks.md`
3. In the "Profile Status Summary" table, change the profile's status from **constrained** to **verified**
4. In the "Recorded Snapshots" section, add an entry with the snapshot data (see template in that file)
5. Commit the change with message: `docs: promote <profile> to verified (artifact: <artifact-name>)`

## Rollback Checklist (Verified -> Constrained)

A profile should be demoted from **verified** to **constrained** when:

- A subsequent CI run fails for the profile
- A dependency in `pyproject.toml` or `requirements/mcp-*.txt` is updated but not re-verified
- The profile's test suite is modified in a way that changes dependency coverage
- A transitive dependency update causes test failures

To demote:

1. Open `docs/mcp_verified_stacks.md`
2. In the "Profile Status Summary" table, change the profile's status from **verified** to **constrained**
3. Do NOT delete the previous snapshot entry — add a note: `(demoted <date>: <reason>)`
4. Commit the change with message: `docs: demote <profile> to constrained (<reason>)`

## Re-verification After Dependency Changes

Any change to the following files invalidates verified status for the affected profiles:

| File Changed | Profiles Affected |
|-------------|-------------------|
| `pyproject.toml` (dependency sections) | All profiles |
| `requirements/mcp-scientific-runtime.txt` | `scientific-runtime` |
| `requirements/mcp-extended-runtime.txt` | `extended-runtime` |

After such a change:
1. Demote affected profiles to **constrained**
2. Wait for a successful CI run with the updated dependencies
3. Re-promote following the upgrade checklist above

## Cross-References

- [Verified Dependency Stacks](mcp_verified_stacks.md) — snapshot records and status table
- [CI Profiles](mcp_ci_profiles.md) — profile details, triggers, troubleshooting
- [Runtime Matrix](mcp_runtime_matrix.md) — tier model and test file reference
- Version snapshot script: `scripts/ci/mcp-report-versions.py`
