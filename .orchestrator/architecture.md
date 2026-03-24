# Architecture Plan

## System Design
- `ngagent review` remains the orchestrator entrypoint, but its `test_cmd`
  becomes a local wrapper script instead of plain local `pytest`.
- The wrapper syncs the active task worktree to a task-scoped scratch directory
  under `/slow/ngagent-review/omicverse/<task-id>` on the Taiwan host and then
  invokes a remote helper over SSH.
- The remote helper owns environment bootstrap and validation execution. It
  creates a task-local virtualenv, reuses `~/micromamba/envs/aliyawak` when
  available, installs `omicverse` plus the `tests` extra, and executes the
  bounded smart-agent regression suite with `OV_AGENT_RUN_HARNESS_TESTS=1`.

## Key Interfaces
- Local review wrapper: `scripts/ci/ngagent_taiwan_review.sh`
- Remote review helper: `scripts/ci/ngagent_taiwan_remote_pytest.sh`
- Orchestrator config: root `.orchestrator/record.json` `config.test_cmd`
- Remote scratch root: `/slow/ngagent-review/omicverse/<task-id>`

## Technology Decisions

| Decision | Choice | Rationale | Date |
|----------|--------|-----------|------|
| Review transport | `rsync` over SSH to Taiwan scratch under `/slow` | Keeps `ngagent review` local while satisfying server-only validation | 2026-03-24 |
| Remote environment | Task-local `venv`, seeded from `~/micromamba/envs/aliyawak` when present | Avoids mutating shared server Python and speeds reuse of `numpy`/`pandas` | 2026-03-24 |
| Default validation target | Bounded smart-agent and harness regression set, with one temporary exclusion for a known baseline failure | Keeps review aligned with the refactor scope while allowing task-007 to establish the bridge before task-001 fixes the failing retry regression | 2026-03-24 |
| Forbidden storage | Never touch `/Yamiyoru`; use `/slow` only | Matches human constraint and server guidance bundle | 2026-03-24 |

## File Structure
```text
scripts/ci/ngagent_taiwan_review.sh
scripts/ci/ngagent_taiwan_remote_pytest.sh
docs/harness/ngagent-review-bridge.md
docs/harness/server-validation.md
```
