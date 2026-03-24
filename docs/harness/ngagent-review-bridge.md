# ngagent Taiwan Review Bridge

This document is the system of record for running `ngagent review` against the
Taiwan server instead of local-only `pytest`.

## Why This Exists

- The human requirement for this project is Taiwan-server validation.
- `ngagent review` only knows how to execute a static `test_cmd` locally in the
  task worktree.
- The bridge keeps `ngagent` state transitions intact by making `test_cmd`
  invoke a local wrapper that syncs the current task worktree to the Taiwan host
  and then runs the real validation there.

## Operating Rules

- Host: `1.34.182.186`
- Port: `5583`
- User: `kblueleaf`
- Preferred storage root: `/slow`
- Forbidden path: `/Yamiyoru`
- Default local bundle path:
  `../taiwan-server-migration-bundle-2026-02-22`
- Default SSH key path:
  `../taiwan-server-migration-bundle-2026-02-22/credentials/key`

## Review Flow

1. `ngagent review <task-id>` runs `./scripts/ci/ngagent_taiwan_review.sh` from
   the task worktree.
2. The wrapper computes a task-scoped remote scratch directory under
   `/slow/ngagent-review/omicverse/<task-id>`.
3. The wrapper uses `rsync` over SSH to mirror the current worktree into
   `<remote-task-root>/repo`.
4. The wrapper invokes `./scripts/ci/ngagent_taiwan_remote_pytest.sh` on the
   Taiwan host.
5. The remote helper creates or refreshes a task-local virtualenv under
   `<remote-task-root>/.ngagent-venv`, preferring
   `~/micromamba/envs/aliyawak/bin/python --system-site-packages` when present.
6. The helper installs `omicverse` plus the `tests` extra with
   `pip install -e ".[tests]"`.
7. The helper exports `OV_AGENT_RUN_HARNESS_TESTS=1` and runs the bounded smart
   agent regression suite.

## Default Remote Validation Set

The remote helper runs the smart-agent refactor safety net instead of the whole
repo:

- `tests/utils/test_agent_initialization.py`
- `tests/utils/test_smart_agent.py`
- `tests/utils/test_agent_backend_streaming.py`
- `tests/utils/test_agent_backend_usage.py`
- `tests/utils/test_agent_backend_providers.py`
- `tests/utils/test_ovagent_run_store.py`
- `tests/utils/test_ovagent_tool_runtime.py`
- `tests/utils/test_ovagent_workflow.py`
- `tests/utils/test_harness_cleanup.py`
- `tests/utils/test_harness_cli.py`
- `tests/utils/test_harness_compaction.py`
- `tests/utils/test_harness_contracts.py`
- `tests/utils/test_harness_runtime_state.py`
- `tests/utils/test_harness_tool_catalog.py`
- `tests/utils/test_harness_web_bridge.py`
- `tests/jarvis/test_session_shared_adata.py`
- `tests/test_claw_cli.py`

Override the list with `OV_NGAGENT_REMOTE_PYTEST_TARGETS` when a task needs a
different Taiwan-only target set.

The default command temporarily excludes
`test_agentic_loop_retries_after_text_only_promise_until_tool_call` because the
current `dev` baseline fails it before any task-007 changes. Remove that
exclusion once task-001 hardens the smart-agent retry contract.

## ngagent Configuration

The root orchestrator state must set:

```bash
./scripts/ci/ngagent_taiwan_review.sh
```

as `config.test_cmd`.

The recommended update path is `ngagent init --test-cmd
"./scripts/ci/ngagent_taiwan_review.sh"` or another `ngagent`-supported state
update flow that preserves existing task metadata.

## First-Run Expectations

- The first Taiwan review run will be slower because the remote virtualenv may
  need to install project dependencies.
- Subsequent reviews reuse the same task-local environment until `pyproject.toml`
  or the remote helper changes.
- Because `ngagent review` has a 300-second command timeout, prewarming the
  remote environment before the first formal review is recommended.

## Environment Overrides

- `TAIWAN_REMOTE_HOST`
- `TAIWAN_REMOTE_PORT`
- `TAIWAN_REMOTE_USER`
- `TAIWAN_SSH_KEY_PATH`
- `TAIWAN_REMOTE_BASE`
- `TAIWAN_REMOTE_TASK_ROOT`
- `OV_NGAGENT_REMOTE_ENV_DIR`
- `OV_NGAGENT_REMOTE_BASE_PYTHON`
- `OV_NGAGENT_REMOTE_PYTEST_TARGETS`
