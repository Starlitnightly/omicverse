# Server Validation

Harness validation is executed only on the Taiwan server.

## Policy

- No local harness tests
- No public-CI harness execution
- All harness scenario, replay, and cleanup checks must be opt-in and server-gated

## Recommended Environment Flag

- `OV_AGENT_RUN_HARNESS_TESTS=1`

## Minimum Validation Set

- harness contract tests
- web bridge compatibility tests
- cleanup report generation tests
- targeted runtime trace tests
- **E2E PBMC pipeline validation** (`tests/utils/test_e2e_pbmc_validation.py`)

## E2E PBMC Pipeline Validation

The E2E validation suite exercises the full OVAgent runtime stack through a
realistic PBMC single-cell analysis scenario.  It uses staged mock LLM
responses to drive the real subsystem instances without requiring API keys
or heavy optional dependencies.

### What it validates

| Subsystem | Exercised? |
|-----------|-----------|
| TurnController (agentic loop) | Yes |
| ToolRuntime (dispatch) | Yes |
| ToolScheduler (batching) | Yes |
| ContextBudgetManager | Yes |
| PermissionPolicy | Yes |
| ExecutionRepairLoop | Yes |
| RuntimeEventEmitter | Yes |
| ToolRegistry | Yes |
| PromptBuilder | Yes |
| FollowUpGate (recovery) | Yes |

### Prerequisites

All satisfied by `pip install -e ".[tests]"`:
- pytest >= 7.0
- numpy (for mock AnnData matrix)
- No LLM API key required (staged mock LLM)
- No scanpy required (lightweight MockAnnData)

### How to run

```bash
pytest -q tests/utils/test_e2e_pbmc_validation.py
```

The aggregate test (`TestE2EAggregateReport::test_aggregate_e2e_validation`)
outputs a structured JSON report to stdout that can be captured as a
server-validation artifact.

### Pipeline stages

1. **QC** — filter cells by gene count threshold
2. **Preprocessing** — log-normalization + 2000 HVG selection
3. **Clustering** — Leiden cluster assignment
4. **Finish** — summary report

Each stage is dispatched as an `execute_code` tool call through the real
TurnController -> ToolScheduler -> ToolRuntime -> ExecutionRepairLoop path.

## Harness CLI Commands

All of the following are server-only:

```bash
OV_AGENT_RUN_HARNESS_TESTS=1 python -m omicverse.utils.verifier replay <trace_id>
OV_AGENT_RUN_HARNESS_TESTS=1 python -m omicverse.utils.verifier scenario <trace_id> --name smoke
OV_AGENT_RUN_HARNESS_TESTS=1 python -m omicverse.utils.verifier cleanup --save-report
```

## Review Execution

Use native `ngagent remote` configuration for Taiwan review execution.
Do not add repository-specific sync wrappers or bridge scripts to `omicverse`.

The stable contract here is:

- harness validation is server-gated
- `ngagent` remote connection details live in local `.orchestrator/record.json`
- `.orchestrator/` is local orchestration state and must not be committed
- verifier commands must run against the server environment

## Native `ngagent remote` Flow

Initialize local orchestration state with remote review configured from the
start:

```bash
ngagent init omicverse \
  --test-cmd "pytest -q" \
  --lint-cmd "ruff check ." \
  --typecheck-cmd "python -m py_compile omicverse/utils/smart_agent.py" \
  --remote-host "<user>@<taiwan-host>" \
  --remote-port <ssh-port> \
  --remote-key "<path-to-ssh-key>" \
  --remote-workspace "<remote-workspace>" \
  --remote-activate "micromamba activate <env-name>"
```

If `.orchestrator/` already exists, update the remote review path in place:

```bash
ngagent remote set \
  --host "<user>@<taiwan-host>" \
  --port <ssh-port> \
  --key "<path-to-ssh-key>" \
  --workspace "<remote-workspace>" \
  --activate "micromamba activate <env-name>"
```

Validate the SSH and activation path before running task review:

```bash
ngagent remote test
```

When a dedicated review environment is needed on the Taiwan host, bootstrap it
with the native remote helper and then update `--activate` to match that
environment:

```bash
ngagent remote setup --python 3.12 --env-name ngagent-dev --install "-e .[tests]"
```

After remote configuration is in place, the normal review entry point stays the
same:

```bash
ngagent review <TASK_ID>
```
