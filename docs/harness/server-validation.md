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

## Harness CLI Commands

All of the following are server-only:

```bash
OV_AGENT_RUN_HARNESS_TESTS=1 python -m omicverse.utils.verifier replay <trace_id>
OV_AGENT_RUN_HARNESS_TESTS=1 python -m omicverse.utils.verifier scenario <trace_id> --name smoke
OV_AGENT_RUN_HARNESS_TESTS=1 python -m omicverse.utils.verifier cleanup --save-report
```

## ngagent Review Bridge

For ngagent-managed task review, use `./scripts/ci/ngagent_taiwan_review.sh`.
That wrapper syncs the current task worktree to `/slow/ngagent-review/...` and
executes the remote validation helper described in
`docs/harness/ngagent-review-bridge.md`.
