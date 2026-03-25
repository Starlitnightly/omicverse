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

## Review Execution

Server-only harness validation may be orchestrated externally, but this
repository does not treat any transport/bootstrap wrapper as part of the OV
runtime contract. The stable contract here is only:

- harness validation is server-gated
- required environment flags must be set explicitly
- verifier commands must run against the server environment
