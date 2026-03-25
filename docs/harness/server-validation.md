# Server Validation

Harness validation is executed only on an operator-managed remote review host.

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

## ngagent Remote Review

### Preferred path: native `ngagent remote`

Configure the remote review environment using the native ngagent CLI:

```bash
# One-time setup (operator-local, not committed to repo)
ngagent remote set --host <REMOTE_HOST> \
                   --user <USER> \
                   --key <SSH_KEY_PATH> \
                   --workspace <REMOTE_WORKSPACE> \
                   --activate "<REMOTE_ACTIVATE_CMD>"

# Verify connectivity
ngagent remote test

# Bootstrap remote environment (installs test deps)
ngagent remote setup

# Run review for a specific task
ngagent review <TASK_ID>
```

Configuration is persisted in `.orchestrator/record.json.config.remote`
(operator-local, never committed).  See `RemoteReviewConfig` in
`omicverse/utils/ovagent/contracts.py` for the configuration shape.

### Constraints

- `ngagent review` enforces a 300-second timeout per command
- Remote workspace, interpreter, and activation command are operator-specific
  and must not be hard-coded into the repository
- Prewarm the remote environment before the first formal review run
- Keep remote-only credentials, hostnames, IPs, and local bundle paths out of
  versioned files

### Legacy path (deprecated)

Repository-specific wrapper scripts are superseded by native `ngagent remote`
and should not be added back to the repo.
