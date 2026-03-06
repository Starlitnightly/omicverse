# Approval And Questions

The current harness already supports approval-driven pause and resume for risky
execution. Arbitrary user questions are not yet a first-class runtime surface.

## Current Approval Contract

- approval requests are emitted as structured harness events
- approval responses are stored in session state
- the runtime broker can block a turn until approval resolves
- replay and session history retain approval-linked trace identifiers

## Current Session State

Session summaries already expose the runtime state that approval-aware clients
need:

- `active_turn_id`
- `trace_count`
- `last_trace_id`
- `pending_approvals`

## Question Flow Direction

When `AskUserQuestion` is added, it should reuse the same pause/resume shape as
approvals:

- attach to `session_id`, `turn_id`, and `trace_id`
- persist request and resolution in session state
- emit structured events instead of free-form text markers
- remain server-gated in harness tests

