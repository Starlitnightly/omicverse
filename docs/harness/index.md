# OVAgent Harness

This directory is the system of record for the OVAgent harness layer.

## Core Docs

- `core-beliefs.md`: non-negotiable harness principles
- `runtime-contract.md`: shared event, trace, replay, and artifact contracts
- `tool-catalog.md`: current runtime catalog and Claude-style substrate mapping
- `tool-loading.md`: static-vs-deferred loading rules and future invariants
- `approval-and-questions.md`: pause/resume contract for approvals and future questions
- `server-validation.md`: remote validation policy and native `ngagent remote` commands
- `cleanup-policy.md`: recurring cleanup and drift-control rules

## Server-Only Entry Points

These commands are intentionally server-gated and should not be treated as
public-CI entrypoints:

- `python -m omicverse.utils.verifier replay <trace_id>`
- `python -m omicverse.utils.verifier scenario <trace_id> --name ...`
- `python -m omicverse.utils.verifier cleanup --save-report`

## Relationship To Existing Plans

The following documents remain useful implementation references, but the harness
contract lives here:

- `../agent_backend_improvement_plan.md`
- `../ovagent_legacy_deprecation_plan.md`
- `../ovagent_web_chatbot_plan.md`
