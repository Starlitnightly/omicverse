# Project Goals

## Objective
Design and stage a safe refactor of `omicverse/utils/smart_agent.py` so the agent runtime becomes easier to maintain without changing user-visible behavior. The current file is a large facade-plus-implementation hybrid even though major runtime pieces already exist under `omicverse/utils/ovagent/`. The goal is to move the remaining high-coupling responsibilities into clearer module boundaries and prepare implementation tasks for worker agents.

## Success Criteria
- [ ] `smart_agent.py` is reduced to a thin public facade/composer instead of owning multiple execution subsystems directly.
- [ ] Public APIs and expected behaviors remain stable for `Agent`, `OmicVerseAgent`, `run_async`, `stream_async`, `restart_session`, `get_session_history`, and `list_supported_models`.
- [ ] `Jarvis`, `Claw`, `MCP`, workflow/run-store integration, and OAuth/auth flows continue to function without interface regressions.
- [ ] Validation for implementation tasks runs through the Taiwan server workflow rather than assuming local-only review execution.
- [ ] Refactor work is decomposed into bounded tasks with explicit acceptance criteria, dependency order, and risk levels before any worker execution starts.
- [ ] Existing targeted tests around smart agent, ovagent runtime, harness tooling, Jarvis, and MCP remain the primary regression guardrails during execution.

## Out of Scope
- New product features or user-facing behavior changes
- Rewriting the agent architecture from scratch
- Unifying `OvIntelligence` into the main `ov.Agent` runtime in this phase
- Direct edits to scientific analysis logic unrelated to agent composition
- Launching implementation workers before human approval of the plan

## Constraints
- Planning and orchestration happen inside the `omicverse` git repository, not the outer wrapper repo.
- The orchestration flow must use `ngagent`; task metadata must not be edited manually.
- Public compatibility is non-negotiable for downstream consumers already wired to the current agent stack.
- High-risk touchpoints include streaming event shape, tool schema visibility, workflow artifact linkage, and authentication/backend setup.
- Validation must honor the Taiwan server requirement; local `pytest` is not sufficient as the final source of truth for task review.
- Current session scope is planning and decomposition only; no implementation code is to be written directly by the main orchestrator.
