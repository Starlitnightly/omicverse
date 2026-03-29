# OVAgent Legacy Path Deprecation Plan

> **Status: COMPLETED (2026-02-28)**
> All legacy Priority 1/2 methods have been removed (~1,182 lines deleted).
> `execute_code` enhanced with pattern-based auto-recovery.
> `stream_async` rewritten to wrap the agentic loop.
> `agent_mode` parameter retained for backward compatibility but deprecated
> and ignored — passing any value other than `"agentic"` emits a
> `DeprecationWarning`.  The agentic tool-calling loop is the sole
> execution mode.

## Scope

This plan deprecates and removes legacy Priority 1/2 execution paths in `omicverse/utils/smart_agent.py`, while preserving useful behavior by migrating it into:

- agentic loop tools (`AGENT_TOOLS` + `_dispatch_tool`)
- reusable internal helpers used by agentic tools
- curated skills (`.claude/skills`) where prompt guidance is the right abstraction

Primary target module: `omicverse/utils/smart_agent.py`

## Original Problem (resolved)

`OmicVerseAgent` previously mixed two architectures:

- Agentic tool loop (now the sole execution mode)
- Legacy single-shot code generation and fallback workflows (Priority 1/2)

This increased maintenance cost, created behavior divergence (`run()` vs `stream_async()`), and left stale docs/tests.  All legacy paths have since been removed; `run()` and `stream_async()` both execute via the agentic tool-calling loop.

## Goals

1. Make agentic loop the only execution architecture.
2. Preserve useful legacy capabilities as tools/helpers/skills.
3. Keep a short compatibility window with explicit deprecation warnings.
4. Align `run()`, `stream_async()`, and web service behavior.

## Non-Goals

1. Rewriting provider backend (`agent_backend.py`) protocol layer.
2. Removing notebook execution or security scanner features.
3. Changing user-facing `ov.Agent(...)` constructor beyond the `agent_mode` deprecation.

## Legacy Inventory and Disposition

| Legacy path | Current role | Decision | Agentic destination |
|---|---|---|---|
| `_run_legacy_mode` | Orchestrates Priority 1/2 | Deprecate then remove | None |
| `_run_registry_workflow` | Priority 1 fast single-shot | Remove | Replace with agentic direct tool-call policy |
| `_run_skills_workflow` | Priority 2 single-shot skills workflow | Remove | Skills remain via `search_skills` tool |
| `run_async_LEGACY` | Old full codegen path | Remove | None |
| `_analyze_task_complexity` | Routes simple/complex for legacy | Remove | None (agentic handles complexity dynamically) |
| `_validate_simple_execution` | Enforces Priority 1 constraints | Remove | None |
| `_select_skill_matches` | Deprecated non-LLM skill matching | Remove | None |
| `stream_async` (legacy implementation) | Streams legacy codegen flow | Rewrite | Agentic event stream |
| `_reflect_on_code` | LLM code review/refinement | Keep and repurpose | New tool: `review_code` (optional) |
| `_review_result` | Post-execution intent validation | Keep and repurpose | New tool: `review_result` (optional) |
| `_apply_execution_error_fix` | Pattern-based fix retries | Keep | Internal helper in `execute_code` tool |
| `_diagnose_error_with_llm` | Error diagnosis + patch generation | Keep and repurpose | New tool: `diagnose_error` or internal retry tier |
| `_generate_completion_code` | Generate missing output file steps | Keep and repurpose | New tool: `complete_outputs` or internal post-check |
| `_extract_python_code` and candidate helpers | Code extraction utility | Keep | Internal utility for diagnosis/review helpers |

## Tool/Skill Migration Decisions

### New/Updated Agentic Tools

1. `execute_code` (enhance existing):
   - Keep current behavior.
   - Add built-in two-stage recovery:
     - stage A: `_apply_execution_error_fix`
     - stage B: `_diagnose_error_with_llm`
   - Return structured status fields (`ok`, `error_type`, `recovered`).

2. `review_result` (new):
   - Wrap `_review_result`.
   - Input: `request`, optional `notes`.
   - Output: match/confidence/issues/recommendation.
   - Use after significant processing blocks.

3. `review_code` (new, optional):
   - Wrap `_reflect_on_code`.
   - Use before risky operations (integration, trajectory, large parameter sweeps).

4. `complete_outputs` (new, optional):
   - Wrap `_validate_outputs` + `_generate_completion_code`.
   - Used when user asks for files/reports and outputs are missing.

### Skills (Prompt Guidance) Instead of Code Paths

Move workflow-level knowledge to skills, not runtime branches:

- legacy “simple task fast path” heuristics -> skill snippets for concise single-step tasks.
- legacy “complex multi-step prompt template” -> workflow skills (preprocess/clustering/DEG/spatial).
- maintain domain best practices in skill documents, not in `_run_skills_workflow`.

## Phased Rollout

### Phase 0: Guardrails and Visibility

1. ~~Add structured warnings when `agent_mode='legacy'` is used.~~ **Done** — `DeprecationWarning` emitted when `agent_mode != "agentic"`.
2. Add telemetry events (`category=deprecation`) for legacy entry points.
3. ~~Update docs to state: agentic is canonical.~~ **Done.**

Exit criteria:
- No hidden legacy path usage in tests or web service. **Met.**

### Phase 1: Behavior Parity in Agentic Mode

1. Integrate retry/recovery helpers into agentic `execute_code`.
2. Add `review_result` tool and wire into system prompt guidance.
3. Rewrite `stream_async` to emit agentic loop events.

Exit criteria:
- `run()` and `stream_async()` both execute via agentic loop.
- web service keeps working through new `stream_async` contract.

### Phase 2: Deprecation Window

1. Mark legacy methods with deprecation decorators/comments.
2. Keep thin wrappers that raise `DeprecationWarning` and forward to agentic equivalents where possible.
3. Update notebook tutorials and web docs.

Exit criteria:
- No direct calls to legacy methods in repo code/tests/docs.

### Phase 3: Removal

1. Delete:
   - `_run_legacy_mode`
   - `_run_registry_workflow`
   - `_run_skills_workflow`
   - `run_async_LEGACY`
   - `_analyze_task_complexity`
   - `_validate_simple_execution`
   - `_select_skill_matches`
2. Simplify `run_async()` to a single execution path.
3. Remove obsolete docs and tests tied to Priority 1/2 wording.

Exit criteria:
- `smart_agent.py` has one orchestration architecture.

## Required Code Changes by File

### `omicverse/utils/smart_agent.py`

1. Add new tools to `AGENT_TOOLS` and `_dispatch_tool`:
   - `review_result`
   - `review_code` (optional)
   - `complete_outputs` (optional)
2. Refactor `_tool_execute_code` to use shared recovery helper pipeline.
3. Replace legacy `stream_async` implementation with agentic event streaming.
4. Add deprecation warnings for legacy mode/methods.
5. Remove legacy methods in Phase 3.

### `omicverse/omicverse_web/services/agent_service.py`

1. Validate compatibility with new agentic `stream_async` event types.
2. Keep `run_agent_stream` parser stable during migration.

### `omicverse/docs/*` and `omicverse_guide/*`

1. Remove Priority 1/2 architecture language.
2. Document new agentic tools and recommended invocation patterns.
3. ~~Update stale API docs that mention `query/query_async`.~~ **Done** — docs now reference `run`/`run_async`/`stream_async`/`generate_code`.

## Test Plan

### New tests to add

1. Agentic tool-loop tests:
   - tool dispatch ordering
   - `finish` handling
   - max-turn termination
2. `execute_code` recovery tests:
   - pattern fix path
   - LLM diagnosis path
3. Agentic streaming tests:
   - event sequence
   - parity with `run()`
4. Deprecation tests:
   - warnings on legacy mode usage

### Existing tests to update/remove

1. Remove assertions that depend on Priority 1/2 wording.
2. Keep provider backend tests unchanged.
3. Keep security scanner tests unchanged.

## Review Checklist (for local review)

1. [x] `smart_agent.py` has a single runtime architecture.
2. [x] No repo references to:
   - “Priority 1”
   - “Priority 2”
   - `run_async_LEGACY`
3. [x] `stream_async` and `run()` produce consistent behavior (both use agentic loop).
4. [x] Web service agent streaming still functions.
5. [x] Docs reflect real APIs (`run`, `run_async`, `stream_async`, `generate_code`).

## Risk Notes

1. `stream_async` rewrite may affect web UI assumptions.
2. Legacy code contains useful recovery logic; do not remove until migrated.
3. Removing complexity classifier may change perceived speed on trivial tasks; mitigate with agentic prompt/tool guidance.

## Suggested Execution Order

1. Phase 0 + Phase 1 in a single feature branch.
2. Ship deprecation warnings (Phase 2) for one release cycle.
3. Remove legacy code (Phase 3) only after usage confirms near-zero legacy reliance.

