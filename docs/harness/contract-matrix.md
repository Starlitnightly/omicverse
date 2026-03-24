# Smart Agent Contract Matrix

> **Purpose**: Enumerate public APIs and behavioral contracts that must survive
> any refactor of `smart_agent.py`.  Each row is guarded by at least one
> regression test in `tests/utils/test_agent_contract.py`.

## Public API Surface

| Symbol | Kind | Signature | Return | Behavioral Contract | Test Guard |
|--------|------|-----------|--------|---------------------|------------|
| `Agent` | factory fn | `Agent(model, api_key, endpoint, …, *, config, reporter, verbose) -> OmicVerseAgent` | `OmicVerseAgent` | Delegates all kwargs to `OmicVerseAgent.__init__`; callers may use positional or keyword style. | `test_agent_factory_returns_omicverseagent` |
| `OmicVerseAgent` | class | `__init__(model, api_key, endpoint, …, *, config, reporter, verbose)` | — | Sets `.model`, `.provider`, `.api_key`, `.endpoint`, `.enable_reflection`, `.use_notebook_execution`; initializes `._llm`, `._config`, `._reporter`. | `test_agent_factory_returns_omicverseagent` |
| `OmicVerseAgent.run` | sync method | `run(request: str, adata: Any) -> Any` | processed adata | Thread-safe sync wrapper: uses `asyncio.run` or spawns a thread if a loop is already running. | `test_run_outside_event_loop`, `test_run_inside_running_loop` (existing) |
| `OmicVerseAgent.run_async` | async method | `run_async(request: str, adata: Any) -> Any` | processed adata | Detects direct Python; otherwise enters agentic loop via `_run_agentic_mode`. | `test_run_async_delegates_to_agentic_loop`, `test_run_async_direct_python_bypasses_llm` |
| `OmicVerseAgent.stream_async` | async generator | `stream_async(request, adata, cancel_event=None, history=None, approval_handler=None)` | yields `dict` | Each yielded dict has `"type"` key. Terminal event is `type="done"`. Errors emit `type="error"` then `type="done"`. | `test_stream_async_yields_typed_events`, `test_stream_async_error_emits_error_then_done` |
| `OmicVerseAgent.restart_session` | method | `restart_session() -> None` | None | Clears `._notebook_executor.current_session` and resets prompt count when notebook execution is enabled; no-op otherwise. | `test_restart_session_clears_executor_state`, `test_restart_session_noop_without_notebook` |
| `OmicVerseAgent.get_session_history` | method | `get_session_history() -> List[Dict]` | list | Returns `._notebook_executor.session_history` when notebook execution is enabled; empty list otherwise. | `test_get_session_history_returns_executor_history`, `test_get_session_history_empty_without_notebook` |
| `list_supported_models` | module fn | `list_supported_models(show_all: bool = False) -> str` | str | Delegates to `ModelConfig.list_supported_models`. | `test_list_supported_models_returns_string` |

## Workflow Injection

| Seam | Contract | Test Guard |
|------|----------|------------|
| `OmicVerseRuntime.compose_system_prompt(base)` | Appends `workflow.build_prompt_block()` to `base` when the workflow document has body or default_tools; returns `base` unchanged otherwise. | `test_compose_system_prompt_injects_workflow`, `test_compose_system_prompt_noop_without_body` |
| `OmicVerseAgent._setup_agent` → `OmicVerseRuntime` | Agent creates an `OmicVerseRuntime` during `_setup_agent` and uses its workflow for system prompt composition. | Covered transitively by workflow injection tests. |

## Downstream Integration Seams

| Consumer | Seam | Contract | Test Guard |
|----------|------|----------|------------|
| **Claw** (`omicverse/claw.py`) | `_load_agent_factory()` → `Agent` | Returns the `Agent` callable from `smart_agent`. `_build_agent(args)` passes `model`, `api_key`, `endpoint`, `enable_reflection`, and disables `enable_result_review`, `use_notebook_execution`, `enable_filesystem_context`, `verbose`. | `test_claw_load_agent_factory_returns_callable`, `test_claw_build_agent_passes_expected_kwargs` |
| **Jarvis** (`omicverse/jarvis/session.py`) | `_load_agent_factory()` → `Agent` | Same `Agent` callable. `SessionManager._build_agent` passes `model` and a Jarvis-specific `kernel_root` configuration. `JarvisSession.agent` holds the returned instance. | Covered by existing `tests/jarvis/test_session_shared_adata.py`. |

## Module Exports

| Module | Exported Symbols | Test Guard |
|--------|------------------|------------|
| `omicverse.utils.smart_agent` | `Agent`, `OmicVerseAgent`, `list_supported_models` (`__all__`) | `test_utils_exports_agent_entrypoints` (existing) |
| `omicverse.utils` | Re-exports `Agent`, `OmicVerseAgent`, `list_supported_models`, `smart_agent` module | `test_utils_exports_agent_entrypoints` (existing) |

## Critical Behavioral Invariants

1. **`run()` is always safe to call from sync or async contexts** — it detects a running event loop and spawns a thread if needed.
2. **`stream_async` always terminates with a `done` event** — even on error, a `done` event is emitted after the `error` event.
3. **`restart_session` never raises** — it prints status and is a no-op when notebook execution is disabled.
4. **`get_session_history` never raises** — returns `[]` when notebook execution is disabled.
5. **`Agent()` is a thin wrapper** — it must remain a simple delegation to `OmicVerseAgent.__init__` with identical parameter lists.
6. **Workflow injection is additive** — `compose_system_prompt` must never truncate or replace the base prompt.
