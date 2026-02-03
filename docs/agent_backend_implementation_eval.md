# OV Agent Backend Improvement — Implementation Evaluation

> Codex-inspired backend refactoring for `omicverse/utils/agent_backend.py`,
> `model_config.py`, `smart_agent.py`, and five new infrastructure modules.

---

## Phase P0 — Foundational Refactors

### P0-1  Provider Registry Pattern

| Metric | Before | After |
|--------|--------|-------|
| Source-of-truth data structures | 4 flat dicts + `OPENAI_COMPAT_BASE_URLS` | Single `PROVIDER_REGISTRY` (`Dict[str, ProviderInfo]`) |
| Provider dispatch in `_run_sync` | 5-branch `if/elif` | `_SYNC_DISPATCH[wire_api]` table lookup |
| Provider dispatch in `_stream_async` | 5-branch `if/elif` | `_STREAM_DISPATCH[wire_api]` table lookup |
| API-key resolution `_resolve_api_key` | 8-branch `if/elif` | Registry `env_key` + `alt_env_keys` loop |
| Adding a new provider | Touch 5+ dicts/methods | Call `register_provider(ProviderInfo(...))` once |

**Files changed**: `model_config.py` (full rewrite), `agent_backend.py` (dispatch tables, resolve_api_key).

**Backward compat**: Legacy dicts (`AVAILABLE_MODELS`, `PROVIDER_API_KEYS`, `PROVIDER_ENDPOINTS`, `PROVIDER_DEFAULT_KEYS`) are auto-computed from the registry.  All downstream code that imports them (including `smart_agent.py`) continues to work unchanged.

**New public API**:
- `WireAPI` enum (5 values)
- `ProviderInfo` dataclass
- `register_provider()` / `get_provider()` functions
- `PROVIDER_REGISTRY` dict

---

### P0-2  Grouped Config Dataclasses

**New file**: `agent_config.py`

| Dataclass | Key fields |
|-----------|-----------|
| `LLMConfig` | model, api_key, endpoint, reasoning_effort |
| `ReflectionConfig` | enabled, iterations (clamped 1-3), result_review |
| `ExecutionConfig` | use_notebook, max_prompts, storage_dir, timeout, sandbox_fallback_policy |
| `ContextConfig` | enabled, storage_dir |
| `AgentConfig` | llm, reflection, execution, context, verbose, history_enabled |

**Backward compat**: `AgentConfig.from_flat_kwargs(**kw)` accepts the original 14 keyword args.

**Integration**: `OmicVerseAgent.__init__` now accepts an optional `config: AgentConfig` keyword-only parameter.  When omitted, `from_flat_kwargs()` builds it from the flat params.

---

### P0-3  Structured Error Hierarchy

**New file**: `agent_errors.py`

```
OVAgentError
├── WorkflowNeedsFallback    # Priority 1 → 2 signal (not a bug)
├── ProviderError             # .provider, .status_code, .retryable
├── ConfigError               # Bad model / missing key
└── ExecutionError
    └── SandboxDeniedError    # Notebook execution denied
```

**Integration**:
- `_run_registry_workflow` raises `WorkflowNeedsFallback` instead of `ValueError` when LLM returns `NEEDS_WORKFLOW`.
- `run_async` catches `(ValueError, WorkflowNeedsFallback)` for fallback.
- `_execute_generated_code` raises `SandboxDeniedError` when policy is `RAISE`.

---

## Phase P1 — Code Organization

### P1-1  Event/Reporter Pattern

**New file**: `agent_reporter.py`

| Reporter | Behavior |
|----------|----------|
| `PrintReporter` | Default — emoji-rich stdout (matches legacy `print()`) |
| `SilentReporter` | `logging` only — ideal for tests / batch |
| `CallbackReporter` | Forwards `AgentEvent` to user-supplied callable |

**Factory**: `make_reporter(verbose=..., callback=..., reporter=...)`

**Integration**: `OmicVerseAgent.__init__` creates a `_reporter` and provides `_emit(level, msg, category)` helper.  The notebook fallback path now uses `_emit` when available, falling back to `print()` for defensive safety.

### P1-2  Execution Method Split (smart_agent.py integration)

`_execute_generated_code` now obeys `SandboxFallbackPolicy`:
- `RAISE`: immediately raises `SandboxDeniedError` on notebook failure
- `WARN_AND_FALLBACK`: emits warning via reporter then falls back to in-process
- `SILENT`: falls back silently (legacy behavior)

### P1-3  Retry De-duplication + GPT-5 Extractor

**Retry**: Added `OmicVerseLLMBackend._retry(func)` instance method that delegates to the module-level `_retry_with_backoff` with config-derived parameters.  All **12 call-sites** replaced: `_retry_with_backoff(_make_sdk_call, max_attempts=..., base_delay=..., factor=..., jitter=...)` → `self._retry(_make_sdk_call)`.

| Before (per call-site) | After |
|------------------------|-------|
| 5-line `_retry_with_backoff(fn, max_attempts=..., ...)` | 1-line `self._retry(fn)` |
| 12 × 5 = 60 lines of boilerplate | 12 × 1 = 12 lines |

**GPT-5 extractor**: Added `_extract_responses_text_from_dict(payload)` static method consolidating the duplicated output-text / output.text / output.content[*].text fallback chain used in both SDK and HTTP paths.

---

## Phase P2 — Advanced Features

### P2-1  Context Window Compaction

**New file**: `context_compactor.py`

- `estimate_tokens(text)` — fast heuristic (4 chars/token ASCII, 2 chars/token CJK)
- `MODEL_CONTEXT_WINDOWS` — maps model IDs to context limits (128K–1M)
- `ContextCompactor.needs_compaction(system_prompt, user_prompt)` — triggers at 75%
- `ContextCompactor.compact(system_prompt)` — LLM-summarized compression

### P2-2  Session Persistence

**New file**: `session_history.py`

- `HistoryEntry` — dataclass (session_id, timestamp, request, generated_code, result_summary, usage, priority_used, success)
- `SessionHistory` — append-only JSONL file at `~/.ovagent/history.jsonl`
  - `.append(entry)`, `.get_session(id)`, `.get_recent(n)`
  - `.build_context_for_llm(session_id, max_entries)` — concise summary for injection into system prompt

### P2-3  Notebook Fallback Policy

**New enum**: `SandboxFallbackPolicy` (in `agent_config.py`)

| Policy | Behavior on notebook failure |
|--------|------------------------------|
| `RAISE` | `SandboxDeniedError` — no fallback |
| `WARN_AND_FALLBACK` | Reporter warning + in-process (default) |
| `SILENT` | Silent fallback (legacy) |

Integrated in `_execute_generated_code` with explicit policy checking.

---

## Phase P3 — Performance

### P3-1  Shared ThreadPoolExecutor

| Before | After |
|--------|-------|
| `with concurrent.futures.ThreadPoolExecutor() as executor:` per stream | Module-level `_get_shared_executor()` singleton |
| Thread pool created/destroyed per streaming call | Single 4-worker pool reused across all calls |
| `atexit` cleanup: none | `atexit.register(executor.shutdown, wait=False)` |

**Fix**: Gemini streaming also changed from `list(response)` → direct `for chunk in response:` iteration to enable true token-by-token streaming instead of buffering the entire response.

---

## Summary of Changes

| File | Status | Change Type |
|------|--------|-------------|
| `omicverse/utils/model_config.py` | **Rewritten** | P0-1: Provider registry, WireAPI enum, ProviderInfo |
| `omicverse/utils/agent_backend.py` | **Major edit** | P0-1 dispatch tables, P1-3 retry/extractor, P3-1 shared executor |
| `omicverse/utils/agent_config.py` | **New** | P0-2: AgentConfig, SandboxFallbackPolicy |
| `omicverse/utils/agent_errors.py` | **New** | P0-3: Error hierarchy |
| `omicverse/utils/agent_reporter.py` | **New** | P1-1: Reporter protocol + 3 built-in implementations |
| `omicverse/utils/context_compactor.py` | **New** | P2-1: Context window compaction |
| `omicverse/utils/session_history.py` | **New** | P2-2: JSONL session persistence |
| `omicverse/utils/smart_agent.py` | **Modified** | P0-2/P0-3/P1-1/P2-3 integration |
| `omicverse/utils/__init__.py` | **Modified** | Export new modules |

**Lines of new code**: ~550 (across 5 new files)
**Lines of boilerplate removed**: ~100 (12 retry sites × 5 lines + 3 dispatch chains)
**Backward compatibility**: Full — all existing constructor signatures, dict imports, and API behaviors preserved.
