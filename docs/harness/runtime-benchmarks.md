# Runtime Benchmarks

Acceptance metrics for the OVAgent runtime roadmap.  Each benchmark maps to
a contract in `omicverse/utils/ovagent/contracts.py` and is enforced by
`tests/utils/test_runtime_contracts.py`.

## Benchmark Thresholds

| Subsystem | Metric | Target | Source |
|-----------|--------|--------|--------|
| Tool dispatch | Registry lookup latency | < 10 ms | `ToolContract` construction + dict round-trip |
| Context budget | Compaction recovery ratio | ≥ 20% of freed tokens | `ContextBudgetManager.compact()` |
| Event stream | Per-event emission overhead | < 1 ms | `EventEmitter.emit()` |
| Recovery loop | Failure envelope construction | < 5 ms | `ExecutionFailureEnvelope` creation + `to_llm_message()` |
| Prompt composition | Full prompt assembly | < 50 ms | `PromptComposer.compose()` |
| Remote review | Config serialization round-trip | < 1 ms | `RemoteReviewConfig.to_dict()` / `from_dict()` |

## Contract Completeness

Each contract must expose a minimum number of fields to be considered stable:

| Contract | Min required fields | Rationale |
|----------|-------------------|-----------|
| `ToolContract` | 3 (name, group, description) | Identity triple must always be present |
| `ContextEntry` | 3 (content, source, importance) | Entry must carry content, provenance, and priority |
| `ExecutionFailureEnvelope` | 4 (tool_name, phase, exception_type, message) | Minimum for the LLM to reason about the failure |
| `PromptLayer` | 2 (kind, content) | Layer must have a type and actual content |

## Baseline Measurements

These are captured from the current codebase as of the contract definition
date.  They serve as the starting point for gap tracking.

| Area | Current state | Target state |
|------|--------------|--------------|
| Tool dispatch | `if/elif` chain in `tool_runtime.py` (~60 KB) | Declarative registry with `ToolContract` |
| Context management | `ContextCompactor` with character-based clipping | Token-aware `ContextBudgetManager` with importance tiers |
| Error recovery | Regex-based `ProactiveCodeTransformer` | Structured `ExecutionFailureEnvelope` → repair loop |
| Event emission | Mix of `print()`, `_emit()`, `build_stream_event()` | Unified `EventEmitter` protocol |
| Prompt assembly | String concatenation in `PromptBuilder` | Layered `PromptComposer` with `PromptLayer` stack |
| Remote review | Repository-specific wrapper scripts | Native `ngagent remote set` / `ngagent review` |

## How to Run

```bash
# Contract + benchmark tests (no server required)
pytest tests/utils/test_runtime_contracts.py -v

# Full harness validation (remote review host only)
OV_AGENT_RUN_HARNESS_TESTS=1 pytest tests/utils/test_harness_contracts.py -v
```
