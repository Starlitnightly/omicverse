# Runtime Contract

The OVAgent harness has four durable runtime assets:

- `HarnessEvent`: normalized stream event emitted by the agent loop
- `StepTrace`: one structured execution step
- `RunTrace`: one end-to-end agent turn
- `ArtifactRef`: a durable pointer to code, notebooks, files, or reports

## Invariants

- `smart_agent.py` emits harness events.
- `omicverse_web/services/agent_service.py` forwards harness events without redefining their schema.
- Session history stores trace identifiers and summaries, not only free-form text.
- Replay and cleanup operate on stored `RunTrace` files.

## Target Runtime Contracts

The following contracts define the target interfaces for the OVAgent runtime
roadmap.  They are codified as Python dataclasses and protocols in
`omicverse/utils/ovagent/contracts.py` and tested by
`tests/utils/test_runtime_contracts.py`.

### Tool Policy Metadata

Each registered tool carries policy metadata beyond its name and schema:

| Field | Type | Purpose |
|-------|------|---------|
| `approval` | `ApprovalClass` (allow/ask/deny) | Whether execution needs user approval |
| `isolation` | `IsolationMode` (in_process/subprocess/worktree) | Execution isolation level |
| `parallel` | `ParallelClass` (safe/unsafe/conditional) | Whether the tool can run concurrently |
| `output_tier` | `OutputTier` (minimal/standard/verbose/unbounded) | Context budget hint for tool output |
| `read_only` | `bool` | Whether the tool has side effects |
| `max_output_tokens` | `Optional[int]` | Hard cap on output tokens |
| `timeout_seconds` | `Optional[float]` | Per-invocation timeout |

Migration path: `ToolDefinition.high_risk` → `approval=ASK`;
`ToolDefinition.server_only` → `isolation=SUBPROCESS`.

### Context Budget

The context budget manager replaces character-based clipping with token-aware
budgeting:

- **ImportanceTier**: critical > high > standard > low > ephemeral
- **ContextEntry**: content + source + importance + token count
- **OverflowPolicy**: truncate_oldest, compact_low_importance,
  summarize_and_drop, reject
- **ContextBudgetConfig**: max_tokens, reserve_tokens,
  compaction_threshold, per_tier_limits

Migration path: `ContextCompactor` → `ContextBudgetManager` protocol.

### Execution Failure Envelope

Structured failure payloads replace regex-based error string parsing:

- **FailurePhase**: pre_exec, execution, post_exec, normalization, timeout
- **RepairHint**: strategy + description + confidence
- **ExecutionFailureEnvelope**: tool_name, phase, exception_type, message,
  stderr_summary, traceback_summary, retry_count, repair_hints

The envelope provides `to_llm_message()` for LLM-consumable formatting and
`retryable` for the repair loop to check.

Migration path: `ProactiveCodeTransformer` error patterns → structured envelopes
fed into a multi-attempt repair loop.

### Event Stream

The `EventEmitter` protocol unifies ad-hoc event emission:

- `emit(event_type, content, category, step_id, metadata)` — general events
- `emit_failure(envelope)` — failure-specific events

Event categories: lifecycle, tool, code, llm, approval, question, task, trace.

Migration path: `_emit()` + `print()` calls → `EventEmitter` implementation
backed by `HarnessEvent`.

### Prompt Composition

The `PromptComposer` protocol replaces string concatenation:

- **PromptLayer**: kind + content + priority + source + token_estimate
- **PromptLayerKind**: base_system, provider, workflow, skill,
  runtime_state, context
- Layers are composed in priority order with token budget awareness

Migration path: `PromptBuilder.build_filesystem_context_instructions()` and
`OmicVerseRuntime.compose_system_prompt()` → `PromptComposer` with typed layers.

### Remote Review

The `RemoteReviewConfig` shape documents the ngagent remote review settings:

- host, user, key_path, workspace, activate_cmd, timeout_seconds
- Serialized via `to_dict()` / `from_dict()` round-trip
- Lives in operator-local config, never committed to the repo

Configuration path: `ngagent init --remote-*` or `ngagent remote set`.
