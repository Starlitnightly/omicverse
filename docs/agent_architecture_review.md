# OmicVerse Agent Architecture Review

> **Scope**: OmicVerseAgent and all supporting modules in `omicverse/utils/ovagent/`,
> excluding OvIntelligence.
>
> **Date**: 2026-03-27

---

## 1. Architecture Overview

OmicVerse Agent adopts an **extracted-module facade** pattern:

- **OmicVerseAgent** (`smart_agent.py`, ~1700 LOC) is a thin facade that wires
  subsystems together.
- Core logic is decomposed into ~32 focused modules under `omicverse/utils/ovagent/`.
- Modules depend on the `AgentContext` Protocol (duck-typing contract), not the
  concrete agent class.

### Component Map

| Component | File(s) | Responsibility |
|-----------|---------|----------------|
| TurnController | `turn_controller.py` | Main agentic loop (LLM call -> tool dispatch -> repeat) |
| ToolRegistry | `tool_registry.py` | Canonical name -> metadata mapping with policy enums |
| ToolRuntime | `tool_runtime.py`, `tool_runtime_*.py` | Tool dispatch hub + handler implementations |
| SubagentController | `subagent_controller.py` | Subagent spawning with isolated runtime |
| PromptBuilder | `prompt_builder.py`, `prompt_templates.py` | Deterministic prompt composition |
| AnalysisExecutor | `analysis_executor.py`, `analysis_sandbox.py` | Code execution with sandboxing |
| RepairLoop | `repair_loop.py`, `analysis_diagnostics.py` | Structured error recovery |
| PermissionPolicy | `permission_policy.py` | Multi-layered tool approval decisions |
| ContextBudgetManager | `context_budget.py` | Token-aware context budgeting |
| SkillRegistry | `skill_registry.py` | Progressive skill discovery and loading |
| OmicVerseLLMBackend | `agent_backend*.py` | Multi-provider LLM abstraction |
| AgentConfig | `agent_config.py` | Hierarchical configuration |

### Initialization Flow

```
AgentConfig creation
    |
Backend resolution (model/provider/endpoint)
    |
Skill registry + Function registry loading
    |
Subsystem bootstrap (notebook, filesystem, security, tracing)
    |
Extracted module instantiation:
  PromptBuilder -> AnalysisExecutor -> ToolRuntime
  -> SubagentController -> TurnController -> CodegenPipeline
```

---

## 2. Strengths

### 2.1 Protocol-Based Decoupling

`AgentContext` uses `@runtime_checkable` Protocol to define the contract between
the facade and extracted modules. This breaks circular imports between
`smart_agent.py` and the `ovagent/` subpackage and enables mock testing with
duck-typed doubles.

**Verdict**: Excellent design choice for a system of this complexity.

### 2.2 Registry Seam Pattern

`ToolRegistry` separates tool metadata (`ToolMetadata`) from runtime handlers.
Handlers are bound via `handler_key` at runtime, creating a seam that supports:

- Pluggable handler implementations
- `validate_handlers()` to detect unbound seams during testing
- Policy enums (`ApprovalClass`, `ParallelClass`, `OutputTier`, `IsolationMode`)
  that drive dispatch, scheduling, and permission decisions independently

**Verdict**: Clean separation of concerns. The frozen `ToolMetadata` dataclass
ensures immutability after registration.

### 2.3 Subagent Isolation

`SubagentRuntime` enforces strong isolation:

- Independent `PermissionPolicy` scoped to allowed tools
- Subagent-local `ContextBudgetManager` (no shared budget with parent)
- Tool schemas snapshotted at creation time (immune to parent mutations)
- Separate `last_usage` tracking
- `can_mutate_adata` flag controls data modification rights

Three subagent types (explore/plan/execute) with progressively wider tool sets.

**Verdict**: Well-designed isolation contract. The dataclass-based approach is
both simple and effective.

### 2.4 Layered Permission System

`PermissionPolicy` resolves decisions through a 6-level priority chain:

1. Explicit deny list
2. Allowlist restriction
3. Per-tool overrides
4. Per-class overrides (by `ApprovalClass`)
5. Registry metadata defaults
6. Unknown-tool fallback

Critically, evaluation is separated from enforcement — the policy returns a
`PermissionDecision` but does not block execution itself. This allows future
process-backed isolation to inject enforcement without changing evaluation logic.

**Verdict**: Forward-compatible design with clear resolution semantics.

### 2.5 Structured Error Recovery

The `RepairLoop` uses `FailureEnvelope` to normalize error context:

```
Code -> ProactiveCodeTransformer (regex fixes)
     -> Sandbox execution
     -> Failure -> FailureEnvelope (phase, exception, traceback, hints)
              -> Regex guardrail attempt
              -> LLM diagnosis (if regex fails)
              -> Retry (max 3)
              -> Escalate to user
```

This tiered approach is cost-effective: cheap regex patterns handle common issues
before invoking expensive LLM diagnosis.

**Verdict**: Pragmatic design that balances cost and repair quality.

### 2.6 Progressive Skill Disclosure

Skills load in two phases:

1. **Startup**: Metadata only (name + description) — minimal overhead
2. **On-demand**: Full SKILL.md body loaded when the agent selects a skill

`SkillRouter` matches queries to skills via keyword cosine similarity.

**Verdict**: Appropriate for the current scale (~40 skills). No over-engineering
with vector embeddings.

### 2.7 Multi-Provider LLM Support

`OmicVerseLLMBackend` provides a unified interface across 8+ providers
(OpenAI, Anthropic, Google Gemini, DashScope, DeepSeek, Moonshot, xAI, Zhipu).
Each provider has a dedicated adapter module. Unified `ChatResponse` and
`ToolCall` types abstract away provider differences.

**Verdict**: Well-executed adapter pattern with good provider coverage.

### 2.8 Domain Integration Depth

- **49 BioContext tools** for external knowledge (UniProt, STRING, Reactome, GO,
  PanglaoDB, PubMed, OpenTargets, etc.)
- **`@register_function` decorator** with `requires`/`produces` tracking for
  AnnData slot dependencies
- **Multilingual aliases** (English + Chinese) for function discovery
- **Auto-fix strategies** (`auto`, `escalate`, `none`) per function

**Verdict**: Deep bioinformatics integration that goes beyond a thin wrapper.

---

## 3. Issues and Recommendations

### 3.1 AgentContext Protocol Surface Is Too Large (Severity: Medium)

**Problem**: The Protocol exposes ~30 attributes and ~15 methods, most prefixed
with `_` (private convention). A Protocol should describe a public contract, not
internal implementation details.

**Impact**: Any refactoring of `smart_agent.py` internals forces Protocol
changes, which cascade to all extracted modules. The large surface also makes
it harder to create focused test doubles.

**Recommendation**: Apply the Interface Segregation Principle. Split into
smaller Protocols:

```python
class LLMContext(Protocol):
    model: str
    provider: str
    _llm: Optional[OmicVerseLLMBackend]

class ToolContext(Protocol):
    skill_registry: Optional[SkillRegistry]
    def _get_visible_agent_tools(self, ...) -> list: ...
    def _tool_blocked_in_plan_mode(self, ...) -> bool: ...

class SecurityContext(Protocol):
    _security_config: SecurityConfig
    _security_scanner: CodeSecurityScanner
```

Each extracted module then depends only on the sub-protocols it actually uses.

### 3.2 smart_agent.py Remains Oversized (Severity: Medium)

**Problem**: At ~1700 lines, the facade still carries significant initialization
and wiring logic in `__init__`.

**Recommendation**: Extract a `AgentBuilder` or `AgentFactory` that handles
subsystem assembly. The `__init__` should only accept pre-built dependencies
(pure dependency injection).

### 3.3 Tool Handler Registration Is Not Self-Service (Severity: Medium)

**Problem**: `ToolRuntime` directly imports handler functions from four
`tool_runtime_*.py` modules. Adding a new tool category requires modifying the
central dispatch module.

**Recommendation**: Define a `ToolHandler` Protocol. Each handler module
registers itself into the `ToolRegistry` at import time. The dispatcher becomes
a pure router with no handler knowledge.

### 3.4 Legacy/Catalog Tool Duality (Severity: Low-Medium)

**Problem**: Two parallel tool systems coexist:
- Catalog tools via `ToolDefinition` with structured schemas
- Legacy tools via raw dict schemas

Three separate mapping tables (`_CATALOG_TOOL_POLICIES`, `_LEGACY_TOOL_POLICIES`,
`_LEGACY_CATALOG_ALIASES`) increase cognitive overhead.

**Recommendation**: The `migration_notes` field on `ToolMetadata` is a good
starting point. Set explicit deprecation dates for legacy tools and track
migration progress. The goal should be a single catalog-based system.

### 3.5 Skill Routing May Not Scale (Severity: Low)

**Problem**: `SkillRouter` uses keyword tokenization + cosine similarity without
embeddings. Synonyms and domain abbreviations (DEG vs DGE vs "differential
expression") may not match well.

**Recommendation**: Current approach is adequate for ~40 skills. If the skill
count grows beyond 100, consider adding a lightweight embedding model or
expanding the keyword synonym mapping.

### 3.6 Unit Test Coverage Gap (Severity: Medium)

**Problem**: Tests are primarily E2E provider validations requiring real API keys.
Pure logic components (`ToolRegistry`, `PermissionPolicy`, `ContextBudgetManager`,
`FailureEnvelope`) lack dedicated unit tests despite being highly testable.

**Recommendation**: The Protocol-based design was built for testability — use it.
Add unit tests with mock `AgentContext` doubles for each extracted module.
Priority targets:

1. `PermissionPolicy.check()` — verify all 6 resolution levels
2. `ToolRegistry` — alias resolution, handler validation
3. `ContextBudgetManager` — budget allocation and compaction
4. `FailureEnvelope.from_exception()` — envelope construction

### 3.7 Configuration Source Tracing (Severity: Low)

**Problem**: Configuration values come from multiple sources — `AgentConfig`
dataclasses, `WORKFLOW.md` YAML front-matter, environment variables, and
constructor kwargs. It can be difficult to determine where a specific value
originated.

**Recommendation**: Add a `config_sources` dict to `AgentConfig` that records
the provenance of each resolved value (e.g., `{"model": "kwarg", "max_turns":
"WORKFLOW.md", "api_key": "env:OPENAI_API_KEY"}`).

---

## 4. Workflow Assessment

### 4.1 Main Agentic Loop

```
User request
  -> PromptBuilder: system prompt + overlays (deterministic ordering by priority)
  -> TurnController.run_agentic_loop():
       -> LLM call with messages + visible tools
       -> Parse tool calls from response
       -> For each tool call:
            -> ToolRegistry.resolve_name() (alias normalization)
            -> PermissionPolicy.check() (6-level evaluation)
            -> ToolRuntime.dispatch_tool() (handler execution)
            -> Capture result
       -> Append results in provider-specific format
       -> FollowUpGate + ConvergenceMonitor check
       -> Repeat until `finish` tool or max_turns
  -> Return: AnnData + summary + artifacts
```

**Assessment**: Clean turn-based loop with explicit termination conditions.
`FollowUpGate` and `ConvergenceMonitor` prevent infinite loops. The
provider-specific message formatting is correctly handled per-adapter rather
than in the loop itself.

### 4.2 Subagent Workflow

```
Parent agent
  -> SubagentController.spawn(agent_type)
  -> Create SubagentRuntime:
       - Snapshot tool schemas
       - Create scoped PermissionPolicy
       - Create subagent-local ContextBudgetManager
  -> Independent turn loop (restricted tool set)
  -> Result returned to parent
```

**Assessment**: Isolation is well-designed. The three subagent types
(explore/plan/execute) cover common decomposition patterns.

**Limitation**: No direct subagent-to-subagent communication. All coordination
goes through the parent agent. This is acceptable for the current use cases but
could become a bottleneck for complex multi-step workflows.

### 4.3 Code Execution & Repair

```
Code
  -> ProactiveCodeTransformer (regex-based fixes)
  -> Sandbox execution (isolated namespace)
  -> On failure:
       -> FailureEnvelope (normalized error context)
       -> Stage A: Regex guardrail recovery
       -> Stage B: LLM diagnosis (if Stage A fails)
       -> Auto-package install (if ImportError)
       -> Retry (max 3 attempts)
       -> Escalate to user (if all retries fail)
```

**Assessment**: The tiered recovery strategy is cost-effective and well-bounded.
`FailureEnvelope` provides sufficient context for LLM diagnosis (phase,
exception type, traceback excerpt, domain-specific hints). The auto-package
install for missing dependencies is a practical UX improvement.

---

## 5. Scoring Summary

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Modularity** | 4/5 | Extracted-module pattern is excellent; facade still oversized |
| **Testability** | 4/5 | Protocol design enables mocking; actual test coverage is low |
| **Security** | 5/5 | Multi-layer permissions, sandbox isolation, subagent state isolation |
| **Extensibility** | 4/5 | Registry seam + provider adapters; handler registration could be more flexible |
| **Code Quality** | 4/5 | Thorough docstrings, clear naming, comprehensive type annotations |
| **Workflow Design** | 5/5 | Turn loop + repair cycle + subagent isolation form a mature pipeline |
| **Domain Fit** | 5/5 | Deep bioinformatics integration via skills, BioContext, function registry |

**Overall: 4.3 / 5**

This is an architecturally mature agent system. The core design decisions —
Protocol-based decoupling, Registry seam, subagent isolation, layered permissions
— are sound and well-executed. The main areas for improvement are: reducing the
`AgentContext` surface area, completing the legacy-to-catalog tool migration,
and adding unit test coverage for pure-logic components.
