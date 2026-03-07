# OVAgent 本地化 Symphony 与 `smart_agent.py` 模块化执行计划

## Summary

本计划将 Symphony 的优点本地化为 OVAgent 的分析任务控制层，而不是引入 issue tracker、调度队列或 PR 自动化。首版只面向数据科学与生物信息学分析。

交付目标有两条主线：

1. 把 `smart_agent.py` 从巨型单文件拆成新的 runtime/controller 架构。
2. 引入面向分析任务的 `WORKFLOW.md`、`AnalysisRun`、proof bundle 与 CLI-first 控制面。

## Target State

- `ov.Agent()` / `OmicVerseAgent` 外部接口保持兼容。
- `smart_agent.py` 退化为兼容 facade，不再承载主要工具实现和执行引擎。
- 仓库根目录有 repo-owned `WORKFLOW.md`，可驱动 agent 行为。
- 每次分析任务都能生成本地 run 记录、trace 关联与 proof bundle。
- 所有回归验证继续只在台湾服务器执行。

## Phase 0: 基线冻结与职责映射

### Goals

- 冻结兼容边界：`ov.Agent()`、`OmicVerseAgent`、现有 web chat/session/trace/approval/question 接口均保持可用。
- 识别 `smart_agent.py` 当前职责并完成迁移归属。

### Deliverables

- 新模块职责图。
- 旧方法到新模块的迁移映射表。
- 明确哪些行为保留在 facade，哪些迁入 runtime/controller。

### Exit Criteria

- 没有未归属的核心职责。
- 所有高风险兼容点都记录清楚。

## Phase 1: Runtime / Controller 骨架

### Goals

- 新增内部核心类：
  - `OmicVerseRuntime`
  - `TurnController`
  - `ToolRuntime`
  - `PromptBuilder`
  - `AnalysisExecutor`
  - `SubagentController`
- `smart_agent.py` 开始委派，而不是继续扩张。

### Deliverables

- `omicverse/utils/ovagent/` 主包。
- turn loop 迁入 `TurnController`。
- prompt 组装迁入 `PromptBuilder`。

### Exit Criteria

- `TurnController` 能驱动一次完整 agent turn。
- `smart_agent.py` 不再是唯一 turn loop 实现位置。

## Phase 2: ToolRuntime 与执行引擎拆分

### Goals

- 把工具派发从 `smart_agent.py` 拆到 `ToolRuntime`。
- 把代码执行、自修复、notebook/in-process fallback、artifact 采集拆到 `AnalysisExecutor`。

### Deliverables

- `ToolRuntime` 统一管理 Claude-style tools 与 legacy aliases。
- `AnalysisExecutor` 独立管理执行、修复、artifact、provenance。

### Exit Criteria

- `smart_agent.py` 不再包含主要 `_tool_*` 实现。
- 执行引擎可独立测试。

## Phase 3: Repo-owned `WORKFLOW.md`

### Goals

- 增加仓库级分析策略契约。
- 让 prompt/tool/approval/test policy 从代码里抽离到 repo 文档。

### Required Front Matter

- `domain`
- `default_tools`
- `approval_policy`
- `max_turns`
- `execution_mode`
- `required_artifacts`
- `validation_commands`
- `completion_criteria`
- `compaction_policy`

### Constraints

- 首版 `domain` 仅允许：
  - `data-science`
  - `bioinformatics`

### Exit Criteria

- 修改 `WORKFLOW.md` 能改变 agent 行为。
- 无需改 Python 代码即可调整分析策略。

## Phase 4: `AnalysisRun` 与 proof bundle

### Goals

- 新增 `AnalysisRun` 与 `RunStore`。
- 把 Symphony 的 proof-of-work 本地化成分析交付证明。

### Run Assets

- `run_id`
- workflow snapshot
- trace ids
- input provenance
- artifact manifest
- warnings
- final summary

### Default Location

- `~/.ovagent/runs/<run_id>/`

### Exit Criteria

- 单个分析任务能产出 `summary.md` 与 `bundle.json`。
- replay / resume 能基于 run store 工作。

## Phase 5: CLI-first 控制面

### Goals

- 不单开新 CLI，扩展现有 verifier 入口。

### Commands

- `workflow show`
- `workflow validate`
- `run start`
- `run status`
- `run resume`
- `run replay`
- `run bundle`

### Exit Criteria

- 可以通过 CLI 启动、查看、恢复、回放分析 run。
- workflow 配置与 proof bundle 都能直接审阅。

## Phase 6: Web 最小接入与收口

### Goals

- 现有 chat/harness 页面只做最小增强。
- 不做 Symphony 风格 operator console。

### Deliverables

- 显示 `run_id`
- 关联 trace
- 提供 proof bundle 链接
- 更新 `docs/harness/` 文档

### Exit Criteria

- web 不破坏现有 chat 交互。
- run 信息可见但不演化成调度台。

## Acceptance Criteria

- `smart_agent.py` 不再承载主要工具实现与执行引擎。
- `WORKFLOW.md` 真正驱动分析行为。
- 单个分析任务能生成可审阅的 proof bundle。
- verifier CLI 能管理 workflow 与 runs。
- 所有验证仍遵循台湾服务器测试策略。

## Assumptions

- 首版不做 GitHub/Linear tracker、scheduler、queue、operator dashboard。
- “本地化 Symphony” 保留的是 workflow contract、run isolation、proof-of-work、restartable runs、controller/runtime 分层。
- OVAgent 的目标域固定为数据科学与生物信息学分析。
