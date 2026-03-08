---
domain: bioinformatics
default_tools:
  - inspect_data
  - execute_code
  - run_snippet
  - search_functions
  - search_skills
  - WebFetch
  - WebSearch
  - ToolSearch
approval_policy: guarded
max_turns: 15
execution_mode: notebook_preferred
required_artifacts:
  - summary.md
  - bundle.json
  - trace_linkage
validation_commands:
  - pytest
completion_criteria:
  - Produce a reproducible analysis outcome or a precise blocked summary.
  - Preserve provenance for inputs, outputs, and key decisions.
  - Record enough artifacts for another analyst to audit the run.
compaction_policy: summarize_recent_failures_and_context
---

# OVAgent Workflow Policy

OVAgent in this repository is a domain-focused analysis agent. It is not a
general-purpose coding agent.

## Scope

- Single-cell, spatial, multi-omics, and related bioinformatics analysis
- Data-science workflows where the primary output is an analysis result
- Reproducible generation of tables, figures, notebooks, summaries, and h5ad outputs

## Core Rules

1. Prefer analyzing data over performing general repository maintenance.
2. Treat provenance as a first-class output. Record where inputs came from and what files were produced.
3. When a request is blocked, finish with a precise explanation of what is missing rather than continuing to search indefinitely.
4. Prefer concise, auditable outputs over long narrative answers.
5. When code executes, preserve enough context to understand what changed in `adata` and what artifacts were created.

## Completion Standard

A task is complete when one of the following is true:

- The requested analysis has been performed and the key outputs are saved.
- The task is blocked, and the agent has produced a clear blocked summary with the missing dependency, asset, permission, or user decision.

## Proof Bundle Expectation

Each substantial run should leave behind:

- a short summary
- artifact references
- trace linkage
- warnings and notable assumptions
- basic provenance for the input data
