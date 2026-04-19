---
title: OmicVerse Claw CLI
---

# OmicVerse Claw CLI

`omicverse claw` is the code-only entry point built on top of `ov.Agent`.

Use it when you want:

- Python code only
- no direct execution on the current machine
- a fast CLI workflow that can be called from shells, scripts, or external tools

If you want a chat-style gateway workflow, use `omicclaw` or `omicverse gateway`.
If you want a tool server for Claude Code or other MCP clients, use `omicverse mcp`.

## 1. Install

For local development:

```bash
pip install -e .
```

Or install the published package:

```bash
pip install -U omicverse
```

Check that the CLI is available:

```bash
omicverse claw --help
```

## 2. Basic Usage

The simplest form is:

```bash
omicverse claw "basic qc and clustering"
```

This prints generated Python code to `stdout`.

More examples:

```bash
omicverse claw "annotate lung scRNA-seq with a minimal workflow"
omicverse claw "find marker genes for each leiden cluster"
omicverse claw "write a basic PCA + neighbors + UMAP pipeline"
```

## 3. Save Output to a File

```bash
omicverse claw "basic qc and clustering" --output workflow.py
```

This keeps generated code clean on `stdout` while writing the same code to `workflow.py`.

## 4. Common Options

Choose a model:

```bash
omicverse claw --model gpt-5.2 "basic qc and clustering"
```

Use an explicit API key:

```bash
omicverse claw --api-key "$OPENAI_API_KEY" "basic qc and clustering"
```

Use a custom endpoint:

```bash
omicverse claw \
  --endpoint http://127.0.0.1:11434/v1 \
  --model my-model \
  "basic qc and clustering"
```

Disable the lightweight reflection pass:

```bash
omicverse claw --no-reflection "basic qc and clustering"
```

## 5. Debug Mode

Use `--debug-registry` when you want to inspect initialization, runtime registry hits, and internal progress.

```bash
omicverse claw --debug-registry "basic qc and clustering"
```

In debug mode:

- generated Python code still goes to `stdout`
- initialization logs go to `stderr`
- matched runtime registry entries go to `stderr`
- a `tqdm` progress bar shows the current stage

Typical stages include:

- `init agent`
- `inspect registry`
- `prepare prompt`
- `request model`
- `extract code`
- `review code`
- `rewrite scanpy`
- `finalize`

## 6. Runtime Behavior

`omicverse claw` initializes `OmicVerseAgent` first, then enters code-only mode.

That means you will see the normal Agent initialization log before code generation, for example:

```text
🧭 Loaded 32 skills (progressive disclosure) (32 built-in)
Model: OpenAI GPT-5.2
Provider: Openai
Endpoint: https://api.openai.com/v1
✅ OpenAI API key available
📚 Function registry loaded: 90 functions in 4 categories
...
✅ Smart Agent initialized successfully!
```

The CLI then asks the initialized agent to return code only, without executing analysis code.

## 7. Daemon Mode

If repeated startup cost is noticeable, run a persistent local daemon:

```bash
omicverse claw --daemon
```

Then send requests to the daemon:

```bash
omicverse claw --use-daemon "basic qc and clustering"
```

With debug mode:

```bash
omicverse claw --use-daemon --debug-registry "basic qc"
```

The daemon keeps OmicVerse and the default `OmicVerseAgent` in memory, so repeated calls avoid cold-start import and agent initialization overhead.

Stop the daemon with:

```bash
omicverse claw --stop-daemon
```

## 8. Custom Socket Path

By default the daemon socket is:

```text
~/.cache/omicverse/claw.sock
```

You can override it:

```bash
omicverse claw --daemon --socket /tmp/ov-claw.sock
omicverse claw --use-daemon --socket /tmp/ov-claw.sock "basic qc"
omicverse claw --stop-daemon --socket /tmp/ov-claw.sock
```

## 9. Recommended Usage Patterns

Single one-off generation:

```bash
omicverse claw "basic qc and clustering"
```

Frequent local experimentation:

```bash
omicverse claw --daemon
omicverse claw --use-daemon "basic qc and clustering"
omicverse claw --use-daemon "find marker genes"
```

Debugging runtime behavior:

```bash
omicverse claw --debug-registry "basic qc and clustering"
```

## 10. When to Use Claw vs Jarvis

Use `claw` when:

- you want code only
- you want to inspect or edit the generated script yourself
- you want to call OmicVerse from another CLI or automation layer

Use `jarvis` when:

- you want a message-bot workflow
- you want session memory and interactive follow-up
- you want a human-facing chat interface

## 11. Related Pages

- OpenClaw integration: [OpenClaw Integration](t_claw_openclaw.md)
- Gateway overview: [OmicClaw Gateway Overview](t_msg_bot_overview.md)
- MCP quick start: [OmicVerse MCP Quick Start](../Tutorials-llm/t_mcp_quickstart.md)
