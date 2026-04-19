---
title: OmicClaw Session Workflow
---

# OmicClaw Session Workflow

The current OmicClaw deployment model is a shared gateway runtime:

- the web UI runs inside gateway mode
- optional channels such as Telegram, Feishu, iMessage, and QQ connect to the same runtime
- interactive state is still stored under `~/.ovjarvis`

## 1. Recommended Workflow

1. Start `omicclaw` or `omicverse gateway`.
2. If needed, add a channel with `--channel ...`.
3. Upload a `.h5ad` file in chat or through the web workspace.
4. Load the dataset and issue natural-language analysis requests.
5. Inspect status and kernel health.
6. Export results with `/save` or from the web UI.

## 2. Current Session Root

Default root:

```text
~/.ovjarvis
```

Common entries:

- `config.json`
- `auth.json`
- `workspace/`
- `sessions/`
- `context/`
- `memory/`

## 3. Message-Channel Commands

Current command set includes:

- `/workspace`
- `/ls [path]`
- `/find <pattern>`
- `/load <filename>`
- `/save`
- `/status`
- `/usage`
- `/model [name]`
- `/memory`
- `/cancel`
- `/reset`
- `/kernel`
- `/kernel ls`
- `/kernel new <name>`
- `/kernel use <name>`

## 4. Gateway and Web Behavior

When started through `omicclaw` or `omicverse gateway`:

- the web gateway stays available even without a channel
- if a channel is configured, channel turns and web runtime share the same launcher stack
- missing channel credentials do not block gateway mode; the launcher can fall back to web-only mode

## 5. Code-Only Mode Is Separate

This session workflow does not apply to one-shot code generation.

For code only, use:

```bash
omicverse claw -q "basic qc and clustering"
```
