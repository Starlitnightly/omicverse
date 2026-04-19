---
title: OmicClaw Gateway Overview
---

# OmicClaw Gateway Overview

OmicClaw is now documented as a unified gateway product rather than an older "message bot only" launcher.

The current entry points are:

- `omicclaw`: branded OmicClaw launcher. It starts gateway mode and forces the web login flow.
- `omicverse gateway`: generic gateway launcher. It starts the web UI and auto-starts configured channels.
- `omicverse claw`: gateway mode by default. Use `-q/--question` only when you want one-shot code generation.

## 1. Recommended Launch Modes

Use `omicclaw` when you want the OmicClaw-branded product experience:

```bash
omicclaw
```

Use `omicverse gateway` when you want the same gateway runtime without OmicClaw branding:

```bash
omicverse gateway
```

Use `omicverse claw -q` only for code-only generation:

```bash
omicverse claw -q "basic qc and clustering"
```

## 2. What Each Entry Point Actually Does

| Entry point | Current behavior | Recommended use |
| --- | --- | --- |
| `omicclaw` | Starts gateway mode with forced login and OmicClaw branding | Main product entry for users |
| `omicverse gateway` | Starts gateway web UI and background channel runtime | Generic deployment / ops |
| `omicverse claw` | Starts gateway mode unless `-q`, `--daemon`, `--use-daemon`, or `--stop-daemon` is given | Mixed CLI, especially code-only mode |

## 3. Gateway-Only vs Channel-Backed Mode

If you launch gateway mode without `--channel`, OmicVerse keeps the web UI running in web-only mode.

Example:

```bash
omicclaw
```

or:

```bash
omicverse gateway
```

If you add a channel, the same gateway runtime starts the web UI and the selected message channel together.

Example:

```bash
omicclaw --channel telegram --token "$TELEGRAM_BOT_TOKEN"
```

## 4. Shared Runtime Layout

The current gateway/channel stack still stores runtime state under:

```text
~/.ovjarvis
```

Typical contents include:

- `config.json`: persisted launcher and channel defaults
- `auth.json`: saved auth state
- `workspace/`: user-visible files and prompts
- `sessions/`: runtime session data
- `context/`: cached context
- `memory/`: daily summaries and memory artifacts

## 5. Recommended Reading Order

1. [Setup and Auth](t_setup_auth.md)
2. Channel tutorial for your deployment target
3. [Session Workflow](t_session_commands.md)
4. [Common Issues](t_troubleshooting.md)

Channel pages:

- [Telegram Tutorial](t_channel_telegram.md)
- [Feishu Tutorial](t_channel_feishu.md)
- [iMessage Tutorial](t_channel_imessage.md)
- [QQ Tutorial](t_channel_qq.md)

## 6. Related Pages

- Setup: [Setup and Auth](t_setup_auth.md)
- Session workflow: [Session Workflow](t_session_commands.md)
