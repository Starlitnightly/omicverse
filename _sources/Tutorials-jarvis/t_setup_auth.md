---
title: OmicClaw Setup and Auth
---

# OmicClaw Setup and Auth

This page documents the current setup path for the new OmicClaw / gateway launcher stack.

## 1. Install

For development:

```bash
pip install -e ".[jarvis]"
```

For regular usage:

```bash
pip install "omicverse[jarvis]"
```

For macOS iMessage support, also install:

```bash
brew install steipete/tap/imsg
```

## 2. Recommended First Run

OmicClaw-branded first run:

```bash
omicclaw --setup --setup-language zh
```

Generic gateway first run:

```bash
omicverse gateway --setup --setup-language zh
```

## 3. Persisted Config and Auth Files

By default the launcher stores state under:

```text
~/.ovjarvis
```

Important files:

- `~/.ovjarvis/config.json`: saved launcher, model, and channel defaults
- `~/.ovjarvis/auth.json`: saved provider auth or OAuth state

The setup wizard can be redirected with:

```bash
omicclaw \
  --setup \
  --config-file ~/.ovjarvis/config.json \
  --auth-file ~/.ovjarvis/auth.json
```

## 4. Authentication Sources

The current runtime supports:

- provider environment variables such as `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `GEMINI_API_KEY`
- saved provider auth written by the setup wizard
- OpenAI Codex OAuth saved into `auth.json`
- custom OpenAI-compatible endpoints through `--endpoint`

Example:

```bash
export ANTHROPIC_API_KEY="your_api_key"
omicclaw --model claude-sonnet-4-6 --channel telegram --token "$TELEGRAM_BOT_TOKEN"
```

## 5. Current Launcher Behavior

The new entry logic is:

- `omicclaw`: starts gateway mode with forced login and OmicClaw branding
- `omicverse gateway`: starts gateway mode with the generic OmicVerse brand
- `omicverse claw`: starts gateway mode by default; use `-q/--question` only for code-only generation

If gateway mode is started without `--channel`, the web UI runs in web-only mode.

## 6. Common Runtime Flags

- `--channel`: `telegram`, `feishu`, `imessage`, or `qq`
- `--model`: LLM model name
- `--api-key`: explicit provider key
- `--auth-mode`: `environment`, `openai_oauth`, `saved_api_key`, or `no_auth`
- `--endpoint`: custom OpenAI-compatible base URL
- `--session-dir`: session root directory
- `--max-prompts`: prompt quota before kernel restart; `0` disables auto-restart
- `--web-host` / `--web-port`: gateway web bind settings
- `--no-browser`: keep the launcher from auto-opening a browser
- `--verbose`: enable detailed logs

## 7. Recommended Usage Patterns

Web-first OmicClaw product entry:

```bash
omicclaw
```

Gateway plus a channel:

```bash
omicclaw --channel feishu --feishu-connection-mode websocket
```

Code-only generation:

```bash
omicverse claw -q "basic qc and clustering"
```
