---
title: J.A.R.V.I.S. Setup and Auth
---

# J.A.R.V.I.S. Setup and Auth

This page covers baseline setup shared by all channels.

## 1. Install

```bash
pip install -e ".[jarvis]"
```

Or:

```bash
pip install "omicverse[jarvis]"
```

For iMessage on macOS:

```bash
brew install steipete/tap/imsg
```

## 2. Minimal Start Command

```bash
omicverse jarvis --setup
```

## 3. Full Setup Command

```bash
omicverse jarvis \
  --setup \
  --setup-language zh \
  --config-file ~/.ovjarvis/config.json \
  --auth-file ~/.ovjarvis/auth.json \
  --verbose
```

Parameter explanation:

- `--setup`: runs the interactive setup wizard before launch.
- `--setup-language`: setup UI language (`en` or `zh`).
- `--config-file`: path to persisted Jarvis config.
- `--auth-file`: path to saved auth state (API keys/OAuth tokens).
- `--verbose`: enables debug-level logs.

## 4. LLM Authentication

```bash
export ANTHROPIC_API_KEY="your_api_key"
# or OPENAI_API_KEY / GEMINI_API_KEY
```

You can also pass a key directly:

```bash
omicverse jarvis --api-key "your_api_key"
```

## 5. Common Runtime Parameters

- `--channel`: `telegram`, `feishu`, `imessage`, or `qq`.
- `--model`: LLM model name.
- `--api-key`: explicit API key.
- `--auth-mode`: auth strategy (`environment`, `openai_oauth`, `saved_api_key`, `no_auth`).
- `--endpoint`: custom OpenAI-compatible base URL (for example Ollama).
- `--session-dir`: base runtime directory (default `~/.ovjarvis`).
- `--max-prompts`: max prompts per kernel session (`0` disables auto-restart).
- `--verbose`: verbose logging.

## 6. Default Runtime Directory

```text
~/.ovjarvis
```

Typical contents:

- `workspace/`
- `sessions/`
- `context/`
- `current.h5ad`
