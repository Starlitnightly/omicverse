# OmicVerse Jarvis User Guide

This guide explains how to start and use `omicverse jarvis` (a Telegram assistant for single-cell analysis).

## 1. Install Dependencies

Run in the project root:

```bash
pip install -e ".[jarvis]"
```

If you do not want editable install:

```bash
pip install "omicverse[jarvis]"
```

## 2. Create a Telegram Bot

Jarvis requires a Telegram Bot Token.

1. Open `@BotFather` in Telegram.
2. Send `/newbot` and provide:
   - Bot display name
   - Bot username (must end with `bot`, for example `omicverse_jarvis_bot`)
3. BotFather returns a token like `123456:ABC-...`; this is your `TELEGRAM_BOT_TOKEN`.
4. Optional setup:
   - `/setdescription`
   - `/setabouttext`
   - `/setuserpic`
   - `/setcommands` (for example `start/help/status/load/save/reset`)

Security tips:

- Never commit your token to Git.
- If leaked, revoke it immediately via BotFather `/revoke`.
- In production, use `--allowed-user` to restrict access.

## 3. Prepare Credentials

### 3.1 Telegram Bot Token

Use either environment variable or CLI argument:

```bash
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
```

Or:

```bash
omicverse jarvis --token "your_telegram_bot_token"
```

### 3.2 LLM API Key

Jarvis checks these environment variables in order:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

Example:

```bash
export ANTHROPIC_API_KEY="your_api_key"
```

You can also pass `--api-key`.

## 4. Start Jarvis

Minimal start:

```bash
omicverse jarvis
```

Telegram channel explicitly:

```bash
omicverse jarvis --channel telegram --token "$TELEGRAM_BOT_TOKEN"
```

Feishu webhook channel:

```bash
omicverse jarvis \
  --channel feishu \
  --feishu-app-id "$FEISHU_APP_ID" \
  --feishu-app-secret "$FEISHU_APP_SECRET" \
  --feishu-host 0.0.0.0 \
  --feishu-port 8080 \
  --feishu-path /feishu/events
```

Full example (all current CLI arguments):

```bash
omicverse jarvis \
  --token "$TELEGRAM_BOT_TOKEN" \
  --model claude-sonnet-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  --max-prompts 0 \
  --session-dir ~/.ovjarvis \
  --allowed-user your_telegram_username \
  --allowed-user 123456789 \
  --verbose
```

Parameter reference:

- `--token`: Telegram token (or `TELEGRAM_BOT_TOKEN`)
- `--channel`: backend channel (`telegram` or `feishu`, default: `telegram`)
- `--feishu-app-id`: Feishu app id (or `FEISHU_APP_ID`)
- `--feishu-app-secret`: Feishu app secret (or `FEISHU_APP_SECRET`)
- `--feishu-host/--feishu-port/--feishu-path`: Feishu webhook listen settings
- `--model`: model name (default: `claude-sonnet-4-6`)
- `--api-key`: LLM key (or `ANTHROPIC_API_KEY/OPENAI_API_KEY/GEMINI_API_KEY`)
- `--max-prompts`: max prompts per kernel session (`0` disables auto-restart; default: 0)
- `--session-dir`: session root directory (default: `~/.ovjarvis`)
- `--allowed-user`: allowed username/ID (repeatable)
- `--verbose`: verbose logging

## 5. Session Directory Layout

Each Telegram user gets `~/.ovjarvis/<user_id>/`:

- `workspace/`: data and context files
- `sessions/`: notebook/kernel session data
- `context/`: agent context cache
- `current.h5ad`: current loaded data snapshot
- `kernels/<name>/...`: additional named kernels (created via `/kernel new <name>`)

## 6. Common Telegram Commands

### 6.1 Data & Files

- `/workspace` show workspace overview
- `/ls [path]` list files
- `/find <pattern>` find files (for example `/find *.h5ad`)
- `/load <filename>` load dataset
- `/save` export current `h5ad`
- Uploading `.h5ad` directly also triggers loading

### 6.2 Session & Model

- `/status` current status
- `/kernel` active kernel health and prompt usage
- `/kernel ls` list kernels
- `/kernel new <name>` create and switch to a new kernel
- `/kernel use <name>` switch active kernel
- `/usage` latest token usage
- `/model [name]` view/switch model
- `/memory` recent analysis memory (last 2 days)
- `/cancel` cancel running analysis
- `/reset` reset session and restart kernel

### 6.3 Whitelisted Shell Commands

`/shell` only allows: `ls find cat head wc file du pwd tree`

Examples:

```text
/shell ls -lh
/shell find . -name "*.h5ad"
```

## 7. Recommended Workflow

1. Upload `.h5ad` to `workspace/` (or send it directly via Telegram).
2. Run `/workspace` and `/load <file>`.
3. Send natural-language analysis requests.
4. Monitor with `/status`, `/kernel`, `/usage`.
5. Download results with `/save`.

## 8. Customize Agent Behavior

Put your preferences in `workspace/AGENTS.md` (language/style/plot preferences). Jarvis injects it automatically for each request.

Use `workspace/MEMORY.md` for long-term memory. Daily summaries are auto-written to `workspace/memory/YYYY-MM-DD.md`.

## 9. FAQ

### 9.1 Missing telegram dependency

```bash
pip install -e ".[jarvis]"
```

### 9.2 Missing token error

Set `TELEGRAM_BOT_TOKEN` or pass `--token`.

### 9.3 Model switch does not take effect

After `/model <new_model>`, run `/reset` to apply the model in a new kernel.

### 9.4 Interrupted analysis or lost variables

Set `--max-prompts 0` to avoid auto-restart and keep one long-lived kernel by default. Use `/reset` when you explicitly want a fresh kernel.
