---
title: J.A.R.V.I.S. Telegram Tutorial
---

# J.A.R.V.I.S. Telegram Tutorial

## 1. Create a Telegram Bot

1. Open `@BotFather` in Telegram.
2. Send `/newbot`.
3. Provide bot display name and username (username must end with `bot`).
4. Save the token as `TELEGRAM_BOT_TOKEN`.

## 2. Environment Variables

```bash
export TELEGRAM_BOT_TOKEN="123456:ABC-..."
export ANTHROPIC_API_KEY="your_api_key"
```

## 3. Minimal Start Command

```bash
omicverse jarvis --channel telegram --token "$TELEGRAM_BOT_TOKEN"
```

## 4. Full Start Command

```bash
omicverse jarvis \
  --channel telegram \
  --token "$TELEGRAM_BOT_TOKEN" \
  --model claude-sonnet-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  --auth-mode environment \
  --session-dir ~/.ovjarvis \
  --max-prompts 0 \
  --allowed-user your_telegram_username \
  --allowed-user 123456789 \
  --verbose
```

Parameter explanation:

- `--channel telegram`: selects Telegram backend.
- `--token`: Telegram bot token (or `TELEGRAM_BOT_TOKEN`).
- `--model`: model name used by Jarvis.
- `--api-key`: explicit LLM provider key.
- `--auth-mode environment`: read auth from env vars.
- `--session-dir`: runtime root directory.
- `--max-prompts 0`: disables automatic kernel restart.
- `--allowed-user`: restricts access to specific users (repeatable).
- `--verbose`: enables detailed logs.

## 5. Common Commands

- `/workspace`
- `/load <filename>`
- `/save`
- `/status`
- `/kernel`
- `/kernel ls`
- `/kernel new <name>`
- `/kernel use <name>`
- `/cancel`
- `/reset`

## 6. Troubleshooting

1. Missing Telegram dependency  
   Run `pip install -e ".[jarvis]"`.

2. Missing token error  
   Check `TELEGRAM_BOT_TOKEN` or `--token`.

3. Polling `409 Conflict`  
   Stop other processes using the same bot token.
