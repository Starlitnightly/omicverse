---
title: OmicClaw Telegram Tutorial
---

# OmicClaw Telegram Tutorial

Telegram is now documented as a gateway-backed channel, not as a standalone old Jarvis entry.

## 1. Create a Telegram Bot

1. Open `@BotFather` in Telegram.
2. Send `/newbot`.
3. Provide the display name and username.
4. Save the returned token as `TELEGRAM_BOT_TOKEN`.

## 2. Environment Variables

```bash
export TELEGRAM_BOT_TOKEN="123456:ABC-..."
export ANTHROPIC_API_KEY="your_api_key"
```

## 3. Recommended Start Command

OmicClaw product entry:

```bash
omicclaw --channel telegram --token "$TELEGRAM_BOT_TOKEN"
```

Generic gateway entry:

```bash
omicverse gateway --channel telegram --token "$TELEGRAM_BOT_TOKEN"
```

## 4. Full Start Command

```bash
omicclaw \
  --channel telegram \
  --token "$TELEGRAM_BOT_TOKEN" \
  --model claude-sonnet-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  --auth-mode environment \
  --session-dir ~/.ovjarvis \
  --max-prompts 0 \
  --allowed-user your_telegram_username \
  --allowed-user 123456789 \
  --web-port 5050 \
  --verbose
```

## 5. Notes on Current Behavior

- Starting through `omicclaw` or `omicverse gateway` also launches the web gateway.
- If you want code only, do not use this path; use `omicverse claw -q ...`.

## 6. Common Commands

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

## 7. Troubleshooting

1. Missing Telegram dependency
   Run `pip install -e ".[jarvis]"` or `pip install "omicverse[jarvis]"`.

2. Missing token error
   Check `TELEGRAM_BOT_TOKEN` or `--token`.

3. `409 Conflict`
   Stop other processes using the same bot token.
