---
title: J.A.R.V.I.S. Troubleshooting
---

# J.A.R.V.I.S. Troubleshooting

## 1. General

1. Setup wizard did not start  
   Run `omicverse jarvis --setup --setup-language zh`.

2. Model change did not apply  
   Run `/model <new_model>`, then `/reset`.

3. Session restarts unexpectedly  
   Use `--max-prompts 0`.

## 2. Telegram

1. Missing dependency  
   Run `pip install -e ".[jarvis]"`.

2. Missing token  
   Check `TELEGRAM_BOT_TOKEN` or `--token`.

3. `409 Conflict`  
   Stop other running processes using the same bot token.

## 3. Feishu

1. WebSocket SDK missing  
   Run `pip install lark-oapi`.

2. Webhook verification fails  
   Verify callback URL and `--feishu-host/--feishu-port/--feishu-path`.

3. Can receive text but cannot send images/files  
   Check Feishu app permissions.

## 4. iMessage

1. `imsg` not found  
   Check `which imsg`.

2. Cannot access `chat.db`  
   Verify DB path and macOS permissions.

## 5. QQ

1. Missing credentials  
   Check `QQ_APP_ID` and `QQ_CLIENT_SECRET`.

2. Image sending fails  
   Check `--qq-image-host` public reachability.

3. Markdown not working  
   Verify markdown permission in QQ Open Platform.
