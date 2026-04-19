---
title: OmicClaw Troubleshooting
---

# OmicClaw Troubleshooting

## 1. Launcher Behavior

1. `omicverse claw` opened the web UI instead of returning code
   This is now expected. `omicverse claw` defaults to gateway mode unless you pass `-q`, `--question`, or daemon flags.

2. I want code only
   Use:

   ```bash
   omicverse claw -q "basic qc and clustering"
   ```

3. `omicclaw` always asks for login
   This is also expected. The `omicclaw` entrypoint enables forced-login web behavior by design.

4. Gateway started but no bot is online
   If no `--channel` is configured, gateway runs in web-only mode.

## 2. Setup and Auth

1. The setup wizard did not start
   Run `omicclaw --setup --setup-language zh` or `omicverse gateway --setup --setup-language zh`.

2. Model or provider settings did not apply
   Check `~/.ovjarvis/config.json` and `~/.ovjarvis/auth.json`, then restart the launcher.

3. Provider key not found
   Export the matching environment variable or use `--api-key`.

## 3. Telegram

1. Missing dependency
   Run `pip install -e ".[jarvis]"` or `pip install "omicverse[jarvis]"`.

2. Missing token
   Check `TELEGRAM_BOT_TOKEN` or `--token`.

3. `409 Conflict`
   Another process is already using the same bot token.

## 4. Feishu

1. WebSocket SDK missing
   Install `lark-oapi`.

2. Webhook verification fails
   Verify callback URL and `--feishu-host/--feishu-port/--feishu-path`.

3. Can receive text but not images/files
   Check Feishu app permissions and deployment reachability.

## 5. iMessage

1. `imsg` not found
   Check `which imsg`.

2. Cannot access `chat.db`
   Verify the database path and macOS permissions.

## 6. QQ

1. Missing credentials
   Check `QQ_APP_ID` and `QQ_CLIENT_SECRET`.

2. Image sending fails
   Check `--qq-image-host` public reachability.

3. Markdown does not work
   Verify that the QQ bot has markdown message permission.
