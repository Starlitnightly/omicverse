---
title: J.A.R.V.I.S. Msg Bot Overview
---

# J.A.R.V.I.S. Msg Bot Overview

This section shows how to run Jarvis as a message bot across channels:

- Telegram (English)
- Feishu (中文)
- iMessage (English)
- QQ (中文)

## 1. Install

```bash
pip install -e ".[jarvis]"
```

Or:

```bash
pip install "omicverse[jarvis]"
```

## 2. First-Time Setup (Recommended)

```bash
omicverse jarvis --setup --setup-language zh
```

The setup wizard writes:

- `~/.ovjarvis/config.json` (channel/model defaults)
- `~/.ovjarvis/auth.json` (auth data)

## 3. Common Environment Variables

LLM key (example):

```bash
export ANTHROPIC_API_KEY="your_api_key"
# or OPENAI_API_KEY / GEMINI_API_KEY
```

Channel credentials:

```bash
export TELEGRAM_BOT_TOKEN="123456:ABC-..."
export FEISHU_APP_ID="cli_xxx"
export FEISHU_APP_SECRET="xxx"
# optional for Feishu webhook verification/encryption
export FEISHU_VERIFICATION_TOKEN="xxx"
export FEISHU_ENCRYPT_KEY="xxx"
export QQ_APP_ID="your_qq_app_id"
export QQ_CLIENT_SECRET="your_qq_client_secret"
```

## 4. Quick Channel Entry Points

- Telegram tutorial: `t_channel_telegram.md`
- Feishu tutorial: `t_channel_feishu.md`
- iMessage tutorial: `t_channel_imessage.md`
- QQ tutorial: `t_channel_qq.md`
- Shared commands/session flow: `t_session_commands.md`
- Troubleshooting: `t_troubleshooting.md`
