---
title: J.A.R.V.I.S. iMessage Tutorial
---

# J.A.R.V.I.S. iMessage Tutorial

iMessage channel works on macOS and requires `imsg`.

## 1. Install Dependency

```bash
brew install steipete/tap/imsg
```

## 2. Minimal Start Command

```bash
omicverse jarvis \
  --channel imessage \
  --imessage-cli-path "$(which imsg)" \
  --imessage-db-path ~/Library/Messages/chat.db
```

## 3. Full Start Command

```bash
omicverse jarvis \
  --channel imessage \
  --imessage-cli-path "$(which imsg)" \
  --imessage-db-path ~/Library/Messages/chat.db \
  --imessage-include-attachments \
  --model claude-sonnet-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  --auth-mode environment \
  --session-dir ~/.ovjarvis \
  --max-prompts 0 \
  --verbose
```

Parameter explanation:

- `--channel imessage`: selects iMessage backend.
- `--imessage-cli-path`: path to `imsg` executable.
- `--imessage-db-path`: path to macOS Messages SQLite DB.
- `--imessage-include-attachments`: include attachment metadata in inbound events.
- `--model`: model name used for analysis.
- `--api-key`: explicit LLM provider key.
- `--auth-mode environment`: read auth from env vars.
- `--session-dir`: runtime root directory.
- `--max-prompts 0`: disables automatic kernel restart.
- `--verbose`: enables detailed logs.

## 4. Troubleshooting

1. `imsg` not found  
   Check `which imsg`.

2. Cannot read `chat.db`  
   Verify DB path and macOS permissions.
