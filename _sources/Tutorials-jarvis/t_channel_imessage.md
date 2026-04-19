---
title: OmicClaw iMessage Tutorial
---

# OmicClaw iMessage Tutorial

iMessage remains a macOS-only channel, but it is now typically launched through the gateway stack.

## 1. Install Dependency

```bash
brew install steipete/tap/imsg
```

## 2. Recommended Start Command

```bash
omicclaw \
  --channel imessage \
  --imessage-cli-path "$(which imsg)" \
  --imessage-db-path ~/Library/Messages/chat.db
```

Equivalent generic gateway entry:

```bash
omicverse gateway \
  --channel imessage \
  --imessage-cli-path "$(which imsg)" \
  --imessage-db-path ~/Library/Messages/chat.db
```

## 3. Full Start Command

```bash
omicclaw \
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

## 4. Current Notes

- Launching through `omicclaw` or `omicverse gateway` keeps the web gateway available.
- Use `omicverse claw -q ...` only for code generation, not for iMessage bot runtime.

## 5. Troubleshooting

1. `imsg` not found
   Check `which imsg`.

2. Cannot read `chat.db`
   Verify the DB path and macOS permission prompts.

3. Attachments are missing
   Start with `--imessage-include-attachments`.
