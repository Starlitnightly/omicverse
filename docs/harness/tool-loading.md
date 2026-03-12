# Tool Loading

OVAgent does not yet implement Claude-style deferred tool loading. The current
runtime exposes one static catalog at agent construction time.

## Current State

- the visible tool set is static for a turn
- the runtime does not expose `ToolSearch`
- the web handshake reports harness capabilities, not per-session loaded tools

## Required Direction

When deferred loading is added, these rules should hold:

- tool visibility must be explicit per session
- risky tools must not appear silently
- traces should record which tools were visible for a turn
- server-only tools should remain gated by environment and approval policy

## Existing Backing Pieces

- session summaries already expose turn and trace state
- harness traces already persist structured turn metadata
- the approval broker already pauses and resumes risky execution

