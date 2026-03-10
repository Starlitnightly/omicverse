---
title: J.A.R.V.I.S. Session and Commands
---

# J.A.R.V.I.S. Session and Commands

## 1. Recommended Workflow

1. Upload a `.h5ad` file to chat or workspace.
2. Run `/workspace` and `/load <filename>`.
3. Send your analysis request in natural language.
4. Monitor with `/status` and `/kernel`.
5. Export results with `/save`.

## 2. Common Commands

- `/workspace`
- `/ls [path]`
- `/find <pattern>`
- `/load <filename>`
- `/save`
- `/status`
- `/usage`
- `/model [name]`
- `/memory`
- `/cancel`
- `/reset`
- `/kernel`
- `/kernel ls`
- `/kernel new <name>`
- `/kernel use <name>`

## 3. Session Directory Layout

Default root:

```text
~/.ovjarvis
```

Typical entries:

- `workspace/`
- `sessions/`
- `context/`
- `current.h5ad`
- `kernels/<name>/...`

## 4. Behavior Customization

Files in workspace:

- `AGENTS.md`: assistant behavior rules.
- `MEMORY.md`: long-term memory.
- `memory/YYYY-MM-DD.md`: daily memory summaries.
