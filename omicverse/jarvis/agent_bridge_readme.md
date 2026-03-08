# AgentBridge — Architecture & Channel Integration

## Overview

`AgentBridge` is the thin async middleware between a channel handler (Telegram / Feishu / QQ) and `ov.Agent`. It drives `agent.stream_async()`, fires UI callbacks on each event, and harvests output artifacts (figures, reports, files) into a single `AgentRunResult` object.

```
User message
     |
     v
channel._run_analysis()
     |
     +---> AgentBridge(session.agent, progress_cb, llm_chunk_cb)
     |              |
     |              v
     |     bridge.run(request, adata)
     |              |
     |              v
     |     async for event in agent.stream_async():
     |              |
     |              +-- "llm_chunk"   --> llm_chunk_cb(chunk)
     |              +-- "code"        --> progress_cb(description)
     |              +-- "tool_call"   --> progress_cb("tool: name(args)")
     |              +-- "tool_result" --> diagnostics (error detection)
     |              +-- "result"      --> result.adata + _harvest_figures()
     |              +-- "finish"      --> result.summary
     |              +-- "done"        --> result.summary (final)
     |              +-- "error"       --> result.error
     |              +-- "usage"       --> result.usage
     |
     v
AgentRunResult
  .adata       -- updated AnnData (or None)
  .figures     -- list[bytes]  PNG images from notebook outputs / disk
  .reports     -- list[str]    freshly written .md files
  .artifacts   -- list[AgentArtifact]  .csv/.pdf/.xlsx etc.
  .summary     -- str   agent's final narrative
  .error       -- str | None
  .diagnostics -- list[str]  tool errors, empty-arg warnings, max-turn alerts
  .usage       -- token usage object from LLM
     |
     v
channel renders result to user
```

---

## Event Stream Reference

| Event type    | Payload                     | AgentBridge action                              |
|---------------|-----------------------------|-------------------------------------------------|
| `llm_chunk`   | raw LLM token string        | fire `llm_chunk_cb(chunk)`                      |
| `code`        | code string + description   | fire `progress_cb(description or first_line)`   |
| `tool_call`   | `{name, arguments}`         | fire `progress_cb("tool: name(arg_keys)")`      |
| `tool_result` | `{name, output}`            | parse errors into `result.diagnostics`          |
| `status`      | `{follow_up_exhausted}`     | push diagnostic if agent gave up                |
| `result`      | updated adata               | set `result.adata`; call `_harvest_figures()`   |
| `finish`      | summary string              | set `result.summary`                            |
| `done`        | final summary / error       | set `result.summary`; set `result.error` if err |
| `error`       | error string                | set `result.error`                              |
| `usage`       | token usage object          | set `result.usage`                              |

> **Note on `search_functions`:** This is an OmicVerse agent tool registered in `tool_runtime.py`. When the agent calls it, AgentBridge receives a `tool_call` event with `name="search_functions"` and fires `progress_cb("tool: search_functions(query)")`. The channel then surfaces this to the user via its progress mechanism.

---

## Artifact Harvesting

After each `result` event (and again at the end of the run), AgentBridge scans:

1. **Notebook cell outputs** — last 10 cells for `display_data / image/png` outputs
2. **Filesystem** — `session_dir/`, `workspace_dir/`, and `cwd/` for `*.png` files newer than run start
3. **Reports** — `*.md` files newer than run start (up to 5, capped at 12 000 chars each)
4. **Artifacts** — `.md .pdf .csv .tsv .txt .html .xlsx` files newer than run start (up to 12)

Deduplication is key-based (path + size + mtime for files; base64 prefix for notebook images).

---

## Channel Comparison

### Callbacks during analysis

| Callback       | Telegram                               | Feishu                                  | QQ                                          |
|----------------|----------------------------------------|-----------------------------------------|---------------------------------------------|
| `llm_chunk_cb` | append to buffer; edit draft (1.5 s throttle) | append to buffer; edit card (1.2 s throttle) | append to buffer only (not shown live)  |
| `progress_cb`  | update `last_progress`; edit draft (forced) | update `last_progress`; edit card (forced) | send text message (12 s throttle)        |

### Draft message strategy

| Channel  | Draft mechanism                                           | Can edit in-place? |
|----------|-----------------------------------------------------------|--------------------|
| Telegram | `bot.send_message("思考中...")` → `editMessageText` HTML  | Yes                |
| Feishu   | `send_markdown_card("思考中...", color=grey)` → `edit_card` PATCH | Yes         |
| QQ       | None (no draft) — ack message sent before analysis starts | No                 |

### Result delivery (after `bridge.run()` returns)

#### Telegram

```
Error?
  yes --> edit draft to error_message() HTML (or send new if edit fails)
  no  -->
       Complex result (figures / reports / artifacts / long summary)?
         yes --> edit draft to "正在发送..."
                 _dispatch_final_blocks():
                   reports    --> send_prose() / send_code()
                   figures    --> sendPhoto (InputMediaPhoto)
                   artifacts  --> sendDocument
                   summary    --> md_to_html() + keyboard
                 edit draft to "分析完成"
         no  --> edit draft to final HTML
                 if too long / has <pre> --> send_prose()
                 send keyboard as separate message
```

#### Feishu

```
Error?
  yes --> edit_card(red) "分析失败"  (fallback: edit_text)
  no  -->
       edit draft card to "分析完成，正在发送结果..."
       for each report (<=4800 chars):
           send_markdown_card("分析报告")
       for each figure:
           send_image(file_key)
       for each artifact:
           send_file(file_key)
       summary (<=4800 chars):
           send_markdown_card("分析完成", color=green)
       edit draft card to "分析完成" (color=green)
```

#### QQ

```
Error?
  yes --> _send_markdown(err_text)   [msg_type=2 or fallback plain text]
  no  -->
       for each report chunk:
           _send_markdown(chunk)
       for each figure (if image server configured):
           _ImageServer.host_image() --> public URL
           send_image(url)
           sleep(10s) --> remove from server
       summary:
           _send_markdown(summary)

Note: every send auto-allocates a unique msg_seq per msg_id
      (prevents silent message drops from duplicate seq)
      msg_id expiry (>5 min) --> retry without msg_id (proactive send)
```

---

## Session State Updated After Each Run

```python
session.adata         = result.adata        # persisted to workspace/current.h5ad
session.prompt_count += 1
session.last_usage    = result.usage
session.append_memory_log(
    request   = user_text,
    summary   = result.summary,
    adata_info= "N cells x M genes",
)
```

---

## Key Files

| File                                          | Role                                              |
|-----------------------------------------------|---------------------------------------------------|
| `omicverse/jarvis/agent_bridge.py`            | AgentBridge class, event loop, artifact harvest   |
| `omicverse/utils/ovagent/tool_runtime.py`     | Agent tool implementations (search_functions etc.)|
| `omicverse/jarvis/channels/telegram.py`       | Telegram channel; HTML draft streaming            |
| `omicverse/jarvis/channels/feishu.py`         | Feishu channel; interactive card streaming        |
| `omicverse/jarvis/channels/qq.py`             | QQ channel; throttled progress text; msg_seq mgmt |
| `omicverse/jarvis/session.py`                 | SessionManager; per-user ov.Agent + workspace     |
