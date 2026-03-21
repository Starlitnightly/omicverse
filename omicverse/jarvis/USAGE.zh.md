# OmicVerse Jarvis 使用教程

本文档介绍如何启动和使用 `omicverse jarvis`（多渠道单细胞分析助手）。

## 1. 安装依赖

在项目根目录执行：

```bash
pip install -e ".[jarvis]"
```

如果你不使用 editable 安装，也可使用：

```bash
pip install "omicverse[jarvis]"
```

如果要启用 macOS iMessage，还需要额外安装 [`imsg`](https://github.com/steipete/imsg)：

```bash
brew install steipete/tap/imsg
```

## 2. Telegram Bot 申请教程

Jarvis 依赖 Telegram Bot Token。可按以下步骤申请：

1. 在 Telegram 搜索并打开 `@BotFather`。
2. 发送 `/newbot`，按提示填写：
   - Bot 显示名称（可含空格）
   - Bot 用户名（必须以 `bot` 结尾，如 `omicverse_jarvis_bot`）
3. 创建成功后，BotFather 会返回 token（形如 `123456:ABC-...`），即 `TELEGRAM_BOT_TOKEN`。
4. 可选配置：
   - `/setdescription`
   - `/setabouttext`
   - `/setuserpic`
   - `/setcommands`（如 `start/help/status/load/save/reset`）

安全建议：

- 不要把 token 提交到 Git。
- token 泄露后，立刻在 BotFather 执行 `/revoke`。
- 生产环境建议使用 `--allowed-user` 限制访问。

## 3. 准备密钥

### 3.1 Telegram Bot Token

二选一：

```bash
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
```

或启动时传参：

```bash
omicverse jarvis --token "your_telegram_bot_token"
```

### 3.2 LLM 认证

支持以下环境变量之一（按顺序自动读取）：

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

示例：

```bash
export ANTHROPIC_API_KEY="your_api_key"
```

也可用 `--api-key` 显式传入。

如果你想走 OpenAI 浏览器登录而不是手动填 API key，直接运行：

```bash
omicverse jarvis --setup
```

首次向导可以完成：

- 选择 `iMessage`、`Telegram` 或 `Feishu`
- 把渠道配置保存到 `~/.ovjarvis/config.json`
- 配置 OpenAI Codex OAuth（ChatGPT 登录）或各 provider 的 API Key
- 选择 OpenAI、Claude、千问、Kimi、DeepSeek、智谱、Gemini、Grok、Ollama 以及其他 OpenAI 兼容端点上的默认模型
- 为 Ollama 或其他 OpenAI 兼容网关保存自定义 endpoint

## 4. 启动 Jarvis

最小启动：

```bash
omicverse jarvis
```

首次交互式配置：

```bash
omicverse jarvis --setup
```

强制指定向导语言：

```bash
omicverse jarvis --setup --setup-language zh
```

显式使用 Telegram 渠道：

```bash
omicverse jarvis --channel telegram --token "$TELEGRAM_BOT_TOKEN"
```

显式使用 Discord 渠道：

```bash
omicverse jarvis --channel discord --discord-token "$DISCORD_BOT_TOKEN"
```

Feishu Webhook 渠道：

```bash
omicverse jarvis \
  --channel feishu \
  --feishu-app-id "$FEISHU_APP_ID" \
  --feishu-app-secret "$FEISHU_APP_SECRET" \
  --feishu-verification-token "$FEISHU_VERIFICATION_TOKEN" \
  --feishu-encrypt-key "$FEISHU_ENCRYPT_KEY" \
  --feishu-host 0.0.0.0 \
  --feishu-port 8080 \
  --feishu-path /feishu/events
```

Feishu 长连接渠道（WebSocket 事件订阅）：

```bash
omicverse jarvis \
  --channel feishu \
  --feishu-connection-mode websocket \
  --feishu-app-id "$FEISHU_APP_ID" \
  --feishu-app-secret "$FEISHU_APP_SECRET" \
  --feishu-verification-token "$FEISHU_VERIFICATION_TOKEN" \
  --feishu-encrypt-key "$FEISHU_ENCRYPT_KEY"
```

`websocket` 模式下，请在飞书后台将事件订阅配置为“长连接订阅”；`--feishu-host/--feishu-port/--feishu-path` 不生效。

iMessage 渠道：

```bash
omicverse jarvis \
  --channel imessage \
  --imessage-cli-path "$(which imsg)" \
  --imessage-db-path ~/Library/Messages/chat.db
```

### 4.1 飞书专用部署教程（Webhook 模式）

本节是飞书渠道的最小可用 checklist，按顺序完成即可。

#### A. 飞书应用准备

1. 在飞书开放平台创建企业自建应用。  
2. 获取并保存：
   - `App ID`
   - `App Secret`
3. 将机器人添加到目标群聊（或可私聊使用的场景）。

#### B. 事件订阅与回调

1. 在“事件订阅”中开启订阅。  
2. 回调地址填写：
   - `http://<你的公网域名或IP>:8080/feishu/events`
   - 若你改过参数，请与 `--feishu-host/--feishu-port/--feishu-path` 保持一致。  
3. 订阅事件：
   - `im.message.receive_v1`
4. 保存后，飞书会发起 URL 验证（`challenge`）；Jarvis 内置已支持该握手。

#### C. 权限建议（按飞书后台实际命名勾选）

至少确保机器人具备：
- 读取消息内容（用于接收文本/命令）
- 发送消息（文本）
- 上传并发送图片
- 上传并发送文件
- 下载文件（用于接收 `.h5ad`）

说明：飞书权限名在不同版本控制台中可能有细微差异，按“消息读取/发送、图片文件上传下载”对应项勾选即可。

#### D. 启动命令（推荐）

```bash
omicverse jarvis \
  --channel feishu \
  --feishu-app-id "$FEISHU_APP_ID" \
  --feishu-app-secret "$FEISHU_APP_SECRET" \
  --feishu-host 0.0.0.0 \
  --feishu-port 8080 \
  --feishu-path /feishu/events \
  --api-key "$ANTHROPIC_API_KEY" \
  --model claude-sonnet-4-6 \
  --max-prompts 0 \
  --verbose
```

#### E. 联调验证步骤

1. 飞书发送 `/status`，应返回当前会话状态。  
2. 飞书发送 `/kernel`，应看到 kernel 状态。  
3. 上传一个 `.h5ad` 文件，应提示“已上传并加载”。  
4. 发送一句分析请求，应先出现“思考中”草稿，再出现结果文本/图片/附件。

#### F. 飞书通道当前支持能力

- Draft 流式预览（消息编辑）
- 图片发送（analysis figures）
- 文件附件发送（report/csv/pdf/h5ad 等）
- `.h5ad` 上传并自动加载
- `/cancel` `/status` `/reset`
- `/kernel` `/kernel ls` `/kernel new` `/kernel use`

#### G. 常见故障排查（飞书）

1. 回调验证失败  
   - 检查回调 URL 与启动参数一致。  
   - 确认端口可从公网访问（必要时通过反向代理/隧道）。  

2. 能收消息但不能发图片/文件  
   - 通常是权限不足；检查“图片/文件上传发送”相关权限是否已开并生效。  

3. 上传 `.h5ad` 后未自动加载  
   - 检查文件后缀是否为 `.h5ad`。  
   - 查看 `--verbose` 日志中是否有下载或读取报错。  

4. `/cancel` 无效  
   - 仅取消当前正在运行的分析任务；若任务已结束会提示“当前没有运行中的分析”。

完整参数示例（覆盖当前全部 CLI 参数）：

```bash
omicverse jarvis \
  --token "$TELEGRAM_BOT_TOKEN" \
  --model claude-sonnet-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  --max-prompts 0 \
  --session-dir ~/.ovjarvis \
  --allowed-user your_telegram_username \
  --allowed-user 123456789 \
  --verbose
```

参数总览：

- `--token`: Telegram Bot Token（或 `TELEGRAM_BOT_TOKEN`）
- `--discord-token`: Discord Bot Token（或 `DISCORD_BOT_TOKEN`）
- `--setup`: 运行交互式设置向导
- `--setup-language`: 向导语言（`en` 或 `zh`，默认界面为英文）
- `--config-file`: 配置文件路径（默认 `~/.ovjarvis/config.json`）
- `--auth-file`: 本地认证状态文件（默认 `~/.ovjarvis/auth.json`）
- `--channel`: 渠道后端（`telegram`、`discord`、`feishu`、`imessage` 或 `qq`；若已配置则优先使用配置）
- `--feishu-app-id`: Feishu app id（或 `FEISHU_APP_ID`）
- `--feishu-app-secret`: Feishu app secret（或 `FEISHU_APP_SECRET`）
- `--feishu-connection-mode`: `webhook` 或 `websocket`（默认 `websocket`）
- `--feishu-verification-token`: Webhook 校验 token（或 `FEISHU_VERIFICATION_TOKEN`）
- `--feishu-encrypt-key`: Webhook 加密密钥（用于加密回调，或 `FEISHU_ENCRYPT_KEY`）
- `--feishu-host/--feishu-port/--feishu-path`: Feishu Webhook 监听参数（仅 `webhook` 模式使用）
- `--imessage-cli-path`: `imsg` 可执行文件路径
- `--imessage-db-path`: `chat.db` 路径
- `--imessage-include-attachments`: 启用 iMessage 入站附件元数据订阅
- `--auth-mode`: 保存到本地配置中的认证方式（`environment`、`openai_oauth` 表示 OpenAI Codex OAuth、`openai_api_key`、`saved_api_key`、`no_auth`）
- `--model`: 默认模型（若已配置则优先使用配置）
- `--api-key`: LLM API Key（或已保存的 provider 认证 / 对应 provider 的环境变量）
- `--endpoint`: Ollama 或其他 OpenAI 兼容端点的 API Base URL
- `--max-prompts`: 单 kernel 最大请求数（`0` 表示不自动重启，默认 0）
- `--session-dir`: 会话根目录（默认 `~/.ovjarvis`）
- `--allowed-user`: 允许的用户名/ID（可重复）
- `--verbose`: 详细日志

## 5. 会话目录结构

默认每个 Telegram 用户位于 `~/.ovjarvis/<user_id>/`：

- `workspace/`: 数据与上下文目录
- `sessions/`: notebook/kernel 数据
- `context/`: Agent 上下文缓存
- `current.h5ad`: 当前加载数据快照
- `kernels/<name>/...`: 额外命名 kernel（通过 `/kernel new <name>` 创建）

## 6. Telegram 内常用命令

### 6.1 数据与文件

- `/workspace` 查看工作区
- `/ls [路径]` 列出文件
- `/find <模式>` 搜索文件（如 `/find *.h5ad`）
- `/load <文件名>` 加载数据
- `/save` 导出当前 `h5ad`
- 直接上传 `.h5ad` 也可触发加载

### 6.2 会话与模型

- `/status` 当前状态
- `/kernel` 当前 kernel 健康与 prompt 用量
- `/kernel ls` 列出 kernels
- `/kernel new <name>` 新建并切换 kernel
- `/kernel use <name>` 切换 active kernel
- `/usage` 最近一次 token 用量
- `/model [名称]` 查看/切换模型
- `/memory` 近两天分析历史
- `/cancel` 取消当前分析
- `/reset` 重置会话并重启 kernel

### 6.3 Shell 命令（白名单）

`/shell` 仅允许：`ls find cat head wc file du pwd tree`

示例：

```text
/shell ls -lh
/shell find . -name "*.h5ad"
```

## 7. 推荐使用流程

1. 上传 `.h5ad` 到 `workspace/`（或直接发到 Telegram）。
2. 在 Telegram 执行 `/workspace` 与 `/load 文件名`。
3. 发送自然语言分析请求（中英都可以）。
4. 用 `/status`、`/kernel`、`/usage` 监控状态。
5. 用 `/save` 下载结果。

## 8. 自定义 Agent 行为

在 `workspace/AGENTS.md` 写入个性化规则（语言、分析风格、图偏好等），每次请求会自动注入。

`workspace/MEMORY.md` 用于长期记忆；`workspace/memory/YYYY-MM-DD.md` 会自动记录每日摘要。

## 9. 常见问题

### 9.1 启动报错：缺少 telegram 依赖

```bash
pip install -e ".[jarvis]"
```

### 9.2 提示缺少 Token

确认设置了 `TELEGRAM_BOT_TOKEN`，或通过 `--token` 传入。

### 9.3 模型切换后没生效

执行 `/model 新模型` 后，再执行 `/reset` 让新模型在新 kernel 生效。

### 9.4 分析被中断或变量丢失

建议使用 `--max-prompts 0` 关闭自动重启，默认维持单一长生命周期 kernel。只有你明确需要新环境时再用 `/reset`。
