# OmicVerse Jarvis 使用教程

本文档介绍如何启动和使用 `omicverse jarvis`（Telegram 单细胞分析助手）。

## 1. 安装依赖

在项目根目录执行：

```bash
pip install -e ".[jarvis]"
```

如果你不使用 editable 安装，也可使用：

```bash
pip install "omicverse[jarvis]"
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

### 3.2 LLM API Key

支持以下环境变量之一（按顺序自动读取）：

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

示例：

```bash
export ANTHROPIC_API_KEY="your_api_key"
```

也可用 `--api-key` 显式传入。

## 4. 启动 Jarvis

最小启动：

```bash
omicverse jarvis
```

完整参数示例（覆盖当前全部 CLI 参数）：

```bash
omicverse jarvis \
  --token "$TELEGRAM_BOT_TOKEN" \
  --model claude-sonnet-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  --max-prompts 50 \
  --session-dir ~/.ovjarvis \
  --allowed-user your_telegram_username \
  --allowed-user 123456789 \
  --verbose
```

参数总览：

- `--token`: Telegram Bot Token（或 `TELEGRAM_BOT_TOKEN`）
- `--model`: 默认 `claude-sonnet-4-6`
- `--api-key`: LLM key（或 `ANTHROPIC_API_KEY/OPENAI_API_KEY/GEMINI_API_KEY`）
- `--max-prompts`: 单 kernel 最大请求数（默认 50）
- `--session-dir`: 会话根目录（默认 `~/.ovjarvis`）
- `--allowed-user`: 允许的用户名/ID（可重复）
- `--verbose`: 详细日志

## 5. 会话目录结构

默认每个 Telegram 用户位于 `~/.ovjarvis/<user_id>/`：

- `workspace/`: 数据与上下文目录
- `sessions/`: notebook/kernel 数据
- `context/`: Agent 上下文缓存
- `current.h5ad`: 当前加载数据快照

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
- `/kernel` kernel 健康与 prompt 用量
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

当接近 `--max-prompts` 上限时 kernel 可能重启。可提高 `--max-prompts`，并定期 `/save`。
