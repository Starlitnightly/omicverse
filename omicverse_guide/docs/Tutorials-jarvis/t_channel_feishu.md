---
title: J.A.R.V.I.S. Feishu Tutorial
---

# J.A.R.V.I.S. Feishu 教程

Jarvis 支持飞书两种接入模式：

- `websocket`（默认，推荐）
- `webhook`（需要公网回调地址）

## 1. 飞书应用准备

1. 在飞书开放平台创建企业自建应用。
2. 获取并保存 `App ID`、`App Secret`。
3. 在事件订阅中至少勾选 `im.message.receive_v1`。
4. 将机器人添加到目标群聊或可私聊场景。

## 2. 环境变量

```bash
export FEISHU_APP_ID="cli_xxx"
export FEISHU_APP_SECRET="xxx"
export FEISHU_VERIFICATION_TOKEN="xxx"
export FEISHU_ENCRYPT_KEY="xxx"
export ANTHROPIC_API_KEY="your_api_key"
```

## 3. 最小启动命令（WebSocket）

```bash
omicverse jarvis \
  --channel feishu \
  --feishu-connection-mode websocket \
  --feishu-app-id "$FEISHU_APP_ID" \
  --feishu-app-secret "$FEISHU_APP_SECRET"
```

## 4. 完整启动命令（WebSocket）

```bash
omicverse jarvis \
  --channel feishu \
  --feishu-connection-mode websocket \
  --feishu-app-id "$FEISHU_APP_ID" \
  --feishu-app-secret "$FEISHU_APP_SECRET" \
  --feishu-verification-token "$FEISHU_VERIFICATION_TOKEN" \
  --feishu-encrypt-key "$FEISHU_ENCRYPT_KEY" \
  --model claude-sonnet-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  --auth-mode environment \
  --session-dir ~/.ovjarvis \
  --max-prompts 0 \
  --verbose
```

参数逐项解释：

- `--channel feishu`：选择飞书通道。
- `--feishu-connection-mode websocket`：使用长连接事件订阅模式。
- `--feishu-app-id`：飞书应用 App ID。
- `--feishu-app-secret`：飞书应用 App Secret。
- `--feishu-verification-token`：事件 token 校验值（可选但推荐）。
- `--feishu-encrypt-key`：加密回调解密密钥（启用加密时必须）。
- `--model`：Jarvis 使用的模型名称。
- `--api-key`：显式指定 LLM API key。
- `--auth-mode environment`：从环境变量读取认证信息。
- `--session-dir`：会话根目录。
- `--max-prompts 0`：关闭自动重启，保持长生命周期 kernel。
- `--verbose`：输出详细日志。

## 5. 完整启动命令（Webhook）

```bash
omicverse jarvis \
  --channel feishu \
  --feishu-connection-mode webhook \
  --feishu-app-id "$FEISHU_APP_ID" \
  --feishu-app-secret "$FEISHU_APP_SECRET" \
  --feishu-verification-token "$FEISHU_VERIFICATION_TOKEN" \
  --feishu-encrypt-key "$FEISHU_ENCRYPT_KEY" \
  --feishu-host 0.0.0.0 \
  --feishu-port 8080 \
  --feishu-path /feishu/events \
  --model claude-sonnet-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  --auth-mode environment \
  --session-dir ~/.ovjarvis \
  --max-prompts 0 \
  --verbose
```

Webhook 额外参数解释：

- `--feishu-host`：Webhook 监听地址。
- `--feishu-port`：Webhook 监听端口。
- `--feishu-path`：Webhook 回调路径。

示例回调地址：

```text
http://<你的公网域名或IP>:8080/feishu/events
```

## 6. 常见问题

1. WebSocket 报错缺少 SDK  
   执行 `pip install lark-oapi`。

2. Webhook 验证失败  
   检查回调 URL 与 `--feishu-host/--feishu-port/--feishu-path` 是否一致，并确认公网可达。

3. 能收消息但不能发图/文件  
   检查飞书应用权限是否已开通并生效。
