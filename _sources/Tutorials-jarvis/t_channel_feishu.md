---
title: OmicClaw Feishu Tutorial
---

# OmicClaw Feishu 教程

飞书现在推荐通过 `omicclaw` 或 `omicverse gateway` 启动，这两种方式都会进入统一 gateway 入口。

## 1. 飞书应用准备

1. 在飞书开放平台创建企业自建应用。
2. 获取并保存：
   - `App ID`
   - `App Secret`
3. 在事件订阅中至少勾选 `im.message.receive_v1`。
4. 将机器人加入目标群聊或允许私聊的场景。

## 2. 环境变量

```bash
export FEISHU_APP_ID="cli_xxx"
export FEISHU_APP_SECRET="xxx"
export FEISHU_VERIFICATION_TOKEN="xxx"
export FEISHU_ENCRYPT_KEY="xxx"
export ANTHROPIC_API_KEY="your_api_key"
```

## 3. 推荐启动方式

OmicClaw 品牌入口：

```bash
omicclaw \
  --channel feishu \
  --feishu-connection-mode websocket \
  --feishu-app-id "$FEISHU_APP_ID" \
  --feishu-app-secret "$FEISHU_APP_SECRET"
```

通用 gateway 入口：

```bash
omicverse gateway \
  --channel feishu \
  --feishu-connection-mode websocket \
  --feishu-app-id "$FEISHU_APP_ID" \
  --feishu-app-secret "$FEISHU_APP_SECRET"
```

## 4. WebSocket 模式

```bash
omicclaw \
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
  --web-port 5050 \
  --verbose
```

说明：

- `websocket` 是当前推荐模式。
- 该模式下不需要 `--feishu-host/--feishu-port/--feishu-path`。
- 通过 `omicclaw` 或 `omicverse gateway` 启动时，web UI 会和飞书通道一起运行。

## 5. Webhook 模式

```bash
omicclaw \
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

回调地址示例：

```text
http://<你的公网域名或IP>:8080/feishu/events
```

## 6. 当前入口逻辑

- `omicclaw`：OmicClaw 品牌入口，强制登录 web 界面
- `omicverse gateway`：OmicVerse 品牌 gateway 入口

## 7. 常见问题

1. WebSocket 报缺少 SDK
   安装 `lark-oapi`。

2. Webhook 验证失败
   检查公网回调地址与 `--feishu-host/--feishu-port/--feishu-path` 是否一致。

3. 文字能收发，但图片或文件失败
   检查飞书应用权限是否已生效。
