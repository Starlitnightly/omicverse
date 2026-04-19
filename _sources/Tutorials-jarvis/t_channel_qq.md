---
title: OmicClaw QQ Tutorial
---

# OmicClaw QQ Tutorial

QQ is now documented as part of the unified gateway launcher flow.

## 1. 准备 QQ Bot 凭据

至少需要：

- `QQ_APP_ID`
- `QQ_CLIENT_SECRET`

环境变量示例：

```bash
export QQ_APP_ID="your_qq_app_id"
export QQ_CLIENT_SECRET="your_qq_client_secret"
```

## 2. 推荐启动命令

```bash
omicclaw \
  --channel qq \
  --qq-app-id "$QQ_APP_ID" \
  --qq-client-secret "$QQ_CLIENT_SECRET"
```

通用 gateway 入口：

```bash
omicverse gateway \
  --channel qq \
  --qq-app-id "$QQ_APP_ID" \
  --qq-client-secret "$QQ_CLIENT_SECRET"
```

## 3. 图片与 Markdown（可选）

```bash
omicclaw \
  --channel qq \
  --qq-app-id "$QQ_APP_ID" \
  --qq-client-secret "$QQ_CLIENT_SECRET" \
  --qq-image-host "http://YOUR_PUBLIC_IP:8081" \
  --qq-image-server-port 8081 \
  --qq-markdown \
  --model claude-sonnet-4-6 \
  --api-key "$ANTHROPIC_API_KEY" \
  --auth-mode environment \
  --session-dir ~/.ovjarvis \
  --max-prompts 0 \
  --verbose
```

参数说明：

- `--qq-image-host`：QQ 访问图像的公网地址
- `--qq-image-server-port`：本地图片服务端口
- `--qq-markdown`：启用 markdown 消息格式

## 4. 当前入口逻辑

- `omicclaw`：主推荐入口
- `omicverse gateway`：等价的通用 gateway 入口

## 5. 常见问题

1. 启动时报凭据缺失
   检查 `QQ_APP_ID`、`QQ_CLIENT_SECRET` 或对应 CLI 参数。

2. 文字能发但图片发送失败
   检查 `--qq-image-host` 是否公网可达。

3. Markdown 不生效
   检查 QQ 开放平台是否已经开通 markdown 消息权限。
