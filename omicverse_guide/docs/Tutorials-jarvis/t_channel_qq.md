---
title: J.A.R.V.I.S. QQ Tutorial
---

# J.A.R.V.I.S. QQ Tutorial

## 1. 准备 QQ Bot 凭据

至少需要：

- `QQ_APP_ID`
- `QQ_CLIENT_SECRET`

环境变量示例：

```bash
export QQ_APP_ID="your_qq_app_id"
export QQ_CLIENT_SECRET="your_qq_client_secret"
```

## 2. 启动 Jarvis（QQ）

```bash
omicverse jarvis \
  --channel qq \
  --qq-app-id "$QQ_APP_ID" \
  --qq-client-secret "$QQ_CLIENT_SECRET"
```

## 3. 图片与 Markdown（可选）

```bash
omicverse jarvis \
  --channel qq \
  --qq-app-id "$QQ_APP_ID" \
  --qq-client-secret "$QQ_CLIENT_SECRET" \
  --qq-image-host "http://YOUR_PUBLIC_IP:8081" \
  --qq-image-server-port 8081 \
  --qq-markdown
```

参数说明：

- `--qq-image-host`：QQ 访问图片的公网地址（不配置时图像发送可能受限）
- `--qq-image-server-port`：本地图片服务端口（默认 `8081`）
- `--qq-markdown`：启用 markdown 消息格式（需要平台权限）

## 4. 常见问题

1. 启动时报凭据缺失  
   检查 `QQ_APP_ID`、`QQ_CLIENT_SECRET` 或对应 CLI 参数

2. 文字能发但图片发送失败  
   检查 `--qq-image-host` 是否为公网可访问地址

3. Markdown 不生效  
   检查 QQ 开放平台机器人是否已开通 markdown 消息权限
