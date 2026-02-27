### Openclaw → fun-ai-station-api 转发（Webhook）

`fun-ai-station-api` 提供一个 webhook 用于接收 openclaw 从企业微信/其他渠道收进来的消息，并转交给 `fun-agent-service` 执行。

### 架构图

见：`docs/openclaw-architecture.md`

### 1) 接口地址

如果你按本仓库的 Nginx 模板部署，对外地址是：

- `POST http://<域名或公网IP>/api/webhooks/openclaw`

说明：Nginx 会把 `/api/` 反代到 FastAPI（8001）根路径，因此 FastAPI 实际路由是 `/webhooks/openclaw`。

### 2) 配置

编辑 `fun-ai-station-api/configs/fun-ai-station-api.env`：

- `OPENCLAW_WEBHOOK_SECRET`: HMAC 签名密钥（务必改成强随机）
- `OPENCLAW_MAX_SKEW_SECONDS`: 时间戳容忍偏移（默认 300 秒）
- `OPENCLAW_DEFAULT_AGENT`: 默认智能体（默认 `general`，以 `GET /api/agent-service/agents` 输出为准）

改完后重启 `fun-ai-station-api` 生效。

### 3) 鉴权（签名算法）

请求头（必需）：

- `x-openclaw-timestamp`: unix 秒级时间戳（int）
- `x-openclaw-signature`: hex 字符串

签名计算（HMAC-SHA256）：

- 待签名消息：把 `timestamp`（十进制字符串）、一个点号 `.`、以及 **原始 HTTP body 字节** 直接拼接
- `signature = hex(hmac_sha256(secret, message_bytes))`

### 4) 请求体（建议）

建议 openclaw 转发时至少包含：

```json
{
  "event_id": "unique-message-id",
  "agent": "general",
  "text": "用户发来的消息内容",
  "context": {
    "channel": "wecom",
    "from": "user_id"
  }
}
```

### 5) curl 快速验证

下面示例用 `python` 现算签名（你也可以在 openclaw 里用同样算法实现）：

```bash
ts=$(python -c "import time; print(int(time.time()))")
body='{"event_id":"e1","agent":"general","text":"你好"}'
sig=$(python - << 'PY'
import hmac, hashlib, os, sys
secret = os.environ["OPENCLAW_WEBHOOK_SECRET"].encode("utf-8")
ts = os.environ["TS"].encode("utf-8")
body = os.environ["BODY"].encode("utf-8")
msg = ts + b"." + body
print(hmac.new(secret, msg, hashlib.sha256).hexdigest())
PY
)

curl -X POST "http://<域名或公网IP>/api/webhooks/openclaw" \
  -H "content-type: application/json" \
  -H "x-openclaw-timestamp: $ts" \
  -H "x-openclaw-signature: $sig" \
  --data "$body"
```

注意：Windows PowerShell 下变量/引号略有差异，但签名算法相同。

