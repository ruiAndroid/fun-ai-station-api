### 消息闭环（OpenClaw 生成回复并回到企业微信）

你已经完成了“OpenClaw 收到企微消息 → 转发到 `fun-ai-station-api` webhook”的单向链路。
要做“闭环”（用户发消息后能收到 AI 回复），最稳的方式是让 OpenClaw **把 `fun-ai-station-api` 当作 LLM 后端**：

- OpenClaw 仍然负责：企业微信回调、加解密、被动回复/流式占位等
- `fun-ai-station-api` 负责：把消息交给 `fun-agent-service` 执行，并把文本结果作为 “LLM 输出” 返回给 OpenClaw

这样无需在企微侧做额外的“出站发消息”配置，闭环天然成立。

---

### 1) fun-ai-station-api：开启 OpenAI 兼容接口

本项目已提供 OpenAI 兼容接口：

- `GET /api/openai/v1/models`
- `POST /api/openai/v1/chat/completions`

鉴权方式：`Authorization: Bearer <OPENAI_API_KEY>`

配置在：`fun-ai-station-api/configs/fun-ai-station-api.env`

- `OPENAI_API_KEY`
- `OPENAI_DEFAULT_AGENT`（默认用哪个 agent 来处理消息）

改完后重启 `fun-ai-station-api`。

---

### 2) OpenClaw：把模型 baseUrl 指向本项目

在 OpenClaw 服务器的 `~/.openclaw/openclaw.json` 里，把你正在用的模型配置改成类似：

```json
{
  "models": {
    "default": {
      "baseUrl": "http://47.118.27.59/api/openai/v1",
      "apiKey": "填 fun-ai-station-api.env 的 OPENAI_API_KEY",
      "model": "fun-agent"
    }
  }
}
```

说明：
- `baseUrl` 末尾不要带 `/chat/completions`，只需要到 `/openai/v1`
- `model` 可以随便填；若你希望显式指定 agent，可用：`agent:<agent_code>`（例如 `agent:attendance`）

改完后重启 OpenClaw Gateway（你当前是 systemd user unit）：

```bash
XDG_RUNTIME_DIR=/run/user/0 systemctl --user restart openclaw-gateway
```

---

### 3) 验证

在 OpenClaw 服务器上验证 models：

```bash
curl -sS "http://47.118.27.59/api/openai/v1/models" \
  -H "Authorization: Bearer <OPENAI_API_KEY>" | head
```

验证 chat/completions：

```bash
curl -sS "http://47.118.27.59/api/openai/v1/chat/completions" \
  -H "Authorization: Bearer <OPENAI_API_KEY>" \
  -H "content-type: application/json" \
  --data '{"model":"fun-agent","messages":[{"role":"user","content":"你好"}]}' | head
```

