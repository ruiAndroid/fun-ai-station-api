### OpenClaw（企业微信）→ fun-ai-station：整体架构图（当前部署）

下面是我们这次最终跑通的生产链路（按你现在的部署：企业微信回调走 `studio.fun.tv` 的 HTTPS，openclaw 机器不需要 HTTPS，通过 Nginx `mirror` 把回调复制给本机转发器，再加签转给 `fun-ai-station-api`）。

### 架构图（Mermaid）

```mermaid
flowchart LR
  %% ========= External =========
  subgraph WeCom[企业微信 WeCom]
    W1[用户/群聊消息]
    W2[企业微信回调服务器]
  end

  %% ========= Java HTTPS entry =========
  subgraph Studio[Java 服务器（HTTPS 入口）]
    S1[Nginx / 网关\nstudio.fun.tv]
    S2[Java Service（业务服务）]
  end

  %% ========= OpenClaw server =========
  subgraph OpenClaw[OpenClaw 服务器 43.128.113.123]
    O1[Nginx (HTTP)\n:18790]
    O2[openclaw-gateway\n(Node)\n:18789]
    O3[wecom 插件\n/wecom/bot]
    O4[本机转发器\n解密 WeCom + 加签\n127.0.0.1:9100]
  end

  %% ========= FunAIStation server =========
  subgraph FunAI[fun-ai-station 服务器 47.118.27.59]
    F1[Nginx\n:80\n/api/* -> 127.0.0.1:8001]
    F2[fun-ai-station-api\nFastAPI\n:8001\n/webhooks/openclaw]
    F3[fun-agent-service\nNode+Python\n:4010\n/agents/:agent/execute]
  end

  %% ========= Flows =========
  W1 --> W2

  %% WeCom callback must be HTTPS
  W2 -->|HTTPS 回调\nhttps://studio.fun.tv/wecom/bot| S1

  %% Studio forwards to OpenClaw (HTTP is OK)
  S1 -->|HTTP 透明反代\nhttp://43.128.113.123:18790/wecom/bot| O1

  %% OpenClaw nginx proxies to gateway
  O1 -->|proxy_pass| O2
  O2 --> O3

  %% Mirror copy to local forwarder (does NOT affect main response)
  O1 -.->|mirror 子请求\nPOST 同一份 body| O4

  %% Forwarder decrypts wecom encrypt, extracts text, signs and forwards
  O4 -->|HTTP + HMAC-SHA256 签名\nPOST http://47.118.27.59/api/webhooks/openclaw| F1
  F1 -->|proxy_pass /api/| F2
  F2 -->|HTTP| F3

  %% ========= Notes =========
  classDef note fill:#f8f8f8,stroke:#ddd,color:#333;
```

### 关键点（简要）

- **企业微信回调必须是 HTTPS**：我们用 `studio.fun.tv` 作为受信任 HTTPS 入口。
- **openclaw 服务器可保持 HTTP**：`studio.fun.tv` 反代到 `43.128.113.123:18790`。
- **Nginx mirror 的作用**：不影响 openclaw 正常处理企微回调，同时复制一份请求给本机转发器做“出站转发”。
- **本机转发器职责**：
  - 校验/解密 WeCom `encrypt`（Bot 模式）
  - 提取文本（或将非文本转换为可读摘要）
  - 生成 `x-openclaw-timestamp` + `x-openclaw-signature`（HMAC-SHA256）
  - POST 到 `fun-ai-station-api` 的 `/api/webhooks/openclaw`
- **fun-ai-station-api webhook**：验签 + 幂等去重（短 TTL）+ 转发给 `fun-agent-service` 执行，返回 `output`。

