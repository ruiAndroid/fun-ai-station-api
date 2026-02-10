import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.core.config import get_settings

router = APIRouter(prefix="/openai/v1", tags=["openai-compat"])


def _require_bearer(request: Request) -> None:
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = auth.removeprefix("Bearer ").strip()
    if token != settings.OPENAI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _pick_last_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    # find last user message content
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        # OpenAI allows rich content blocks; degrade to json
        try:
            return json.dumps(c, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(c)
    return ""


async def _agent_execute(*, agent: str, user_input: str, context: Dict[str, Any], trace_id: str) -> str:
    settings = get_settings()
    base = settings.FUN_AGENT_SERVICE_URL.rstrip("/")
    url = f"{base}/agents/{agent}/execute"
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json={"input": user_input, "context": context}, headers={"x-trace-id": trace_id})
    if resp.status_code >= 500:
        raise HTTPException(status_code=502, detail="Agent service unavailable")
    try:
        data = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid agent service response")
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Invalid agent service response")
    if "error" in data and resp.status_code >= 400:
        raise HTTPException(status_code=400, detail=data)
    return str(data.get("output") or "")


@router.get("/models")
async def list_models(request: Request):
    _require_bearer(request)
    # Minimal response: OpenClaw typically only needs "some model exists".
    settings = get_settings()
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": "fun-agent",
                "object": "model",
                "created": now,
                "owned_by": "fun-ai-station",
            },
            {
                "id": settings.OPENAI_DEFAULT_AGENT or "attendance",
                "object": "model",
                "created": now,
                "owned_by": "fun-ai-station",
            },
        ],
    }


@router.post("/chat/completions")
async def chat_completions(request: Request):
    _require_bearer(request)
    settings = get_settings()

    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        payload = {}

    trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())

    messages = payload.get("messages")
    user_input = _pick_last_user_text(messages) or ""

    # We accept model but default to configured agent.
    model = payload.get("model")
    agent = settings.OPENAI_DEFAULT_AGENT or settings.OPENCLAW_DEFAULT_AGENT or "attendance"
    if isinstance(model, str) and model.strip():
        # If caller passes model like "agent:attendance", respect it.
        m = model.strip()
        if m.startswith("agent:") and m.split("agent:", 1)[1].strip():
            agent = m.split("agent:", 1)[1].strip()

    stream = bool(payload.get("stream"))

    context: Dict[str, Any] = {
        "trace_id": trace_id,
        "source": "openai-compat",
        "openai": {
            "model": model,
            "messages": messages if isinstance(messages, list) else None,
        },
    }

    output = await _agent_execute(agent=agent, user_input=user_input, context=context, trace_id=trace_id)

    created = int(time.time())
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"

    if not stream:
        return JSONResponse(
            status_code=200,
            content={
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model or "fun-agent",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": output},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def _sse() -> AsyncIterator[bytes]:
        # Minimal SSE streaming compatible with OpenAI: one delta chunk + DONE.
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model or "fun-agent",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": output}, "finish_reason": "stop"}
            ],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")

