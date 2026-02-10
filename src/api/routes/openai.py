import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.core.config import get_settings
from src.core.db import get_db
from src.core.security import hash_password
from src.models.agent import Agent
from src.models.base import utcnow
from src.models.chat import ChatMessage, ChatSession
from src.models.user import User
from sqlalchemy.orm import Session

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

def _ensure_system_user(db: Session, *, email: str, password: str) -> User:
    user = db.query(User).filter(User.email == email).first()
    if user:
        return user
    user = User(email=email, hashed_password=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _ensure_agent_row(db: Session, *, agent_code: str) -> Agent:
    row = db.query(Agent).filter(Agent.code == agent_code).first()
    if row:
        return row
    row = Agent(
        code=agent_code,
        name=agent_code,
        handle=f"@{agent_code}",
        description=f"Auto-created for OpenAI-compat: {agent_code}",
        tags=["openai-compat"],
        capabilities=[],
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _persist_pair(
    db: Session,
    *,
    user: User,
    agent_row: Agent,
    title: str,
    user_text: str,
    assistant_text: str,
) -> str:
    # reuse session by (user_id, agent_id, title)
    session = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.id, ChatSession.agent_id == agent_row.id, ChatSession.title == title)
        .first()
    )
    if not session:
        session = ChatSession(user_id=user.id, agent_id=agent_row.id, title=title)
        db.add(session)
        db.commit()
        db.refresh(session)

    session.updated_at = utcnow()
    db.add(ChatMessage(session_id=session.id, role="user", content=user_text))
    db.commit()

    session.updated_at = utcnow()
    db.add(ChatMessage(session_id=session.id, role="assistant", content=assistant_text))
    db.commit()
    return session.id


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
async def chat_completions(request: Request, db: Session = Depends(get_db)):
    _require_bearer(request)
    settings = get_settings()

    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        payload = {}

    trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())

    messages = payload.get("messages")
    user_input = _pick_last_user_text(messages) or ""
    if len(user_input) > 8000:
        user_input = user_input[:7999] + "…"

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
    output_text = str(output or "")
    if len(output_text) > 12000:
        output_text = output_text[:11999] + "…"

    # Persist under one system user (testing)
    system_email = (settings.OPENCLAW_SYSTEM_USER_EMAIL or "").strip()
    if system_email:
        system_user = _ensure_system_user(
            db,
            email=system_email,
            password=settings.OPENCLAW_SYSTEM_USER_PASSWORD or "",
        )
        agent_row = _ensure_agent_row(db, agent_code=agent)
        session_title = f"openclaw:llm:{agent}"
        _persist_pair(
            db,
            user=system_user,
            agent_row=agent_row,
            title=session_title[:255],
            user_text=user_input,
            assistant_text=output_text,
        )

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
                        "message": {"role": "assistant", "content": output_text},
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
                {"index": 0, "delta": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}
            ],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


@router.post("/completions")
async def completions(request: Request, db: Session = Depends(get_db)):
    """
    OpenAI legacy completions compatibility.
    Some clients/providers still call /v1/completions instead of /v1/chat/completions.
    We map `prompt` -> agent input and return a minimal text completion response.
    """
    _require_bearer(request)
    settings = get_settings()

    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        payload = {}

    trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())

    prompt = payload.get("prompt", "")
    if isinstance(prompt, list) and prompt:
        user_input = str(prompt[-1])
    else:
        user_input = str(prompt or "")
    if len(user_input) > 8000:
        user_input = user_input[:7999] + "…"

    model = payload.get("model")
    agent = settings.OPENAI_DEFAULT_AGENT or settings.OPENCLAW_DEFAULT_AGENT or "attendance"
    if isinstance(model, str) and model.strip():
        m = model.strip()
        if m.startswith("agent:") and m.split("agent:", 1)[1].strip():
            agent = m.split("agent:", 1)[1].strip()

    context: Dict[str, Any] = {
        "trace_id": trace_id,
        "source": "openai-compat",
        "openai": {
            "model": model,
            "prompt": prompt if isinstance(prompt, (str, list)) else None,
        },
    }

    output = await _agent_execute(agent=agent, user_input=user_input, context=context, trace_id=trace_id)
    output_text = str(output or "")
    if len(output_text) > 12000:
        output_text = output_text[:11999] + "…"

    system_email = (settings.OPENCLAW_SYSTEM_USER_EMAIL or "").strip()
    if system_email:
        system_user = _ensure_system_user(
            db,
            email=system_email,
            password=settings.OPENCLAW_SYSTEM_USER_PASSWORD or "",
        )
        agent_row = _ensure_agent_row(db, agent_code=agent)
        session_title = f"openclaw:llm:{agent}"
        _persist_pair(
            db,
            user=system_user,
            agent_row=agent_row,
            title=session_title[:255],
            user_text=user_input,
            assistant_text=output_text,
        )

    created = int(time.time())
    completion_id = f"cmpl_{uuid.uuid4().hex}"
    return JSONResponse(
        status_code=200,
        content={
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model or "fun-agent",
            "choices": [
                {
                    "index": 0,
                    "text": output_text,
                    "finish_reason": "stop",
                }
            ],
        },
    )

