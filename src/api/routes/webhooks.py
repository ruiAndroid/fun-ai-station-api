import asyncio
import hashlib
import hmac
import json
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from src.core.config import get_settings
from src.core.agent_routing import AgentLike, build_dispatch_plan_auto
from src.core.agent_service_client import list_agent_service_agents
from src.core.dispatch_fallback import run_plan_with_dependencies
from src.core.orchestrator_client import dispatch_execute, dispatch_plan
from src.core.db import get_db
from src.core.security import hash_password
from src.models.agent import Agent
from src.models.chat import ChatMessage, ChatSession
from src.models.openclaw_event import OpenclawEvent
from src.models.user import User
from src.models.base import utcnow
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Best-effort idempotency (in-memory, per-process).
# - Works for webhook retries in a short window.
# - Not durable across restarts, and not shared between multiple workers.
_DEDUP_TTL_SECONDS = 600
_DEDUP_MAX_SIZE = 2048
_dedup_lock = asyncio.Lock()
_seen: "OrderedDict[str, int]" = OrderedDict()


def _now_ts() -> int:
    return int(time.time())


async def _dedup_check_and_mark(key: str) -> bool:
    """
    Return True if key was already seen (duplicate); otherwise mark and return False.
    """
    if not key:
        return False
    now = _now_ts()
    async with _dedup_lock:
        # purge expired
        expired_before = now - _DEDUP_TTL_SECONDS
        while _seen:
            (_, ts) = next(iter(_seen.items()))
            if ts >= expired_before:
                break
            _seen.popitem(last=False)

        if key in _seen:
            _seen.move_to_end(key, last=True)
            return True

        _seen[key] = now
        _seen.move_to_end(key, last=True)
        while len(_seen) > _DEDUP_MAX_SIZE:
            _seen.popitem(last=False)
        return False


def _extract_text(payload: Any) -> str:
    if isinstance(payload, dict):
        for k in ("input", "text", "content", "message"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # common nested shapes
        msg = payload.get("msg") or payload.get("data") or payload.get("event")
        if isinstance(msg, dict):
            for k in ("text", "content", "message"):
                v = msg.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    # fallback: stable-ish json
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(payload)


def _select_dependencies(results: List[Dict[str, str]], depends_on: Any) -> List[Dict[str, str]]:
    if not isinstance(depends_on, list) or not depends_on:
        return list(results)
    dep_set = {str(d).strip() for d in depends_on if isinstance(d, str) and d.strip()}
    if not dep_set:
        return list(results)
    return [r for r in results if str(r.get("agent") or "").strip() in dep_set]


def _build_synthesis_input(user_input: str, results: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for r in results:
        name = str(r.get("agent_name") or r.get("agent") or "").strip()
        out = str(r.get("output") or "").strip()
        if not name or not out:
            continue
        if len(out) > 2000:
            out = out[:1999] + "…"
        lines.append(f"- {name}: {out}")
    joined = "\n".join(lines) if lines else "(无有效输出)"
    return (
        "你是多智能体协作的汇总助手。请基于用户原始问题与各智能体输出，给出最终答复。\n"
        "要求：1) 只输出最终结论/行动项，不逐条复述每个智能体；2) 如有冲突，指出并给出最合理方案；"
        "3) 保持与用户一致的语言与语气。\n\n"
        f"【用户原始问题】\n{user_input}\n\n"
        f"【各智能体输出】\n{joined}"
    )


def _parse_signature_headers(request: Request) -> Tuple[Optional[int], Optional[str]]:
    ts_raw = request.headers.get("x-openclaw-timestamp")
    sig = request.headers.get("x-openclaw-signature")
    if not ts_raw or not sig:
        return None, None
    try:
        ts = int(ts_raw)
    except Exception:
        return None, None
    sig = sig.strip()
    if not sig:
        return None, None
    return ts, sig


def _verify_signature(*, secret: str, ts: int, body: bytes, signature_hex: str) -> bool:
    msg = str(ts).encode("utf-8") + b"." + (body or b"")
    expected = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_hex)


def _safe_str(v: Any, max_len: int = 500) -> str:
    s = v if isinstance(v, str) else str(v)
    s = s.strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _payload_summary(payload: Any) -> Dict[str, Any]:
    """
    Build a small summary of payload to avoid passing large bodies downstream.
    """
    if isinstance(payload, dict):
        keys = list(payload.keys())
        summary: Dict[str, Any] = {"type": "dict", "keys": keys[:50]}
        # capture a few common lightweight fields (if present)
        for k in ("event_id", "id", "message_id", "agent", "agent_code", "agent_name", "text", "content", "msgtype"):
            if k in payload and payload.get(k) is not None:
                summary[k] = _safe_str(payload.get(k), 300)
        return summary
    if isinstance(payload, list):
        return {"type": "list", "len": len(payload)}
    if isinstance(payload, str):
        return {"type": "str", "preview": _safe_str(payload, 500), "len": len(payload)}
    return {"type": type(payload).__name__}


async def _call_agent_service(*, agent: str, user_input: str, context: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    settings = get_settings()
    base = settings.FUN_AGENT_SERVICE_URL.rstrip("/")
    url = f"{base}/agents/{agent}/execute"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            json={"input": user_input, "context": context},
            headers={"x-trace-id": trace_id},
        )
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    if resp.status_code >= 500:
        raise HTTPException(status_code=502, detail="Agent service unavailable")
    if resp.status_code >= 400:
        raise HTTPException(status_code=400, detail={"agent_service_error": data})
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Invalid agent service response")
    return data


def _should_skip_dispatch_item(item: Dict[str, Any], _index: int, total: int) -> bool:
    text = str(item.get("text") or "")
    return (not text.strip()) and total > 1


async def _execute_plan_fallback(
    *,
    plan: List[Dict[str, Any]],
    user_input: str,
    context: Dict[str, Any],
    trace_id: str,
    forced_agent: str,
    max_parallel: int,
) -> List[Dict[str, Any]]:
    dispatch_mode = "forced" if forced_agent else ("mentions" if "@" in user_input else "auto")

    async def _run_item(
        item: Dict[str, Any],
        index: int,
        total: int,
        prior_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        agent_code = str(item.get("agent") or "").strip()
        agent_text = str(item.get("text") or "")
        if not agent_text.strip() and total == 1:
            agent_text = user_input
        depends_on = item.get("depends_on")
        dependencies = _select_dependencies(prior_results, depends_on)
        step_ctx = {
            **context,
            "dispatch": {
                "mode": dispatch_mode,
                "index": index,
                "total": total,
                "agent": agent_code,
                "agent_name": item.get("agent_name") or agent_code,
                "original_input": user_input,
                "depends_on": depends_on if isinstance(depends_on, list) else [],
                "dependencies": dependencies,
                "previous": [
                    {"agent": r.get("agent"), "output": r.get("output", "")} for r in prior_results
                ],
            },
        }
        r = await _call_agent_service(agent=agent_code, user_input=agent_text, context=step_ctx, trace_id=trace_id)
        out = r.get("output") if isinstance(r, dict) else None
        return {
            "agent": agent_code,
            "agent_name": item.get("agent_name") or agent_code,
            "output": str(out or ""),
            "raw": r,
        }

    return await run_plan_with_dependencies(
        plan=plan,
        max_parallel=max_parallel,
        run_item=_run_item,
        should_skip=_should_skip_dispatch_item,
    )


@router.post("/openclaw")
async def openclaw_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Receive openclaw-forwarded messages.

    Auth headers (HMAC-SHA256):
    - x-openclaw-timestamp: unix seconds
    - x-openclaw-signature: hex(hmac_sha256(secret, f"{ts}.{raw_body}"))
    """
    settings = get_settings()
    if not settings.OPENCLAW_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="OPENCLAW_WEBHOOK_SECRET not configured")

    ts, sig = _parse_signature_headers(request)
    if ts is None or sig is None:
        raise HTTPException(status_code=401, detail="Missing or invalid signature headers")

    now = _now_ts()
    if abs(now - ts) > int(settings.OPENCLAW_MAX_SKEW_SECONDS or 0):
        raise HTTPException(status_code=401, detail="Signature timestamp expired")

    raw_body = await request.body()
    if not _verify_signature(secret=settings.OPENCLAW_WEBHOOK_SECRET, ts=ts, body=raw_body, signature_hex=sig):
        raise HTTPException(status_code=401, detail="Bad signature")

    trace_id = request.headers.get("x-trace-id")
    if not trace_id:
        trace_id = str(uuid.uuid4())

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    # Best-effort dedup
    event_id = None
    if isinstance(payload, dict):
        event_id = payload.get("event_id") or payload.get("id") or payload.get("message_id")
    event_id = event_id or request.headers.get("x-openclaw-event-id")
    if isinstance(event_id, str) and event_id.strip():
        event_id = event_id.strip()
    else:
        # fallback: signature-derived id
        event_id = hashlib.sha256((str(ts) + ":" + sig).encode("utf-8")).hexdigest()

    if await _dedup_check_and_mark(str(event_id)):
        return {"ok": True, "deduped": True, "trace_id": trace_id, "event_id": event_id}

    forced_agent = None
    if isinstance(payload, dict):
        forced_agent = payload.get("agent") or payload.get("agent_code") or payload.get("agent_name")
    forced_agent = (forced_agent or "").strip()

    # ---- DB idempotency (works across restarts and multiple workers) ----
    try:
        db.add(OpenclawEvent(event_id=str(event_id), trace_id=trace_id))
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.query(OpenclawEvent).filter(OpenclawEvent.event_id == str(event_id)).first()
        return {
            "ok": True,
            "deduped": True,
            "trace_id": trace_id,
            "event_id": event_id,
            "session_id": existing.session_id if existing else None,
        }

    user_input = _extract_text(payload)
    # keep DB content reasonable
    if len(user_input) > 8000:
        user_input = user_input[:7999] + "…"

    # ---- Decide dispatch plan ----
    if forced_agent:
        plan = [
            {"agent": forced_agent, "agent_name": forced_agent, "text": user_input, "depends_on": []}
        ]
    else:
        plan = []

        # Prefer orchestrator service (lives in fun-agent-service for now).
        items0 = await dispatch_plan(
            text=user_input,
            default_agent=(settings.OPENCLAW_DEFAULT_AGENT or "general"),
            mode=getattr(settings, "ROUTER_MODE", "hybrid"),
            trace_id=trace_id,
        )
        if isinstance(items0, list):
            for it in items0:
                if not isinstance(it, dict):
                    continue
                agent = str(it.get("agent") or "").strip()
                if not agent:
                    continue
                plan.append(
                    {
                        "agent": agent,
                        "agent_name": str(it.get("agent_name") or agent),
                        "text": str(it.get("text") or ""),
                        "depends_on": it.get("depends_on") if isinstance(it.get("depends_on"), list) else [],
                    }
                )

        if not plan:
            # Fallback: local router (kept for resiliency / local dev).
            # Route among known agents.
            # - Prefer DB agents (for display/metadata)
            # - Also merge live fun-agent-service agents so new agents work even if DB hasn't been synced yet.
            rows = db.query(Agent).all()
            agent_likes = [
                AgentLike(
                    code=a.code,
                    name=a.name,
                    handle=a.handle or f"@{a.code}",
                    description=a.description or "",
                )
                for a in rows
                if a.code
            ]
            existing_codes = {a.code for a in agent_likes if a.code}
            service_agents = await list_agent_service_agents(trace_id=trace_id)
            for item in service_agents:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("code") or item.get("name") or "").strip()
                if not code or code in existing_codes:
                    continue
                name = str(item.get("display_name") or item.get("name") or code)
                handle = str(item.get("handle") or f"@{code}")
                desc = str(item.get("description") or "")
                agent_likes.append(AgentLike(code=code, name=name, handle=handle, description=desc))
                existing_codes.add(code)
            items = await build_dispatch_plan_auto(
                text=user_input,
                agents=agent_likes,
                default_agent_code=(settings.OPENCLAW_DEFAULT_AGENT or "general"),
                trace_id=trace_id,
                mode=getattr(settings, "ROUTER_MODE", "hybrid"),
            )
            plan = [
                {
                    "agent": it.agent_code,
                    "agent_name": it.agent_name,
                    "text": it.text,
                    "depends_on": it.depends_on,
                }
                for it in items
            ]

    if not plan:
        raise HTTPException(status_code=400, detail="No dispatch plan")

    primary_agent = plan[0]["agent"]

    extra_context: Dict[str, Any] = {}
    if isinstance(payload, dict) and isinstance(payload.get("context"), dict):
        # user-provided context from openclaw; keep it under a dedicated key
        extra_context = dict(payload.get("context") or {})
    context: Dict[str, Any] = {
        "trace_id": trace_id,
        "source": "openclaw",
        **({"channel_context": extra_context} if extra_context else {}),
        "openclaw": {
            "event_id": event_id,
            "timestamp": ts,
            "content_type": request.headers.get("content-type"),
            "body_bytes": len(raw_body or b""),
            "body_sha256": hashlib.sha256(raw_body or b"").hexdigest(),
            "payload_summary": _payload_summary(payload),
        },
    }

    # ---- Persist chat (testing: map all OpenClaw messages to one system user) ----
    system_email = (settings.OPENCLAW_SYSTEM_USER_EMAIL or "").strip()
    system_password = settings.OPENCLAW_SYSTEM_USER_PASSWORD or ""
    if not system_email:
        raise HTTPException(status_code=500, detail="OPENCLAW_SYSTEM_USER_EMAIL not configured")

    user = db.query(User).filter(User.email == system_email).first()
    if not user:
        user = User(email=system_email, hashed_password=hash_password(system_password))
        db.add(user)
        db.commit()
        db.refresh(user)

    agent_row = db.query(Agent).filter(Agent.code == primary_agent).first()
    if not agent_row:
        # Create a minimal placeholder agent record to satisfy FK.
        agent_row = Agent(
            code=primary_agent,
            name=primary_agent,
            handle=f"@{primary_agent}",
            description=f"Auto-created for OpenClaw: {primary_agent}",
            tags=["openclaw"],
            capabilities=[],
        )
        db.add(agent_row)
        db.commit()
        db.refresh(agent_row)

    # Reuse a stable session per (channel, from/chat_id, agent).
    channel = str(extra_context.get("channel") or "openclaw")
    sender = str(extra_context.get("from") or extra_context.get("userid") or extra_context.get("sender") or "unknown")
    chat_id = str(extra_context.get("chat_id") or extra_context.get("chatId") or "")
    title = f"{channel}:{sender}" + (f":{chat_id}" if chat_id and chat_id != "None" else "")
    if len(title) > 255:
        title = title[:255]

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

    # update event -> session mapping (best-effort)
    try:
        ev = db.query(OpenclawEvent).filter(OpenclawEvent.event_id == str(event_id)).first()
        if ev:
            if not ev.session_id:
                ev.session_id = session.id
            if not ev.agent_code:
                ev.agent_code = primary_agent
            db.commit()
    except Exception:
        db.rollback()

    # Insert user message
    session.updated_at = utcnow()
    db.add(ChatMessage(session_id=session.id, role="user", content=user_input))
    db.commit()

    results: list = []
    output_text: str = ""

    # Prefer orchestrator execution (lives in fun-agent-service for now).
    orch = await dispatch_execute(
        text=user_input,
        context=context,
        default_agent=(settings.OPENCLAW_DEFAULT_AGENT or "general"),
        mode=getattr(settings, "ROUTER_MODE", "hybrid"),
        trace_id=trace_id,
        items=plan,
        forced_agent=forced_agent,
    )
    if isinstance(orch, dict) and isinstance(orch.get("results"), list) and isinstance(orch.get("output"), str):
        results = orch.get("results") or []
        output_text = orch.get("output") or ""
    else:
        # Fallback: execute agents with dependency-aware parallelism (API side).
        results = await _execute_plan_fallback(
            plan=plan,
            user_input=user_input,
            context=context,
            trace_id=trace_id,
            forced_agent=forced_agent,
            max_parallel=getattr(settings, "DISPATCH_MAX_PARALLEL", 1),
        )

        if len(results) == 1:
            output_text = results[0]["output"]
        else:
            output_text = "\n\n".join([f"【{r['agent_name']}】{r['output']}" for r in results])
            try:
                synth_input = _build_synthesis_input(user_input, results)
                synth_ctx = {
                    **context,
                    "dispatch": {
                        "mode": "synthesizer",
                        "index": len(results),
                        "total": len(results) + 1,
                        "agent": "synthesizer",
                        "agent_name": "汇总助手",
                        "original_input": user_input,
                        "depends_on": [r["agent"] for r in results if r.get("agent")],
                        "dependencies": results,
                        "previous": results,
                    },
                }
                synth_out = await _call_agent_service(
                    agent="synthesizer",
                    user_input=synth_input,
                    context=synth_ctx,
                    trace_id=trace_id,
                )
                if isinstance(synth_out, dict):
                    synth_text = str(synth_out.get("output") or "").strip()
                    if synth_text:
                        output_text = synth_text
            except Exception:
                pass

    if len(output_text) > 12000:
        output_text = output_text[:11999] + "…"

    # Insert assistant message
    session.updated_at = utcnow()
    db.add(ChatMessage(session_id=session.id, role="assistant", content=output_text))
    db.commit()

    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "trace_id": trace_id,
            "event_id": event_id,
            "agent": primary_agent,
            "agents": [r["agent"] for r in results],
            "output": output_text,
            "session_id": session.id,
            "user_id": user.id,
            "agent_db_id": agent_row.id,
            "agent_service": results,
        },
    )

