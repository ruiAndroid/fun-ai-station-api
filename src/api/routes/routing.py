from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.core.agent_routing import AgentLike, build_dispatch_plan_auto
from src.core.config import get_settings
from src.core.db import get_db
from src.core.orchestrator_client import dispatch_plan_full
from src.models.agent import Agent


router = APIRouter(prefix="/routing", tags=["routing"])


@router.post("/plan")
async def route_plan(request: Request, db: Session = Depends(get_db)):
    """
    Return an ordered dispatch plan for a user input.

    - Uses the same routing logic as enterprise WeCom (OpenClaw) entrypoints.
    - Controlled by settings.ROUTER_MODE (hybrid/llm/keywords).
    """
    settings = get_settings()
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        payload = {}

    text = payload.get("text") or payload.get("input") or payload.get("message") or ""
    text = text if isinstance(text, str) else str(text)
    text = text.strip()

    trace_id = request.headers.get("x-trace-id") or ""

    req_default_agent = payload.get("default_agent") or payload.get("default_agent_code") or ""
    req_default_agent = req_default_agent if isinstance(req_default_agent, str) else str(req_default_agent or "")
    req_default_agent = req_default_agent.strip()
    default_agent = req_default_agent or (settings.OPENCLAW_DEFAULT_AGENT or settings.OPENAI_DEFAULT_AGENT or "attendance")

    # Prefer orchestrator service (lives in fun-agent-service for now).
    plan0 = await dispatch_plan_full(
        text=text,
        default_agent=default_agent,
        mode=getattr(settings, "ROUTER_MODE", "hybrid"),
        trace_id=trace_id,
    )
    items0 = plan0.get("items") if isinstance(plan0, dict) else None
    if isinstance(items0, list) and items0:
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "mode": getattr(settings, "ROUTER_MODE", "hybrid"),
                "strategy": str(plan0.get("strategy") or ""),
                "debug": plan0.get("debug") if isinstance(plan0.get("debug"), dict) else None,
                "agents": [str(it.get("agent") or "").strip() for it in items0 if isinstance(it, dict)],
                "items": [
                    {
                        "agent": str(it.get("agent") or "").strip(),
                        "agent_name": str(it.get("agent_name") or ""),
                        "text": str(it.get("text") or ""),
                        "reason": str(it.get("reason") or ""),
                        "depends_on": it.get("depends_on") if isinstance(it.get("depends_on"), list) else [],
                    }
                    for it in items0
                    if isinstance(it, dict) and str(it.get("agent") or "").strip()
                ],
            },
        )

    rows: List[Agent] = db.query(Agent).all()
    agents = [
        AgentLike(
            code=a.code,
            name=a.name,
            handle=a.handle or f"@{a.code}",
            description=a.description or "",
        )
        for a in rows
        if a.code
    ]

    items = await build_dispatch_plan_auto(
        text=text,
        agents=agents,
        default_agent_code=default_agent,
        trace_id=trace_id,
        mode=getattr(settings, "ROUTER_MODE", "hybrid"),
    )

    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "mode": getattr(settings, "ROUTER_MODE", "hybrid"),
            "strategy": "fallback_local",
            # Backward compatible: "agents" list remains.
            "agents": [it.agent_code for it in items],
            # New: per-agent subtask text.
            "items": [
                {
                    "agent": it.agent_code,
                    "text": it.text,
                    "reason": "",
                    "depends_on": it.depends_on,
                }
                for it in items
            ],
        },
    )

