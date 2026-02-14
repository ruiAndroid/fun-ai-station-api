from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.core.agent_routing import AgentLike, build_dispatch_plan_auto
from src.core.config import get_settings
from src.core.db import get_db
from src.models.agent import Agent


router = APIRouter(prefix="/routing", tags=["routing"])


@router.post("/plan")
async def route_plan(request: Request, db: Session = Depends(get_db)):
    """
    Return an ordered list of agent codes for a user input.

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
        default_agent_code=(settings.OPENCLAW_DEFAULT_AGENT or settings.OPENAI_DEFAULT_AGENT or "attendance"),
        trace_id=trace_id,
        mode=getattr(settings, "ROUTER_MODE", "hybrid"),
    )

    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "mode": getattr(settings, "ROUTER_MODE", "hybrid"),
            "agents": [it.agent_code for it in items],
        },
    )

