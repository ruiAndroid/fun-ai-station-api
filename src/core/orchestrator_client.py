from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from src.core.config import get_settings


async def dispatch_plan(
    *,
    text: str,
    default_agent: str,
    mode: str,
    trace_id: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Call orchestrator to build a sequential dispatch plan:
      items: [{agent, agent_name?, text, reason?}]
    """
    data = await dispatch_plan_full(
        text=text,
        default_agent=default_agent,
        mode=mode,
        trace_id=trace_id,
        context=context,
    )
    items = data.get("items") if isinstance(data, dict) else None
    return items if isinstance(items, list) else []


async def dispatch_plan_full(
    *,
    text: str,
    default_agent: str,
    mode: str,
    trace_id: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call orchestrator to build a sequential dispatch plan (full response).
    Response is expected to contain:
      {ok, mode, default_agent, strategy?, debug?, items:[{agent, agent_name?, text, reason?}]}
    """
    settings = get_settings()
    base = (settings.ORCHESTRATOR_URL or settings.FUN_AGENT_SERVICE_URL or "").rstrip("/")
    if not base:
        return {}

    headers: Optional[Dict[str, str]] = {"x-trace-id": trace_id} if trace_id else None
    payload: Dict[str, Any] = {
        "text": text,
        "default_agent": default_agent,
        "mode": mode,
        "context": context or {},
    }
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(f"{base}/dispatch/plan", json=payload, headers=headers)
    except Exception:
        return {}

    if resp.status_code >= 400:
        return {}

    try:
        data = resp.json()
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


async def dispatch_execute(
    *,
    text: str,
    context: Dict[str, Any],
    default_agent: str,
    mode: str,
    trace_id: str = "",
    forced_agent: str = "",
    items: Optional[List[Dict[str, Any]]] = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """
    Call orchestrator to execute (plan + run agents sequentially):
      {items, results, output}
    """
    settings = get_settings()
    base = (settings.ORCHESTRATOR_URL or settings.FUN_AGENT_SERVICE_URL or "").rstrip("/")
    if not base:
        return {}

    headers: Optional[Dict[str, str]] = {"x-trace-id": trace_id} if trace_id else None
    payload: Dict[str, Any] = {
        "text": text,
        "context": context or {},
        "default_agent": default_agent,
        "mode": mode,
    }
    if forced_agent:
        payload["agent"] = forced_agent
    if items is not None:
        payload["items"] = items

    try:
        async with httpx.AsyncClient(timeout=max(1, int(timeout_seconds or 60))) as client:
            resp = await client.post(f"{base}/dispatch/execute", json=payload, headers=headers)
    except Exception:
        return {}

    if resp.status_code >= 400:
        return {}

    try:
        data = resp.json()
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}
