from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from src.core.config import get_settings


async def list_agent_service_agents(*, trace_id: str = "") -> List[Dict[str, Any]]:
    """
    Fetch agents from fun-agent-service without relying on local DB sync.
    Shape returned by fun-agent-service:
      [{code,name,display_name,handle,description}, ...]
    """
    settings = get_settings()
    base = (settings.FUN_AGENT_SERVICE_URL or "").rstrip("/")
    if not base:
        return []

    headers: Optional[Dict[str, str]] = {"x-trace-id": trace_id} if trace_id else None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base}/agents", headers=headers)
    except Exception:
        return []

    if resp.status_code >= 400:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    return data if isinstance(data, list) else []

