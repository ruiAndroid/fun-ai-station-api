from __future__ import annotations

from typing import List, Set

from sqlalchemy.orm import Session

from src.core.agent_routing import AgentLike
from src.core.agent_service_client import list_agent_service_agents
from src.models.agent import Agent


async def list_agent_likes(*, db: Session, trace_id: str = "") -> List[AgentLike]:
    """
    Build the agent list used for routing/orchestration.

    Priority:
    - Local DB agents (so UI & metadata stay consistent)
    - Merge in any agents exposed by fun-agent-service (/agents) not yet in DB
      (so routing does not depend on DB sync being perfect).

    This function lives in fun-ai-station-api, but is intentionally framework-free
    and can be moved into a standalone orchestrator service later.
    """
    rows = db.query(Agent).all()
    agent_likes: List[AgentLike] = []
    existing: Set[str] = set()

    for a in rows:
        code = (a.code or "").strip()
        if not code:
            continue
        agent_likes.append(
            AgentLike(
                code=code,
                name=a.name,
                handle=a.handle or f"@{code}",
                description=a.description or "",
            )
        )
        existing.add(code)

    service_agents = await list_agent_service_agents(trace_id=trace_id)
    for item in service_agents:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or item.get("name") or "").strip()
        if not code or code in existing:
            continue
        name = str(item.get("display_name") or item.get("name") or code)
        handle = str(item.get("handle") or f"@{code}")
        desc = str(item.get("description") or "")
        agent_likes.append(AgentLike(code=code, name=name, handle=handle, description=desc))
        existing.add(code)

    return agent_likes

