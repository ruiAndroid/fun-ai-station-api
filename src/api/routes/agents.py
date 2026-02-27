from typing import List

import httpx
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.core.db import get_db
from src.models.agent import Agent
from src.schemas.agent import AgentOut

router = APIRouter(prefix="/agents", tags=["agents"])


def _service_base() -> str:
    settings = get_settings()
    return settings.FUN_AGENT_SERVICE_URL.rstrip("/")


def _sync_agent_service_agents(db: Session) -> None:
    base = _service_base()
    if not base:
        return
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{base}/agents")
    except httpx.HTTPError:
        return
    if resp.status_code >= 400:
        return
    try:
        data = json.loads(resp.content.decode("utf-8"))
    except Exception:
        return
    if not isinstance(data, list):
        return

    codes = []
    for item in data:
        name = item.get("name") if isinstance(item, dict) else None
        if not name:
            continue
        code = str(item.get("code") or name)
        codes.append(code)
        desc = item.get("description", "") if isinstance(item, dict) else ""
        display_name = item.get("display_name") if isinstance(item, dict) else None
        handle = item.get("handle") if isinstance(item, dict) else None
        normalized_name = str(display_name or name)
        normalized_handle = str(handle or f"@{normalized_name}")
        agent = db.query(Agent).filter(Agent.code == code).first()
        if agent:
            agent.name = normalized_name
            agent.handle = normalized_handle
            agent.description = desc
            agent.tags = ["agent-service"]
            agent.capabilities = []
        else:
            db.add(
                Agent(
                    code=code,
                    name=normalized_name,
                    handle=normalized_handle,
                    description=desc,
                    tags=["agent-service"],
                    capabilities=[],
                )
            )

    # Prune stale agent-service entries
    if codes:
        db.query(Agent).filter(
            Agent.tags.isnot(None),
            Agent.tags.contains(["agent-service"]),
            ~Agent.code.in_(codes),
        ).delete(synchronize_session=False)

    db.commit()


@router.get("", response_model=List[AgentOut])
def list_agents(db: Session = Depends(get_db)):
    _sync_agent_service_agents(db)
    agents = db.query(Agent).order_by(Agent.code.asc()).all()
    return [
        AgentOut(
            id=a.id,
            code=a.code,
            name=a.name,
            handle=a.handle,
            description=a.description,
            tags=a.tags or [],
            capabilities=a.capabilities or [],
        )
        for a in agents
    ]


@router.get("/{agent_id}", response_model=AgentOut)
def get_agent(
    agent_id: int, db: Session = Depends(get_db)
):
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentOut(
        id=agent.id,
        code=agent.code,
        name=agent.name,
        handle=agent.handle,
        description=agent.description,
        tags=agent.tags or [],
        capabilities=agent.capabilities or [],
    )


