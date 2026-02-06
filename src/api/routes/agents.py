from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.deps import get_current_user
from src.core.db import get_db
from src.models.agent import Agent
from src.models.user import User
from src.schemas.agent import AgentOut

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("", response_model=List[AgentOut])
def list_agents(
    db: Session = Depends(get_db), _: User = Depends(get_current_user)
):
    agents = db.query(Agent).all()
    return [
        AgentOut(
            id=a.id,
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
    agent_id: str, db: Session = Depends(get_db), _: User = Depends(get_current_user)
):
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentOut(
        id=agent.id,
        name=agent.name,
        handle=agent.handle,
        description=agent.description,
        tags=agent.tags or [],
        capabilities=agent.capabilities or [],
    )


