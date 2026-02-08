from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ChatSessionCreate(BaseModel):
    agent_id: Optional[int] = None
    title: Optional[str] = None


class ChatSessionOut(BaseModel):
    id: str
    user_id: str
    agent_id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageCreate(BaseModel):
    role: str  # user/assistant/system
    content: str


class ChatMessageOut(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime
    updated_at: datetime

