from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class LongTaskOrchestratorExecuteCreate(BaseModel):
    title: Optional[str] = None
    text: str
    context: Dict[str, Any] = Field(default_factory=dict)
    default_agent: Optional[str] = None
    mode: Optional[str] = None


class LongTaskOut(BaseModel):
    id: int
    user_id: str
    kind: str
    title: str
    status: str
    trace_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[str] = None
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    cancel_requested: bool
    attempt: int
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

