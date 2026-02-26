from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ScheduledTaskCreate(BaseModel):
    name: str
    enabled: bool = True
    schedule_type: str = "cron"  # cron | interval | once
    schedule_expr: str = ""
    timezone: str = "UTC"
    payload: Dict[str, Any] = Field(default_factory=dict)
    next_run_at: Optional[datetime] = None


class ScheduledTaskUpdate(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    schedule_type: Optional[str] = None
    schedule_expr: Optional[str] = None
    timezone: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    next_run_at: Optional[datetime] = None


class ScheduledTaskOut(BaseModel):
    id: int
    user_id: str
    name: str
    enabled: bool
    schedule_type: str
    schedule_expr: str
    timezone: str
    payload: Dict[str, Any]
    next_run_at: Optional[datetime]
    last_run_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class ScheduledTaskRunOut(BaseModel):
    id: int
    task_id: int
    user_id: str
    status: str
    trace_id: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime]
    error: Optional[str] = None
    result: Dict[str, Any] = Field(default_factory=dict)

