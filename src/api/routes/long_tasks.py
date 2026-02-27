from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.api.deps import get_current_user
from src.core.config import get_settings
from src.core.db import get_db
from src.models.long_task import LongTask
from src.models.user import User
from src.schemas.long_task import LongTaskOrchestratorExecuteCreate, LongTaskOut


router = APIRouter(prefix="/long-tasks", tags=["long-tasks"])


def _as_utc_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        # DB stores naive UTC; tag it so clients can interpret correctly
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _task_out(t: LongTask) -> LongTaskOut:
    return LongTaskOut(
        id=t.id,
        user_id=t.user_id,
        kind=t.kind,
        title=t.title,
        status=t.status,
        trace_id=t.trace_id,
        payload=t.payload or {},
        output=t.output,
        result=t.result or {},
        error=t.error,
        cancel_requested=bool(t.cancel_requested),
        attempt=int(t.attempt or 0),
        started_at=_as_utc_aware(t.started_at),
        finished_at=_as_utc_aware(t.finished_at),
        created_at=_as_utc_aware(t.created_at) or datetime.now(timezone.utc),
        updated_at=_as_utc_aware(t.updated_at) or datetime.now(timezone.utc),
    )


@router.get("", response_model=List[LongTaskOut])
def list_long_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tasks = db.execute(
        select(LongTask)
        .where(LongTask.user_id == current_user.id)
        .order_by(desc(LongTask.updated_at))
        .limit(50)
    ).scalars().all()
    return [_task_out(t) for t in tasks]


@router.get("/{task_id}", response_model=LongTaskOut)
def get_long_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    t = db.get(LongTask, task_id)
    if not t or t.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_out(t)


@router.post("/orchestrator-execute", response_model=LongTaskOut)
def create_orchestrator_execute_task(
    payload: LongTaskOrchestratorExecuteCreate,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    settings = get_settings()

    default_agent = (
        (payload.default_agent or "").strip()
        or (settings.OPENAI_DEFAULT_AGENT or "").strip()
        or (settings.OPENCLAW_DEFAULT_AGENT or "").strip()
        or "attendance"
    )
    mode = (payload.mode or "").strip().lower() or (settings.ROUTER_MODE or "hybrid").strip().lower()

    trace_id = (request.headers.get("x-trace-id") or "").strip() or uuid.uuid4().hex

    title = (payload.title or "").strip()
    if not title:
        title = text.strip().replace("\n", " ")
        if len(title) > 80:
            title = title[:79] + "…"
        title = title or "Long Task"

    t = LongTask(
        user_id=current_user.id,
        kind="orchestrator_execute",
        title=title,
        status="pending",
        trace_id=trace_id,
        payload={
            "text": text,
            "context": payload.context or {},
            "default_agent": default_agent,
            "mode": mode,
        },
        cancel_requested=False,
        attempt=0,
        started_at=None,
        finished_at=None,
        locked_by=None,
        locked_until=None,
        output=None,
        result={},
        error=None,
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return _task_out(t)


@router.post("/{task_id}/cancel", response_model=LongTaskOut)
def cancel_long_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    t = db.get(LongTask, task_id)
    if not t or t.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")

    if (t.status or "").lower() in ("success", "failed", "canceled"):
        return _task_out(t)

    t.cancel_requested = True
    db.commit()
    db.refresh(t)
    return _task_out(t)


@router.delete("/{task_id}")
def delete_long_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    t = db.get(LongTask, task_id)
    if not t or t.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")

    if (t.status or "").strip().lower() == "running":
        raise HTTPException(status_code=409, detail="Task is running; cancel first")

    db.delete(t)
    db.commit()
    return {"ok": True}
