from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from croniter import croniter
from sqlalchemy import delete, desc, select
from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo

from src.api.deps import get_current_user
from src.core.db import get_db
from src.models.scheduled_task import ScheduledTask, ScheduledTaskRun
from src.models.user import User
from src.schemas.scheduled_task import (
    ScheduledTaskCreate,
    ScheduledTaskOut,
    ScheduledTaskRunOut,
    ScheduledTaskUpdate,
)

router = APIRouter(prefix="/scheduled-tasks", tags=["scheduled-tasks"])

_PAYLOAD_DISALLOWED_KEYS = {
    # disallow forcing a specific agent / precomputed plan (keep routing consistent)
    "agent",
    "forced_agent",
    "items",
    # disallow per-task routing overrides for now (use global settings)
    "default_agent",
    "mode",
}


def _sanitize_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {}
    out = dict(payload)
    for k in _PAYLOAD_DISALLOWED_KEYS:
        out.pop(k, None)
    return out


def _to_utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _compute_next_run(schedule_type: str, schedule_expr: str, tz_name: str, *, after_utc: datetime):
    st = (schedule_type or "").strip().lower()
    expr = (schedule_expr or "").strip()
    tz_name = (tz_name or "UTC").strip() or "UTC"

    if st == "once":
        return None

    if st == "interval":
        try:
            seconds = int(expr)
        except Exception:
            seconds = 0
        if seconds <= 0:
            seconds = 60
        return after_utc + timedelta(seconds=seconds)

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")

    base_local_aware = after_utc.replace(tzinfo=timezone.utc).astimezone(tz)
    base_local_naive = base_local_aware.replace(tzinfo=None)
    it = croniter(expr or "* * * * *", base_local_naive)
    next_local_naive = it.get_next(datetime)
    next_local_aware = next_local_naive.replace(tzinfo=tz)
    return _to_utc_naive(next_local_aware)


def _task_out(t: ScheduledTask) -> ScheduledTaskOut:
    return ScheduledTaskOut(
        id=t.id,
        user_id=t.user_id,
        name=t.name,
        enabled=bool(t.enabled),
        schedule_type=t.schedule_type,
        schedule_expr=t.schedule_expr,
        timezone=t.timezone,
        payload=t.payload or {},
        next_run_at=t.next_run_at,
        last_run_at=t.last_run_at,
        created_at=t.created_at,
        updated_at=t.updated_at,
    )


def _run_out(r: ScheduledTaskRun) -> ScheduledTaskRunOut:
    return ScheduledTaskRunOut(
        id=r.id,
        task_id=r.task_id,
        user_id=r.user_id,
        status=r.status,
        trace_id=r.trace_id,
        started_at=r.started_at,
        finished_at=r.finished_at,
        error=r.error,
        result=r.result or {},
    )


@router.get("", response_model=List[ScheduledTaskOut])
def list_tasks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tasks = db.execute(
        select(ScheduledTask)
        .where(ScheduledTask.user_id == current_user.id)
        .order_by(desc(ScheduledTask.updated_at))
    ).scalars().all()
    return [_task_out(t) for t in tasks]


@router.post("", response_model=ScheduledTaskOut)
def create_task(
    payload: ScheduledTaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    now = datetime.utcnow()
    schedule_type = (payload.schedule_type or "cron").strip()
    schedule_expr = (payload.schedule_expr or "").strip()
    tz_name = (payload.timezone or "UTC").strip() or "UTC"
    next_run_at = payload.next_run_at
    if next_run_at is None and schedule_type.strip().lower() != "once":
        next_run_at = _compute_next_run(schedule_type, schedule_expr, tz_name, after_utc=now)

    task = ScheduledTask(
        user_id=current_user.id,
        name=payload.name.strip() or "Scheduled Task",
        enabled=payload.enabled,
        schedule_type=schedule_type,
        schedule_expr=schedule_expr,
        timezone=tz_name,
        payload=_sanitize_payload(payload.payload),
        next_run_at=next_run_at,
        last_run_at=None,
        locked_by=None,
        locked_until=None,
        created_at=now,
        updated_at=now,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return _task_out(task)


@router.put("/{task_id}", response_model=ScheduledTaskOut)
def update_task(
    task_id: int,
    payload: ScheduledTaskUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    task = db.get(ScheduledTask, task_id)
    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")

    if payload.name is not None:
        task.name = payload.name.strip() or task.name
    if payload.enabled is not None:
        task.enabled = bool(payload.enabled)
    if payload.schedule_type is not None:
        task.schedule_type = (payload.schedule_type or "").strip() or task.schedule_type
    if payload.schedule_expr is not None:
        task.schedule_expr = (payload.schedule_expr or "").strip()
    if payload.timezone is not None:
        task.timezone = (payload.timezone or "UTC").strip() or "UTC"
    if payload.payload is not None:
        task.payload = _sanitize_payload(payload.payload)
    if payload.next_run_at is not None:
        task.next_run_at = payload.next_run_at

    # clear lease on edits (avoid confusing stuck locks)
    task.locked_by = None
    task.locked_until = None

    db.commit()
    db.refresh(task)
    return _task_out(task)


@router.delete("/{task_id}")
def delete_task(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    task = db.get(ScheduledTask, task_id)
    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")
    # keep DB clean: remove run history together with the task
    db.execute(delete(ScheduledTaskRun).where(ScheduledTaskRun.task_id == task_id))
    db.delete(task)
    db.commit()
    return {"ok": True}


@router.get("/{task_id}/runs", response_model=List[ScheduledTaskRunOut])
def list_task_runs(
    task_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    task = db.get(ScheduledTask, task_id)
    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")

    runs = db.execute(
        select(ScheduledTaskRun)
        .where(ScheduledTaskRun.task_id == task_id)
        .order_by(desc(ScheduledTaskRun.started_at))
        .limit(50)
    ).scalars().all()
    return [_run_out(r) for r in runs]
