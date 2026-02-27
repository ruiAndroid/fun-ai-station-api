from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from croniter import croniter
from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo

from src.api.deps import get_current_user
from src.core.config import get_settings
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
    # disallow per-task routing overrides (use global settings)
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


def _as_utc_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        # DB stores naive UTC; tag it so clients can interpret correctly
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_interval_seconds(expr: str) -> int:
    try:
        return int(str(expr or "").strip())
    except Exception:
        raise HTTPException(status_code=400, detail="interval schedule_expr must be integer seconds")


def _count_enabled_tasks(db: Session, *, user_id: str, exclude_task_id: int | None = None) -> int:
    q = (
        select(func.count())
        .select_from(ScheduledTask)
        .where(ScheduledTask.user_id == user_id, ScheduledTask.enabled.is_(True))
    )
    if exclude_task_id is not None:
        q = q.where(ScheduledTask.id != exclude_task_id)
    try:
        return int(db.execute(q).scalar_one() or 0)
    except Exception:
        return 0


def _compute_next_run(schedule_type: str, schedule_expr: str, tz_name: str, *, after_utc: datetime):
    st = (schedule_type or "").strip().lower()
    expr = (schedule_expr or "").strip()
    tz_name = (tz_name or "UTC").strip() or "UTC"

    if st == "once":
        return None

    if st == "interval":
        seconds = _parse_interval_seconds(expr)
        if seconds <= 0:
            seconds = 60
        min_seconds = max(1, int(get_settings().SCHEDULED_TASK_INTERVAL_MIN_SECONDS or 10))
        if seconds < min_seconds:
            seconds = min_seconds
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
        next_run_at=_as_utc_aware(t.next_run_at),
        last_run_at=_as_utc_aware(t.last_run_at),
        created_at=_as_utc_aware(t.created_at),
        updated_at=_as_utc_aware(t.updated_at),
    )


def _run_out(r: ScheduledTaskRun) -> ScheduledTaskRunOut:
    return ScheduledTaskRunOut(
        id=r.id,
        task_id=r.task_id,
        user_id=r.user_id,
        status=r.status,
        trace_id=r.trace_id,
        started_at=_as_utc_aware(r.started_at),
        finished_at=_as_utc_aware(r.finished_at),
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
    settings = get_settings()
    now = datetime.utcnow()
    schedule_type = (payload.schedule_type or "cron").strip()
    schedule_expr = (payload.schedule_expr or "").strip()
    tz_name = (payload.timezone or "UTC").strip() or "UTC"

    if schedule_type.strip().lower() == "interval":
        seconds = _parse_interval_seconds(schedule_expr)
        if seconds <= 0:
            seconds = 60
        if seconds < settings.SCHEDULED_TASK_INTERVAL_MIN_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"interval must be >= {settings.SCHEDULED_TASK_INTERVAL_MIN_SECONDS}s",
            )
        schedule_expr = str(seconds)

    max_enabled = int(settings.SCHEDULED_TASKS_MAX_ENABLED_PER_USER or 0)
    if payload.enabled and max_enabled > 0:
        enabled_count = _count_enabled_tasks(db, user_id=current_user.id)
        if enabled_count >= max_enabled:
            raise HTTPException(
                status_code=409,
                detail=f"too many enabled scheduled tasks (max {max_enabled})",
            )

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
    settings = get_settings()
    task = db.get(ScheduledTask, task_id)
    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")

    if payload.name is not None:
        task.name = payload.name.strip() or task.name
    if payload.enabled is not None:
        new_enabled = bool(payload.enabled)
        if new_enabled and not bool(task.enabled):
            max_enabled = int(settings.SCHEDULED_TASKS_MAX_ENABLED_PER_USER or 0)
            if max_enabled > 0:
                enabled_count = _count_enabled_tasks(db, user_id=current_user.id, exclude_task_id=int(task.id))
                if enabled_count >= max_enabled:
                    raise HTTPException(
                        status_code=409,
                        detail=f"too many enabled scheduled tasks (max {max_enabled})",
                    )
        task.enabled = new_enabled
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

    if (task.schedule_type or "").strip().lower() == "interval":
        seconds = _parse_interval_seconds(task.schedule_expr)
        if seconds <= 0:
            seconds = 60
        if seconds < settings.SCHEDULED_TASK_INTERVAL_MIN_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"interval must be >= {settings.SCHEDULED_TASK_INTERVAL_MIN_SECONDS}s",
            )
        task.schedule_expr = str(seconds)

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
