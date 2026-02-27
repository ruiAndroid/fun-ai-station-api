from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from croniter import croniter
from sqlalchemy import and_, asc, or_, select
from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo

from src.core.config import get_settings
from src.core.db import SessionLocal
from src.core.orchestrator_client import dispatch_execute
from src.models.scheduled_task import ScheduledTask, ScheduledTaskRun


logger = logging.getLogger("scheduler-worker")


def utcnow() -> datetime:
    return datetime.utcnow()


def _to_utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        # Assume already UTC naive
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def compute_next_run(task: ScheduledTask, *, after_utc: datetime) -> Optional[datetime]:
    st = (task.schedule_type or "").strip().lower()
    expr = (task.schedule_expr or "").strip()
    tz_name = (task.timezone or "UTC").strip() or "UTC"

    if st == "once":
        return None

    if st == "interval":
        try:
            seconds = int(expr)
        except Exception:
            seconds = 0
        if seconds <= 0:
            seconds = 60
        min_seconds = max(1, int(get_settings().SCHEDULED_TASK_INTERVAL_MIN_SECONDS or 10))
        if seconds < min_seconds:
            seconds = min_seconds
        return after_utc + timedelta(seconds=seconds)

    # default: cron
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")

    base_local_aware = after_utc.replace(tzinfo=timezone.utc).astimezone(tz)
    # croniter behaves best with naive datetimes (interpreted in the given tz)
    base_local_naive = base_local_aware.replace(tzinfo=None)
    it = croniter(expr or "* * * * *", base_local_naive)
    next_local_naive = it.get_next(datetime)
    next_local_aware = next_local_naive.replace(tzinfo=tz)
    return _to_utc_naive(next_local_aware)


def parse_payload(
    task: ScheduledTask,
) -> Tuple[str, Dict[str, Any], str, str, str, Optional[List[Dict[str, Any]]]]:
    """
    Normalize task.payload into orchestrator dispatch_execute parameters.
    """
    settings = get_settings()
    raw = task.payload or {}
    text = str(raw.get("text") or raw.get("input") or "")
    context = raw.get("context") if isinstance(raw.get("context"), dict) else {}

    # Do NOT allow per-task forcing/overrides; only use scheduler defaults.
    default_agent = str(
        (settings.SCHEDULER_DEFAULT_AGENT or "").strip()
        or (settings.OPENCLAW_DEFAULT_AGENT or "").strip()
        or "attendance"
    )
    mode = str(
        (settings.SCHEDULER_ROUTER_MODE or "").strip().lower()
        or (settings.ROUTER_MODE or "").strip().lower()
        or "hybrid"
    )
    forced_agent = ""
    items_list = None
    return text, context, default_agent, mode, forced_agent, items_list


@dataclass
class TickResult:
    claimed: int = 0
    executed: int = 0
    ok: int = 0
    failed: int = 0


def claim_due_tasks(
    db: Session,
    *,
    now: datetime,
    limit: int,
    lease_seconds: int,
    worker_id: str,
) -> List[int]:
    lease_until = now + timedelta(seconds=max(10, lease_seconds))
    q = (
        select(ScheduledTask)
        .where(
            ScheduledTask.enabled.is_(True),
            ScheduledTask.next_run_at.is_not(None),
            ScheduledTask.next_run_at <= now,
            or_(ScheduledTask.locked_until.is_(None), ScheduledTask.locked_until <= now),
        )
        .order_by(asc(ScheduledTask.next_run_at), asc(ScheduledTask.id))
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    tasks = db.execute(q).scalars().all()
    ids: List[int] = []
    for t in tasks:
        t.locked_by = worker_id
        t.locked_until = lease_until
        ids.append(int(t.id))
    if ids:
        db.commit()
    return ids


def _finish_task(
    db: Session,
    *,
    task: ScheduledTask,
    now: datetime,
    ok: bool,
    next_run_at: Optional[datetime],
    worker_id: str,
) -> None:
    # release lock only if we own it (avoid clobbering in weird edge cases)
    if task.locked_by == worker_id:
        task.locked_by = None
        task.locked_until = None
    task.last_run_at = now
    if task.schedule_type.strip().lower() == "once":
        if ok:
            task.enabled = False
            task.next_run_at = None
        else:
            task.next_run_at = next_run_at
    else:
        task.next_run_at = next_run_at
    db.commit()


def run_one_task(task_id: int, *, worker_id: str) -> bool:
    now = utcnow()
    with SessionLocal() as db:
        task = db.get(ScheduledTask, task_id)
        if not task:
            return True
        if not task.enabled:
            return True
        if task.locked_by != worker_id or not task.locked_until or task.locked_until < now:
            # lease lost
            return True

        trace_id = uuid.uuid4().hex
        run = ScheduledTaskRun(
            task_id=task.id,
            user_id=task.user_id,
            status="running",
            trace_id=trace_id,
            started_at=now,
            finished_at=None,
            error=None,
            result={},
        )
        db.add(run)
        db.commit()
        db.refresh(run)

        text, context, default_agent, mode, forced_agent, items = parse_payload(task)
        # include multi-tenant info in context for downstream auditing (best-effort)
        if isinstance(context, dict):
            context = {**context, "user_id": task.user_id, "scheduled_task_id": task.id}

        ok = False
        err: Optional[str] = None
        result: Dict[str, Any] = {}
        finished_at = utcnow()
        try:
            result = asyncio.run(
                dispatch_execute(
                    text=text,
                    context=context,
                    default_agent=default_agent,
                    mode=mode,
                    trace_id=trace_id,
                    forced_agent=forced_agent,
                    items=items,
                )
            )
            ok = True
        except Exception as exc:
            err = str(exc)
        finally:
            finished_at = utcnow()

        with SessionLocal() as db2:
            run2 = db2.get(ScheduledTaskRun, run.id)
            task2 = db2.get(ScheduledTask, task.id)
            if run2:
                run2.status = "success" if ok else "failed"
                run2.finished_at = finished_at
                run2.error = err
                run2.result = result or {}
                db2.commit()

            if task2:
                # avoid hot-loop retries: basic backoff on failure
                next_run = (
                    compute_next_run(task2, after_utc=finished_at)
                    if ok
                    else (finished_at + timedelta(seconds=60))
                )
                _finish_task(
                    db2,
                    task=task2,
                    now=finished_at,
                    ok=ok,
                    next_run_at=next_run,
                    worker_id=worker_id,
                )

        return ok


def tick(*, batch: int, lease_seconds: int, worker_id: str) -> TickResult:
    now = utcnow()
    with SessionLocal() as db:
        ids = claim_due_tasks(
            db,
            now=now,
            limit=batch,
            lease_seconds=lease_seconds,
            worker_id=worker_id,
        )

    res = TickResult(claimed=len(ids))
    for task_id in ids:
        res.executed += 1
        ok = run_one_task(task_id, worker_id=worker_id)
        if ok:
            res.ok += 1
        else:
            res.failed += 1
    return res


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="fun-ai-station-api scheduler worker")
    parser.add_argument("--once", action="store_true", help="Run one tick and exit")
    parser.add_argument("--poll", type=int, default=5, help="Poll interval seconds (loop mode)")
    parser.add_argument("--batch", type=int, default=10, help="Max tasks claimed per tick")
    parser.add_argument("--lease", type=int, default=120, help="Lease seconds for claimed tasks")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    logger.info("starting scheduler-worker orchestrator_base=%s", settings.ORCHESTRATOR_URL)

    host = socket.gethostname()
    worker_id = f"{host}:{uuid.uuid4().hex[:8]}"

    if args.once:
        res = tick(batch=args.batch, lease_seconds=args.lease, worker_id=worker_id)
        logger.info(
            "tick claimed=%s executed=%s ok=%s failed=%s",
            res.claimed,
            res.executed,
            res.ok,
            res.failed,
        )
        return 0

    while True:
        try:
            res = tick(batch=args.batch, lease_seconds=args.lease, worker_id=worker_id)
            if res.claimed:
                logger.info(
                    "tick claimed=%s executed=%s ok=%s failed=%s",
                    res.claimed,
                    res.executed,
                    res.ok,
                    res.failed,
                )
        except Exception:
            logger.exception("tick failed")
        time.sleep(max(1, int(args.poll)))


if __name__ == "__main__":
    raise SystemExit(main())
