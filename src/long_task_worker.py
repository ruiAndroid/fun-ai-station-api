from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import asc, or_, select
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.core.db import SessionLocal
from src.core.orchestrator_client import dispatch_execute
from src.models.long_task import LongTask


logger = logging.getLogger("long-task-worker")


def utcnow() -> datetime:
    return datetime.utcnow()


def _normalize_payload(task: LongTask) -> tuple[str, Dict[str, Any], str, str, str, Optional[List[Dict[str, Any]]]]:
    settings = get_settings()
    raw = task.payload or {}

    text = str(raw.get("text") or raw.get("input") or "")
    context = raw.get("context") if isinstance(raw.get("context"), dict) else {}

    default_agent = str(raw.get("default_agent") or "").strip() or settings.OPENCLAW_DEFAULT_AGENT or "general"
    mode = str(raw.get("mode") or "").strip().lower() or settings.ROUTER_MODE or "hybrid"

    forced_agent = str(raw.get("agent") or raw.get("forced_agent") or "").strip()
    items_list = raw.get("items") if isinstance(raw.get("items"), list) else None
    return text, context, default_agent, mode, forced_agent, items_list


@dataclass
class TickResult:
    claimed: int = 0
    executed: int = 0
    ok: int = 0
    failed: int = 0
    canceled: int = 0


def claim_tasks(
    db: Session,
    *,
    now: datetime,
    limit: int,
    lease_seconds: int,
    worker_id: str,
) -> List[int]:
    lease_until = now + timedelta(seconds=max(10, lease_seconds))
    q = (
        select(LongTask)
        .where(
            LongTask.status.in_(("pending", "running")),
            or_(LongTask.locked_until.is_(None), LongTask.locked_until <= now),
        )
        .order_by(asc(LongTask.created_at), asc(LongTask.id))
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    tasks = db.execute(q).scalars().all()
    ids: List[int] = []
    for t in tasks:
        if t.cancel_requested:
            # If user already requested cancel, mark it best-effort here.
            t.status = "canceled"
            t.finished_at = now
            t.locked_by = None
            t.locked_until = None
            continue

        t.locked_by = worker_id
        t.locked_until = lease_until
        t.status = "running"
        if t.started_at is None:
            t.started_at = now
        t.attempt = int(t.attempt or 0) + 1
        ids.append(int(t.id))

    if tasks:
        db.commit()
    return ids


def _release_lock(task: LongTask) -> None:
    task.locked_by = None
    task.locked_until = None


def run_one_task(task_id: int, *, worker_id: str) -> bool:
    settings = get_settings()
    now = utcnow()

    with SessionLocal() as db:
        task = db.get(LongTask, task_id)
        if not task:
            return True

        # Only the lease owner should mutate state.
        if task.locked_by != worker_id or not task.locked_until or task.locked_until < now:
            return True

        if task.cancel_requested:
            task.status = "canceled"
            task.finished_at = now
            _release_lock(task)
            db.commit()
            return True

        text, context, default_agent, mode, forced_agent, items = _normalize_payload(task)
        if not text.strip():
            task.status = "failed"
            task.error = "empty task text"
            task.finished_at = now
            _release_lock(task)
            db.commit()
            return False

        trace_id = (task.trace_id or "").strip() or uuid.uuid4().hex
        task.trace_id = trace_id
        db.commit()

    ok = False
    err: Optional[str] = None
    result: Dict[str, Any] = {}
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
                timeout_seconds=getattr(settings, "LONG_TASK_EXECUTE_TIMEOUT_SECONDS", 600),
            )
        )
        if isinstance(result, dict) and result:
            ok = True
        else:
            err = "orchestrator returned empty response"
    except Exception as exc:
        err = str(exc)

    finished_at = utcnow()
    with SessionLocal() as db2:
        task2 = db2.get(LongTask, task_id)
        if not task2:
            return ok

        # Lost lease => do not overwrite.
        if task2.locked_by != worker_id:
            return ok

        if task2.cancel_requested:
            task2.status = "canceled"
            task2.finished_at = finished_at
            _release_lock(task2)
            db2.commit()
            return True

        task2.finished_at = finished_at
        if ok:
            task2.status = "success"
            output = result.get("output") if isinstance(result, dict) else None
            task2.output = str(output) if output is not None else ""
            task2.result = result or {}
            task2.error = None
        else:
            task2.status = "failed"
            task2.output = ""
            task2.result = result or {}
            task2.error = err or "failed"

        _release_lock(task2)
        db2.commit()
        return ok


def tick(*, batch: int, lease_seconds: int, worker_id: str) -> TickResult:
    now = utcnow()
    with SessionLocal() as db:
        ids = claim_tasks(
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
    parser = argparse.ArgumentParser(description="fun-ai-station-api long task worker")
    parser.add_argument("--once", action="store_true", help="Run one tick and exit")
    parser.add_argument("--poll", type=int, default=2, help="Poll interval seconds (loop mode)")
    parser.add_argument("--batch", type=int, default=10, help="Max tasks claimed per tick")
    parser.add_argument("--lease", type=int, default=600, help="Lease seconds for claimed tasks")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    logger.info(
        "starting long-task-worker orchestrator_base=%s timeout=%ss",
        settings.ORCHESTRATOR_URL,
        getattr(settings, "LONG_TASK_EXECUTE_TIMEOUT_SECONDS", 600),
    )

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
