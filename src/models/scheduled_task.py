from __future__ import annotations

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin, utcnow


class ScheduledTask(TimestampMixin, Base):
    __tablename__ = "scheduled_tasks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # multi-tenant
    user_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # schedule:
    # - cron:     schedule_expr is a cron expression
    # - interval: schedule_expr is integer seconds
    # - once:     schedule_expr is optional; next_run_at drives execution
    schedule_type: Mapped[str] = mapped_column(String(16), nullable=False, default="cron")
    schedule_expr: Mapped[str] = mapped_column(Text, nullable=False, default="")
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")

    # execution input for orchestrator (free-form JSON)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    next_run_at: Mapped[object] = mapped_column(DateTime(timezone=False), nullable=True, index=True)
    last_run_at: Mapped[object] = mapped_column(DateTime(timezone=False), nullable=True)

    # simple lease to avoid duplicate runs in multi-worker setups
    locked_by: Mapped[str] = mapped_column(String(64), nullable=True)
    locked_until: Mapped[object] = mapped_column(DateTime(timezone=False), nullable=True, index=True)


class ScheduledTaskRun(TimestampMixin, Base):
    __tablename__ = "scheduled_task_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    trace_id: Mapped[str] = mapped_column(String(64), nullable=True)

    started_at: Mapped[object] = mapped_column(DateTime(timezone=False), nullable=False, default=utcnow)
    finished_at: Mapped[object] = mapped_column(DateTime(timezone=False), nullable=True)

    error: Mapped[str] = mapped_column(Text, nullable=True)
    result: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

