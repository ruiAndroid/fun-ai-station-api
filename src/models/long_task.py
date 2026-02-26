from __future__ import annotations

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class LongTask(TimestampMixin, Base):
    __tablename__ = "long_tasks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # multi-tenant
    user_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)

    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="orchestrator_execute")
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="Long Task")

    # status: pending | running | success | failed | canceled
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    trace_id: Mapped[str] = mapped_column(String(64), nullable=True)

    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    output: Mapped[str] = mapped_column(Text, nullable=True)
    result: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str] = mapped_column(Text, nullable=True)

    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    started_at: Mapped[object] = mapped_column(DateTime(timezone=False), nullable=True)
    finished_at: Mapped[object] = mapped_column(DateTime(timezone=False), nullable=True)

    # lease to avoid duplicate execution
    locked_by: Mapped[str] = mapped_column(String(64), nullable=True)
    locked_until: Mapped[object] = mapped_column(DateTime(timezone=False), nullable=True, index=True)
