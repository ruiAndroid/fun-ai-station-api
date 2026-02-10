import uuid

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class OpenclawEvent(TimestampMixin, Base):
    """
    Database-level idempotency for OpenClaw webhook events.

    `event_id` should be stable for the same upstream message (e.g. WeCom MsgId or hash of encrypt),
    so retries won't cause duplicate chat persistence.
    """

    __tablename__ = "openclaw_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(36), nullable=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=True)
    agent_code: Mapped[str] = mapped_column(String(64), nullable=True)

