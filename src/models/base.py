from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow():
    # MySQL DATETIME doesn't store timezone; keep values naive UTC to avoid driver issues
    return datetime.utcnow()


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, onupdate=utcnow
    )

