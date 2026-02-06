from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class Agent(TimestampMixin, Base):
    __tablename__ = "agents"

    # matches frontend static ids like "general", "dev"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    handle: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # store arrays as JSON for simplicity
    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    capabilities: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

