import uuid
from typing import List

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin


class ChatSession(TimestampMixin, Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    agent_id: Mapped[str] = mapped_column(String(64), ForeignKey("agents.id"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Chat")

    messages: Mapped[List["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(TimestampMixin, Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chat_sessions.id"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # user/assistant/system
    content: Mapped[str] = mapped_column(Text, nullable=False)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")

