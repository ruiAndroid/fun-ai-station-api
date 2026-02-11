"""use DATETIME(6) for message ordering stability

Revision ID: 0005_datetime_fsp6
Revises: 0004_openclaw_events
Create Date: 2026-02-11
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0005_datetime_fsp6"
down_revision = "0004_openclaw_events"
branch_labels = None
depends_on = None


def _try(sql: str) -> None:
    try:
        op.execute(sa.text(sql))
    except Exception:
        # best-effort; some tables/columns may differ by existing DB state
        pass


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "mysql":
        return

    # Improve timestamp precision so user+assistant messages inserted in quick succession
    # won't get the same second-level created_at.
    for table in ("chat_messages", "chat_sessions", "users", "agents", "openclaw_events"):
        _try(f"ALTER TABLE {table} MODIFY COLUMN created_at DATETIME(6) NOT NULL")
        _try(f"ALTER TABLE {table} MODIFY COLUMN updated_at DATETIME(6) NOT NULL")


def downgrade() -> None:
    raise NotImplementedError("Downgrade not supported for datetime precision migration.")

