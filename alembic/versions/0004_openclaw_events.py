"""openclaw events idempotency table

Revision ID: 0004_openclaw_events
Revises: 0003_utf8mb4_tables
Create Date: 2026-02-10
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0004_openclaw_events"
down_revision = "0003_utf8mb4_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "openclaw_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("event_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=True),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("agent_code", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False),
    )
    op.create_index("ix_openclaw_events_event_id", "openclaw_events", ["event_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_openclaw_events_event_id", table_name="openclaw_events")
    op.drop_table("openclaw_events")

