"""add long_tasks table

Revision ID: 0007_long_tasks
Revises: 0006_scheduled_tasks
Create Date: 2026-02-26
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision = "0007_long_tasks"
down_revision = "0006_scheduled_tasks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    dt6 = mysql.DATETIME(fsp=6)

    op.create_table(
        "long_tasks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column(
            "kind",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'orchestrator_execute'"),
        ),
        sa.Column(
            "title",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("'Long Task'"),
        ),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("output", sa.Text(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("started_at", dt6, nullable=True),
        sa.Column("finished_at", dt6, nullable=True),
        sa.Column("locked_by", sa.String(length=64), nullable=True),
        sa.Column("locked_until", dt6, nullable=True),
        sa.Column("created_at", dt6, nullable=False),
        sa.Column("updated_at", dt6, nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_long_tasks_user_id", "long_tasks", ["user_id"])
    op.create_index("ix_long_tasks_status", "long_tasks", ["status"])
    op.create_index("ix_long_tasks_locked_until", "long_tasks", ["locked_until"])


def downgrade() -> None:
    op.drop_index("ix_long_tasks_locked_until", table_name="long_tasks")
    op.drop_index("ix_long_tasks_status", table_name="long_tasks")
    op.drop_index("ix_long_tasks_user_id", table_name="long_tasks")
    op.drop_table("long_tasks")

