"""scheduled tasks tables (user-specific cron/interval jobs)

Revision ID: 0006_scheduled_tasks
Revises: 0005_datetime_fsp6
Create Date: 2026-02-26
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision = "0006_scheduled_tasks"
down_revision = "0005_datetime_fsp6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    dt6 = mysql.DATETIME(fsp=6)

    op.create_table(
        "scheduled_tasks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("schedule_type", sa.String(length=16), nullable=False, server_default=sa.text("'cron'")),
        sa.Column("schedule_expr", sa.Text(), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False, server_default=sa.text("'UTC'")),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("next_run_at", dt6, nullable=True),
        sa.Column("last_run_at", dt6, nullable=True),
        sa.Column("locked_by", sa.String(length=64), nullable=True),
        sa.Column("locked_until", dt6, nullable=True),
        sa.Column("created_at", dt6, nullable=False),
        sa.Column("updated_at", dt6, nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_scheduled_tasks_user_id", "scheduled_tasks", ["user_id"])
    op.create_index("ix_scheduled_tasks_next_run_at", "scheduled_tasks", ["next_run_at"])
    op.create_index("ix_scheduled_tasks_locked_until", "scheduled_tasks", ["locked_until"])

    op.create_table(
        "scheduled_task_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default=sa.text("'running'")),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("started_at", dt6, nullable=False),
        sa.Column("finished_at", dt6, nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", dt6, nullable=False),
        sa.Column("updated_at", dt6, nullable=False),
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )
    op.create_index("ix_scheduled_task_runs_task_id", "scheduled_task_runs", ["task_id"])
    op.create_index("ix_scheduled_task_runs_user_id", "scheduled_task_runs", ["user_id"])
    op.create_index("ix_scheduled_task_runs_started_at", "scheduled_task_runs", ["started_at"])


def downgrade() -> None:
    op.drop_index("ix_scheduled_task_runs_started_at", table_name="scheduled_task_runs")
    op.drop_index("ix_scheduled_task_runs_user_id", table_name="scheduled_task_runs")
    op.drop_index("ix_scheduled_task_runs_task_id", table_name="scheduled_task_runs")
    op.drop_table("scheduled_task_runs")

    op.drop_index("ix_scheduled_tasks_locked_until", table_name="scheduled_tasks")
    op.drop_index("ix_scheduled_tasks_next_run_at", table_name="scheduled_tasks")
    op.drop_index("ix_scheduled_tasks_user_id", table_name="scheduled_tasks")
    op.drop_table("scheduled_tasks")

