"""ensure utf8mb4 charset for core tables

Revision ID: 0003_utf8mb4_tables
Revises: 0002_agent_id_bigint
Create Date: 2026-02-08
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0003_utf8mb4_tables"
down_revision = "0002_agent_id_bigint"
branch_labels = None
depends_on = None


def _try_execute(sql: str):
    try:
        op.execute(sa.text(sql))
    except Exception:
        # best-effort; table may not exist depending on DB state
        pass


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "mysql":
        return

    _try_execute("ALTER TABLE agents CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    _try_execute("ALTER TABLE users CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    _try_execute("ALTER TABLE chat_sessions CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    _try_execute("ALTER TABLE chat_messages CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")


def downgrade() -> None:
    # No downgrade: avoid data loss from charset conversion.
    raise NotImplementedError("Downgrade not supported for charset migration.")
