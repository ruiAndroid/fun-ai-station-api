"""agents id -> bigint autoincrement

Revision ID: 0002_agent_id_bigint
Revises: 0001_init
Create Date: 2026-02-07
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0002_agent_id_bigint"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def _get_fk_name(bind, table: str, column: str, ref_table: str):
    # MySQL stores FK names in information_schema; we fetch dynamically because names may differ by environment.
    row = bind.execute(
        sa.text(
            """
            SELECT CONSTRAINT_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = :table
              AND COLUMN_NAME = :column
              AND REFERENCED_TABLE_NAME = :ref_table
            LIMIT 1
            """
        ),
        {"table": table, "column": column, "ref_table": ref_table},
    ).fetchone()
    return row[0] if row else None


def _try_execute(sql: str):
    try:
        op.execute(sa.text(sql))
    except Exception:
        # best-effort (index/constraint may not exist depending on DB state)
        pass


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "mysql":
        raise RuntimeError("This migration currently supports MySQL only.")

    # 1) Drop FK chat_sessions.agent_id -> agents.id (varchar) if exists, and drop index.
    fk_name = _get_fk_name(bind, "chat_sessions", "agent_id", "agents")
    if fk_name:
        op.execute(sa.text(f"ALTER TABLE chat_sessions DROP FOREIGN KEY `{fk_name}`"))
    _try_execute("DROP INDEX ix_chat_sessions_agent_id ON chat_sessions")

    # 2) Agents: rename old id -> code, then switch primary key to bigint autoincrement id.
    op.execute(sa.text("ALTER TABLE agents CHANGE id code VARCHAR(64) NOT NULL"))
    op.execute(sa.text("ALTER TABLE agents DROP PRIMARY KEY"))
    op.execute(sa.text("ALTER TABLE agents ADD COLUMN id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST"))
    _try_execute("ALTER TABLE agents ADD UNIQUE KEY uq_agents_code (code)")

    # 3) chat_sessions: preserve old string id as agent_code, add bigint agent_id, backfill from agents.code.
    op.execute(sa.text("ALTER TABLE chat_sessions CHANGE agent_id agent_code VARCHAR(64) NOT NULL"))
    op.execute(sa.text("ALTER TABLE chat_sessions ADD COLUMN agent_id BIGINT NULL AFTER user_id"))
    op.execute(
        sa.text(
            """
            UPDATE chat_sessions cs
            JOIN agents a ON a.code = cs.agent_code
            SET cs.agent_id = a.id
            """
        )
    )

    missing = bind.execute(sa.text("SELECT COUNT(*) FROM chat_sessions WHERE agent_id IS NULL")).scalar_one()
    if missing and int(missing) > 0:
        raise RuntimeError(
            f"Migration aborted: {missing} chat_sessions rows could not be mapped to agents by agent_code."
        )

    op.execute(sa.text("ALTER TABLE chat_sessions MODIFY agent_id BIGINT NOT NULL"))
    op.execute(
        sa.text(
            "ALTER TABLE chat_sessions ADD CONSTRAINT fk_chat_sessions_agent_id "
            "FOREIGN KEY (agent_id) REFERENCES agents(id)"
        )
    )
    op.execute(sa.text("CREATE INDEX ix_chat_sessions_agent_id ON chat_sessions (agent_id)"))

    # 4) Drop the temporary agent_code column now that bigint FK is in place.
    op.execute(sa.text("ALTER TABLE chat_sessions DROP COLUMN agent_code"))


def downgrade() -> None:
    raise NotImplementedError("Downgrade not supported for agent id type migration.")

