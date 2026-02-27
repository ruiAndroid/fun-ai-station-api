"""remove deprecated agents (attendance/expense/admin)

Revision ID: 0008_remove_deprecated_agents
Revises: 0007_long_tasks
Create Date: 2026-02-27
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0008_remove_deprecated_agents"
down_revision = "0007_long_tasks"
branch_labels = None
depends_on = None


REMOVED_CODES = ("attendance", "expense", "admin")
REPLACEMENT_CODE = "general"


def _expanding_in_clause(param_name: str):
    return sa.bindparam(param_name, expanding=True)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "mysql":
        # This project uses MySQL in production; keep migration as a no-op for other dialects.
        return

    # 1) Ensure replacement agent exists (so we can safely re-point chat sessions).
    general_id = bind.execute(
        sa.text("SELECT id FROM agents WHERE code=:code LIMIT 1"),
        {"code": REPLACEMENT_CODE},
    ).scalar()

    if general_id is None:
        bind.execute(
            sa.text(
                """
                INSERT INTO agents(code, name, handle, description, tags, capabilities, created_at, updated_at)
                VALUES(:code, :name, :handle, :description, :tags, :capabilities, NOW(6), NOW(6))
                """
            ),
            {
                "code": REPLACEMENT_CODE,
                "name": "通用智能体",
                "handle": "@通用智能体",
                "description": "通用智能体：通用问答/澄清需求/给出下一步建议",
                "tags": "[]",
                "capabilities": "[]",
            },
        )
        general_id = bind.execute(
            sa.text("SELECT id FROM agents WHERE code=:code LIMIT 1"),
            {"code": REPLACEMENT_CODE},
        ).scalar()

    if general_id is None:
        raise RuntimeError("Migration aborted: failed to create/find replacement agent 'general'.")

    # 2) Re-point any chat sessions bound to removed agents.
    old_rows = bind.execute(
        sa.text("SELECT id FROM agents WHERE code IN :codes").bindparams(_expanding_in_clause("codes")),
        {"codes": list(REMOVED_CODES)},
    ).fetchall()
    old_ids = [int(r[0]) for r in old_rows if r and r[0] is not None]

    if old_ids:
        bind.execute(
            sa.text("UPDATE chat_sessions SET agent_id=:new_id WHERE agent_id IN :old_ids").bindparams(
                _expanding_in_clause("old_ids")
            ),
            {"new_id": int(general_id), "old_ids": old_ids},
        )

    # 3) Remove deprecated agent rows.
    bind.execute(
        sa.text("DELETE FROM agents WHERE code IN :codes").bindparams(_expanding_in_clause("codes")),
        {"codes": list(REMOVED_CODES)},
    )


def downgrade() -> None:
    raise NotImplementedError("Downgrade not supported for deprecated agents removal.")
