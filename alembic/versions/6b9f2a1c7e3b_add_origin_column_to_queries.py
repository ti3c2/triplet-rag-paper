"""add origin column to queries

Revision ID: 6b9f2a1c7e3b
Revises: 5e1b2f7a4d9a
Create Date: 2025-10-17 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6b9f2a1c7e3b"
down_revision: Union[str, None] = "5e1b2f7a4d9a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add column as nullable to support SQLite; backfill, then (optionally) enforce non-null in DBs that support it
    op.add_column("queries", sa.Column("origin", sa.String(), nullable=True))
    op.create_index("ix_queries_origin", "queries", ["origin"], unique=False)

    # Backfill existing rows to default 'dataset'
    op.execute("UPDATE queries SET origin='dataset' WHERE origin IS NULL")

    # Note: We intentionally keep the column nullable=True for SQLite compatibility
    # If you use PostgreSQL and want NOT NULL, you can uncomment the next line:
    # op.alter_column('queries', 'origin', existing_type=sa.String(), nullable=False)


def downgrade() -> None:
    op.drop_index("ix_queries_origin", table_name="queries")
    op.drop_column("queries", "origin")
