"""add sequential column to queries

Revision ID: 4f6d9a3b2c14
Revises: 3aa7f8d4e567
Create Date: 2025-10-07 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "4f6d9a3b2c14"
down_revision = "3aa7f8d4e567"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add the new column with a server_default so existing rows are backfilled to "single"
    op.add_column(
        "queries",
        sa.Column("sequential", sa.String(), nullable=False, server_default="single"),
    )

    # Create an index to match ORM "index=True"
    op.create_index("ix_queries_sequential", "queries", ["sequential"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_queries_sequential", table_name="queries")
    op.drop_column("queries", "sequential")
