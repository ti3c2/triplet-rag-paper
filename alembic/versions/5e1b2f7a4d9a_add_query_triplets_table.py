"""add query_triplets table

Revision ID: 5e1b2f7a4d9a
Revises: 4f6d9a3b2c14
Create Date: 2025-10-16 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5e1b2f7a4d9a"
down_revision: Union[str, None] = "4f6d9a3b2c14"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "query_triplets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("evaluation_set_id", sa.Integer(), nullable=False),
        sa.Column("eval_query_id", sa.Integer(), nullable=False),
        sa.Column("gold_answer_id", sa.Integer(), nullable=True),
        sa.Column("contexts_by_k", sa.JSON(), nullable=False),
        sa.Column("rag_answers_by_k", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("archived_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["evaluation_set_id"], ["evaluation_sets.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["eval_query_id"], ["queries.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["gold_answer_id"], ["answers.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_query_triplets_evaluation_set_id",
        "query_triplets",
        ["evaluation_set_id"],
        unique=False,
    )
    op.create_index(
        "ix_query_triplets_eval_query_id",
        "query_triplets",
        ["eval_query_id"],
        unique=False,
    )
    op.create_index(
        "ix_query_triplets_gold_answer_id",
        "query_triplets",
        ["gold_answer_id"],
        unique=False,
    )
    op.create_index(
        "uq_query_triplet_evalset_query",
        "query_triplets",
        ["evaluation_set_id", "eval_query_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_query_triplet_evalset_query", table_name="query_triplets")
    op.drop_index("ix_query_triplets_gold_answer_id", table_name="query_triplets")
    op.drop_index("ix_query_triplets_eval_query_id", table_name="query_triplets")
    op.drop_index("ix_query_triplets_evaluation_set_id", table_name="query_triplets")
    op.drop_table("query_triplets")
