"""Add CASCADE to foreign key constraints

Revision ID: 2ff4e6b8c123
Revises: 158c7a11133b
Create Date: 2025-08-18 16:45:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2ff4e6b8c123"
down_revision: Union[str, Sequence[str], None] = "158c7a11133b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add CASCADE to foreign key constraints."""
    # Drop existing foreign key constraints and recreate with CASCADE

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # SQLite: Enable foreign keys and handle differently
        op.execute("PRAGMA foreign_keys=ON")
        # For SQLite, constraint modification is complex - skip for now
        return

    # PostgreSQL: Standard approach
    # answers.query_id -> queries.id
    op.drop_constraint("answers_query_id_fkey", "answers", type_="foreignkey")
    op.create_foreign_key(
        "answers_query_id_fkey",
        "answers",
        "queries",
        ["query_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # query_vectors.parent_id -> queries.id
    op.drop_constraint(
        "query_vectors_parent_id_fkey", "query_vectors", type_="foreignkey"
    )
    op.create_foreign_key(
        "query_vectors_parent_id_fkey",
        "query_vectors",
        "queries",
        ["parent_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # chunk_vectors.parent_id -> chunks.id
    op.drop_constraint(
        "chunk_vectors_parent_id_fkey", "chunk_vectors", type_="foreignkey"
    )
    op.create_foreign_key(
        "chunk_vectors_parent_id_fkey",
        "chunk_vectors",
        "chunks",
        ["parent_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # rag_responses.query_id -> queries.id
    op.drop_constraint(
        "rag_responses_query_id_fkey", "rag_responses", type_="foreignkey"
    )
    op.create_foreign_key(
        "rag_responses_query_id_fkey",
        "rag_responses",
        "queries",
        ["query_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # rag_responses.evaluation_set_id -> evaluation_sets.id
    op.drop_constraint(
        "rag_responses_evaluation_set_id_fkey", "rag_responses", type_="foreignkey"
    )
    op.create_foreign_key(
        "rag_responses_evaluation_set_id_fkey",
        "rag_responses",
        "evaluation_sets",
        ["evaluation_set_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Remove CASCADE from foreign key constraints."""
    # Revert to non-CASCADE foreign keys

    # answers.query_id -> queries.id
    op.drop_constraint("answers_query_id_fkey", "answers", type_="foreignkey")
    op.create_foreign_key(
        "answers_query_id_fkey", "answers", "queries", ["query_id"], ["id"]
    )

    # query_vectors.parent_id -> queries.id
    op.drop_constraint(
        "query_vectors_parent_id_fkey", "query_vectors", type_="foreignkey"
    )
    op.create_foreign_key(
        "query_vectors_parent_id_fkey",
        "query_vectors",
        "queries",
        ["parent_id"],
        ["id"],
    )

    # chunk_vectors.parent_id -> chunks.id
    op.drop_constraint(
        "chunk_vectors_parent_id_fkey", "chunk_vectors", type_="foreignkey"
    )
    op.create_foreign_key(
        "chunk_vectors_parent_id_fkey", "chunk_vectors", "chunks", ["parent_id"], ["id"]
    )

    # rag_responses.query_id -> queries.id
    op.drop_constraint(
        "rag_responses_query_id_fkey", "rag_responses", type_="foreignkey"
    )
    op.create_foreign_key(
        "rag_responses_query_id_fkey", "rag_responses", "queries", ["query_id"], ["id"]
    )

    # rag_responses.evaluation_set_id -> evaluation_sets.id
    op.drop_constraint(
        "rag_responses_evaluation_set_id_fkey", "rag_responses", type_="foreignkey"
    )
    op.create_foreign_key(
        "rag_responses_evaluation_set_id_fkey",
        "rag_responses",
        "evaluation_sets",
        ["evaluation_set_id"],
        ["id"],
    )
