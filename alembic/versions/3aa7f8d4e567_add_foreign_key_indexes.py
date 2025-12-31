"""Add indexes on foreign key columns for performance

Revision ID: 3aa7f8d4e567
Revises: 2ff4e6b8c123
Create Date: 2025-08-18 17:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3aa7f8d4e567"
down_revision: Union[str, Sequence[str], None] = "2ff4e6b8c123"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add indexes on foreign key columns for performance."""

    # Indexes for CASCADE performance
    op.create_index("ix_answers_query_id", "answers", ["query_id"])
    op.create_index("ix_query_vectors_parent_id", "query_vectors", ["parent_id"])
    op.create_index("ix_chunk_vectors_parent_id", "chunk_vectors", ["parent_id"])
    op.create_index("ix_rag_responses_query_id", "rag_responses", ["query_id"])
    op.create_index(
        "ix_rag_responses_evaluation_set_id", "rag_responses", ["evaluation_set_id"]
    )

    # Additional useful indexes for queries
    op.create_index("ix_queries_doc_id", "queries", ["doc_id"])
    op.create_index("ix_queries_chunk_id", "queries", ["chunk_id"])
    op.create_index("ix_queries_prompt_id", "queries", ["prompt_id"])
    op.create_index("ix_queries_llm", "queries", ["llm"])
    op.create_index("ix_queries_qa_type", "queries", ["qa_type"])
    op.create_index("ix_queries_text_hash", "queries", ["text_hash"])

    # Indexes for chunks
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("ix_chunks_text_hash", "chunks", ["text_hash"])

    # Indexes for docs
    op.create_index("ix_docs_dataset", "docs", ["dataset"])
    op.create_index("ix_docs_text_hash", "docs", ["text_hash"])


def downgrade() -> None:
    """Remove the indexes."""

    # Drop indexes
    op.drop_index("ix_answers_query_id", "answers")
    op.drop_index("ix_query_vectors_parent_id", "query_vectors")
    op.drop_index("ix_chunk_vectors_parent_id", "chunk_vectors")
    op.drop_index("ix_rag_responses_query_id", "rag_responses")
    op.drop_index("ix_rag_responses_evaluation_set_id", "rag_responses")

    op.drop_index("ix_queries_doc_id", "queries")
    op.drop_index("ix_queries_chunk_id", "queries")
    op.drop_index("ix_queries_prompt_id", "queries")
    op.drop_index("ix_queries_llm", "queries")
    op.drop_index("ix_queries_qa_type", "queries")
    op.drop_index("ix_queries_text_hash", "queries")

    op.drop_index("ix_chunks_doc_id", "chunks")
    op.drop_index("ix_chunks_text_hash", "chunks")

    op.drop_index("ix_docs_dataset", "docs")
    op.drop_index("ix_docs_text_hash", "docs")
