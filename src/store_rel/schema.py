import hashlib
import logging
from datetime import datetime
from typing import TypeVar

import numpy as np
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import select

from src.config.utils import get_text_hash

logger = logging.getLogger(__name__)

Base = declarative_base()


class NumpyArray(TypeDecorator):
    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            # Handle both numpy arrays and bytes
            if isinstance(value, bytes):
                return value
            else:
                # Convert numpy array to bytes
                return value.astype(np.float32).tobytes()
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return np.frombuffer(value, dtype=np.float32)
        return value


class ChunkVector(Base):
    __tablename__ = "chunk_vectors"
    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey("chunks.id"), nullable=False, index=True)
    vector = Column(NumpyArray, nullable=False)
    emb_model = Column(String, nullable=False)

    parent = relationship("Chunk", back_populates="vectors")


class QueryVector(Base):
    __tablename__ = "query_vectors"
    id = Column(Integer, primary_key=True)
    parent_id = Column(Integer, ForeignKey("queries.id"), nullable=False, index=True)
    vector = Column(NumpyArray, nullable=False)
    emb_model = Column(String, nullable=False)

    parent = relationship("Query", back_populates="vectors")


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, ForeignKey("docs.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    text_hash = Column(
        String(64), unique=True, nullable=False, index=True
    )  # SHA-256 hash of text content
    hf_id = Column(String, nullable=True)
    meta = Column(JSON, nullable=True)

    doc = relationship("Doc", back_populates="chunks")
    vectors = relationship(
        "ChunkVector", back_populates="parent", cascade="all, delete-orphan"
    )
    queries = relationship("Query", back_populates="chunk")


class Doc(Base):
    __tablename__ = "docs"
    id = Column(Integer, primary_key=True)
    dataset = Column(String, nullable=False, index=True)  # nq | squad | multihop
    title = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    text_hash = Column(
        String(64), unique=True, nullable=False, index=True
    )  # SHA-256 hash of text content
    hf_id = Column(String, nullable=True)
    is_virtual = Column(Boolean, nullable=False, default=False)
    meta = Column(JSON, nullable=True)

    queries = relationship("Query", back_populates="doc")
    chunks = relationship("Chunk", back_populates="doc")


class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    text_hash = Column(
        String(64), unique=True, nullable=False
    )  # SHA-256 hash of text content

    queries = relationship("Query", back_populates="prompt")


class Answer(Base):
    __tablename__ = "answers"
    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    temperature = Column(Float, default=None, nullable=True)

    query = relationship("Query", back_populates="answer")


class Query(Base):
    __tablename__ = "queries"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, ForeignKey("docs.id"), nullable=False, index=True)
    chunk_id = Column(Integer, ForeignKey("chunks.id"), nullable=True, index=True)
    text = Column(Text, nullable=False)
    text_hash = Column(
        String(64), unique=False, nullable=False, index=True
    )  # SHA-256 hash of text content
    qa_type = Column(String, nullable=False, default="q", index=True)  # q or qa
    llm = Column(String, nullable=True, index=True)  # LLM used to generate the query. None if taken from dataset # fmt: skip
    temperature = Column(Float, default=None, nullable=True)
    prompt_id = Column(
        Integer, ForeignKey("prompts.id"), nullable=True, index=True
    )  # prompt id used for query generation. None if taken from dataset
    hf_id = Column(String, nullable=True)
    sequential = Column(String, nullable=False, default="single", index=True)
    # Origin: dataset, quote, eval
    origin = Column(String, nullable=False, default="dataset", index=True)
    meta = Column(JSON, nullable=True)

    doc = relationship("Doc", back_populates="queries")
    chunk = relationship("Chunk", back_populates="queries")
    prompt = relationship("Prompt", back_populates="queries")
    answer = relationship(
        "Answer", back_populates="query", cascade="all, delete-orphan"
    )
    vectors = relationship(
        "QueryVector", back_populates="parent", cascade="all, delete-orphan"
    )
    rag_responses = relationship(
        "RagResponseDB", back_populates="query", cascade="all, delete-orphan"
    )


class EvaluationSet(Base):
    """
    Evaluation set is an abstraction for referencing a set of rag responses for a given experiment.
    It stores all the settings in JSON format, because there are numerous ways we may want to experiments,
    and we want to be able to easily change them.
    """

    __tablename__ = "evaluation_sets"
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime, nullable=False)
    dataset = Column(String, nullable=False)
    metrics = Column(JSON, nullable=True)
    settings = Column(JSON, nullable=True)

    rag_responses = relationship(
        "RagResponseDB", back_populates="evaluation_set", cascade="all, delete-orphan"
    )
    rag_triplets = relationship(
        "QueryTriplet", back_populates="evaluation_set", cascade="all, delete-orphan"
    )

    def set_metrics(self, metrics: dict):
        self.metrics = metrics


class RagResponseDB(Base):
    __tablename__ = "rag_responses"
    id = Column(Integer, primary_key=True)
    evaluation_set_id = Column(
        Integer,
        ForeignKey("evaluation_sets.id"),
        nullable=False,
        index=True,
    )
    retrieval_result = Column(JSON, nullable=False)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False, index=True)

    evaluation_set = relationship("EvaluationSet", back_populates="rag_responses")
    query = relationship("Query", back_populates="rag_responses")


class QueryTriplet(Base):
    __tablename__ = "query_triplets"
    __table_args__ = (
        Index(
            "uq_query_triplet_evalset_query",
            "evaluation_set_id",
            "eval_query_id",
            unique=True,
        ),
    )

    id = Column(Integer, primary_key=True)
    evaluation_set_id = Column(
        Integer,
        ForeignKey("evaluation_sets.id"),
        nullable=False,
        index=True,
    )
    # The eval-origin query id that produced this triplet
    eval_query_id = Column(
        Integer, ForeignKey("queries.id"), nullable=False, index=True
    )
    # Optional link to gold answer of the eval query, if present
    gold_answer_id = Column(
        Integer, ForeignKey("answers.id"), nullable=True, index=True
    )

    # JSON payloads storing all contexts and answers grouped by k
    # contexts_by_k: {"k": [{"type": "chunk|query", "doc_id": int, "doc_title": str|null, "chunk_id": int|null, "text": str, "score": float, "rank": int}]}
    contexts_by_k = Column(JSON, nullable=False)
    # rag_answers_by_k: {"k": str}
    rag_answers_by_k = Column(JSON, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    archived_at = Column(DateTime, nullable=True)

    evaluation_set = relationship("EvaluationSet", back_populates="rag_triplets")
    eval_query = relationship("Query", foreign_keys=[eval_query_id])


async def init_db(database_url: str):
    engine = create_async_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


T = TypeVar("T", bound=Base)


async def get_or_create_item(
    session,
    model_class: type[T],
    force_create: bool = False,
    **kwargs,
) -> tuple[T, bool]:
    """Get existing item by content hash or create new one."""
    from sqlalchemy.exc import IntegrityError

    text_hash = get_text_hash(kwargs["text"])
    item = None
    if not force_create:
        item = await session.execute(select(model_class).filter_by(text_hash=text_hash))
        item = item.scalars().first()
    created_new = False
    if item is None or force_create:
        kwargs["text_hash"] = text_hash
        item = model_class(**kwargs)
        session.add(item)
        try:
            await session.flush()
            await session.refresh(item)
            created_new = True
        except IntegrityError:
            # Race condition: another process created the item
            # Roll back and fetch the existing item
            await session.rollback()
            item = await session.execute(select(model_class).filter_by(text_hash=text_hash))
            item = item.scalars().first()
            if item is None:
                # If still not found, re-raise the original error
                raise
            created_new = False
    return item, created_new


async def get_prompt(session, prompt_name: str) -> Prompt:
    prompt_db = await session.execute(select(Prompt).where(Prompt.name == prompt_name))
    prompts = prompt_db.scalars().all()
    if len(prompts) == 0 or prompts[0] is None:
        raise ValueError(f"Prompt {prompt_name} not found in the database.")
    prompt = sorted(prompts, key=lambda x: x.id)[-1]
    if len(prompts) > 1:
        logger.warning(
            f"Multiple prompts found for {prompt_name}. Using the latest one. \n{prompt.text}"
        )
    return prompt
