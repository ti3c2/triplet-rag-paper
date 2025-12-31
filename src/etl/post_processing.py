import logging
from typing import Literal, Optional

from sqlalchemy import select

from ..config.settings import settings
from ..config.utils import get_text_hash
from ..store_rel.entry import get_db
from ..store_rel.schema import (
    Answer,
    Chunk,
    Doc,
    Prompt,
    Query,
    get_or_create_item,
    get_prompt,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)


async def convert_q_queries_to_qa(
    dataset: str = settings.rag_dataset,
    llm: str = settings.question_generation_model,
    qa_type: Literal["qa", "qc"] = settings.question_generation_mode,
    n_commited: int = settings.rag_max_rows // 10,
    prompt_name: Optional[str] = settings.question_generation_prompt_name,
    max_items: int = -1,
) -> tuple[int, int]:
    """Convert existing queries with qa_type='q' to qa_type='qa' by appending their answers.

    Args:
        dataset: Dataset to convert queries for (None for all datasets)
        llm: LLM to filter queries by (None for all LLMs)
        n_commited: Number of queries to commit after each batch
        max_items: Maximum number of queries to convert (None for all)

    Returns:
        tuple[int, int]: Total number of queries found, number of queries successfully converted
    """
    logger.info(
        f"Converting q-type queries to qa-type for dataset {dataset} with model {llm}"
    )
    logger.info(f"Settings:\n{settings.model_dump_json(indent=2)}")

    prompt = settings.get_prompt(prompt_name)
    prompt_hash = get_text_hash(prompt.text)

    async with get_db() as session:
        if not prompt.text:
            prompt_db = await get_prompt(session, prompt.name)
            prompt_hash = str(prompt_db.text_hash)
        # Query for q-type queries with their associated answers from Answer or Chunk tables
        if qa_type == "qa":
            q_queries_result = await session.execute(
                select(
                    Query.id,
                    Query.text,
                    Query.text_hash,
                    Query.doc_id,
                    Query.chunk_id,
                    Query.llm,
                    Query.temperature,
                    Query.prompt_id,
                    Query.hf_id,
                    Query.meta,
                    Query.origin,
                    Answer.text.label("answer_text"),
                    Prompt.text_hash.label("prompt_hash"),
                )
                .join(Doc, Query.doc_id == Doc.id)
                .join(Prompt, Query.prompt_id == Prompt.id)
                .outerjoin(Answer, Query.id == Answer.query_id)
                .where(
                    Query.qa_type == "q",
                    Doc.dataset == dataset,
                    Query.llm == llm,
                    Prompt.text_hash == prompt_hash,
                    Query.origin.in_(settings.rag_retrieval_query_origins),
                )
                .limit(max_items if max_items > 0 else None)
            )
        elif qa_type == "qc":
            q_queries_result = await session.execute(
                select(
                    Query.id,
                    Query.text,
                    Query.text_hash,
                    Query.doc_id,
                    Query.chunk_id,
                    Query.llm,
                    Query.temperature,
                    Query.prompt_id,
                    Query.hf_id,
                    Query.meta,
                    Query.origin,
                    Chunk.text.label("answer_text"),
                )
                .join(Doc, Query.doc_id == Doc.id)
                .outerjoin(Chunk, Query.chunk_id == Chunk.id)
                .where(
                    Query.qa_type == "q",
                    Doc.dataset == dataset,
                    Query.llm == llm,
                    Query.origin.in_(settings.rag_retrieval_query_origins),
                )
                .limit(max_items if max_items > 0 else None)
            )
        else:
            raise ValueError(f"Invalid qa_type: {qa_type}")

        # TODO: Optimize for SQL-only queries and maybe do streaming iteration
        existing_queries_result = await session.execute(
            select(Query.text_hash)
            .join(Doc, Query.doc_id == Doc.id)
            .where(
                Query.qa_type == qa_type,
                Query.llm == llm,
                Doc.dataset == dataset,
                Query.origin.in_(settings.rag_retrieval_query_origins),
            )
        )
        existing_queries = set(row.text_hash for row in existing_queries_result.all())
        rows = q_queries_result.all()
        q_queries_rows = [row for row in rows if row.text_hash not in existing_queries]
        n_queries_to_convert = len(q_queries_rows)

        successfully_converted = 0
        logger.info(
            f"Found {len(rows)} queries with qa_type='q' to convert to qa_type='{qa_type}'. "
            f"Processing {n_queries_to_convert} queries as {len(existing_queries)} already exist."
        )

        for i, row in enumerate(q_queries_rows):
            try:
                if not row.answer_text:
                    logger.warning(f"Query {row.id} has no associated answer, skipping")
                    continue

                answer_text = row.answer_text
                qa_text = f"{row.text}{settings.qa_separator}{answer_text}"

                qa_query, created = await get_or_create_item(
                    session,
                    Query,
                    force_create=True,
                    doc_id=row.doc_id,
                    chunk_id=row.chunk_id,
                    text=qa_text,
                    qa_type=qa_type,
                    llm=row.llm,
                    temperature=row.temperature,
                    prompt_id=row.prompt_id,
                    hf_id=row.hf_id,
                    meta=row.meta,
                    origin=row.origin,
                )

                if created:
                    new_answer = Answer(
                        query_id=qa_query.id,
                        text=answer_text,
                    )
                    session.add(new_answer)
                    successfully_converted += 1
                    logger.debug(
                        f"Converted query {row.id} to qa-type query {qa_query.id}"
                    )
                else:
                    logger.debug(f"Query {row.id} already has qa equivalent, skipping")

                if (i + 1) % n_commited == 0:
                    await session.commit()
                    logger.info(f"Committed {i + 1} queries.")

            except Exception as e:
                logger.error(f"Error converting query {row.id}: {e}")
                continue

        try:
            await session.commit()
            logger.info(
                f"Successfully converted {successfully_converted}/{n_queries_to_convert} q-type queries to qa-type"
            )
        except Exception as e:
            await session.rollback()
            logger.error(f"Error committing conversions: {e}")
            successfully_converted = 0

        return n_queries_to_convert, successfully_converted
