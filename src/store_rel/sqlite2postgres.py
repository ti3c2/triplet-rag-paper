import logging
from typing import Optional

from sqlalchemy import func, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.sql import select

from ..config.settings import settings
from .schema import (
    Answer,
    Base,
    Chunk,
    ChunkVector,
    Doc,
    EvaluationSet,
    Prompt,
    Query,
    QueryVector,
    RagResponseDB,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def export_to_postgres(
    source_database_url: Optional[str] = None,
    target_database_url: Optional[str] = None,
    recreate_tables: bool = False,
    commit_every: int = 1000,
    skip_orphaned_records: bool = True,
    exclude_tables: Optional[list[str]] = None,
) -> dict:
    """
    Export all data from source SQLAlchemy storage to PostgreSQL.

    Args:
        source_database_url: Source database URL. If None, uses sqlite from settings.
        target_database_url: Target PostgreSQL URL. If None, uses postgres from settings.
        recreate_tables: Whether to drop and recreate tables in target database.
        commit_every: Commit every N records to avoid large transactions.
        skip_orphaned_records: Skip records with invalid foreign key references.
        exclude_tables: Tables to exclude from export. Only rag_responses is supported for now.
    Returns:
        dict: Statistics about the export operation.
    """
    # Default URLs from settings
    if source_database_url is None:
        source_database_url = settings.sqlite_database_url
    if target_database_url is None:
        target_database_url = settings.postgres_database_url_async

    logger.info(f"Starting export from {source_database_url} to {target_database_url}")

    source_engine = create_async_engine(source_database_url)
    target_engine = create_async_engine(target_database_url)

    # Create session makers
    SourceSession = async_sessionmaker(
        autocommit=False, autoflush=False, bind=source_engine
    )
    TargetSession = async_sessionmaker(
        autocommit=False, autoflush=False, bind=target_engine
    )

    stats = {
        "docs": 0,
        "chunks": 0,
        "chunk_vectors": 0,
        "prompts": 0,
        "queries": 0,
        "query_vectors": 0,
        "answers": 0,
        "evaluation_sets": 0,
        "rag_responses": 0,
        "skipped_orphaned": 0,
    }

    try:
        # Create tables in target database if needed
        if recreate_tables:
            async with target_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Recreated target database tables")
        else:
            async with target_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Ensured target database tables exist")

        async with SourceSession() as source_session, TargetSession() as target_session:
            # Export in dependency order to maintain foreign key relationships

            # 1. Export Docs (no dependencies)
            logger.info("Exporting docs...")
            docs_result = await source_session.execute(select(Doc))
            docs = docs_result.scalars().all()
            for i, doc in enumerate(docs):
                # Create new doc instance for target database
                new_doc = Doc(
                    id=doc.id,
                    dataset=doc.dataset,
                    title=doc.title,
                    text=doc.text,
                    text_hash=doc.text_hash,
                    hf_id=doc.hf_id,
                    is_virtual=doc.is_virtual,
                    meta=doc.meta,
                )
                await target_session.merge(new_doc)
                if i % commit_every == 0:
                    await target_session.commit()
                    logger.info(f"Exported {stats['docs']} docs")
                stats["docs"] += 1

            await target_session.commit()
            logger.info(f"Exported {stats['docs']} docs")

            # 2. Export Chunks (depends on Docs)
            logger.info("Exporting chunks...")
            chunks_result = await source_session.execute(select(Chunk))
            chunks = chunks_result.scalars().all()
            for i, chunk in enumerate(chunks):
                new_chunk = Chunk(
                    id=chunk.id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    text_hash=chunk.text_hash,
                    hf_id=chunk.hf_id,
                    meta=chunk.meta,
                )
                await target_session.merge(new_chunk)
                if i % commit_every == 0:
                    await target_session.commit()
                    logger.info(f"Exported {stats['chunks']} chunks")
                stats["chunks"] += 1

            await target_session.commit()
            logger.info(f"Exported {stats['chunks']} chunks")

            # 3. Export ChunkVectors (depends on Chunks)
            logger.info("Exporting chunk vectors...")

            if skip_orphaned_records:
                # Only export chunk vectors with valid chunk references
                chunk_vectors_result = await source_session.execute(
                    select(ChunkVector).where(
                        ChunkVector.parent_id.in_(select(Chunk.id))
                    )
                )
                chunk_vectors = chunk_vectors_result.scalars().all()
                logger.info(f"Found {len(chunk_vectors)} valid chunk vectors")
            else:
                chunk_vectors_result = await source_session.execute(select(ChunkVector))
                chunk_vectors = chunk_vectors_result.scalars().all()

            for i, cv in enumerate(chunk_vectors):
                new_cv = ChunkVector(
                    id=cv.id,
                    parent_id=cv.parent_id,
                    vector=cv.vector,
                    emb_model=cv.emb_model,
                )
                await target_session.merge(new_cv)
                if i % commit_every == 0:
                    await target_session.commit()
                    logger.info(f"Exported {stats['chunk_vectors']} chunk vectors")
                stats["chunk_vectors"] += 1

            await target_session.commit()
            logger.info(f"Exported {stats['chunk_vectors']} chunk vectors")

            # 4. Export Prompts (no dependencies)
            logger.info("Exporting prompts...")
            prompts_result = await source_session.execute(select(Prompt))
            prompts = prompts_result.scalars().all()
            for prompt in prompts:
                new_prompt = Prompt(
                    id=prompt.id,
                    name=prompt.name,
                    text=prompt.text,
                    text_hash=prompt.text_hash,
                )
                await target_session.merge(new_prompt)
                stats["prompts"] += 1

            await target_session.commit()
            logger.info(f"Exported {stats['prompts']} prompts")

            # 5. Export Queries (depends on Docs, Chunks, Prompts)
            logger.info("Exporting queries...")
            queries_result = await source_session.execute(select(Query))
            queries = queries_result.scalars().all()
            for i, query in enumerate(queries):
                new_query = Query(
                    id=query.id,
                    doc_id=query.doc_id,
                    chunk_id=query.chunk_id,
                    text=query.text,
                    text_hash=query.text_hash,
                    qa_type=query.qa_type,
                    llm=query.llm,
                    temperature=query.temperature,
                    prompt_id=query.prompt_id,
                    hf_id=query.hf_id,
                    meta=query.meta,
                )
                await target_session.merge(new_query)
                if i % commit_every == 0:
                    await target_session.commit()
                    logger.info(f"Exported {stats['queries']} queries")
                stats["queries"] += 1

            await target_session.commit()
            logger.info(f"Exported {stats['queries']} queries")

            # 6. Export QueryVectors (depends on Queries)
            logger.info("Exporting query vectors...")

            if skip_orphaned_records:
                # Only export query vectors with valid query references
                query_vectors_result = await source_session.execute(
                    select(QueryVector).where(
                        QueryVector.parent_id.in_(select(Query.id))
                    )
                )
                query_vectors = query_vectors_result.scalars().all()
                logger.info(f"Found {len(query_vectors)} valid query vectors")
            else:
                query_vectors_result = await source_session.execute(select(QueryVector))
                query_vectors = query_vectors_result.scalars().all()

            for i, qv in enumerate(query_vectors):
                new_qv = QueryVector(
                    id=qv.id,
                    parent_id=qv.parent_id,
                    vector=qv.vector,
                    emb_model=qv.emb_model,
                )
                await target_session.merge(new_qv)
                if i % commit_every == 0:
                    await target_session.commit()
                    logger.info(f"Exported {stats['query_vectors']} query vectors")
                stats["query_vectors"] += 1

            await target_session.commit()
            logger.info(f"Exported {stats['query_vectors']} query vectors")

            # 7. Export Answers (depends on Queries)
            logger.info("Exporting answers...")

            if skip_orphaned_records:
                # Only export answers with valid query references (simple SQLAlchemy way)
                answers_result = await source_session.execute(
                    select(Answer).where(Answer.query_id.in_(select(Query.id)))
                )
                answers = answers_result.scalars().all()
                logger.info(
                    f"Found {len(answers)} valid answers (with existing queries)"
                )
            else:
                # Export all answers (may fail on orphaned records)
                answers_result = await source_session.execute(select(Answer))
                answers = answers_result.scalars().all()

            for i, answer in enumerate(answers):
                new_answer = Answer(
                    id=answer.id,
                    query_id=answer.query_id,
                    text=answer.text,
                    temperature=answer.temperature,
                )
                await target_session.merge(new_answer)
                if i % commit_every == 0:
                    await target_session.commit()
                    logger.info(f"Exported {stats['answers']} answers")
                stats["answers"] += 1

            await target_session.commit()
            logger.info(f"Exported {stats['answers']} answers")

            if skip_orphaned_records:
                # Calculate how many orphaned records were skipped (simple way)
                total_answers_result = await source_session.execute(
                    select(func.count(Answer.id))
                )
                total_answers = total_answers_result.scalar()
                stats["skipped_orphaned"] = total_answers - stats["answers"]
                if stats["skipped_orphaned"] > 0:
                    logger.warning(
                        f"Skipped {stats['skipped_orphaned']} orphaned answers"
                    )

            # 8. Export EvaluationSets (no dependencies)
            logger.info("Exporting evaluation sets...")
            eval_sets_result = await source_session.execute(select(EvaluationSet))
            eval_sets = eval_sets_result.scalars().all()
            for i, eval_set in enumerate(eval_sets):
                new_eval_set = EvaluationSet(
                    id=eval_set.id,
                    datetime=eval_set.datetime,
                    dataset=eval_set.dataset,
                    metrics=eval_set.metrics,
                    settings=eval_set.settings,
                )
                await target_session.merge(new_eval_set)
                if i % commit_every == 0:
                    await target_session.commit()
                    logger.info(f"Exported {stats['evaluation_sets']} evaluation sets")
                stats["evaluation_sets"] += 1

            await target_session.commit()
            logger.info(f"Exported {stats['evaluation_sets']} evaluation sets")

            # 9. Export RagResponseDB (depends on EvaluationSets and Queries)
            if "rag_responses" not in exclude_tables:
                logger.info("Exporting RAG responses...")
                rag_responses_result = await source_session.execute(
                    select(RagResponseDB)
                )
                rag_responses = rag_responses_result.scalars().all()
                for i, rag_response in enumerate(rag_responses):
                    new_rag_response = RagResponseDB(
                        id=rag_response.id,
                        evaluation_set_id=rag_response.evaluation_set_id,
                        retrieval_result=rag_response.retrieval_result,
                        query_id=rag_response.query_id,
                    )
                    await target_session.merge(new_rag_response)
                    if i % commit_every == 0:
                        await target_session.commit()
                        logger.info(f"Exported {stats['rag_responses']} RAG responses")
                    stats["rag_responses"] += 1

                await target_session.commit()
                logger.info(f"Exported {stats['rag_responses']} RAG responses")

        # Fix PostgreSQL sequences after migration
        logger.info("Fixing PostgreSQL sequences...")
        async with target_engine.begin() as conn:
            tables_with_sequences = [
                ("docs", "docs_id_seq"),
                ("chunks", "chunks_id_seq"),
                ("chunk_vectors", "chunk_vectors_id_seq"),
                ("prompts", "prompts_id_seq"),
                ("queries", "queries_id_seq"),
                ("query_vectors", "query_vectors_id_seq"),
                ("answers", "answers_id_seq"),
                ("evaluation_sets", "evaluation_sets_id_seq"),
                ("rag_responses", "rag_responses_id_seq"),
            ]

            for table_name, sequence_name in tables_with_sequences:
                try:
                    result = await conn.execute(
                        text(f"SELECT COALESCE(MAX(id), 0) FROM {table_name}")
                    )
                    max_id = result.scalar()
                    if max_id > 0:
                        await conn.execute(
                            text(f"SELECT setval('{sequence_name}', {max_id}, true)")
                        )
                        logger.info(f"Reset {sequence_name} to start from {max_id + 1}")
                except Exception as e:
                    logger.warning(f"Could not reset {sequence_name}: {e}")

    except Exception as e:
        logger.error(f"Error during export: {e}")
        raise
    finally:
        await source_engine.dispose()
        await target_engine.dispose()

    logger.info(f"Export completed successfully. Stats: {stats}")
    return stats
