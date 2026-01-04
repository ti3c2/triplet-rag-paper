import asyncio
import logging
from functools import partial
from typing import Awaitable, Callable, List, Literal, Optional, Union

import numpy as np
from sqlalchemy import or_, select

from ..config.prompts import Answers, QAPair, QAPairs, Questions
from ..config.settings import settings
from ..config.utils import (
    create_chat_completion,
    create_embeddings,
    crop_text_to_max_tokens,
    execute_with_semaphore,
    get_text_hash,
)
from ..store_rel.entry import get_db
from ..store_rel.schema import (
    Answer,
    Chunk,
    ChunkVector,
    Doc,
    Prompt,
    Query,
    QueryVector,
    get_or_create_item,
    get_prompt,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)


async def generate_qa_sequential_batch(
    chunk_text: str,
    prompt: str,
    prompt_answer: str = settings.prompt_sequential_answer,
    qgen_temperature: float = settings.question_generation_temperature,
    answer_temperature: float = settings.answer_generation_temperature,
    chat_completion_func: Awaitable = create_chat_completion,
) -> QAPairs:
    logger.debug(f"Prompt: {prompt}. Chunk text: {chunk_text}")

    if "sequential" not in (pname := settings.question_generation_prompt_name):
        raise ValueError(f"Sequential generation is not supported for prompt {pname}")

    logger.info(f"Generating questions")
    questions_resp = await chat_completion_func(
        messages=[
            {
                "role": "system",
                "content": settings.prompt_system_qgen,
            },
            {
                "role": "user",
                "content": prompt.format(context=chunk_text),
            },
        ],
        temperature=qgen_temperature,
        response_format=Questions,
    )
    questions = (
        questions_resp.choices[0].message.parsed
        if questions_resp
        else Questions(questions=[])
    )
    # NOTE: Now answers are generated for all the questions at once
    # TODO: Consider approach with 1q-1a generation in a single request
    questions_context = "\n".join(
        [
            f"Question {i+1}: {question}"
            for i, question in enumerate(questions.questions)
        ]
    )
    logger.debug(f"Questions: {questions}")
    logger.info(f"Generating answers")
    answers_resp = await chat_completion_func(
        messages=[
            {
                "role": "user",
                "content": prompt_answer.format(
                    context=chunk_text,
                    context_questions=questions_context,
                ),
            }
        ],
        temperature=answer_temperature,
        response_format=Answers,
    )
    answers = (
        answers_resp.choices[0].message.parsed if answers_resp else Answers(answers=[])
    )
    qa_pairs = QAPairs(
        items=[
            QAPair(question=question, answer=answer)
            for question, answer in zip(questions.questions, answers.answers)
        ]
    )
    logger.debug(f"QA Pairs: {qa_pairs}")
    logger.info(f"Generated {len(qa_pairs.items)} QA pairs")
    return qa_pairs


async def generate_qa_sequential(
    chunk_text: str,
    prompt: str,
    prompt_answer: str = settings.prompt_sequential_answer,
    qgen_temperature: float = settings.question_generation_temperature,
    answer_temperature: float = settings.answer_generation_temperature,
    chat_completion_func: Awaitable = create_chat_completion,
) -> QAPairs:
    pass


async def generate_qa_single(
    chunk_text: str,
    prompt: str,
    qgen_temperature: float = settings.question_generation_temperature,
    chat_completion_func: Awaitable = create_chat_completion,
) -> QAPairs:
    logger.info(f"Generating questions using single request")
    logger.debug(f"Prompt: {prompt}. Chunk text: {chunk_text}")

    if "sequential" in (pname := settings.question_generation_prompt_name):
        raise ValueError(f"Sequential generation is not supported for prompt {pname}")

    response = await chat_completion_func(
        messages=[
            {
                "role": "system",
                "content": settings.prompt_system_qgen,
            },
            {
                "role": "user",
                "content": prompt.format(context=chunk_text),
            },
        ],
        temperature=qgen_temperature,
        response_format=QAPairs,
    )
    llm_output = response.choices[0].message.parsed or QAPairs(items=[])
    logger.debug(f"QA Pairs: {llm_output}")
    logger.info(f"Generated {len(llm_output.items)} QA pairs")
    return llm_output


async def generate_questions_for_doc(
    doc_id: int,
    chunks: list,
    prompt: str,
    prompt_id: int,
    session,
    model: str = settings.question_generation_model,
    qa_type: Literal["qa", "qc"] = settings.question_generation_mode,
    sequential: str = settings.qa_sequential_generation,
    max_concurrency: int = settings.openai_chat_completion_max_concurrency,
    origin: Literal["dataset", "quote", "eval"] = settings.rag_retrieval_query_origins[
        0
    ],
) -> list[Query | Answer]:
    """Generate questions for a document using LLM with structured output."""
    db_items = []
    chunk_texts = [chunk["text"] for chunk in chunks]
    generation_function = dict(
        single=generate_qa_single,
        sequential=generate_qa_sequential,
        sequential_batch=generate_qa_sequential_batch,
    ).get(sequential, generate_qa_single)
    logger.info(f"Using generation function: {generation_function.__name__}")
    chat_completion_func = partial(  # Predefine arguments for later use
        create_chat_completion,
        client=settings.get_openai_client(),
        parse=True,
        model=model,
        max_completion_tokens=settings.question_generation_max_completion_tokens,
    )
    try:
        tasks = [
            generation_function(
                chunk_text=chunk_text,
                prompt=prompt,
                chat_completion_func=chat_completion_func,
            )
            for chunk_text in chunk_texts
        ]
        # List of QAPairs
        responses = await execute_with_semaphore(tasks, max_concurrency)

        # TODO: Refactor this to separate code for query generation and adding them to the database
        for response, chunk in zip(responses, chunks):
            if response is None:
                logger.error(f"Error generating questions for doc {doc_id}")
                return []
            llm_output = response or QAPairs(items=[])
            logger.debug(f"LLM Output: {llm_output}")
            qa_pairs = llm_output.items
            for qa in qa_pairs:
                query_texts = []
                modes = settings.question_generation_mode.split(",")
                for mode in modes:
                    if mode == "q":
                        query_texts.append(qa.question)
                    elif mode in ["qa", "qc"]:
                        if qa_type == "qa":
                            answer = qa.answer
                        elif qa_type == "qc":
                            answer = chunk["text"]
                        else:
                            raise ValueError(f"Invalid qa_type: {qa_type}")
                        query_texts.append(
                            f"{qa.question}{settings.qa_separator}{answer}"
                        )

                for query_text, qa_type in zip(query_texts, modes):
                    chunk_id = chunk["chunk_id"]
                    query, _ = await get_or_create_item(
                        session,
                        Query,
                        force_create=True,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        text=query_text,
                        llm=model,
                        temperature=settings.question_generation_temperature,
                        prompt_id=prompt_id,
                        qa_type=qa_type,
                        origin=origin,
                    )
                    logger.debug(f"Added query `{query.text}, {query.llm}`")
                    query_id = int(query.id)
                    answer = Answer(
                        query_id=query_id,
                        text=qa.answer,
                        temperature=settings.question_generation_temperature,
                    )
                    session.add(answer)
                    db_items.append(query)
                    db_items.append(answer)
            logger.info(f"Added {len(qa_pairs)} queries for doc {doc_id}")
        return db_items
    except Exception as e:
        logger.error(f"Error processing doc {doc_id}: {e}", exc_info=True)
        return []


async def _process_single_doc_with_session(
    doc_info: dict,
    prompt_text: str,
    prompt_name: str,
    model: str,
    max_concurrency: int,
    origin: Literal["dataset", "quote", "eval"],
) -> None:
    """Process a single document with its own database session."""
    async with get_db() as session:
        # Get or create prompt for this session
        prompt_db, _ = await get_or_create_item(
            session,
            Prompt,
            text=prompt_text,
            name=prompt_name,
        )
        await session.refresh(prompt_db)
        prompt_db_id = int(prompt_db.id)

        # Generate questions for this document
        await generate_questions_for_doc(
            doc_id=doc_info["doc_id"],
            chunks=doc_info["chunks"],
            prompt=prompt_text,
            prompt_id=prompt_db_id,
            session=session,
            model=model,
            max_concurrency=max_concurrency,
            origin=origin,
        )
        # Commit immediately after processing this document
        await session.commit()
        logger.info(f"Committed doc {doc_info['doc_id']}")


async def generate_questions_for_docs(
    docs: Optional[List[Doc]] = None,
    dataset: Optional[str] = settings.rag_dataset,
    docs_commited: int = settings.question_generation_docs_commited,
    max_concurrency: int = settings.openai_chat_completion_max_concurrency,
    max_docs: int = -1,
    model: str = settings.question_generation_model,
    prompt_name: Optional[str] = settings.question_generation_prompt_name,
    origin: Literal["dataset", "quote", "eval"] = settings.rag_retrieval_query_origins[
        0
    ],
):
    logger.info(f"Settings:\n{settings.model_dump_json(indent=2)}")
    if not await settings.ping_model(settings.question_generation_model):
        raise ValueError(
            f"Question generation model {settings.question_generation_model} is not available"
        )

    async with get_db() as session:
        prompt = settings.get_prompt(prompt_name)
        # If prompt text is empty, fetch it from the DB using the resolved name
        if not prompt.text:
            prompt_db = await get_prompt(session, prompt.name)
            prompt.text = str(prompt_db.text)
        prompt_hash = get_text_hash(prompt.text)

        # Get doc IDs to process
        if docs is None:
            docs_result = await session.execute(
                select(Doc.id, Doc.dataset, Doc.is_virtual).where(
                    Doc.dataset == dataset
                )
            )
            doc_rows = docs_result.all()
        else:
            # If docs were provided, get their IDs and avoid lazy loading
            doc_ids = [doc.id for doc in docs]
            docs_result = await session.execute(
                select(Doc.id, Doc.dataset, Doc.is_virtual).where(Doc.id.in_(doc_ids))
            )
            doc_rows = docs_result.all()

        if not doc_rows:
            logger.warning(f"No documents found for dataset {dataset}")
            return

        doc_ids = [row.id for row in doc_rows]

        # Pre-fetch all queries for these documents
        queries_result = await session.execute(
            select(Query.doc_id, Query.llm)
            .join(Prompt, Query.prompt_id == Prompt.id)
            .where(
                Query.doc_id.in_(doc_ids),
                Prompt.text_hash == prompt_hash,
                Query.origin == origin,
            )
        )
        queries_by_doc = {}
        for row in queries_result.all():
            if row.doc_id not in queries_by_doc:
                queries_by_doc[row.doc_id] = []
            queries_by_doc[row.doc_id].append({"llm": str(row.llm)})

        # Pre-fetch all chunks for these documents
        chunks_result = await session.execute(
            select(Chunk.doc_id, Chunk.id, Chunk.text).where(Chunk.doc_id.in_(doc_ids))
        )
        chunks_by_doc = {}
        for row in chunks_result.all():
            if row.doc_id not in chunks_by_doc:
                chunks_by_doc[row.doc_id] = []
            chunks_by_doc[row.doc_id].append(
                {
                    "text": str(row.text),
                    "chunk_id": int(row.id),
                }
            )

        # Build doc_data without lazy loading
        doc_data = []
        for row in doc_rows:
            doc_info = {
                "doc_id": int(row.id),
                "dataset": str(row.dataset),
                "is_virtual": bool(row.is_virtual),
                "queries": queries_by_doc.get(row.id, []),
                "chunks": chunks_by_doc.get(row.id, []),
            }
            doc_data.append(doc_info)

        if max_docs > 0:
            doc_data = doc_data[:max_docs]

        # Use the previously resolved prompt (code or DB)
        logger.info(f"Using prompt: {prompt.name}\n{prompt.text}")

        # Filter documents that need processing
        docs_to_process = []
        for doc_info in doc_data:
            doc_dataset = doc_info["dataset"]
            doc_is_virtual = doc_info["is_virtual"]
            doc_queries = doc_info["queries"]
            doc_llms = [q["llm"] for q in doc_queries]

            # Skip non-virtual docs in multihoprag dataset
            if doc_dataset == "multihoprag" and not doc_is_virtual:
                continue

            # Skip docs that already have queries by this model
            if model in doc_llms:
                continue

            docs_to_process.append(doc_info)

        total_docs = len(docs_to_process)
        logger.info(f"Processing {total_docs} docs for LLM question generation...")

        # Process documents in batches with concurrent sessions
        # Cap batch size to available DB connections to prevent pool exhaustion
        max_pool_connections = settings.sql_pool_size + settings.sql_max_overflow
        batch_size = docs_commited if docs_commited > 0 else total_docs
        # Use smaller of: requested batch size or available connections (with safety margin)
        batch_size = min(
            batch_size, max_pool_connections - 5
        )  # Reserve 5 for other queries

        logger.info(
            f"Batch size: {batch_size} (limited by pool size: {max_pool_connections})"
        )

        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            batch = docs_to_process[batch_start:batch_end]

            logger.info(
                f"Processing batch: docs {batch_start+1}-{batch_end}/{total_docs}"
            )

            # Create tasks for concurrent processing, each with its own session
            tasks = [
                _process_single_doc_with_session(
                    doc_info=doc_info,
                    prompt_text=prompt.text,
                    prompt_name=prompt.name,
                    model=model,
                    max_concurrency=max_concurrency,
                    origin=origin,
                )
                for doc_info in batch
            ]

            # Execute batch concurrently with semaphore limiting
            await execute_with_semaphore(tasks, max_concurrency)
            logger.info(
                f"Completed batch: docs {batch_start+1}-{batch_end}/{total_docs}"
            )


async def embed_db_items(
    db_items: Optional[Union[List[Query], List[Chunk]]] = None,
    db_items_types: List[Literal["query", "chunk"]] = ["query", "chunk"],
    include_default_queries: bool = True,
    llm: str = settings.question_generation_model,
    emb_model: str = settings.openai_embedding_model,
    dataset: str = settings.rag_dataset,
    batch_size: int = settings.openai_embedding_batch_size,
    max_items: int = -1,
    qa_type: str = settings.question_generation_mode,
    ignore_missing_prompt: bool = True,
) -> Union[List[QueryVector], List[ChunkVector]]:
    logger.info(f"Settings:\n{settings.model_dump_json(indent=2)}")
    if not await settings.ping_model(emb_model):
        raise ValueError(f"Embedding model {emb_model} is not available")

    client = settings.get_openai_client(embedding=True)
    all_vector_objects = []

    prompt = settings.question_generation_default_prompt
    logger.info(f"Using prompt: {prompt.name}\n{prompt.text}")
    async with get_db() as session:
        prompt_db = await get_prompt(session, prompt.name, ignore_missing_prompt)
        if prompt_db is None:
            logger.warning(f"Prompt {prompt.name} not found in the database")
            prompt_id = -1
        else:
            prompt_id = prompt_db.id

        # Prepare items for processing
        items_to_process = []
        if db_items is None:
            logger.info(f"Getting all queries for dataset {dataset} with model {llm}")
            if "query" in db_items_types:
                db_items_result_q = await session.execute(
                    select(Query.id, Query.text)
                    .join(Doc)
                    .where(
                        # include both generated and initial queries
                        or_(
                            Query.llm == llm,
                            Query.llm.is_(None) if include_default_queries else False,
                        ),
                        Doc.dataset == dataset,
                        or_(
                            Query.qa_type.in_(
                                qa_type.split(",") + ["q"]
                                if include_default_queries
                                else []
                            ),
                        ),
                        or_(
                            Query.prompt_id == prompt_id,
                            Query.prompt_id.is_(None),
                        ),
                    )
                )
                # Convert to (id, text, type) tuples
                items_to_process.extend(
                    [(row.id, row.text, "query") for row in db_items_result_q.all()]
                )
            if "chunk" in db_items_types:
                db_items_result_c = await session.execute(
                    select(Chunk.id, Chunk.text).join(Doc).where(Doc.dataset == dataset)
                )
                items_to_process.extend(
                    (row.id, row.text, "chunk") for row in db_items_result_c.all()
                )
        else:
            # Convert mixed Query/Chunk objects to tuples
            items_to_process = []
            for item in db_items:
                if isinstance(item, Query):
                    items_to_process.append((item.id, item.text, "query"))
                elif isinstance(item, Chunk):
                    items_to_process.append((item.id, item.text, "chunk"))
                else:
                    raise ValueError(f"Unsupported item type: {type(item)}")

        if max_items > 0:
            items_to_process = items_to_process[:max_items]

        if not items_to_process:
            logger.warning("No items to process")
            await client.close()
            return all_vector_objects

        logger.info(f"Embedding {len(items_to_process)} items")

        # Pre-fetch all existing vector IDs for efficiency
        query_ids = [
            item_id
            for item_id, _, item_type in items_to_process
            if item_type == "query"
        ]
        chunk_ids = [
            item_id
            for item_id, _, item_type in items_to_process
            if item_type == "chunk"
        ]

        existing_vector_ids = set()

        # Filter out items that already have vectors

        async def parent_ids_in_batch(
            ids: list[int], table: QueryVector | ChunkVector
        ) -> list[int]:
            result = await session.execute(
                select(table.parent_id).where(
                    table.parent_id.in_(ids),
                    table.emb_model == emb_model,
                )
            )
            return result.all()

        # Process queries and chunks IDs in batches
        # Batch size to stay under PostgreSQL's 32767 parameter limit
        QUERY_BATCH_SIZE = 30000
        tasks = []
        if query_ids:
            for i in range(0, len(query_ids), QUERY_BATCH_SIZE):
                batch_query_ids = query_ids[i : i + QUERY_BATCH_SIZE]
                tasks.append(parent_ids_in_batch(batch_query_ids, QueryVector))

        # Process chunk IDs in batches
        if chunk_ids:
            for i in range(0, len(chunk_ids), QUERY_BATCH_SIZE):
                batch_chunk_ids = chunk_ids[i : i + QUERY_BATCH_SIZE]
                tasks.append(parent_ids_in_batch(batch_chunk_ids, ChunkVector))

        for result in await asyncio.gather(*tasks):
            existing_vector_ids.update(ids for ids, in result)

        # Finally, do the filtering
        items_to_embed = [
            (item_id, text, item_type)
            for item_id, text, item_type in items_to_process
            if item_id not in existing_vector_ids
        ]

        if not items_to_embed:
            logger.info("All items already have vectors, skipping embedding")
            await client.close()
            return all_vector_objects

        logger.info(
            f"Embedding {len(items_to_embed)} new items (skipped {len(items_to_process) - len(items_to_embed)} existing)"
        )

        # Process in batches
        for i in range(0, len(items_to_embed), batch_size):
            batch = items_to_embed[i : i + batch_size]
            item_ids = [item[0] for item in batch]
            texts = [str(item[1]) for item in batch]
            item_types = [item[2] for item in batch]

            # Crop texts to max tokens
            texts = [crop_text_to_max_tokens(text, emb_model) for text in texts]

            try:
                logger.debug(
                    f"Requesting embeddings for batch {i//batch_size + 1} with {len(batch)} items"
                )
                response = await create_embeddings(
                    client,
                    model=emb_model,
                    input=texts,
                )

                if response is None:
                    logger.error(
                        f"Embedding API returned None for batch {i//batch_size + 1}"
                    )
                    continue

                vector_objects = []
                for item_id, item_type, embedding_data in zip(
                    item_ids, item_types, response.data
                ):
                    vector_table = QueryVector if item_type == "query" else ChunkVector
                    vector_obj = vector_table(
                        parent_id=item_id,
                        vector=np.array(embedding_data.embedding, dtype=np.float32),
                        emb_model=emb_model,
                    )
                    vector_objects.append(vector_obj)

                if vector_objects:
                    all_vector_objects.extend(vector_objects)
                    session.add_all(vector_objects)
                    await session.commit()
                    logger.info(
                        f"Embedded batch {i//batch_size + 1}/{(len(items_to_embed)-1)//batch_size + 1} with {len(vector_objects)} vectors"
                    )

            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {e}")
                continue

    await client.close()
    return all_vector_objects
