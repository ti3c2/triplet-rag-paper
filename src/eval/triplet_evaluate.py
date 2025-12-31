import json
import logging
from datetime import datetime as dt
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func, or_, select
from tqdm import tqdm

from ..config.settings import settings
from ..config.utils import execute_with_semaphore
from ..rag.models import (
    EvaluationEmbeddingData,
    EvaluationMetrics,
    EvaluationQueryData,
    EvaluationResult,
    GroundTruthItem,
    RagResponse,
    RetrievalResult,
)
from ..rag.pipeline import RAGPipeline
from ..rag.triplet_store import (
    build_result_from_triplet,
    fetch_triplets_by_eval_query_ids,
    find_triplets_for_retrieval_by_eval_query,
    index_triplets_by_chunk,
    index_triplets_by_eval_query,
    load_triplets_from_json,
)
from ..store_rel.entry import get_db
from ..store_rel.schema import (
    Answer,
    Chunk,
    Doc,
    EvaluationSet,
    Query,
    QueryVector,
    RagResponseDB,
)
from .evaluate import calculate_metrics

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)


async def run_triplet_evaluation(
    dataset: str = settings.rag_dataset,
    batch_size: int = 10,
    ks: List[int] = settings.rag_retrieval_ks,
    rag_llm: str = settings.rag_generation_model,
    qgen_llm: str = settings.question_generation_model,
    emb_model: str = settings.openai_embedding_model,
    retrieval_sources: List[Literal["chunk", "query"]] = settings.rag_retrieval_sources,
    rag_qa_type: str = settings.question_generation_mode,
    deduplication_factor: int = settings.rag_deduplication_factor,
    max_queries: Optional[int] = settings.rag_max_rows,
    generate_answer_retrieval: bool = False,  # NOTE: Disabled for using retrieval only pipeline
    generate_answer_triplets: bool = True,
    write_json: bool = settings.rag_write_json,
    write_txt: bool = settings.rag_write_txt,
    write_spreadsheet: bool = False,  # keep spreadsheet disabled by default for triplet eval
    spreadsheet_sheet_name: Optional[str] = None,
) -> EvaluationResult:
    """Evaluate using dataset-origin questions and an index built from eval-origin queries.

    For each dataset question, we attempt to locate a matching triplet (DB or JSON)
    and use its contexts (and reference QA when available) to generate the final answer.
    Triplet contexts are persisted as the retrieval_result for transparency.
    """
    logger.info(f"Settings (triplet eval):\n{settings.model_dump_json(indent=2)}")
    logger.info(
        f"Preparing dataset-origin questions for triplet evaluation on {dataset}"
    )
    datetime_str = dt.now().strftime("%Y%m%d_%H%M%S")

    async with get_db() as session:
        # Fetch dataset-origin queries and their embeddings
        query_sql = (
            select(
                Query.id,
                Query.text,
                Query.text_hash,
                Query.chunk_id,
                Query.doc_id,
                QueryVector.vector,
                Query.origin,
                Doc.meta.label("doc_meta"),
                func.coalesce(Doc.title, "").label("doc_title"),
            )
            .join(QueryVector, Query.id == QueryVector.parent_id)
            .join(Chunk, Query.chunk_id == Chunk.id)
            .join(Doc, Chunk.doc_id == Doc.id)
            .where(
                Doc.dataset == dataset,
                or_(
                    Query.llm.is_(None),
                    Query.origin.in_(settings.eval_whitelist_origins),
                ),
                QueryVector.emb_model == emb_model,
                Query.origin == "dataset",
            )
            .limit(max_queries)
            .order_by(Query.id)
        )
        results = await session.execute(query_sql)
        rows = results.all()
        if not rows:
            raise ValueError(
                f"No dataset-origin queries found for dataset {dataset} with emb model {emb_model}."
            )

        queries_data: List[EvaluationQueryData] = []
        embeddings_data: List[EvaluationEmbeddingData] = []
        for row in rows:
            qd = EvaluationQueryData(**row._asdict())
            queries_data.append(qd)
            embeddings_data.append(EvaluationEmbeddingData(vector=row.vector))

        logger.info(f"Ready to evaluate {len(queries_data)} dataset-origin queries")

        # Prepare pipeline with index built from eval-origin queries
        logger.info("Preparing pipeline and building index from eval-origin queries")
        pipeline = RAGPipeline(
            k=max(ks),
            emb_model=emb_model,
            dataset=dataset,
            rag_llm=rag_llm,
            qgen_llm=qgen_llm,
            qa_type=rag_qa_type,
            retrieval_sources=retrieval_sources,
            retrieval_query_origins=["eval"],
            deduplication_factor=deduplication_factor,
            generate_answer=generate_answer_retrieval,
        )
        await pipeline.fit(session)
        logger.info("Pipeline fitted")

        # Load triplets: JSON-backed or DB-backed (from eval-origin queries used in index)
        triplet_map = {}
        if settings.rag_triplet_json_path:
            triplet_map = load_triplets_from_json(settings.rag_triplet_json_path)
        else:
            eval_query_ids = [
                item.query_id
                for item in getattr(pipeline.retriever, "fit_items", [])
                if hasattr(item, "query_id")
            ]
            triplet_map = await fetch_triplets_by_eval_query_ids(
                session, eval_query_ids
            )

        triplet_chunk_index = index_triplets_by_chunk(triplet_map)
        triplet_eval_index = triplet_map
        logger.info(
            f"Triplets ready: {len(triplet_map)} (indexed by {len(triplet_chunk_index)} chunk ids)"
        )

        # Evaluation bookkeeping
        results_dirname = (
            f"{datetime_str}_triplet_"
            f"{dataset}_"
            f"{emb_model.split('/')[-1]}_"
            f"{rag_llm.split('/')[-1]}_"
            f"{qgen_llm.split('/')[-1]}"
        )

        evaluation_set = EvaluationSet(
            dataset=dataset,
            datetime=dt.now(),
            metrics=None,
            settings=dict(
                results_dirname=results_dirname, **settings.model_dump(mode="json")
            ),
        )
        session.add(evaluation_set)
        await session.flush()
        await session.refresh(evaluation_set)
        evaluation_set_id = evaluation_set.id

        all_rag_responses: List[RagResponse] = []
        ground_truths = []

        # Pre-fetch chunks and answers maps
        query_ids = [q.id for q in queries_data]
        chunk_ids_present = [q.chunk_id for q in queries_data if q.chunk_id]

        # TODO: Refactor this mess
        chunk_texts_map = {}
        if chunk_ids_present:
            in_clause_batch_size = 10000
            for start_index in range(0, len(chunk_ids_present), in_clause_batch_size):
                batch_chunk_ids = chunk_ids_present[
                    start_index : start_index + in_clause_batch_size
                ]
                batch_chunks_result = await session.execute(
                    select(Chunk.id, Chunk.text).where(Chunk.id.in_(batch_chunk_ids))
                )
                for row in batch_chunks_result.all():
                    chunk_texts_map[row.id] = row.text

        answer_texts_map = {}
        if query_ids:
            in_clause_batch_size = 10000
            for start_index in range(0, len(query_ids), in_clause_batch_size):
                batch_query_ids = query_ids[
                    start_index : start_index + in_clause_batch_size
                ]
                batch_answers_result = await session.execute(
                    select(Answer.id, Answer.query_id, Answer.text).where(
                        Answer.query_id.in_(batch_query_ids)
                    )
                )
                for row in batch_answers_result.all():
                    answer_texts_map[row.query_id] = row.text

        # Run in batches
        tasks = []
        for i in tqdm(range(0, len(queries_data), batch_size)):
            batch_q = queries_data[i : i + batch_size]
            batch_emb = embeddings_data[i : i + batch_size]
            questions = [str(q.text) for q in batch_q]
            emb = np.array([e.vector for e in batch_emb])
            if np.isnan(emb).any():
                logger.warning(f"Found null embeddings in batch {i // batch_size + 1}")
                emb = None

            # Retrieve once with max k
            retrieval_results = pipeline.retrieve(
                emb
                if not np.isnan(emb).any()
                else await pipeline.embed_queries(questions)
            )

            # For each query, for each k, build response using triplet if matched
            for idx, (query, retres) in enumerate(zip(batch_q, retrieval_results)):
                # Build ground truth entry for this query (single-k; reused across ks)
                chunk_ids = [query.chunk_id] if query.chunk_id else []
                chunk_texts = (
                    [chunk_texts_map.get(query.chunk_id, "")]
                    if query.chunk_id
                    else [""]
                )
                ground_truths.append(
                    [
                        GroundTruthItem(
                            doc_ids=[int(query.doc_id)],
                            doc_titles=[str(getattr(query, "doc_title", ""))],
                            chunk_ids=chunk_ids,
                            chunk_texts=chunk_texts,
                            query=str(query.text),
                            answer=answer_texts_map.get(query.id),
                        )
                    ]
                )

                # Persist one retrieval_result per query (we'll use k=ks[0] for storage)
                stored_retrieval = None

                # Attempt triplet match by overlapping chunk ids
                # Prefer deterministic eval-query match; fallback to chunk-overlap
                matched_triplets = find_triplets_for_retrieval_by_eval_query(
                    triplet_eval_index, retres
                )

                # For each k, prepare response
                for k in ks:
                    if matched_triplets and any(
                        t is not None for t in matched_triplets
                    ):
                        triplets = [t for t in matched_triplets if t is not None]
                        logger.debug(
                            f"Generating response for query {query.id} with {len(triplets)} triplets"
                        )
                        tasks.append(
                            pipeline.generator.generate_triplets(
                                question=str(query.text),
                                triplets=triplets,  # type: ignore
                                k=k,
                                dataset=(
                                    retres.items[0].metadata.dataset
                                    if retres.items
                                    else dataset
                                ),
                                emb_model=(
                                    retres.items[0].metadata.emb_model
                                    if retres.items
                                    else emb_model
                                ),
                                generate_answer=generate_answer_triplets,
                            )
                        )
                    else:
                        # Fallback to top-k slices
                        rr = retres
                        if len(rr.items) > k:
                            rr = RetrievalResult(items=rr.items[:k])
                        tasks.append(
                            pipeline.generate(
                                question=str(query.text),
                                retrieval_result=rr,
                                generate_answer=True,
                            )
                        )
        responses = await execute_with_semaphore(tasks)
        all_rag_responses.extend(responses)

        logger.info("Calculating metrics for each k value")

        triplet_ks = list(set(response.n_context for response in all_rag_responses))
        metrics_result = calculate_metrics(
            triplet_ks,
            all_rag_responses,
            ground_truths,
            dataset=dataset,
            deduplication_factor=deduplication_factor,
        )

        evaluation_result = EvaluationResult(
            rag_responses=all_rag_responses,
            ground_truth=ground_truths,
            metrics=metrics_result.metrics,
            experiment_name=results_dirname,
            eval_llm=settings.eval_llm,
            eval_embedding_model=settings.eval_embedding_model,
        )

        evaluation_set.set_metrics(evaluation_result.model_dump(include={"metrics"}))
        await session.commit()

    logger.info("Saving results")
    dirpath = settings.path_eval / results_dirname
    dirpath.mkdir(parents=True, exist_ok=True)

    if write_spreadsheet:
        from ..config.spreadsheets import write_evaluation_to_spreadsheet
        try:
            retrieval_sources_str = (
                "+".join(retrieval_sources)
                if len(retrieval_sources) > 1
                else retrieval_sources[0]
            )
            write_evaluation_to_spreadsheet(
                evaluation_result=evaluation_result,
                experiment_name=results_dirname,
                sheet_name=spreadsheet_sheet_name,
                retrieval_sources=retrieval_sources_str,
            )
            logger.info("Successfully wrote evaluation results to spreadsheet")
        except Exception as e:
            logger.error(f"Failed to write to spreadsheet: {str(e)}", exc_info=True)

    if write_json:
        with open(dirpath / f"{datetime_str}.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_result.model_dump(), f, indent=2, ensure_ascii=False)

    if write_txt:
        with open(dirpath / f"{datetime_str}.txt", "w", encoding="utf-8") as f:
            f.write(evaluation_result.format_str())

    df_metrics = pd.DataFrame(
        [m.model_dump(exclude_none=True) for m in evaluation_result.metrics]
    )
    df_metrics.to_csv(dirpath / f"{datetime_str}_metrics.csv", index=False)
    logger.info(f"Results saved to {dirpath.relative_to(settings.path_root)}")

    return evaluation_result
