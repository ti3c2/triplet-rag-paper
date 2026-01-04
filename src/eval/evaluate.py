import hashlib
import json
import logging
from datetime import datetime as dt
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sqlalchemy import and_, func, or_, select
from tqdm import tqdm

from ..config.settings import settings
from ..config.spreadsheets import write_evaluation_to_spreadsheet
from ..rag.models import (
    EvaluationEmbeddingData,
    EvaluationMetrics,
    EvaluationQueryData,
    EvaluationResult,
    GroundTruthItem,
    RagResponse,
    RetrievalResult,
    Triplet,
    TripletContext,
    Triplets,
)
from ..rag.pipeline import RAGPipeline
from ..store_rel.entry import get_db
from ..store_rel.schema import (
    Answer,
    Chunk,
    Doc,
    EvaluationSet,
    Query,
    QueryTriplet,
    QueryVector,
    RagResponseDB,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)


def res2ids(
    retres: RetrievalResult,
    type: Literal["doc", "chunk"] = "doc",
) -> List[int]:
    if type == "doc":
        return [item.metadata.doc_id for item in retres.items]
    elif type == "chunk":
        return [item.metadata.chunk_id for item in retres.items]
    else:
        raise ValueError(f"Invalid type: {type}")


def context_accuracy(
    k: int,
    retrieval_results: List[RetrievalResult],
    ground_truths: List[List[GroundTruthItem]],
) -> float:
    metric = 0
    for retrieval_result, ground_truth in zip(retrieval_results, ground_truths):
        retrieved_ids = res2ids(retrieval_result, type="chunk")
        relevant_ids = [gt.chunk_id for gt in ground_truth]
        logger.debug(f"Retrieved/Relevant: {retrieved_ids[:k]} / {relevant_ids}")
        if set(retrieved_ids[:k]).intersection(set(relevant_ids)):
            metric += 1
    return 0 if len(retrieval_results) == 0 else metric / len(retrieval_results)


def title_accuracy(
    k: int,
    retrieval_results: List[RetrievalResult],
    ground_truths: List[List[GroundTruthItem]],
) -> float:
    metric = 0
    for retrieval_result, ground_truth in zip(retrieval_results, ground_truths):
        retrieved_titles = retrieval_result.doc_titles
        relevant_titles = [gt.doc_title for gt in ground_truth]
        logger.debug(f"Retrieved/Relevant: {retrieved_titles[:k]} / {relevant_titles}")
        if set(retrieved_titles[:k]).intersection(set(relevant_titles)):
            metric += 1
    return 0 if len(retrieval_results) == 0 else metric / len(retrieval_results)


# TODO: Check multihop metrics
def full_match_accuracy(
    k: int,
    retrieval_results: List[RetrievalResult],
    ground_truths: List[List[GroundTruthItem]],
) -> float:
    metric = 0
    for retrieval_result, ground_truth in zip(retrieval_results, ground_truths):
        retrieved_ids = res2ids(retrieval_result)
        relevant_ids = [doc_id for gt in ground_truth for doc_id in gt.doc_ids]
        logger.debug(f"Retrieved/Relevant: {retrieved_ids[:k]} / {relevant_ids}")
        if set(relevant_ids).issubset(set(retrieved_ids[:k])):
            metric += 1
    return 0 if len(retrieval_results) == 0 else metric / len(retrieval_results)


def partial_match_accuracy(
    k: int,
    retrieval_results: List[RetrievalResult],
    ground_truths: List[List[GroundTruthItem]],
) -> float:  # NOTE: Same as context accuracy?
    metric = 0
    for retrieval_result, ground_truth in zip(retrieval_results, ground_truths):
        retrieved_ids = res2ids(retrieval_result)
        relevant_ids = [doc_id for gt in ground_truth for doc_id in gt.doc_ids]
        logger.debug(f"Retrieved/Relevant: {retrieved_ids[:k]} / {relevant_ids}")
        if set(relevant_ids).intersection(set(retrieved_ids[:k])):
            metric += 1
    return 0 if len(retrieval_results) == 0 else metric / len(retrieval_results)


def mrr(
    k: int,
    retrieval_results: List[RetrievalResult],
    ground_truths: List[List[GroundTruthItem]],
) -> float:
    metric = 0
    for retrieval_result, ground_truth in zip(retrieval_results, ground_truths):
        retrieved_ids = res2ids(retrieval_result, type="chunk")[:k]
        relevant_ids = [chunk_id for gt in ground_truth for chunk_id in gt.chunk_ids]
        rel_ret = set(relevant_ids).intersection(set(retrieved_ids))
        if not rel_ret:
            continue
        relevant_id_rank = min([retrieved_ids.index(doc_id) + 1 for doc_id in rel_ret])
        metric += 1 / relevant_id_rank
    return 0 if metric == 0 else metric / len(retrieval_results)


def ndcg(
    k: int,
    retrieval_results: List[RetrievalResult],
    ground_truths: List[List[GroundTruthItem]],
) -> float:
    metric = 0
    for retrieval_result, ground_truth in zip(retrieval_results, ground_truths):
        retrieved_ids = res2ids(retrieval_result, type="chunk")[:k]
        relevant_ids = [chunk_id for gt in ground_truth for chunk_id in gt.chunk_ids]
        rel_ret = set(relevant_ids).intersection(set(retrieved_ids))
        if not rel_ret:
            continue
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in rel_ret:
                metric += (
                    1
                    / np.log2(i + 2)  # one-based index + 1
                    / sum(1 / np.log2(j + 2) for j in range(len(retrieved_ids)))
                )
    return 0 if metric == 0 else metric / len(retrieval_results)


def queries_ratio(
    k: int,
    retrieval_results: List[RetrievalResult],
    ground_truths: List[List[GroundTruthItem]],
) -> float:
    q_ratio = 0
    for retrieval_result in retrieval_results:
        items = retrieval_result.items[:k]
        if n_items := len(items):
            n_queries = sum(item.metadata.type == "query" for item in items)
            q_ratio += n_queries / n_items
    return q_ratio / len(retrieval_results) if q_ratio > 0 else 0


def calculate_metrics(
    ks: List[int],
    rag_responses: List[RagResponse],
    ground_truths: List[List[GroundTruthItem]],
    dataset: str = settings.rag_dataset,
    deduplication_factor: int = settings.rag_deduplication_factor,
) -> EvaluationResult:
    if dataset in [
        "squad",
        "natural-questions",
        "natural-questions-kilt",
        "tech-qa",
        "trec-covid",
    ]:
        metrics_funcs = [context_accuracy, title_accuracy, mrr, ndcg, queries_ratio]
    elif dataset in ["multihoprag"]:
        metrics_funcs = [full_match_accuracy, partial_match_accuracy, queries_ratio]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    eval_metrics = []
    for k in ks:
        # Filter responses for this specific k value
        k_responses = [
            response for response in rag_responses if response.n_context == k
        ]

        if not k_responses:
            logger.warning(
                f"No responses found for k={k}. Available k values: {sorted(set(r.n_context for r in rag_responses))}"
            )
            continue

        retrieval_results = [
            rag_response.retrieval_result for rag_response in k_responses
        ]

        k_metrics = EvaluationMetrics(
            **{
                func.__name__: func(k, retrieval_results, ground_truths)
                for func in metrics_funcs
            },
            k=k,
            deduplication_factor=deduplication_factor,
        )
        eval_metrics.append(k_metrics)
        logger.info(
            f"Calculated metrics for k={k}: {k_metrics.model_dump(exclude_none=True)}"
        )

    eval_result = EvaluationResult(
        rag_responses=rag_responses,
        ground_truth=ground_truths,
        metrics=eval_metrics,
        eval_llm=settings.eval_llm,
        eval_embedding_model=settings.eval_embedding_model,
    )
    return eval_result


def recalculate_metrics(
    exp_name: str,
    ks: List[int] = settings.rag_retrieval_ks,
    write_to_file: bool = True,
) -> EvaluationResult:
    # Match exp name with path
    exp_path = None
    for path in settings.path_eval.glob("*"):
        if path.is_dir() and exp_name in path.name:
            exp_path = path
            break
    if not exp_path:
        raise ValueError(f"Experiment {exp_name} not found")
    eval_res_path = exp_path / f"{exp_name}.json"
    eval_res = EvaluationResult(**json.loads(eval_res_path.read_text(encoding="utf-8")))
    logger.info(
        f"Loaded evaluation result from {eval_res_path}. Recalculating metrics..."
    )
    # Get values from existing metrics
    deduplication_factor = eval_res.metrics[0].deduplication_factor
    dataset = eval_res.rag_responses[0].retrieval_result.dataset
    metrics = calculate_metrics(
        ks,
        eval_res.rag_responses,
        eval_res.ground_truth,
        dataset=dataset,
        deduplication_factor=deduplication_factor,
    )
    # Write new metrics to file
    if write_to_file:
        new_metrics_path = (
            exp_path / f"metrics_upd_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df_metrics = pd.DataFrame(
            [m.model_dump(exclude_none=True) for m in metrics.metrics]
        )
        df_metrics.to_csv(new_metrics_path, index=False)
        logger.info(f"Metrics saved to {new_metrics_path}")
    return metrics


async def run_evaluation(
    dataset: str = settings.rag_dataset,
    batch_size: int = 10,
    ks: List[int] = settings.rag_retrieval_ks,
    rag_llm: str = settings.rag_generation_model,
    qgen_llm: str = settings.question_generation_model,
    emb_model: str = settings.openai_embedding_model,
    retrieval_sources: List[Literal["chunk", "query"]] = settings.rag_retrieval_sources,
    rag_qa_type: str = settings.question_generation_mode,
    origins: List[str] = settings.rag_retrieval_query_origins,
    deduplication_factor: int = settings.rag_deduplication_factor,
    max_queries: Optional[int] = None,
    generate_answer: bool = True,
    read_only: bool = settings.rag_readonly_eval,
    write_json: bool = settings.rag_write_json,
    write_txt: bool = settings.rag_write_txt,
    write_spreadsheet: bool = settings.rag_write_spreadsheet,
    persist_triplets: bool = settings.rag_persist_triplets,
    spreadsheet_sheet_name: Optional[str] = None,
) -> EvaluationResult:
    logger.info(f"Settings:\n{settings.model_dump_json(indent=2)}")
    logger.info(f"Preparing questions for evaluation on dataset {dataset}")
    datetime_str = dt.now().strftime("%Y%m%d_%H%M%S")
    if read_only:
        logger.warning("Running in read-only mode")
    # Collect triplets for optional JSON export
    triplets_for_json: list[Triplet] = []

    async with get_db() as session:
        query_q_sql = (
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
                QueryVector.emb_model == emb_model,
                or_(
                    Query.prompt_id.is_(None),
                    Query.origin.in_(settings.eval_whitelist_origins),
                ),
                Query.origin.in_(origins),
                or_(
                    and_(
                        "eval" in origins,
                        Query.llm == settings.question_generation_model,
                    ),
                    and_(
                        "eval" in origins,
                        Query.llm.is_(None),
                    ),
                    "eval" not in origins,
                ),
                # Query.qa_type == "q",
            )
            .order_by(Query.id)
            # .distinct(Query.text_hash) # Use all the queries even if there are duplicates
            .limit(max_queries)
        )
        logger.info(
            f"Query SQL:\n{query_q_sql.compile(compile_kwargs={'literal_binds': True})}"
        )
        results = await session.execute(query_q_sql)
        results = results.all()

        if not results:
            raise ValueError(
                f"No queries found for dataset {dataset} with emb model {emb_model}. "
                "Make sure you added the embeddings to the database."
            )

        # Convert to Pydantic models to avoid lazy loading
        queries_data = []
        embeddings_data = []
        for row in results:
            query_data = EvaluationQueryData(**row._asdict())
            queries_data.append(query_data)

            embedding_data = EvaluationEmbeddingData(vector=row.vector)
            embeddings_data.append(embedding_data)

        n_duplicates = len(results) - len(set(row.text_hash for row in results))
        logger.info(
            f"Ready to evaluate {len(queries_data)} queries with {n_duplicates} duplicates"
        )

        logger.info("Preparing pipeline and building index")
        rag_pipeline = RAGPipeline(
            k=max(ks),
            emb_model=emb_model,
            dataset=dataset,
            rag_llm=rag_llm,
            qgen_llm=qgen_llm,
            qa_type=rag_qa_type,
            retrieval_sources=retrieval_sources,
            deduplication_factor=deduplication_factor,
            generate_answer=generate_answer,
        )
        await rag_pipeline.fit(session)
        logger.info("Pipeline fitted")

        # We'll collect all responses in a single list
        all_rag_responses = []

        logger.info("Running evaluation in batches")
        results_dirname = (
            f"{datetime_str}_"
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
                results_dirname=results_dirname,
                **settings.model_dump(mode="json"),
            ),
        )
        evaluation_set_id = 0
        if not settings.rag_readonly_eval:
            session.add(evaluation_set)
            await session.flush()
            await session.refresh(evaluation_set)
            evaluation_set_id = evaluation_set.id

        # We'll collect ground truths only once since they're the same for all k values
        ground_truths = []
        for i in tqdm(range(0, len(queries_data), batch_size)):
            logger.info(f"Evaluating batch {i // batch_size + 1} of {len(queries_data) // batch_size}")  # fmt: skip
            batch_query = queries_data[i : i + batch_size]
            batch_embeddings = embeddings_data[i : i + batch_size]
            questions = [str(query.text) for query in batch_query]
            batch_embeddings = np.array(
                [embedding.vector for embedding in batch_embeddings]
            )
            if np.isnan(batch_embeddings).any():
                logger.warning(f"Found null embeddings in batch {i // batch_size + 1}")
                batch_embeddings = None
            # Get responses for all k values at once
            batch_responses_by_k = await rag_pipeline.batch_query(
                questions, batch_embeddings, ks=ks
            )

            # Pre-fetch chunk and answer data for this batch to avoid lazy loading
            batch_query_ids = [query.id for query in batch_query]
            batch_chunk_ids = [
                query.chunk_id for query in batch_query if query.chunk_id
            ]

            # Get chunk texts
            chunk_texts_map = {}
            if batch_chunk_ids:
                chunks_result = await session.execute(
                    select(Chunk.id, Chunk.text).where(Chunk.id.in_(batch_chunk_ids))
                )
                chunk_texts_map = {row.id: row.text for row in chunks_result.all()}

            # Get answer ids and texts
            answers_result = await session.execute(
                select(Answer.id, Answer.query_id, Answer.text).where(
                    Answer.query_id.in_(batch_query_ids)
                )
            )
            answer_rows = answers_result.all()
            answer_texts_map = {row.query_id: row.text for row in answer_rows}
            answer_id_map = {row.query_id: row.id for row in answer_rows}

            # Collect all responses into a single list
            for k in ks:
                all_rag_responses.extend(batch_responses_by_k[k])

            # Pre-fetch eval-origin reference answers for this batch (from retrieved query items)
            eval_query_ids_in_batch: set[int] = set()
            first_k_responses = batch_responses_by_k[ks[0]]
            for rag_resp in first_k_responses:
                for it in rag_resp.retrieval_result.items:
                    meta = it.metadata
                    if (
                        meta.type == "query"
                        and getattr(meta, "origin", None) == "eval"
                        and isinstance(meta.query_id, int)
                    ):
                        eval_query_ids_in_batch.add(int(meta.query_id))
            eval_answer_texts_map: dict[int, str] = {}
            if eval_query_ids_in_batch:
                eval_ans_res = await session.execute(
                    select(Answer.query_id, Answer.text).where(
                        Answer.query_id.in_(list(eval_query_ids_in_batch))
                    )
                )
                eval_answer_texts_map = {
                    row.query_id: row.text for row in eval_ans_res.all()
                }

            # Process the first k for database storage (we only store once)
            retrieved_batch = batch_responses_by_k[ks[0]]  # Use first k for DB storage
            for idx, (query, retrieved_result) in enumerate(
                zip(batch_query, retrieved_batch)
            ):
                retrieval_result_db = RagResponseDB(
                    evaluation_set_id=evaluation_set_id,
                    retrieval_result=retrieved_result.model_dump(),
                    query_id=query.id,
                )

                doc_meta = query.doc_meta
                ground_truths_intermediate = []

                if (
                    doc_meta and len(doc_ids := doc_meta.get("doc_ids", [])) > 0
                ):  # Multihop
                    docs_result = await session.execute(
                        select(Doc.id, Doc.title).where(Doc.id.in_(doc_ids))
                    )
                    docs_data = docs_result.all()
                    doc_ids = [int(row.id) for row in docs_data]
                    doc_titles = [str(row.title) for row in docs_data]
                    # TODO: Figure out getting chunk_ids for multihop
                    chunk_ids = [query.chunk_id] if query.chunk_id else []
                    chunk_texts = (
                        [chunk_texts_map.get(query.chunk_id, "")]
                        if query.chunk_id
                        else []
                    )
                    logger.debug(f"Doc: {doc_ids} - {doc_titles}")
                    ground_truths_intermediate.append(
                        GroundTruthItem(
                            doc_ids=doc_ids,
                            doc_titles=doc_titles,
                            chunk_ids=chunk_ids,
                            chunk_texts=chunk_texts,
                            query=str(query.text),
                            answer=answer_texts_map.get(query.id),
                        )
                    )
                else:
                    chunk_ids = [query.chunk_id] if query.chunk_id else []
                    chunk_texts = (
                        [chunk_texts_map.get(query.chunk_id, "")]
                        if query.chunk_id
                        else [""]
                    )

                    # Get similar queries with their chunk data
                    similar_queries_result = await session.execute(
                        select(
                            Query.id,
                            Query.chunk_id,
                            Query.text,
                            Chunk.text.label("chunk_text"),
                        )
                        .outerjoin(Chunk, Query.chunk_id == Chunk.id)
                        .where(
                            Query.text_hash == query.text_hash,
                            Query.id != query.id,
                        )
                    )
                    similar_queries_data = similar_queries_result.all()
                    logger.debug(
                        f"Similar queries: {[(row.id, row.chunk_id, row.text) for row in similar_queries_data]}"
                    )

                    for row in similar_queries_data:
                        if row.chunk_id != query.chunk_id and row.chunk_id is not None:
                            logger.debug(
                                f"Adding chunk {row.chunk_id} to ground truth with text {row.chunk_text}"
                            )
                            chunk_ids.append(row.chunk_id)
                            chunk_texts.append(row.chunk_text or "")

                    logger.debug(f"Chunk: {chunk_ids} - {chunk_texts}")
                    ground_truths_intermediate.append(
                        GroundTruthItem(
                            doc_ids=[int(query.doc_id)],
                            doc_titles=[str(query.doc_title)],
                            chunk_ids=chunk_ids,
                            chunk_texts=chunk_texts,
                            query=str(query.text),
                            answer=answer_texts_map.get(query.id),
                        )
                    )
                logger.debug(f"Ground truth: `{ground_truths_intermediate}`")
                ground_truths.append(ground_truths_intermediate)
                if not settings.rag_readonly_eval:
                    session.add(retrieval_result_db)

                # Build triplet payload (always when persist_triplets=True)
                if persist_triplets:
                    contexts_by_k: dict = {}
                    rag_answers_by_k: dict = {}
                    for k in ks:
                        rag_resp = batch_responses_by_k[k][idx]
                        contexts_by_k[k] = [
                            TripletContext(
                                type=item.metadata.type,
                                doc_id=item.metadata.doc_id,
                                doc_title=item.metadata.doc_title,
                                chunk_id=item.metadata.chunk_id,
                                text=item.metadata.chunk_text or item.text,
                                score=float(item.score),
                                rank=rank + 1,
                            )
                            for rank, item in enumerate(rag_resp.retrieval_result.items)
                        ]
                        rag_answers_by_k[k] = rag_resp.answer or ""

                    gold_answer_id = answer_id_map.get(query.id)
                    triplets_for_json.append(
                        Triplet(
                            evaluation_set_id=evaluation_set_id,
                            eval_query_id=query.id,
                            gold_answer_id=gold_answer_id,
                            contexts_by_k=contexts_by_k,
                            rag_answers_by_k=rag_answers_by_k,
                            ref_q=query.text.split(settings.qa_separator)[0].strip(),
                            ref_a=answer_texts_map.get(query.id),
                        )
                    )

                    # Also persist to DB when not in read-only mode
                    if not settings.rag_readonly_eval:
                        triplet = QueryTriplet(
                            evaluation_set_id=evaluation_set_id,
                            eval_query_id=query.id,
                            gold_answer_id=gold_answer_id,
                            contexts_by_k=contexts_by_k,
                            rag_answers_by_k=rag_answers_by_k,
                        )
                        session.add(triplet)
            if not settings.rag_readonly_eval:
                await session.commit()
        logger.info("Evaluation set saved")

        logger.info("Calculating metrics for each k value")

        # Calculate metrics using all responses (will filter by n_context internally)
        metrics_result = calculate_metrics(
            ks,
            all_rag_responses,
            ground_truths,
            dataset=dataset,
            deduplication_factor=deduplication_factor,
        )

        # Create the final evaluation result
        evaluation_result = EvaluationResult(
            rag_responses=all_rag_responses,
            ground_truth=ground_truths,
            metrics=metrics_result.metrics,
            experiment_name=results_dirname,
            eval_llm=settings.eval_llm,
            eval_embedding_model=settings.eval_embedding_model,
        )
        if not settings.rag_readonly_eval:
            evaluation_set.set_metrics(
                evaluation_result.model_dump(include={"metrics"})
            )
            await session.commit()
        logger.info("Metrics calculated")

    logger.info("Saving results")
    dirpath = settings.path_eval / results_dirname
    dirpath.mkdir(parents=True, exist_ok=True)

    if write_spreadsheet:
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
            # Don't raise the exception to avoid failing the entire evaluation

    if write_json:
        with open(dirpath / f"{datetime_str}.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_result.model_dump(), f, indent=2, ensure_ascii=False)

    # Save triplets in a JSON format compatible with load_triplets_from_json
    if not settings.rag_persist_triplets:
        logger.warning("Triplets will not be saved")

    if settings.rag_persist_triplets:
        if not len(triplets_for_json):
            logger.warning("No triplets to save")
        else:
            triplets_payload = Triplets(triplets=triplets_for_json)
            with open(
                dirpath / f"{datetime_str}_triplets.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    triplets_payload.model_dump(), f, indent=2, ensure_ascii=False
                )

    if write_txt:
        with open(dirpath / f"{datetime_str}.txt", "w", encoding="utf-8") as f:
            f.write(evaluation_result.format_str())

    df_metrics = pd.DataFrame(
        [m.model_dump(exclude_none=True) for m in evaluation_result.metrics]
    )
    df_metrics.to_csv(dirpath / f"{datetime_str}_metrics.csv", index=False)
    logger.info(f"Results saved to {dirpath.relative_to(settings.path_root)}")

    return evaluation_result
