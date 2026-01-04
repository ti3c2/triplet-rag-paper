import asyncio
import json
import logging
from functools import partial
from typing import List, Literal, Optional

import fire
from sqlalchemy import delete, func, select

from .config.settings import settings
from .etl import (
    convert_q_queries_to_qa,
    embed_db_items,
    etl_csv_queries,
    etl_multihop_to_sql,
    etl_nq_to_sql,
    etl_squad_to_sql,
    etl_techqa_to_sql,
    etl_treccovid_to_sql,
    generate_questions_for_docs,
)
from .eval.evaluate import recalculate_metrics, run_evaluation
from .eval.ragas_evaluation import run_ragas_evaluation
from .eval.triplet_evaluate import (
    run_triplet_evaluation,
    run_two_stage_triplet_evaluation,
)
from .rag.models import EvaluationResult
from .rag.pipeline import RAGPipeline
from .store_rel.entry import get_db
from .store_rel.schema import (
    Answer,
    Chunk,
    ChunkVector,
    Doc,
    EvaluationSet,
    Prompt,
    Query,
    QueryTriplet,
    QueryVector,
    RagResponseDB,
)
from .store_rel.sqlite2postgres import export_to_postgres

logger = logging.getLogger(__name__)


class ETLCommands:
    """Commands for ETL operations - loading datasets into the database.

    Available Flags for each command:
        --n_rows: Number of rows from the dataset to process (100 by default)
        --n_commited: Number of rows to commit at once (100 by default)
    """

    n_rows: int = settings.rag_max_rows
    n_commited: int = settings.rag_max_rows // 10

    def fill_db(
        self,
        datasets: Optional[List[str]] = None,
        n_rows: int = n_rows,
        n_commited: int = n_commited,
    ):
        """Fill the database with data from the datasets.

        This function will load the datasets into the database.
        It will first load the natural-questions dataset, then the squad dataset,
        then the multihop dataset.

        Args:
            datasets: List of datasets to load
            n_rows: Number of rows to process per dataset (0 for all)
            n_commited: Number of rows to commit at once
        """
        logger.info("Filling database with data from the datasets")
        datasets = datasets or [settings.rag_dataset]
        func_map = {
            "natural-questions": etl_nq_to_sql,
            "natural-questions-kilt": partial(etl_nq_to_sql, use_kilt=True),
            "squad": etl_squad_to_sql,
            "tech-qa": etl_techqa_to_sql,
            "trec-covid": etl_treccovid_to_sql,
            "multihoprag": etl_multihop_to_sql,
        }
        for dataset in datasets:
            func = func_map.get(dataset)
            if func is not None:
                asyncio.run(func(n_rows=n_rows, n_commited=n_commited))
            else:
                logger.error(f"Dataset {dataset} not found")
        logger.info("Database filled successfully")


class QuestionCommands:
    """Commands for generating questions using LLMs."""

    async def generate(
        self,
        dataset: str = settings.rag_dataset,
        max_docs: int = -1,
        batch_size: int = settings.openai_chat_completion_max_concurrency,
        origin: Literal["dataset", "eval", "quote"] = "dataset",
    ):
        """Generate questions for documents using LLMs.

        Args:
            dataset: Dataset to generate questions for (None for all)
            max_docs: Maximum number of documents to process (-1 for all)
            batch_size: Batch size for processing
        """
        logger.info("Starting question generation")
        await generate_questions_for_docs(
            dataset=dataset,
            max_docs=max_docs,
            max_concurrency=batch_size,
            origin=origin,
        )

        logger.info("Question generation completed")

    async def convert_q_to_qa(
        self,
        dataset: str = settings.rag_dataset,
        llm: str = settings.question_generation_model,
        qa_type: Literal["qa", "qc"] = settings.question_generation_mode,
    ):
        """Convert existing queries with qa_type='q' to qa_type='qa' by appending their answers.

        This avoids having to regenerate queries when you want to create question-answer
        pairs from standalone questions.

        Args:
            dataset: Dataset to convert queries for (None for all datasets)
            llm: LLM to filter queries by (None for all LLMs)
        """
        logger.info("Starting conversion of q-type queries to qa-type")

        total_found, successfully_converted = await convert_q_queries_to_qa(
            dataset=dataset,
            llm=llm,
            qa_type=qa_type,
        )

        logger.info(
            f"Conversion completed: {successfully_converted}/{total_found} "
            f"q-type queries converted to qa-type"
        )

        if total_found == 0:
            logger.warning("No q-type queries found to convert")
        elif successfully_converted < total_found:
            logger.warning(
                f"{total_found - successfully_converted} queries could not be converted "
                "(likely missing answers or already converted)"
            )

    async def load_csv(
        self,
        csv_paths: List[str],
        dataset: str = settings.rag_dataset,
        n_rows: int = 0,
        n_committed: int = 100,
        prompt_name: str = "ragas_queries",
        prompt_text: Optional[str] = None,
        llm: str = settings.question_generation_model,
    ):
        """Load queries from CSV files into the database.

        This command loads query-answer pairs from CSV files that were generated
        from evaluation datasets. It maps queries to existing chunks using the
        original_row_idx field.

        Expected CSV format:
        - original_row_idx: Maps to hf_id in chunks table (required)
        - user_input: The query text (required)
        - reference: The reference/ground truth answer (optional)
        - response: The model response (optional)
        - multi_responses: Multiple model responses (optional)
        - reference_contexts: Reference contexts for the query (optional)
        - retrieved_contexts: Retrieved contexts for the query (optional)
        - rubrics: Evaluation rubrics (optional)
        - persona_name: Name of the persona (optional)
        - persona_role_description: Description of the persona role (optional)
        - node_id: Node identifier (optional)
        - source_chunk_idx: Source chunk index (optional)

        Args:
            csv_paths: List of paths to CSV files or directories containing CSV files
            dataset: Dataset name to associate queries with
            n_rows: Maximum number of rows to process per file (0 for all)
            n_committed: Commit every N items
            prompt_name: Name of the prompt to associate with CSV queries (default: "ragas_queries")
            prompt_text: Description of the prompt for CSV queries (default: prompt_name)
            llm: LLM model to associate with CSV queries (default: question_generation_model)
        """
        # Handle single string input
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]

        prompt_text = prompt_text or prompt_name

        logger.info(f"Loading queries from {len(csv_paths)} CSV path(s): {csv_paths}")

        await etl_csv_queries(
            csv_paths=csv_paths,
            dataset_name=dataset,
            n_rows=n_rows,
            n_committed=n_committed,
            prompt_name=prompt_name,
            prompt_text=prompt_text,
            llm=llm,
        )

        logger.info("CSV loading completed!")


class EmbeddingCommands:
    """Commands for generating embeddings."""

    async def chunks(
        self,
        dataset: str = settings.rag_dataset,
        max_items: int = -1,
        batch_size: int = settings.openai_embedding_batch_size,
        emb_model: str = settings.openai_embedding_model,
    ):
        """Generate embeddings for chunks.

        Args:
            dataset: Dataset to embed chunks for (None for all)
            max_items: Maximum number of items to embed (-1 for all)
            batch_size: Batch size for embedding
            emb_model: Embedding model to use
        """

        logger.info("Starting chunk embedding")
        await embed_db_items(
            db_items=None,
            db_items_types=["chunk"],
            emb_model=emb_model,
            dataset=dataset,
            batch_size=batch_size,
            max_items=max_items,
        )

    async def queries(
        self,
        dataset: str = settings.rag_dataset,
        llm: str = settings.question_generation_model,
        max_items: int = -1,
        batch_size: int = settings.openai_embedding_batch_size,
        emb_model: str = settings.openai_embedding_model,
        qa_type: str = settings.question_generation_mode,
        include_default_queries: bool = True,
    ):
        """Generate embeddings for queries.

        Args:
            dataset: Dataset to embed queries for (None for all)
            llm: Language model to use for embedding
            max_items: Maximum number of items to embed (-1 for all)
            batch_size: Batch size for embedding
            emb_model: Embedding model to use
        """
        logger.info("Starting query embedding")
        await embed_db_items(
            db_items=None,
            db_items_types=["query"],
            qa_type=qa_type,
            include_default_queries=include_default_queries,
            llm=llm,
            emb_model=emb_model,
            dataset=dataset,
            batch_size=batch_size,
            max_items=max_items,
        )

    async def all(
        self,
        dataset: str = settings.rag_dataset,
        max_items: int = -1,
        llm: str = settings.question_generation_model,
        batch_size: int = settings.openai_embedding_batch_size,
        emb_model: str = settings.openai_embedding_model,
        qa_type: str = settings.question_generation_mode,
        include_default_queries: bool = True,
    ):
        """Generate embeddings for both chunks and queries.

        Args:
            dataset: Dataset to embed for (None for all)
            max_items: Maximum number of items to embed (-1 for all)
            llm: Language model to use for embedding
            batch_size: Batch size for embedding
            emb_model: Embedding model to use
        """
        logger.info("Starting embedding for chunks and queries")
        await embed_db_items(
            db_items=None,
            db_items_types=["chunk", "query"],
            qa_type=qa_type,
            include_default_queries=include_default_queries,
            llm=llm,
            emb_model=emb_model,
            dataset=dataset,
            batch_size=batch_size,
            max_items=max_items,
        )
        logger.info("All embedding completed")


class RAGCommands:
    """Commands for RAG operations."""

    def __init__(self):
        self._pipeline = None

    async def _get_pipeline(
        self,
        k: int = settings.rag_retrieval_k,
        emb_model: str = settings.openai_embedding_model,
        dataset: str = "natural-questions",
        retrieval_sources: List[
            Literal["chunk", "query"]
        ] = settings.rag_retrieval_sources,
        use_triplet_contexts: bool = settings.rag_use_triplet_contexts,
        triplet_json_path: Optional[str] = None,
    ) -> RAGPipeline:
        """Get or create RAG pipeline."""
        if self._pipeline is None:
            # If JSON path provided, enable triplet contexts and set path in settings
            if triplet_json_path:
                settings.rag_triplet_json_path = triplet_json_path
                if not use_triplet_contexts:
                    use_triplet_contexts = True
            self._pipeline = RAGPipeline(
                k=k,
                emb_model=emb_model,
                dataset=dataset,
                retrieval_sources=retrieval_sources,
                use_triplet_contexts=use_triplet_contexts,
            )
            async with get_db() as session:
                await self._pipeline.fit(session)
        return self._pipeline

    async def query(
        self,
        question: str,
        k: int = settings.rag_retrieval_k,
        emb_model: str = settings.openai_embedding_model,
        dataset: str = settings.rag_dataset,
        retrieval_sources: List[
            Literal["chunk", "query"]
        ] = settings.rag_retrieval_sources,
        triplet_json_path: Optional[str] = None,
        output_json: bool = False,
    ):
        """Query the RAG pipeline with a single question.

        Args:
            question: Question to ask
            k: Number of documents to retrieve
            emb_model: Embedding model to use
            dataset: Dataset to query against
            retrieval_sources: Sources for retrieval (chunk, query)
            output_json: Whether to output JSON format
        """

        logger.info(f"Querying RAG pipeline with question: {question}")

        pipeline = await self._get_pipeline(
            k=k,
            emb_model=emb_model,
            dataset=dataset,
            retrieval_sources=retrieval_sources,
            triplet_json_path=triplet_json_path,
        )
        response = await pipeline.query(question)

        if response:
            if output_json:
                print(json.dumps(response.model_dump(), indent=2))
            else:
                print(response.format_str())
        else:
            logger.error("No response received from RAG pipeline")

    async def query_batch(
        self,
        questions: List[str],
        k: int = settings.rag_retrieval_k,
        emb_model: str = settings.openai_embedding_model,
        dataset: str = settings.rag_dataset,
        retrieval_sources: List[
            Literal["chunk", "query"]
        ] = settings.rag_retrieval_sources,
        batch_size: int = 10,
        triplet_json_path: Optional[str] = None,
        output_json: bool = False,
    ):
        """Query the RAG pipeline with multiple questions.

        Args:
            questions: List of questions to ask
            k: Number of documents to retrieve
            emb_model: Embedding model to use
            dataset: Dataset to query against
            retrieval_sources: Sources for retrieval (chunk, query)
            batch_size: Batch size for processing
            output_json: Whether to output JSON format
        """

        logger.info(f"Batch querying RAG pipeline with {len(questions)} questions")

        pipeline = await self._get_pipeline(
            k=k,
            emb_model=emb_model,
            dataset=dataset,
            retrieval_sources=retrieval_sources,
            triplet_json_path=triplet_json_path,
        )
        responses = await pipeline.batch_query(questions, batch_size=batch_size)

        if output_json:
            print(json.dumps([r.model_dump() for r in responses], indent=2))
        else:
            for i, response in enumerate(responses):
                print(f"=== Question {i+1} ===")
                print(response.format_str())
                print()


class EvaluationCommands:
    """Commands for evaluating RAG system performance."""

    async def run(
        self,
        dataset: str = settings.rag_dataset,
        batch_size: int = settings.rag_batch_size,
        ks: List[int] = settings.rag_retrieval_ks,
        rag_llm: str = settings.rag_generation_model,
        qgen_llm: str = settings.question_generation_model,
        emb_model: str = settings.openai_embedding_model,
        retrieval_sources: List[
            Literal["chunk", "query"]
        ] = settings.rag_retrieval_sources,
        deduplication_factor: int = settings.rag_deduplication_factor,
        max_queries: int = settings.eval_max_queries,
        max_queries_ragas: int = settings.eval_max_queries_ragas,
        generate_answer: bool = settings.rag_generate_answer,
        triplet_json_path: Optional[str] = None,
        output_json: bool = False,
        silent: bool = True,
        run_ragas: bool = settings.ragas_enabled,
    ):
        """Run evaluation on the RAG system.

        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for processing queries
            k: Number of documents to retrieve
            rag_llm: Language model to use for answer generation
            qgen_llm: Language model to use for question generation
            emb_model: Embedding model to use
            retrieval_sources: Sources for retrieval (chunk, query)
            deduplication_factor: Factor for deduplication
            max_queries: Maximum number of queries to evaluate (-1 for all)
            generate_answer: Whether to generate answers
            output_json: Whether to output results in JSON format
        """
        logger.info(f"Starting evaluation on dataset: {dataset}")

        # If provided, enable triplet contexts via JSON for this evaluation run
        if triplet_json_path:
            settings.rag_triplet_json_path = triplet_json_path
            settings.rag_use_triplet_contexts = True

        max_queries_param = None if max_queries == -1 else max_queries

        result = await run_evaluation(
            dataset=dataset,
            batch_size=batch_size,
            ks=ks,
            rag_llm=rag_llm,
            qgen_llm=qgen_llm,
            emb_model=emb_model,
            retrieval_sources=retrieval_sources,
            deduplication_factor=deduplication_factor,
            max_queries=max_queries_param,
            generate_answer=generate_answer,
        )

        if run_ragas:
            result = await self.run_ragas(
                eval_res=result, max_queries=max_queries_ragas
            )

        if silent:
            return
        if output_json:
            print(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            self._print_evaluation_results(result, dataset)

    async def run_triplet(
        self,
        dataset: str = settings.rag_dataset,
        batch_size: int = settings.rag_batch_size,
        ks: List[int] = settings.rag_retrieval_ks,
        rag_llm: str = settings.rag_generation_model,
        qgen_llm: str = settings.question_generation_model,
        emb_model: str = settings.openai_embedding_model,
        retrieval_sources: List[
            Literal["chunk", "query"]
        ] = settings.rag_retrieval_sources,
        deduplication_factor: int = settings.rag_deduplication_factor,
        max_queries: int = settings.eval_max_queries,
        generate_answer: bool = settings.rag_generate_answer,
        triplet_json_path: Optional[str] = None,
        output_json: bool = False,
        silent: bool = True,
    ):
        """Run triplet-augmented evaluation.

        Uses dataset-origin questions as evaluation inputs and builds the index
        from eval-origin queries. Triplet contexts are injected as retrieval results.
        """
        logger.info(f"Starting triplet evaluation on dataset: {dataset}")

        # Force-enable triplet prompt mode and optionally set JSON source
        settings.rag_use_triplet_contexts = True
        if triplet_json_path:
            settings.rag_triplet_json_path = triplet_json_path

        max_queries_param = None if max_queries == -1 else max_queries

        result = await run_triplet_evaluation(
            dataset=dataset,
            batch_size=batch_size,
            ks=ks,
            rag_llm=rag_llm,
            qgen_llm=qgen_llm,
            emb_model=emb_model,
            retrieval_sources=retrieval_sources,
            deduplication_factor=deduplication_factor,
            max_queries=max_queries_param,
            generate_answer_retrieval=generate_answer,
        )

        if silent:
            return
        if output_json:
            print(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            self._print_evaluation_results(result, dataset)

    async def run_two_stage_triplet(
        self,
        dataset: str = settings.rag_dataset,
        batch_size: int = settings.rag_batch_size,
        ks: List[int] = settings.rag_retrieval_ks,
        rag_llm: str = settings.rag_generation_model,
        qgen_llm: str = settings.question_generation_model,
        emb_model: str = settings.openai_embedding_model,
        deduplication_factor: int = settings.rag_deduplication_factor,
        max_queries: int = settings.eval_max_queries,
        generate_answer: bool = settings.rag_generate_answer,
        k_query: Optional[int] = None,
        k_chunk: Optional[int] = None,
        triplet_json_path: Optional[str] = None,
        output_json: bool = False,
        silent: bool = True,
    ):
        """Run two-stage (query+chunk) triplet-augmented evaluation.

        Stage 1 retrieves query-origin='two-stage' items to locate triplets; stage 2
        retrieves chunk-based contexts. Both stages' contexts are concatenated and
        passed to the generator.
        """
        logger.info(f"Starting two-stage triplet evaluation on dataset: {dataset}")

        # Optionally override triplet JSON source for this run
        if triplet_json_path:
            settings.rag_triplet_json_path = triplet_json_path

        max_queries_param = None if max_queries == -1 else max_queries

        result = await run_two_stage_triplet_evaluation(
            dataset=dataset,
            batch_size=batch_size,
            ks=ks,
            rag_llm=rag_llm,
            qgen_llm=qgen_llm,
            emb_model=emb_model,
            rag_qa_type=settings.question_generation_mode,
            deduplication_factor=deduplication_factor,
            max_queries=max_queries_param,
            generate_answer=generate_answer,
            k_query=k_query,
            k_chunk=k_chunk,
        )

        if silent:
            return
        if output_json:
            print(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            self._print_evaluation_results(result, dataset)

    def _print_evaluation_results(self, result, dataset: str):
        """Print evaluation results in a human-readable format."""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {dataset.upper()}")
        print(f"{'='*60}")

        print(f"\nDataset: {dataset}")
        print(f"Number of queries evaluated: {len(result.rag_responses)}")
        print(f"Number of responses generated: {sum(1 for r in result.rag_responses if r.answer)}")  # fmt: skip

        print(result.format_str())
        print(f"\n{'='*60}")

    def recalculate_metrics(
        self,
        exp_name: str,
        ks: List[int] = settings.rag_retrieval_ks,
        write_to_file: bool = True,
    ):
        """Recalculate metrics for an experiment."""
        metrics = recalculate_metrics(exp_name, ks, write_to_file)
        logger.info(
            "New Metrics:\n" + metrics.model_dump_json(indent=2, include=["metrics"])
        )

    async def run_ragas(
        self,
        eval_res: Optional[EvaluationResult] = None,
        json_path: Optional[str] = None,
        metrics_to_use: List[str] = settings.ragas_metrics,
        ks: List[int] = settings.eval_context_ks,
        max_queries: Optional[int] = settings.eval_max_queries_ragas,
        save_results: bool = True,
        update_existing: bool = False,
        output_json: bool = False,
        experiment_name: Optional[str] = None,
    ):
        """
        Run ragas evaluation on existing evaluation results.
        **Ignore eval_res when using CLI**
        """
        logger.info(f"Starting ragas evaluation for metrics: {metrics_to_use}")

        result = await run_ragas_evaluation(
            eval_res=eval_res,
            json_path=json_path,
            metrics_to_use=metrics_to_use,
            ks=ks,
            max_queries=max_queries,
            update_existing=update_existing,
            save_results=save_results,
            experiment_name=experiment_name,
        )

        if output_json:
            print(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            self._print_ragas_results(result, metrics_to_use)

    def _print_ragas_results(self, result: EvaluationResult, metrics_used: List[str]):
        """Print ragas evaluation results in a human-readable format."""
        print(f"\n{'='*60}")
        print(f"RAGAS EVALUATION RESULTS")
        print(f"{'='*60}")

        print(f"\nMetrics computed: {', '.join(metrics_used)}")
        print(f"Number of queries evaluated: {len(result.rag_responses)}")

        print(f"\nUpdated Metrics (with Ragas):")
        for metric in result.metrics:
            print(f"\nK={metric.k}:")
            ragas_metrics = {
                k: v
                for k, v in metric.model_dump().items()
                if k.startswith("ragas_") and v is not None
            }
            for metric_name, value in ragas_metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        print(f"\n{'='*60}")


class DBCommands:
    """Commands for checking system status."""

    async def delete_queries_by_llm(
        self,
        llm: str = settings.question_generation_model,
        qa_type: str = settings.question_generation_mode,
    ):
        """Remove queries by model."""

        qa_types = qa_type.split(",")

        async with get_db() as session:
            # First: Delete related Answers
            delete_answers_stmt = delete(Answer).where(
                Answer.query_id.in_(
                    select(Query.id).where(
                        Query.llm == llm,
                        Query.qa_type.in_(qa_types),
                    )
                )
            )
            result1 = await session.execute(delete_answers_stmt)
            deleted_answers = result1.rowcount

            # Second: Delete related QueryVectors
            delete_vectors_stmt = delete(QueryVector).where(
                QueryVector.parent_id.in_(
                    select(Query.id).where(
                        Query.llm == llm,
                        Query.qa_type.in_(qa_types),
                    )
                )
            )
            result2 = await session.execute(delete_vectors_stmt)
            deleted_vectors = result2.rowcount

            # Finally: Delete the Queries
            delete_queries_stmt = delete(Query).where(
                Query.llm == llm,
                Query.qa_type.in_(qa_types),
            )
            result3 = await session.execute(delete_queries_stmt)
            deleted_queries = result3.rowcount

            await session.commit()
            logger.info(
                f"Removed {deleted_queries} queries, {deleted_answers} answers, {deleted_vectors} query vectors"
            )

    async def delete_vectors_by_model(
        self,
        emb_model: str = settings.openai_embedding_model,
        llm: str = settings.rag_generation_model,
        delete_chunk_vectors: bool = False,
        qa_type: str = settings.question_generation_mode,
        dataset: str = settings.rag_dataset,
    ):
        """Remove vectors by model."""
        async with get_db() as session:
            # Delete QueryVectors
            delete_query_vectors_stmt = delete(QueryVector).where(
                QueryVector.parent_id.in_(
                    select(Query.id)
                    .join(Doc)
                    .where(
                        Query.llm == llm,
                        Doc.dataset == dataset,
                        Query.qa_type == qa_type,
                    )
                ),
                QueryVector.emb_model == emb_model,
            )
            result1 = await session.execute(delete_query_vectors_stmt)
            deleted_query_vectors = result1.rowcount

            # Delete ChunkVectors
            deleted_chunk_vectors = 0
            if delete_chunk_vectors:
                delete_chunk_vectors_stmt = (
                    delete(ChunkVector)
                    .where(
                        ChunkVector.parent_id.in_(
                            select(Chunk.id).join(Doc).where(Doc.dataset == dataset)
                        )
                    )
                    .where(ChunkVector.emb_model == emb_model)
                )
                result2 = await session.execute(delete_chunk_vectors_stmt)
                deleted_chunk_vectors = result2.rowcount

            await session.commit()
            logger.info(
                f"Removed {deleted_query_vectors} query vectors "
                f"and {deleted_chunk_vectors} chunk vectors "
                f"by model {emb_model} for dataset {dataset} and llm {llm}"
            )

    async def delete_docs_and_queries(
        self,
        dataset: str = settings.rag_dataset,
    ):
        """Remove documents and their related records (chunks, queries, answers, vectors)."""

        datasets = [ds.strip() for ds in dataset.split(",") if ds.strip()]
        if not datasets:
            logger.warning("No dataset specified for deletion.")
            return

        async with get_db() as session:
            doc_ids_subquery = select(Doc.id).where(Doc.dataset.in_(datasets))
            chunk_ids_subquery = select(Chunk.id).where(
                Chunk.doc_id.in_(doc_ids_subquery)
            )
            query_ids_subquery = select(Query.id).where(
                Query.doc_id.in_(doc_ids_subquery)
            )

            delete_rag_responses_stmt = delete(RagResponseDB).where(
                RagResponseDB.query_id.in_(query_ids_subquery)
            )
            rag_res_deleted = (
                await session.execute(delete_rag_responses_stmt)
            ).rowcount or 0

            delete_answers_stmt = delete(Answer).where(
                Answer.query_id.in_(query_ids_subquery)
            )
            answers_deleted = (await session.execute(delete_answers_stmt)).rowcount or 0

            delete_query_vectors_stmt = delete(QueryVector).where(
                QueryVector.parent_id.in_(query_ids_subquery)
            )
            query_vectors_deleted = (
                await session.execute(delete_query_vectors_stmt)
            ).rowcount or 0

            delete_queries_stmt = delete(Query).where(Query.id.in_(query_ids_subquery))
            queries_deleted = (await session.execute(delete_queries_stmt)).rowcount or 0

            delete_chunk_vectors_stmt = delete(ChunkVector).where(
                ChunkVector.parent_id.in_(chunk_ids_subquery)
            )
            chunk_vectors_deleted = (
                await session.execute(delete_chunk_vectors_stmt)
            ).rowcount or 0

            delete_chunks_stmt = delete(Chunk).where(Chunk.id.in_(chunk_ids_subquery))
            chunks_deleted = (await session.execute(delete_chunks_stmt)).rowcount or 0

            delete_docs_stmt = delete(Doc).where(Doc.id.in_(doc_ids_subquery))
            docs_deleted = (await session.execute(delete_docs_stmt)).rowcount or 0

            await session.commit()

            logger.info(
                "Removed datasets %s -> docs=%s, chunks=%s, chunk_vectors=%s, "
                "queries=%s, query_vectors=%s, answers=%s, rag_responses=%s",
                ", ".join(datasets),
                docs_deleted,
                chunks_deleted,
                chunk_vectors_deleted,
                queries_deleted,
                query_vectors_deleted,
                answers_deleted,
                rag_res_deleted,
            )

    async def stats(self):
        """Show database statistics."""

        async with get_db() as session:
            # Basic counts
            docs_result = await session.execute(select(func.count(Doc.id)))
            docs = docs_result.scalar()

            chunks_result = await session.execute(select(func.count(Chunk.id)))
            chunks = chunks_result.scalar()

            queries_result = await session.execute(select(func.count(Query.id)))
            queries = queries_result.scalar()

            answers_result = await session.execute(select(func.count(Answer.id)))
            answers = answers_result.scalar()

            prompts_result = await session.execute(select(func.count(Prompt.id)))
            prompts = prompts_result.scalar()

            context_vectors_result = await session.execute(
                select(func.count(ChunkVector.id))
            )
            context_vectors = context_vectors_result.scalar()

            query_vectors_result = await session.execute(
                select(func.count(QueryVector.id))
            )
            query_vectors = query_vectors_result.scalar()

            evaluation_results_result = await session.execute(
                select(func.count(EvaluationSet.id))
            )
            evaluation_results = evaluation_results_result.scalar()

            print(f"Database Statistics:")
            print(f"  Documents: {docs}")
            print(f"  Chunks: {chunks}")
            print(f"  Queries: {queries}")
            print(f"  Answers: {answers}")
            print(f"  Context Vectors: {context_vectors}")
            print(f"  Query Vectors: {query_vectors}")
            print(f"  Prompts: {prompts}")
            print(f"  Evaluation Results: {evaluation_results}")

            # Show datasets
            dataset_counts_result = await session.execute(
                select(Doc.dataset, func.count(Doc.id).label("doc_count")).group_by(
                    Doc.dataset
                )
            )
            dataset_counts = dataset_counts_result.all()
            print(f"\nDataset Breakdown:")
            for dataset, doc_count in dataset_counts:
                print(f"  {dataset}: {doc_count} documents")

            # Show models breakdown
            llm_counts_result = await session.execute(
                select(
                    Query.llm,
                    func.count(Query.id).label("query_count"),
                ).group_by(Query.llm)
            )
            llm_counts = llm_counts_result.all()

            # For qa_type breakdown per LLM
            llm_qa_counts_result = await session.execute(
                select(
                    Query.llm,
                    Query.qa_type,
                    func.count(Query.id).label("qa_count"),
                ).group_by(Query.llm, Query.qa_type)
            )
            llm_qa_counts = llm_qa_counts_result.all()

            # Create LLM to number mapping derived from llm_counts
            llm_list = [row.llm for row in llm_counts]
            llm_to_num = {
                llm: i + 1
                for i, llm in enumerate([l for l in llm_list if l is not None])
            }
            if None in llm_list:
                llm_to_num[None] = 0

            print(f"\nLLM Breakdown:")
            # Sort to show non-None first, then None
            for llm, count in sorted(llm_counts, key=lambda x: x[0] is None):
                llm_num = llm_to_num.get(llm, 0)
                llm_display = "Dataset queries" if llm is None else llm
                print(f"  #{llm_num} {llm_display}: {count} queries")

                # Print breakdown by query type for this LLM from precomputed counts
                for l, qa_type, qa_count in filter(
                    lambda r: r[0] == llm, llm_qa_counts
                ):
                    print(f"    - {qa_type}: {qa_count} queries")

            # Show embedding models breakdown
            # Query vectors per emb_model
            qvec_counts_result = await session.execute(
                select(
                    QueryVector.emb_model,
                    func.count(QueryVector.id).label("qvec_count"),
                ).group_by(QueryVector.emb_model)
            )
            qvec_counts = {row.emb_model: row.qvec_count for row in qvec_counts_result}

            # Chunk vectors per emb_model
            cvec_counts_result = await session.execute(
                select(
                    ChunkVector.emb_model,
                    func.count(ChunkVector.id).label("cvec_count"),
                ).group_by(ChunkVector.emb_model)
            )
            cvec_counts = {row.emb_model: row.cvec_count for row in cvec_counts_result}

            # LLMs with vectors per emb_model (with qa_type)
            emb_llm_qa_result = await session.execute(
                select(
                    QueryVector.emb_model,
                    Query.llm,
                    Query.qa_type,
                )
                .join(Query, Query.id == QueryVector.parent_id)
                .group_by(QueryVector.emb_model, Query.llm, Query.qa_type)
            )
            emb_to_llmqa: dict = {}
            for row in emb_llm_qa_result:
                emb = row.emb_model
                emb_to_llmqa.setdefault(emb, []).append((row.llm, row.qa_type))

            # Gather all embedding models to display (skip None later)
            all_emb_models = set(qvec_counts.keys()) | set(cvec_counts.keys())
            print(f"\nEmbedding Model Breakdown:")
            for embedding_model in sorted(all_emb_models):
                if not embedding_model:
                    continue
                q_count = qvec_counts.get(embedding_model, 0)
                c_count = cvec_counts.get(embedding_model, 0)

                llm_numbers = []
                for llm, qa_type in emb_to_llmqa.get(embedding_model, []):
                    if llm in llm_to_num:
                        llm_numbers.append(f"{llm_to_num[llm]}.{qa_type}")
                llm_numbers.sort()

                llm_numbers_str = (
                    ", ".join(f"#{num}" for num in llm_numbers)
                    if llm_numbers
                    else "none"
                )

                print(
                    f"  {embedding_model}: {q_count} query vectors, "
                    f"{c_count} chunk vectors ({c_count + q_count} total)\n"
                    f"\tEmbeded llms: {llm_numbers_str}"
                )

            # Show prompts breakdown (with indexes and text previews)
            prompts_list_result = await session.execute(
                select(Prompt).order_by(Prompt.name)
            )
            prompts_list = prompts_list_result.scalars().all()
            prompt_to_num = {p.id: i + 1 for i, p in enumerate(prompts_list)}

            # Count queries per prompt
            prompt_counts_result = await session.execute(
                select(
                    Prompt.id.label("prompt_id"),
                    func.count(Query.id).label("query_count"),
                )
                .outerjoin(Query, Query.prompt_id == Prompt.id)
                .group_by(Prompt.id)
            )
            prompt_counts = {
                row.prompt_id: row.query_count for row in prompt_counts_result
            }

            print(f"\nPrompt Breakdown:")
            for p in prompts_list:
                p_num = prompt_to_num.get(p.id, 0)
                q_count = prompt_counts.get(p.id, 0)
                print(
                    f"  #{p_num} - id={p.id} name={p.name} (queries: {q_count}):\n```\n{p.text}\n```"
                )
                print()

    async def datasets(self):
        """List available datasets."""

        async with get_db() as session:
            # Aggregate doc counts per dataset
            doc_counts_result = await session.execute(
                select(Doc.dataset, func.count(Doc.id).label("doc_count")).group_by(
                    Doc.dataset
                )
            )
            doc_counts = {row.dataset: row.doc_count for row in doc_counts_result}

            # Aggregate query counts per dataset
            query_counts_result = await session.execute(
                select(
                    Doc.dataset,
                    func.count(Query.id).label("query_count"),
                )
                .join(Query)
                .group_by(Doc.dataset)
            )
            query_counts = {row.dataset: row.query_count for row in query_counts_result}

            print("Available datasets:")
            for dataset in sorted(doc_counts.keys() | query_counts.keys()):
                if not dataset:
                    continue
                doc_count = doc_counts.get(dataset, 0)
                query_count = query_counts.get(dataset, 0)
                print(f"  {dataset}: {doc_count} docs, {query_count} queries")

    async def export_to_postgres(
        self,
        source_database_url: str = settings.sqlite_database_url,
        target_database_url: str = settings.postgres_database_url_async,
        recreate_tables: bool = False,
        exclude_tables: Optional[list[str]] = None,
    ):
        """Export all data from source SQLAlchemy storage to PostgreSQL.

        Args:
            source_database_url: Source database URL. If None, uses sqlite from settings.
            target_database_url: Target PostgreSQL URL. If None, uses postgres from settings.
            recreate_tables: Whether to drop and recreate tables in target database.
            exclude_tables: Tables to exclude from export. Only rag_responses is supported for now.
        """
        logger.info("Starting data export to PostgreSQL...")

        stats = await export_to_postgres(
            source_database_url=source_database_url,
            target_database_url=target_database_url,
            recreate_tables=recreate_tables,
            exclude_tables=exclude_tables,
        )

        print(f"\n{'='*60}")

    async def delete_eval_run(self, evaluation_set_id: int):
        """Hard delete an evaluation run and all its dependent data (triplets, rag responses)."""
        async with get_db() as session:
            # Deleting the EvaluationSet will cascade to RagResponseDB and QueryTriplet per FKs
            result = await session.execute(
                select(EvaluationSet).where(EvaluationSet.id == evaluation_set_id)
            )
            eval_set = result.scalars().first()
            if not eval_set:
                logger.error(f"EvaluationSet {evaluation_set_id} not found")
                return
            await session.delete(eval_set)
            await session.commit()
            logger.info(
                f"Deleted evaluation set {evaluation_set_id} and cascaded children"
            )

    async def archive_eval_run(self, evaluation_set_id: int):
        """Soft-archive all triplets for a given evaluation run (keeps raw rag responses)."""
        async with get_db() as session:
            result = await session.execute(
                select(QueryTriplet).where(
                    QueryTriplet.evaluation_set_id == evaluation_set_id
                )
            )
            triplets = result.scalars().all()
            if not triplets:
                logger.warning(
                    f"No triplets found for evaluation set {evaluation_set_id}"
                )
            from datetime import datetime

            for t in triplets:
                t.archived_at = datetime.utcnow()
            await session.commit()
            logger.info(
                f"Archived {len(triplets)} triplets for evaluation set {evaluation_set_id}"
            )

    async def prune_triplets(
        self,
        dataset: Optional[str] = None,
        keep_last: int = 0,
        older_than_days: Optional[int] = None,
        include_archived: bool = False,
    ):
        """Prune triplets by dataset and time or keep_last constraint.

        - keep_last: keep the newest N evaluation sets (by datetime) per dataset; delete others.
        - older_than_days: delete evaluation sets older than N days.
        - include_archived: when False (default), only prune archived triplets; when True, prune active too.
        """
        async with get_db() as session:
            # Identify evaluation sets to delete
            es_query = select(EvaluationSet)
            if dataset:
                es_query = es_query.where(EvaluationSet.dataset == dataset)
            es_query = es_query.order_by(EvaluationSet.datetime.desc())
            result = await session.execute(es_query)
            eval_sets = result.scalars().all()

            to_delete_ids: set[int] = set()
            from datetime import datetime, timedelta

            now = datetime.utcnow()

            if older_than_days is not None:
                cutoff = now - timedelta(days=older_than_days)
                for es in eval_sets:
                    if es.datetime < cutoff:
                        to_delete_ids.add(es.id)

            if keep_last > 0:
                protected = {es.id for es in eval_sets[:keep_last]}
                for es in eval_sets[keep_last:]:
                    to_delete_ids.add(es.id)

            if not to_delete_ids:
                logger.info("Nothing to prune")
                return

            if not include_archived:
                # Only delete evaluation sets whose triplets are archived (or have no triplets)
                safe_to_delete: set[int] = set()
                for es_id in to_delete_ids:
                    triplets_result = await session.execute(
                        select(QueryTriplet).where(
                            QueryTriplet.evaluation_set_id == es_id
                        )
                    )
                    triplets = triplets_result.scalars().all()
                    if all(t.archived_at is not None for t in triplets):
                        safe_to_delete.add(es_id)
                to_delete_ids = safe_to_delete

            if not to_delete_ids:
                logger.info(
                    "No evaluation sets qualified for deletion under current constraints"
                )
                return

            # Bulk delete evaluation sets (cascade removes triplets and rag responses)
            for es_id in to_delete_ids:
                es_res = await session.execute(
                    select(EvaluationSet).where(EvaluationSet.id == es_id)
                )
                es = es_res.scalars().first()
                if es:
                    await session.delete(es)
            await session.commit()
            logger.info(
                f"Deleted {len(to_delete_ids)} evaluation sets: {sorted(to_delete_ids)}"
            )


class RAGCLITool:
    """RAG Research CLI Tool - ETL, Question Generation, Embedding, and RAG Operations.

    Available Commands:
        etl         - ETL operations for loading datasets into the database
        questions   - Generate questions for documents using LLMs
        embed       - Generate embeddings for chunks and queries
        rag         - RAG pipeline operations (query, query_batch)
        eval        - Evaluate RAG system performance with metrics
        db          - Database operations (delete_queries_by_llm, stats, export_to_postgres)

    Examples:
        # Load SQuAD dataset
        quote-cli etl fill_db [squad,natural-questions,multihop] --n_rows=1000

        # Generate questions for documents
        quote-cli questions generate --dataset=squad --max_docs=100

        # Convert q-type queries to qa-type (question+answer)
        quote-cli questions convert_q_to_qa --dataset=squad --llm=gpt-4o-mini

        # Load queries from CSV files
        quote-cli questions load_csv queries.csv --dataset=squad --prompt_name=eval_queries
        quote-cli questions load_csv csv_directory/ --dataset=squad --n_rows=1000 --prompt_name=eval_queries
        quote-cli questions load_csv [file1.csv,file2.csv,dir1/] --dataset=my_dataset --prompt_name=eval_queries
        quote-cli questions load_csv queries.csv --dataset=my_dataset --prompt_name=eval_queries

        # Load queries from CSV files
        rag-cli questions load_csv queries.csv --dataset=earthquake_dataset
        rag-cli questions load_csv csv_directory/ --dataset=my_dataset --n_rows=1000
        rag-cli questions load_csv [file1.csv,file2.csv,dir1/] --dataset=my_dataset
        rag-cli questions load_csv queries.csv --dataset=my_dataset --prompt_name=eval_queries

        # Generate embeddings for chunks
        quote-cli embed chunks --dataset=squad

        # Query the RAG system
        quote-cli rag query "What is the capital of France?"

        # Run evaluation on dataset
        quote-cli eval run --dataset=squad --max_queries=100

        # Run ragas evaluation on existing results
        quote-cli eval run_ragas data/eval/experiment/20250101_120000.json

        # Run ragas evaluation with specific metrics
        quote-cli eval run_ragas data/eval/experiment/20250101_120000.json --metrics_to_use=[context_precision,faithfulness]

        # Check database statistics
        quote-cli db stats

        # Export data from SQLite to PostgreSQL
        quote-cli db export_to_postgres --recreate_tables=True

        # Export data with custom database URLs
        quote-cli db export_to_postgres --source_database_url="sqlite:///custom.db" --target_database_url="postgresql+asyncpg://user:pass@host:port/db"

        # Delete queries by model
        quote-cli db delete_queries_by_llm --llm=Qwen/Qwen2.5-7B-Instruct

        # Get help for specific commands
        quote-cli etl --help
        quote-cli rag --help
        quote-cli eval --help
    """

    def __init__(self):
        self.etl = ETLCommands()
        self.questions = QuestionCommands()
        self.embed = EmbeddingCommands()
        self.rag = RAGCommands()
        self.eval = EvaluationCommands()
        self.db = DBCommands()


def main():
    """Main CLI entry point."""
    # Configure logging
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    fire.Fire(RAGCLITool)


if __name__ == "__main__":
    main()
