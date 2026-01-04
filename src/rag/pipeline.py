import asyncio
import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from ..config.settings import settings
from ..config.utils import create_embeddings, execute_with_semaphore
from .generation import LLMGenerator
from .models import RagResponse, RetrievalResult
from .retrieval import DatabaseKNNRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        k: int = settings.rag_retrieval_k,
        rag_llm: str = settings.rag_generation_model,
        qgen_llm: str = settings.question_generation_model,
        emb_model: str = settings.openai_embedding_model,
        dataset: str = settings.rag_dataset,
        retrieval_sources: List[
            Literal["chunk", "query"]
        ] = settings.rag_retrieval_sources,
        deduplication_factor: int = settings.rag_deduplication_factor,
        generate_answer: bool = settings.rag_generate_answer,
        qa_type: str = settings.question_generation_mode,
        retrieval_query_origins: List[str] = settings.rag_retrieval_query_origins,
    ):
        self.deduplication_factor = deduplication_factor
        self.k = k
        self.rag_llm = rag_llm
        self.qgen_llm = qgen_llm
        self.emb_model = emb_model
        self.dataset = dataset
        self.generate_answer = generate_answer
        self.qa_type = qa_type
        self.retriever = DatabaseKNNRetriever(
            k=self.k,
            metric=settings.rag_retrieval_metric,
            algorithm=settings.rag_retrieval_algorithm,
            model=self.qgen_llm,
            emb_model=self.emb_model,
            dataset=self.dataset,
            retrieval_sources=retrieval_sources,
            deduplication_factor=self.deduplication_factor,
            qa_type=qa_type,
            retrieval_query_origins=retrieval_query_origins,
        )
        self.generator = LLMGenerator()
        self.is_fitted = False

    async def fit(self, session: AsyncSession) -> None:
        await self.retriever.fit_index(session)
        self.is_fitted = True

    async def embed_queries(self, queries: List[str]) -> np.ndarray:
        response = await create_embeddings(
            settings.get_openai_client(embedding=True),
            model=self.emb_model,
            input=queries,
            encoding_format="float",
        )
        if response is None:
            logger.warning(
                f"No embeddings returned from the API for queries: {queries}"
            )
            return np.array([np.nan] * len(queries), dtype=np.float32)
        embeddings = np.array(
            [item.embedding for item in response.data],
            dtype=np.float32,
        )
        return embeddings

    def retrieve(self, query_embeddings: np.ndarray) -> List[RetrievalResult]:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before retrieval")
        return self.retriever.retrieve(query_embeddings)

    async def generate(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        generate_answer: bool = settings.rag_generate_answer,
    ) -> RagResponse:
        return await self.generator.generate_response(
            question,
            retrieval_result,
            generate_answer=generate_answer,
        )

    async def query(
        self,
        question: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> RagResponse | None:  # TODO: Implement retries
        """End-to-end RAG query: embed, retrieve, and generate."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before querying")

        try:
            query_embedding = query_embedding or await self.embed_queries([question])
            retrieval_result = self.retrieve(query_embedding)[0]
            response = await self.generate(question, retrieval_result)
            return response
        except Exception as e:
            logger.exception(f"Error processing RAG query: {e}")
            return

    async def batch_query(
        self,
        questions: List[str],
        query_embeddings: Optional[np.ndarray] = None,
        batch_size: int = settings.rag_batch_size,
        ks: Optional[List[int]] = None,
    ) -> Union[List[RagResponse], Dict[int, List[RagResponse]]]:
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before querying")

        # If ks is None, use the original behavior (single k)
        if ks is None:
            return await self._batch_query_single_k(
                questions, query_embeddings, batch_size
            )

        # Multiple k values: generate separate answers for each k
        return await self._batch_query_multiple_k(
            questions, query_embeddings, batch_size, ks
        )

    async def _batch_query_single_k(
        self,
        questions: List[str],
        query_embeddings: Optional[np.ndarray] = None,
        batch_size: int = settings.rag_batch_size,
    ) -> List[RagResponse]:
        """Original batch_query logic for single k value."""
        results = []
        questions_batches = [
            questions[i : i + batch_size] for i in range(0, len(questions), batch_size)
        ]
        for i, questions_batch in enumerate(questions_batches):
            try:
                # Assuming that query_embeddings is either full of embeddings or None
                embeddings = (
                    query_embeddings[i * batch_size : (i + 1) * batch_size]
                    if query_embeddings is not None
                    else await self.embed_queries(questions_batch)
                )
                retrieval_results = self.retrieve(embeddings)
                responses = await execute_with_semaphore(
                    [
                        self.generate(
                            question,
                            retrieval_result,
                            self.generate_answer,
                        )
                        for question, retrieval_result in zip(
                            questions_batch, retrieval_results
                        )
                    ]
                )
                results.extend(responses)
            except Exception as e:
                logger.exception(f"Error processing questions batch '{questions_batch}': {e}")  # fmt: skip
                results.extend(
                    [
                        RagResponse(
                            answer="",
                            query=question,
                            model=self.generator.model,
                            retrieval_scores=[],
                            n_context=0,
                            retrieval_result=RetrievalResult(items=[]),
                        )
                        for question in questions_batch
                    ]
                )
        return results

    async def _batch_query_multiple_k(
        self,
        questions: List[str],
        query_embeddings: Optional[np.ndarray] = None,
        batch_size: int = settings.rag_batch_size,
        ks: List[int] = None,
    ) -> Dict[int, List[RagResponse]]:
        """Generate separate answers for each k value."""
        if ks is None:
            raise ValueError("ks parameter is required for multiple k query")

        # Initialize results dict
        results_by_k = {k: [] for k in ks}

        questions_batches = [
            questions[i : i + batch_size] for i in range(0, len(questions), batch_size)
        ]

        for i, questions_batch in enumerate(questions_batches):
            try:
                # Get embeddings once per batch
                embeddings = (
                    query_embeddings[i * batch_size : (i + 1) * batch_size]
                    if query_embeddings is not None
                    else await self.embed_queries(questions_batch)
                )

                # Retrieve once with max(ks) - this gives us all needed items
                retrieval_results = self.retrieve(embeddings)

                # For each k, generate responses with sliced retrieval results
                for k in ks:
                    logger.debug(f"Generating responses for k={k}")

                    # Create tasks for this k value
                    k_tasks = []
                    for question, retrieval_result in zip(
                        questions_batch, retrieval_results
                    ):
                        # Slice retrieval result to top-k items
                        sliced_result = self._slice_retrieval_result(
                            retrieval_result, k
                        )
                        k_tasks.append(
                            self.generate(
                                question,
                                sliced_result,
                                self.generate_answer,
                            )
                        )

                    # Execute all tasks for this k
                    k_responses = await execute_with_semaphore(k_tasks)
                    results_by_k[k].extend(k_responses)

            except Exception as e:
                logger.exception(
                    f"Error processing questions batch '{questions_batch}': {e}"
                )
                # Add error responses for all k values
                for k in ks:
                    results_by_k[k].extend(
                        [
                            RagResponse(
                                answer="",
                                query=question,
                                model=self.generator.model,
                                retrieval_scores=[],
                                n_context=0,
                                retrieval_result=RetrievalResult(items=[]),
                            )
                            for question in questions_batch
                        ]
                    )

        return results_by_k

    def _slice_retrieval_result(
        self, retrieval_result: RetrievalResult, k: int
    ) -> RetrievalResult:
        """Create a new RetrievalResult with only the top-k items."""
        return RetrievalResult(items=retrieval_result.items[:k])


class TwoStageRAGPipeline:
    """RAG pipeline that performs two-stage retrieval (query-based + chunk-based).

    Stage 1: retrieve over query vectors (e.g. synthetic/eval queries) with
    ``retrieval_sources=["query"]`` and configurable ``retrieval_query_origins``.

    Stage 2: retrieve over chunk vectors (``retrieval_sources=["chunk"]``) for
    the same questions using the same embeddings.

    Downstream code can fetch both retrieval results separately and/or build a
    merged ``RetrievalResult`` and pass it to the existing ``LLMGenerator``.
    """

    def __init__(
        self,
        k_query: int,
        k_chunk: int,
        rag_llm: str = settings.rag_generation_model,
        qgen_llm: Optional[str] = None,
        emb_model: str = settings.openai_embedding_model,
        dataset: str = settings.rag_dataset,
        deduplication_factor: int = settings.rag_deduplication_factor,
        qa_type: str = settings.question_generation_mode,
        retrieval_query_origins_query: List[str] = ["two-stage"],
    ) -> None:
        self.k_query = k_query
        self.k_chunk = k_chunk
        self.rag_llm = rag_llm
        self.qgen_llm = qgen_llm
        self.emb_model = emb_model
        self.dataset = dataset
        self.deduplication_factor = deduplication_factor
        self.qa_type = qa_type

        # Query-based retriever (stage 1)
        self.query_retriever = DatabaseKNNRetriever(
            k=self.k_query,
            metric=settings.rag_retrieval_metric,
            algorithm=settings.rag_retrieval_algorithm,
            model=self.qgen_llm,
            emb_model=self.emb_model,
            dataset=self.dataset,
            retrieval_sources=["query"],
            deduplication_factor=self.deduplication_factor,
            qa_type=self.qa_type,
            retrieval_query_origins=retrieval_query_origins_query,
        )

        # Chunk-based retriever (stage 2)
        self.chunk_retriever = DatabaseKNNRetriever(
            k=self.k_chunk,
            metric=settings.rag_retrieval_metric,
            algorithm=settings.rag_retrieval_algorithm,
            model=self.qgen_llm,
            emb_model=self.emb_model,
            dataset=self.dataset,
            retrieval_sources=["chunk"],
            deduplication_factor=self.deduplication_factor,
            qa_type=self.qa_type,
            # retrieval_query_origins is ignored when retrieval_sources=["chunk"]
            retrieval_query_origins=None,
        )

        self.generator = LLMGenerator()
        self.is_fitted = False

    async def fit(self, session: AsyncSession) -> None:
        """Fit both underlying retrievers on the database vectors."""
        await self.query_retriever.fit_index(session)
        await self.chunk_retriever.fit_index(session)
        self.is_fitted = True

    async def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Create embeddings for input questions."""
        response = await create_embeddings(
            settings.get_openai_client(embedding=True),
            model=self.emb_model,
            input=queries,
            encoding_format="float",
        )
        if response is None:
            logger.warning(
                f"No embeddings returned from the API for queries: {queries}"
            )
            return np.array([np.nan] * len(queries), dtype=np.float32)
        embeddings = np.array(
            [item.embedding for item in response.data],
            dtype=np.float32,
        )
        return embeddings

    def retrieve(
        self, query_embeddings: np.ndarray
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """Run both stages of retrieval for the provided embeddings.

        Returns a tuple ``(query_results, chunk_results)`` where each element is
        a list of ``RetrievalResult`` aligned by query index.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before retrieval")

        query_results = self.query_retriever.retrieve(query_embeddings)
        chunk_results = self.chunk_retriever.retrieve(query_embeddings)
        return query_results, chunk_results

    async def generate(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        generate_answer: bool = settings.rag_generate_answer,
    ) -> RagResponse:
        """Delegate to the standard LLM generator using a merged RetrievalResult."""
        return await self.generator.generate_response(
            question,
            retrieval_result,
            generate_answer=generate_answer,
        )
