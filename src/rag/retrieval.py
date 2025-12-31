import logging
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config.settings import settings
from ..store_rel.schema import (
    Answer,
    Chunk,
    ChunkVector,
    Doc,
    Prompt,
    Query,
    QueryVector,
    get_prompt,
)
from .models import ChunkData, ChunkMetadata, QueryData, RetrievalItem, RetrievalResult

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)


class KNNRetriever:
    """Exact KNN retriever using sk-learn."""

    def __init__(
        self,
        k: int = settings.rag_retrieval_k,
        metric: str = settings.rag_retrieval_metric,
        algorithm: str = settings.rag_retrieval_algorithm,
    ):
        self.k = k
        self.metric = metric
        self.algorithm = algorithm
        self.index = NearestNeighbors(n_neighbors=k, metric=metric, algorithm=algorithm)

    def fit(self, corpus_data: np.ndarray) -> None:
        """Fit the KNN index on corpus embeddings.
        Args:
            corpus_data: np.ndarray of shape (n_samples, emb_dim)
                The embeddings of the corpus.
        Returns:
            None
        """
        if corpus_data.size == 0:
            raise ValueError("Cannot fit retriever on empty corpus data")

        self.index.fit(corpus_data)
        self.is_fitted = True
        logger.info(f"KNN retriever fitted with {corpus_data.shape[0]} vectors")

    def predict(self, query_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve k nearest neighbors for query vectors.
        Args:
            query_data: np.ndarray of shape (n_samples, emb_dim)
                The embeddings of the query.
        Returns:
            indices: np.ndarray of shape (n_samples, k)
                The indices of the k nearest neighbors.
            scores: np.ndarray of shape (n_samples, k)
                The distances to the k nearest neighbors.
        """
        if not self.is_fitted:
            raise RuntimeError("Retriever must be fitted before retrieval")

        if query_data.ndim == 1:
            query_data = query_data.reshape(1, -1)

        distances, indices = self.index.kneighbors(query_data)

        # Convert distances to similarity scores
        # For cosine sim=1-distance
        # For euclidean and others sim=1/(1+distance)
        if self.metric == "cosine":
            similarity_scores = 1 - distances
        elif self.metric == "euclidean":
            similarity_scores = 1 / (1 + distances)
        else:
            similarity_scores = 1 / (1 + distances)

        return indices, similarity_scores


class DatabaseKNNRetriever(KNNRetriever):
    """KNN retriever for the database."""

    def __init__(
        self,
        k: int = settings.rag_retrieval_k,
        metric: str = settings.rag_retrieval_metric,
        algorithm: str = settings.rag_retrieval_algorithm,
        model: str = settings.question_generation_model,
        emb_model: str = settings.openai_embedding_model,
        dataset: str = settings.rag_dataset,
        retrieval_sources: List[
            Literal["chunk", "query"]
        ] = settings.rag_retrieval_sources,
        deduplication_factor: int = 1,
        qa_type: str = settings.question_generation_mode,
        retrieval_query_origins: List[str] = settings.rag_retrieval_query_origins,
    ):
        super().__init__(k=k * deduplication_factor, metric=metric, algorithm=algorithm)
        self._k = k
        self.deduplication_factor = deduplication_factor
        self.model = model
        self.emb_model = emb_model
        self.dataset = dataset
        self.fit_items: List[Union[ChunkData, QueryData]] = []
        self.retrieval_sources = retrieval_sources
        self.qa_type = qa_type
        self.retrieval_query_origins: List[str] = retrieval_query_origins

    async def fit_index(self, session: AsyncSession) -> None:
        """Fit the retriever using embeddings from the database.
        Args:
            session: SQLAlchemy session
        Returns:
            None
        """
        try:
            embeddings = []
            self.fit_items = []

            logger.info(f"DatabaseKNNRetriever: loading vectors from the database")

            # Load CHUNK vectors with all needed data to avoid lazy loading
            if "chunk" in self.retrieval_sources:
                chunk_q = await session.execute(
                    select(
                        ChunkVector.vector,
                        Chunk.id.label("chunk_id"),
                        Chunk.text.label("chunk_text"),
                        Chunk.doc_id,
                        Doc.dataset,
                        Doc.title.label("doc_title"),
                    )
                    .join(Chunk, ChunkVector.parent_id == Chunk.id)
                    .join(Doc, Chunk.doc_id == Doc.id)
                    .where(
                        Doc.dataset == self.dataset,
                        ChunkVector.emb_model == self.emb_model,
                    )
                )
                chunk_results = chunk_q.all()

                for row in chunk_results:
                    embeddings.append(row.vector)
                    # Create ChunkData model from database row
                    chunk_data = ChunkData(
                        **row._asdict(),
                        type="chunk",
                    )
                    self.fit_items.append(chunk_data)

                logger.info(
                    f"DatabaseKNNRetriever: added {len(chunk_results)} chunk vectors to the index"
                )

            # Load QUERY vectors with all needed data to avoid lazy loading
            if "query" in self.retrieval_sources:
                prompt = settings.question_generation_default_prompt
                prompt_db = await get_prompt(session, prompt.name)
                prompt_id = prompt_db.id
                logger.info(f"Using prompt: id={prompt_id}, name={prompt.name}")

                query_q = await session.execute(
                    select(
                        QueryVector.vector,
                        Query.id.label("query_id"),
                        Query.text.label("query_text"),
                        Query.chunk_id,
                        Query.doc_id,
                        Query.qa_type,
                        Query.llm,
                        Query.prompt_id,
                        Query.origin,
                        Prompt.name.label("prompt_name"),
                        Doc.dataset,
                        Doc.title.label("doc_title"),
                        Doc.text.label("doc_text"),
                        Chunk.text.label("chunk_text"),
                        Answer.text.label("answer_text"),
                    )
                    .join(Query, QueryVector.parent_id == Query.id)
                    .join(Doc, Query.doc_id == Doc.id)
                    .join(Chunk, Query.chunk_id == Chunk.id)
                    .join(Prompt, Query.prompt_id == Prompt.id)
                    .outerjoin(
                        Answer, Query.id == Answer.query_id
                    )  # Left join for answers
                    .where(
                        Doc.dataset == self.dataset,
                        QueryVector.emb_model == self.emb_model,
                        Query.llm == self.model,
                        # # Only include generated queries which do not have hf_id
                        # Query.hf_id.is_(None),
                        Query.qa_type.in_(self.qa_type.split(",")),
                        Query.prompt_id == prompt_id,
                        Query.origin.in_(self.retrieval_query_origins),
                    )
                )
                query_results = query_q.all()

                for row in query_results:
                    embeddings.append(row.vector)
                    # Create QueryData model from database row
                    query_data = QueryData(
                        **row._asdict(),
                        type="query",
                    )
                    self.fit_items.append(query_data)

                if len(query_results) > 0:
                    logger.info(
                        f"DatabaseKNNRetriever: added {len(query_results)} query vectors to the index"
                    )
                else:
                    logger.error("Couldn't find query vectors for this setup")
                    raise ValueError("Couldn't find query vectors for this setup")

            if not embeddings:
                raise ValueError(f"Empty results")

            corpus_data = np.vstack(embeddings)
            self.fit(corpus_data)

            logger.info(f"DatabaseKNNRetriever fitted with {len(embeddings)} vectors")
        except Exception as e:
            logger.exception(f"Error fitting DatabaseKNNRetriever: {e}")
            raise e

    def retrieve(self, query_embeddings: np.ndarray) -> List[RetrievalResult]:
        """Retrieve similar chunks with full context.
        Args:
            query_embeddings: np.ndarray of shape (n_samples, emb_dim)
                The embeddings of the queries.
        Returns:
            List[RetrievalResult]
        """
        logger.debug(f"DatabaseKNNRetriever retrieving {query_embeddings.shape[0]} queries of dim {query_embeddings.shape[1]}")  # fmt:skip
        indices, scores = self.predict(query_embeddings)

        results: List[RetrievalResult] = []
        qa_type_fail_cnt = 0
        for i in range(indices.shape[0]):
            query_indices = indices[i]
            query_scores = scores[i]

            retrieved_items = [self.fit_items[j] for j in query_indices]

            chunk_ids = set()
            doc_ids = set()
            out_items = []
            logger.debug(
                f"Retrieved items: {[(getattr(item, 'query_id', None) or getattr(item, 'chunk_id', None), item.type) for item in retrieved_items]}"
            )

            for j, item in enumerate(retrieved_items):
                meta = {}
                query = None
                query_id = None
                qa_type = None
                qgen_prompt_name = None
                chunk_text = None
                origin = None

                if isinstance(item, QueryData):
                    # Use answer text if available, otherwise use doc text
                    if item.answer_text:
                        text = item.answer_text
                    else:
                        text = item.doc_text
                        logger.warning(
                            f"Query {item.query_id} has no answer, using doc text"
                        )
                    item_type = "query"
                    chunk_id = item.chunk_id
                    query_id = item.query_id
                    query = item.query_text
                    qgen_prompt_name = item.prompt_name
                    chunk_text = item.chunk_text
                    qa_type = item.qa_type
                    origin = item.origin
                elif isinstance(item, ChunkData):
                    qa_type = "c"
                    text = item.chunk_text
                    item_type = "chunk"
                    chunk_id = item.chunk_id
                else:
                    raise ValueError(f"Unknown item type: {type(item)}")

                doc_id = item.doc_id

                # fmt: off
                if self.deduplication_factor > 1:
                    if chunk_id in chunk_ids:
                        logger.debug(f"Deduplication skip by id for item {chunk_id} and doc {doc_id}")
                        logger.debug(f"Skipped item: {text}")
                        continue

                    if text in [retrieved_item.text for retrieved_item in out_items]:
                        logger.debug(f"Deduplication skip by text for item {chunk_id} and doc {doc_id}")
                        logger.debug(f"Skipped item: {text}")
                        continue

                    if (
                        item_type == "query"
                        and str(query)
                        and query in [retrieved_item.metadata.query for retrieved_item in out_items]
                    ):
                        logger.debug(f"Deduplication skip by query for item {chunk_id} and doc {doc_id}")
                        logger.debug(f"Skipped item: {text}")
                        continue
                # fmt: on

                chunk_ids.add(chunk_id)
                doc_ids.add(doc_id)

                # QA Type small fix
                qa_type_written = qa_type
                if qa_type_written is None:
                    qa_type_written = self.qa_type
                    qa_type_fail_cnt += 1

                out_items.append(
                    RetrievalItem(
                        score=query_scores[j],
                        text=text,  # pyright: ignore
                        metadata=ChunkMetadata(
                            chunk_id=chunk_id,  # pyright: ignore
                            doc_id=doc_id,  # pyright: ignore
                            dataset=item.dataset,
                            query_id=query_id,
                            doc_title=item.doc_title,
                            type=item_type,
                            query=query,  # pyright: ignore
                            chunk_text=chunk_text,
                            qa_type=qa_type_written,
                            qgen_model=getattr(item, "llm", None),
                            emb_model=self.emb_model,
                            origin=origin,
                            qgen_prompt_name=qgen_prompt_name,
                            meta=meta,
                        ),
                    )
                )

            log_method = (
                logger.debug if settings.log_retrieved_items == 0 else logger.info
            )
            log_method(f"Retrieved {len(out_items)} items for query {i}")
            final_items = out_items[: self._k]

            results.append(
                RetrievalResult(
                    items=final_items,
                )
            )

        if qa_type_fail_cnt > 0:
            logger.warning(
                f"QA Type not retrieved for {qa_type_fail_cnt} items. Used default value from settings."
            )

        return results
