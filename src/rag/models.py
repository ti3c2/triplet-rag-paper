import logging
from typing import Any, List, Literal, Optional

import numpy as np
from pydantic import BaseModel

from ..config.utils import fill_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_unique_attr(
    items: List[Any],
    attr: str,
    ignore: Optional[List[Any]] = None,
    default: Optional[Any] = None,
) -> Any:
    items = {getattr(item, attr) for item in items}
    items = items.difference({None})
    if ignore is not None:
        items = items.difference(ignore)
    if len(items) > 1:
        logger.error(f"Multiple {attr} found: {items}")
    return items.pop() if items else default


class ChunkData(BaseModel):
    """Pydantic model for retrieval data from database chunks."""

    chunk_id: int
    chunk_text: str
    doc_id: int
    dataset: str
    doc_title: str
    type: Literal["chunk"]


class QueryData(BaseModel):
    """Pydantic model for retrieval data from database queries."""

    query_id: int
    query_text: str
    chunk_id: Optional[int]
    doc_id: int
    qa_type: Optional[str]
    llm: Optional[str]
    dataset: str
    doc_title: str
    doc_text: Optional[str] = None
    chunk_text: Optional[str] = None
    answer_text: Optional[str]
    prompt_name: Optional[str] = None
    origin: Optional[str] = None
    type: Literal["query"]


class EvaluationQueryData(BaseModel):
    """Pydantic model for querying data from database for evaluation."""

    id: int
    text: str
    text_hash: str
    chunk_id: Optional[int]
    doc_id: int
    doc_meta: Optional[dict]
    doc_title: str


class EvaluationEmbeddingData(BaseModel):
    """Pydantic model for evaluation embedding data."""

    vector: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class ChunkMetadata(BaseModel):
    """Pydantic model for metadata of retrieved chunks."""

    chunk_id: int
    doc_id: int
    dataset: str
    query_id: Optional[int] = None
    doc_title: Optional[str] = None  # required for retrieval metrics
    type: Literal["chunk", "query"]
    query: Optional[str] = None  # only for 'query' type chunks
    # context: Optional[str] = None  # only for 'query' type chunks
    qa_type: Optional[str] = None  # only for 'query' type chunks
    qgen_model: Optional[str] = None  # only for 'query' type chunks
    qgen_prompt_name: Optional[str] = None  # only for 'query' type chunks
    chunk_text: Optional[str] = None  # only for 'query' type chunks
    origin: Optional[str] = None  # only for 'query' type chunks
    emb_model: Optional[str] = None
    meta: Optional[dict] = None


class GroundTruthItem(BaseModel):
    """Pydantic model for ground truth items."""

    doc_ids: List[int]
    doc_titles: List[str]
    chunk_ids: Optional[List[int]] = None
    chunk_texts: Optional[List[str]] = None
    query_id: Optional[int] = None
    query: Optional[str] = None
    answer: Optional[str] = None
    meta: Optional[dict] = None

    @property
    def doc_id(self) -> int:  # TODO: Remove this property
        return self.doc_ids[0]

    @property
    def doc_title(self) -> str:
        return self.doc_titles[0]

    @property
    def chunk_id(self) -> Optional[int]:
        if self.chunk_ids:
            return self.chunk_ids[0]
        return None

    @property
    def chunk_text(self) -> Optional[str]:
        if self.chunk_texts:
            return self.chunk_texts[0]
        return None

    def format_str(self) -> str:
        out = []
        out.append(f"Doc={self.doc_ids} chunks={self.chunk_ids} - {self.doc_title}")
        for k, v in self.model_dump(
            exclude_none=True,
            exclude=["doc_ids", "doc_titles", "chunk_ids", "chunk_texts"],
        ).items():
            out.append(f"{k.capitalize()}: {v}")
        if self.chunk_ids:
            for chunk_id, chunk_text in zip(self.chunk_ids, self.chunk_texts):
                out.append(fill_text(f"Chunk: {chunk_id}"))
                out.append(fill_text(chunk_text, indent="\t"))
        return "\n".join(out)


class RetrievalItem(BaseModel):
    """Pydantic model for retrieval item with score."""

    score: float
    text: str
    metadata: ChunkMetadata


class RetrievalResult(BaseModel):  # TODO: Refactor this class in the rest of the code
    items: List[RetrievalItem]

    @property
    def scores(self) -> List[float]:
        return [item.score for item in self.items]

    @property
    def texts(self) -> List[str]:
        return [item.text for item in self.items]

    @property
    def metadata(self) -> List[ChunkMetadata]:
        return [item.metadata for item in self.items]

    @property
    def doc_ids(self) -> List[int]:
        return [item.metadata.doc_id for item in self.items]

    @property
    def doc_titles(self) -> List[Optional[str]]:
        return [item.metadata.doc_title for item in self.items]

    def get_metadata_attr(
        self, attr: str, ignore_items: Optional[List[Any]] = None
    ) -> Any:
        metadatas = [item.metadata for item in self.items]
        return get_unique_attr(metadatas, attr, ignore_items)

    @property
    def qgen_model(self) -> str:
        return self.get_metadata_attr("qgen_model")

    @property
    def emb_model(self) -> str:
        return self.get_metadata_attr("emb_model")

    @property
    def dataset(self) -> str:
        return self.get_metadata_attr("dataset")

    @property
    def qa_type(self) -> str:
        return self.get_metadata_attr("qa_type", ignore_items=["c"])

    @property
    def qgen_prompt_name(self) -> str:
        return self.get_metadata_attr("qgen_prompt_name")


class RagResponse(BaseModel):
    """Pydantic model for RAG response."""

    answer: Optional[str] = None
    query: str
    model: Optional[str] = None
    retrieval_scores: List[float]
    n_context: int
    retrieval_result: RetrievalResult

    def count_items(self, type: Optional[Literal["query", "chunk"]] = None) -> int:
        if type is None:
            return len(self.retrieval_result.items)
        return sum(item.metadata.type == type for item in self.retrieval_result.items)

    def format_str(self) -> str:
        out = []
        out.append(f"Q: {self.query}")
        out.append(f"A: {self.answer}")
        out.append(
            f"N Queries={self.count_items(type='query')}, "
            f"N Chunks={self.count_items(type='chunk')}"
        )
        out.append("R:")
        for item in self.retrieval_result.items:
            if query_text := item.metadata.query:
                query_text = query_text.split("\n")[0]
            else:
                query_text = item.metadata.type
            out.append(f"\tType: `{item.metadata.type}` - {query_text}")
            if qa_type := item.metadata.qa_type:
                out.append(f"\tQA Type: `{qa_type}`")
            out.append(
                f"\tR: doc={item.metadata.doc_id}/chunk={item.metadata.chunk_id} - "
                f"{item.metadata.doc_title} - {item.score}"
            )
            out.append(
                fill_text(
                    item.text,
                    indent="\t",
                )
            )
            out.append("")
        out.append("-" * 100)
        return "\n".join(out)

    @property
    def qgen_model(self) -> str:
        return self.retrieval_result.qgen_model

    @property
    def emb_model(self) -> str:
        return self.retrieval_result.emb_model

    @property
    def dataset(self) -> str:
        return self.retrieval_result.dataset

    @property
    def qa_type(self) -> str:
        return self.retrieval_result.qa_type

    @property
    def qgen_prompt_name(self) -> str:
        return self.retrieval_result.qgen_prompt_name


class EvaluationMetrics(BaseModel):
    """Pydantic model for evaluation metrics."""

    deduplication_factor: int
    k: int

    # Classical retrieval metrics
    context_accuracy: Optional[float] = None
    title_accuracy: Optional[float] = None
    full_match_accuracy: Optional[float] = None
    partial_match_accuracy: Optional[float] = None
    mrr: Optional[float] = None
    ndcg: Optional[float] = None
    queries_ratio: Optional[float] = None

    # Ragas metrics (using ragas naming conventions)
    ragas_n_evals: Optional[int] = None

    # Generation metrics
    ragas_faithfulness: Optional[float] = None
    ragas_answer_relevancy: Optional[float] = None
    ragas_nv_accuracy: Optional[float] = None
    ragas_nv_context_relevance: Optional[float] = None
    ragas_nv_response_groundedness: Optional[float] = None
    ragas_summary_score: Optional[float] = None

    # Classical metrics for generation
    ragas_rouge_score: Optional[float] = None
    ragas_bleu_score: Optional[float] = None
    ragas_non_llm_string_similarity: Optional[float] = None
    ragas_string_present: Optional[float] = None
    ragas_exact_match: Optional[float] = None
    ragas_semantic_similarity: Optional[float] = None
    ragas_factual_correctness: Optional[float] = None

    # Retrieval metrics
    ragas_context_precision: Optional[float] = None
    ragas_context_recall: Optional[float] = None
    ragas_context_entity_recall: Optional[float] = None
    ragas_noise_sensitivity: Optional[float] = None

    def format_str(self) -> str:
        out = []
        for k, v in self.model_dump(exclude_none=True).items():
            out.append(f"{k}: {v}")
        return "\n".join(out)


class EvaluationResult(BaseModel):
    """Pydantic model for evaluation metrics and retrieved data."""

    metrics: List[EvaluationMetrics]
    rag_responses: List[RagResponse]
    ground_truth: List[List[GroundTruthItem]]
    experiment_name: Optional[str] = None  # For spreadsheet updates
    eval_llm: Optional[str] = None
    eval_embedding_model: Optional[str] = None

    def get_retres_attr(self, attr: str, ignore: Optional[List[Any]] = None) -> Any:
        retrieval_results = [
            response.retrieval_result for response in self.rag_responses
        ]
        return get_unique_attr(retrieval_results, attr, ignore)

    def get_ragres_attr(
        self,
        attr: str,
        ignore: Optional[List[Any]] = None,
        default: Optional[Any] = None,
    ) -> Any:
        return get_unique_attr(self.rag_responses, attr, ignore, default)

    @property
    def qgen_prompt_name(self) -> str:
        return self.get_retres_attr("qgen_prompt_name")

    @property
    def qgen_model(self) -> str:
        return self.get_retres_attr("qgen_model")

    @property
    def qa_size(self) -> int:
        return len(self.rag_responses) // len(self.metrics)

    @property
    def qa_type(self) -> str:
        return self.get_ragres_attr("qa_type", default="unk")

    def get_responses_for_k(self, k: int) -> List[RagResponse]:
        """Get responses for a specific k value by filtering the responses list."""
        k_responses = [
            response for response in self.rag_responses if response.n_context == k
        ]
        if not k_responses:
            available_ks = self.get_available_ks()
            raise ValueError(
                f"No responses found for k={k}. Available k values: {available_ks}"
            )
        return k_responses

    def get_available_ks(self) -> List[int]:
        """Get all available k values from the responses."""
        return sorted(list(set(response.n_context for response in self.rag_responses)))

    def format_str(self) -> str:
        out = []
        out.append("Evaluation Results:")

        if self.rag_responses and self.rag_responses[0].retrieval_result.items:
            out.append(
                f"Dataset: {self.rag_responses[0].retrieval_result.items[0].metadata.dataset}"
            )
        out.append("")

        if self.rag_responses:
            out.append("Models:")
            out.extend(
                [
                    f"Embedding Model: {self.rag_responses[0].retrieval_result.emb_model}",
                    f"Question Generation Model: {self.rag_responses[0].retrieval_result.qgen_model}",
                    f"RAG Model: {self.rag_responses[0].model}",
                    f"Evaluation LLM: {self.eval_llm}",
                    f"Evaluation Embedding Model: {self.eval_embedding_model}",
                ]
            )
            out.append("")

        out.append("Metrics:")
        for metric in self.metrics:
            out.append(metric.format_str())
            out.append("")

        # Show available k values
        available_ks = self.get_available_ks()
        if len(available_ks) > 1:
            out.append(f"Available k values: {available_ks}")
            out.append("")

        out.append("=" * 100)

        # Show responses for all available k values
        for k in available_ks:
            out.append(f"RESPONSES FOR k={k}")
            out.append("-" * 50)
            k_responses = self.get_responses_for_k(k)

            for i, (gts, rag_response) in enumerate(
                zip(self.ground_truth, k_responses)
            ):
                out.append(f"Evaluation {i+1} (k={k}):")
                out.append("Ground Truth:")
                for gt in gts:
                    out.append(gt.format_str())
                out.append("")
                out.append(f"RAG Response (k={k}):")
                out.append(rag_response.format_str())
                out.append("")
            out.append("=" * 100)

        return "\n".join(out)


class TripletContext(BaseModel):
    type: Literal["chunk", "query"]
    doc_id: int
    doc_title: Optional[str]
    chunk_id: int
    text: str
    score: float
    rank: int


class Triplet(BaseModel):
    """Pydantic model representing a stored triplet used for triplet-augmented evaluation.

    This mirrors the essential shape of the DB `QueryTriplet` but is independent
    of SQLAlchemy. It can be created from DB rows or JSON and used to build
    retrieval results for a specific k.
    """

    evaluation_set_id: Optional[int] = None
    eval_query_id: int
    gold_answer_id: Optional[int] = None
    contexts_by_k: dict[int, list[TripletContext]]
    rag_answers_by_k: Optional[dict[int, str]] = None
    # Optional: store reference eval question/answer directly in JSON to avoid DB lookups
    ref_q: Optional[str] = None
    ref_a: Optional[str] = None


class Triplets(BaseModel):
    triplets: List[Triplet]

