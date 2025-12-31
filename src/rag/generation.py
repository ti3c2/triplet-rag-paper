import logging
from typing import Iterable, List, Optional

import openai
from pydantic import BaseModel, ConfigDict, Field

from ..config.settings import settings
from ..config.utils import create_chat_completion
from .models import (
    ChunkMetadata,
    RagResponse,
    RetrievalItem,
    RetrievalResult,
    Triplet,
    TripletContext,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)


class LLMGenerator(BaseModel):
    model: str = settings.rag_generation_model
    temperature: float = settings.rag_generation_temperature
    max_tokens: int = settings.rag_generation_max_tokens
    system_prompt: str = settings.prompt_rag_system
    response_prompt: str = settings.prompt_rag_response
    client: openai.AsyncOpenAI = Field(
        default_factory=lambda: settings.get_openai_client(
            base_url=settings.rag_generation_model_api_base
        ),
        exclude=True,
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def format_context(self, retrieval_result: RetrievalResult) -> str:
        """Format retrieved contexts, preserving query/chunk semantics.

        - For query items: emit Question and Answer (and optional source chunk).
        - For chunk items: emit the chunk text.
        """
        parts: list[str] = ["[Retrieved Contexts]"]
        for i, item in enumerate(retrieval_result.items):
            if item.metadata.type == "query":
                # Optional chunk text carried with the query metadata
                maybe_chunk_text = item.metadata.chunk_text
                maybe_chunk_text = (
                    maybe_chunk_text.strip()
                    if isinstance(maybe_chunk_text, str)
                    else None
                )
                if settings.rag_use_chunk_for_qa_context and maybe_chunk_text:
                    parts.append(f"[Context {i+1}]\n{maybe_chunk_text}")
                else:
                    question_text = item.metadata.query or ""
                    # If qa-type 'qa', strip appended answer from question
                    if item.metadata.qa_type == "qa" and question_text:
                        question_text = question_text.split(settings.qa_separator)[0]
                    lines = [
                        f"[QA Context {i+1}]",
                        f"Question: {question_text}",
                    ]
                    if settings.rag_add_chunk_to_query and maybe_chunk_text:
                        lines.append(f"Context: {maybe_chunk_text}")
                    lines.append(f"Answer: {item.text.strip()}")
                    parts.append("\n".join(lines))
            elif item.metadata.type == "chunk":
                parts.append(f"[Context {i+1}]\n{item.text.strip()}")
            else:
                logger.warning(f"Unknown item type: {item.metadata.type}")
                continue
        return "\n\n".join(parts)

    def create_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        def strip_qa(q: str) -> str:
            return q.split(settings.qa_separator)[0].strip()

        question_clean = strip_qa(question) or ""
        return "\n".join(
            [
                self.system_prompt,
                self.response_prompt.format(question=question_clean, context=context),
            ]
        )

    async def generate_response(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        generate_answer: bool = settings.rag_generate_answer,
    ) -> RagResponse:
        try:
            context = self.format_context(retrieval_result)
            prompt = self.create_prompt(question=question, context=context)
            if generate_answer:
                logger.debug(f"Generating answer for question: {question}")
                response = await create_chat_completion(
                    self.client,
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                answer = response.choices[0].message.content or ""
            else:
                logger.debug(f"Skipping answer generation for question: {question}")
                answer = ""
            result = RagResponse(
                answer=answer,
                query=question,
                model=self.model,
                retrieval_scores=retrieval_result.scores,
                n_context=len(retrieval_result.texts),
                retrieval_result=retrieval_result,
            )
            return result

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def _select_triplet_contexts(
        self,
        triplet: Triplet,
        k: Optional[int],
        n_contexts: int = settings.rag_n_triplet_contexts,
    ) -> List[TripletContext]:
        if k is not None and k in triplet.contexts_by_k:
            return triplet.contexts_by_k[k][:n_contexts]
        # fallback: pick closest <= k, else max available
        keys = sorted(triplet.contexts_by_k.keys())
        if not keys:
            return []
        if k is None:
            return triplet.contexts_by_k[keys[-1]]
        leq = [key for key in keys if key <= k]
        use_k = max(leq) if leq else keys[-1]
        return triplet.contexts_by_k[use_k]

    def format_triplet_blocks(
        self,
        triplets: Iterable[Triplet],
        k: Optional[int],
        n_contexts: int = settings.rag_n_triplet_contexts,
    ) -> str:
        parts: list[str] = ["[Retrieved Contexts from Triplets]"]
        for idx, t in enumerate(triplets, start=1):
            ctxs = self._select_triplet_contexts(t, k, n_contexts)
            # Contexts
            for j, c in enumerate(ctxs, start=1):
                parts.append(f"[Triplet {idx} - Context {j}]\n{c.text.strip()}")
            # Reference QA
            if t.ref_q:
                parts.append(f"[Triplet {idx} - Reference Question]\n{t.ref_q.strip()}")
            if t.ref_a:
                parts.append(f"[Triplet {idx} - Reference Answer]\n{t.ref_a.strip()}")
        return "\n\n".join(parts)

    def _triplets_to_retrieval_items(
        self,
        triplets: Iterable[Triplet],
        k: Optional[int],
        n_contexts: int = settings.rag_n_triplet_contexts,
        dataset: Optional[str] = None,
        emb_model: Optional[str] = None,
    ) -> List[RetrievalItem]:
        items: list[RetrievalItem] = []
        for t in triplets:
            for c in self._select_triplet_contexts(t, k, n_contexts):
                meta = ChunkMetadata(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    dataset=dataset or "",
                    doc_title=c.doc_title,
                    type="chunk",
                    emb_model=emb_model,
                )
                items.append(
                    RetrievalItem(
                        score=float(c.score),
                        text=c.text,
                        metadata=meta,
                    )
                )
        return items

    async def generate_triplets(
        self,
        question: str,
        triplets: List[Triplet],
        k: Optional[int] = None,
        n_contexts: int = settings.rag_n_triplet_contexts,
        dataset: str = settings.rag_dataset,
        emb_model: str = settings.openai_embedding_model,
        generate_answer: bool = settings.rag_triplet_generate_answer,
    ) -> RagResponse:
        """Generate an answer using multiple triplets (multiple ref_q/ref_a and contexts).

        Builds a combined context block enumerating triplet contexts and reference QA pairs.
        Also constructs a RetrievalResult from the combined contexts for downstream logging/metrics.
        """
        try:
            multi_context = self.format_triplet_blocks(triplets, k, n_contexts)
            prompt = self.create_prompt(question=question, context=multi_context)

            if generate_answer:
                response = await create_chat_completion(
                    self.client,
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                answer = response.choices[0].message.content or ""
            else:
                answer = ""
                logger.error(f"Skipping answer generation for question: {question}")

            items = self._triplets_to_retrieval_items(
                triplets,
                k,
                n_contexts=n_contexts,
                dataset=dataset,
                emb_model=emb_model,
            )
            retrieval_result = RetrievalResult(items=items)
            return RagResponse(
                answer=answer,
                query=question,
                model=self.model,
                retrieval_scores=[it.score for it in items],
                n_context=len(items),
                retrieval_result=retrieval_result,
            )
        except Exception as e:
            logger.error(f"Error generating response (triplets): {str(e)}")
            raise
