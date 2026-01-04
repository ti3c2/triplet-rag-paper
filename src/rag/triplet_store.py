import json
import logging
from pathlib import Path
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config.settings import settings
from ..store_rel.schema import QueryTriplet
from .models import (
    ChunkMetadata,
    RetrievalItem,
    RetrievalResult,
    Triplet,
    TripletContext,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)


async def fetch_triplet_by_eval_query_id(
    session: AsyncSession, eval_query_id: int
) -> Optional[Triplet]:
    result = await session.execute(
        select(QueryTriplet).where(
            QueryTriplet.eval_query_id == eval_query_id,
            QueryTriplet.archived_at.is_(None),
        )
    )
    row = result.scalars().first()
    if not row:
        return None
    return Triplet(
        evaluation_set_id=row.evaluation_set_id,
        eval_query_id=row.eval_query_id,
        gold_answer_id=row.gold_answer_id,
        contexts_by_k=row.contexts_by_k or {},
        rag_answers_by_k=row.rag_answers_by_k or {},
    )


async def fetch_triplets_by_eval_query_ids(
    session: AsyncSession, eval_query_ids: list[int]
) -> dict[int, Triplet]:
    if not eval_query_ids:
        return {}
    result = await session.execute(
        select(QueryTriplet).where(
            QueryTriplet.eval_query_id.in_(eval_query_ids),
            QueryTriplet.archived_at.is_(None),
        )
    )
    rows = result.scalars().all()
    logger.info(f"Loaded {len(rows)} triplets from DB")
    out: dict[int, Triplet] = {}
    for r in rows:
        t = Triplet(
            evaluation_set_id=r.evaluation_set_id,
            eval_query_id=r.eval_query_id,
            gold_answer_id=r.gold_answer_id,
            contexts_by_k=r.contexts_by_k or {},
            rag_answers_by_k=r.rag_answers_by_k or {},
        )
        out[t.eval_query_id] = t
    return out


def load_triplets_from_json(json_path: str | Path) -> dict[int, Triplet]:
    """Load a simple triplet map from a JSON file created during evaluation.

    Expected structure:
    {
      "triplets": [
        {
          "eval_query_id": 123,
          "contexts_by_k": { "1": [...], "5": [...] },
          "rag_answers_by_k": { "1": "...", "5": "..." }
        },
        ...
      ]
    }
    """
    logger.info(f"Loading triplets from JSON: {json_path}")
    path = settings.find_file(json_path, base=settings.path_eval)
    if path.is_dir():
        path = settings.find_file("triplets", path)
    if path is None or not path.exists():
        logger.error(f"Triplet JSON not found: {path}")
        raise FileNotFoundError(f"Triplet JSON not found: {path}")
    logger.info(f"Found triplets file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        out: dict[int, Triplet] = {}
        for t in data.get("triplets", []):
            qt = Triplet(**t)
            out[qt.eval_query_id] = qt
        logger.info(f"Loaded {len(out)} triplets from JSON")
        return out
    except Exception as e:
        logger.error(f"Failed to load triplets from {path}: {e}")
        return {}


def index_triplets_by_chunk(
    triplet_map: dict[int, Triplet],
) -> dict[int, list[Triplet]]:
    """Build an index from chunk_id -> list of triplets that contain this chunk in any k."""
    chunk_to_triplets: dict[int, list[Triplet]] = {}
    for t in triplet_map.values():
        for ctx_list in t.contexts_by_k.values():
            for c in ctx_list:
                cid = c.chunk_id
                if isinstance(cid, int):
                    chunk_to_triplets.setdefault(cid, []).append(t)
    return chunk_to_triplets


def index_triplets_by_eval_query(
    triplet_map: dict[int, Triplet],
) -> dict[int, Triplet]:
    """Index triplets by their eval_query_id for deterministic matching."""
    return {t.eval_query_id: t for t in triplet_map.values()}


def find_triplet_for_retrieval_by_chunks(
    triplet_chunk_map: dict[int, list[Triplet]],
    retrieval_result: RetrievalResult,
) -> Triplet | None:
    """Pick the first triplet whose contexts include any chunk_id from the retrieval result."""
    for item in retrieval_result.items:
        cid = item.metadata.chunk_id
        if isinstance(cid, int) and cid in triplet_chunk_map:
            # return the first associated triplet
            lst = triplet_chunk_map[cid]
            if lst:
                return lst[0]
    return None


def find_triplets_for_retrieval_by_eval_query(
    triplet_eval_map: dict[int, Triplet],
    retrieval_result: RetrievalResult,
    allowed_origins: List[str] | None = None,
) -> List[Triplet | None]:
    """Match triplet using query ids present in retrieval result.

    By default this matches queries whose ``origin`` field is in
    ``allowed_origins`` (if provided). Existing callers can pass
    ``allowed_origins=[\"eval\"]`` to keep the original behavior.
    """
    logger.debug(retrieval_result.model_dump_json(indent=2))
    out: List[Triplet | None] = []
    for item in retrieval_result.items:
        meta = item.metadata
        origin = getattr(meta, "origin", None)
        if (
            meta.type == "query"
            and isinstance(meta.query_id, int)
            and meta.query_id in triplet_eval_map
            and (allowed_origins is None or origin in allowed_origins)
        ):
            out.append(triplet_eval_map[meta.query_id])
        else:
            out.append(None)
    return out


def _contexts_json_to_items(
    contexts: list[TripletContext],
    dataset: Optional[str],
    emb_model: Optional[str],
) -> List[RetrievalItem]:
    items: List[RetrievalItem] = []
    for c in contexts:
        # We preserve text and identifiers; treat as chunk-type for generator formatting stability
        metadata = ChunkMetadata(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            dataset=dataset or "",
            doc_title=c.doc_title,
            type="chunk",  # TODO: Think about handling queries as well
            emb_model=emb_model,
        )
        item = RetrievalItem(
            score=float(c.score),
            text=c.text,
            metadata=metadata,
        )
        items.append(item)
    return items


def expand_with_triplet_contexts(
    retrieval_result: RetrievalResult,
    triplet: Triplet,
    max_extra: int = -1,
) -> RetrievalResult:
    """Append stored contexts from a triplet to an existing retrieval result.

    Deduplicate by chunk_id/text to avoid repeats.
    """
    existing_chunk_ids = {m.chunk_id for m in retrieval_result.metadata if m.chunk_id}
    existing_texts = {it.text for it in retrieval_result.items}

    all_contexts: list = []
    for k_str, ctx_list in sorted(
        triplet.contexts_by_k.items(), key=lambda x: int(x[0])
    ):
        all_contexts.extend(ctx_list)

    extra_items = _contexts_json_to_items(
        all_contexts,
        dataset=(
            retrieval_result.items[0].metadata.dataset
            if retrieval_result.items
            else None
        ),
        emb_model=(
            retrieval_result.items[0].metadata.emb_model
            if retrieval_result.items
            else None
        ),
    )

    appended: List[RetrievalItem] = []
    for it in extra_items:
        if it.metadata.chunk_id and it.metadata.chunk_id in existing_chunk_ids:
            continue
        if it.text in existing_texts:
            continue
        appended.append(it)
        if max_extra > 0 and len(appended) >= max_extra:
            break

    return RetrievalResult(items=retrieval_result.items + appended)


def build_result_from_triplet(
    triplet: Triplet,
    dataset: Optional[str],
    emb_model: Optional[str],
    k: int,
) -> RetrievalResult:
    """Construct a RetrievalResult purely from a triplet's stored contexts for a given k.

    If exact k is not present, pick the largest available k not exceeding requested; otherwise, pick max.
    """
    # pick best-matching k
    keys = sorted((int(x) for x in triplet.contexts_by_k.keys()))
    use_k = None
    for key in keys:
        if key == k:
            use_k = key
            break
    if use_k is None:
        # choose closest <= k, else fallback to max
        leq = [key for key in keys if key <= k]
        use_k = max(leq) if leq else max(keys)

    ctx = triplet.contexts_by_k.get(use_k, [])
    items = _contexts_json_to_items(ctx, dataset=dataset, emb_model=emb_model)
    return RetrievalResult(items=items)
