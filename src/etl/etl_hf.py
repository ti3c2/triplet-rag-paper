import logging
from typing import Optional

import pandas as pd
from datasets import load_dataset
from sqlalchemy import select

from ..config.settings import settings
from ..config.utils import get_text_hash
from ..store_rel.entry import get_db
from ..store_rel.schema import Answer, Chunk, Doc, Query, get_or_create_item

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)


def shorten_hf_id(hf_id: str):
    return hf_id.split("/")[-1].lower()


async def process_qa_data(
    data: pd.DataFrame,
    dataset_name: Optional[str] = None,
    n_rows: int = settings.rag_max_rows,
    n_commited: int = settings.rag_max_rows // 10,
):
    """
    Process QA data from a datasets with the following columns:
    {
        "title": str,
        "text": str,
        "query": str,
        "answers": str,
    }
    """
    df_data = pd.DataFrame(data).query("text != ''")

    if n_rows > 0:
        df_data = df_data.head(n_rows)

    logger.info(f"Loaded {len(df_data)} rows from {dataset_name} dataset.")

    agg_dict = {
        "title": ("title", "first"),
        "texts": ("text", list),
        "hf_idxs": ("index", list),
        "queries": ("query", list),
    }
    if "answers" in df_data.columns:
        agg_dict["answers"] = ("answers", list)

    df_data_docs = (
        # concatenate all contexts for each doc
        df_data.reset_index()
        .drop_duplicates(subset=["text"])
        .groupby("title")
        .agg(**agg_dict)
        .reset_index(drop=True)
    )

    title2docid = {}
    hash2chunkid = {}
    item_counter = 1
    async with get_db() as session:
        for idx, row in df_data_docs.iterrows():
            text = "\n\n".join(row["texts"])
            doc, _ = await get_or_create_item(
                session,
                Doc,
                dataset=dataset_name,
                title=row["title"],
                text=text,
                hf_id=None,
                is_virtual=True,
                meta={
                    "hf_ids": row["hf_idxs"],
                },
            )
            doc_id = doc.id
            title2docid[row["title"]] = doc_id
            item_counter += 1

            for hf_idx, chunk_text in zip(
                row["hf_idxs"],
                row["texts"],
            ):
                chunk, _ = await get_or_create_item(
                    session,
                    Chunk,
                    doc_id=doc_id,
                    text=chunk_text,
                    hf_id=str(hf_idx),
                )
                chunk_id = chunk.id
                chunk_text_hash = chunk.text_hash
                hash2chunkid[chunk_text_hash] = chunk_id
                item_counter += 1

                if (item_counter) % n_commited == 0:
                    await session.commit()
                    logger.info(f"Committed {item_counter} items.")

        await session.commit()

        existing_queries_result = await session.execute(
            select(Query.hf_id)
            .join(Doc)
            .where(
                Doc.dataset == dataset_name,
                Query.hf_id.isnot(None),
            )
        )
        existing_query_hf_idxs = {hf_id for (hf_id,) in existing_queries_result.all()}
        logger.debug(f"Existing query hf_ids: {existing_query_hf_idxs}")
        item_counter = 1
        for hf_idx, row in df_data.iterrows():
            if pd.isna(row["query"]):
                logger.warning(f"Skipping query {hf_idx} because query text is empty")
                continue
            hf_idx = str(hf_idx)
            if hf_idx in existing_query_hf_idxs:
                logger.debug(f"Skipping query {hf_idx} because it already exists")
                continue
            doc_id = title2docid[row["title"]]
            chunk_hash = get_text_hash(row["text"])
            chunk_id = hash2chunkid[chunk_hash]
            qtext = str(row["query"]).strip()
            qtext = qtext + "?" if not qtext.endswith("?") else qtext
            query, _ = await get_or_create_item(
                session,
                Query,
                force_create=True,
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=qtext,
                llm=None,
                prompt_id=None,
                hf_id=str(hf_idx),
            )
            query_id = int(query.id)
            if "answers" in df_data.columns:
                answer_text = (
                    ", ".join(row["answers"])
                    if isinstance(row["answers"], list)
                    else row["answers"]
                )
                answer = Answer(
                    query_id=query_id,
                    text=answer_text,
                )
                session.add(answer)
                await session.flush()
            else:
                logger.warning(f"No answers found for query {hf_idx}")
            item_counter += 1
            if (item_counter) % n_commited == 0:
                await session.commit()
                logger.info(f"Committed {item_counter} queries.")
        await session.commit()

    logger.info(
        f"ETL step 1 complete: docs, chunks, queries filled for {dataset_name} dataset."
    )


async def etl_kilt_nq_to_sql(
    n_rows: int = settings.rag_max_rows,
    n_commited: int = settings.rag_max_rows // 10,
    return_df: bool = False,
):
    """
    ETL for Natural Questions using KILT:
    - facebook/kilt_tasks, subset 'nq'  (questions, answers, provenance)
    - facebook/kilt_wikipedia           (wiki text)
    We convert it into a dataframe with columns:
        title, text, query, answers
    and then reuse process_qa_data().
    """
    from datasets import load_dataset

    dataset_name = "facebook/kilt_tasks"
    subset_name = "nq"
    logger.info(f"Loading KILT dataset: {dataset_name}, subset={subset_name}")

    kilt_nq = load_dataset(
        dataset_name,
        subset_name,
        split="train",
        cache_dir=str(settings.path_data_datasets),
    )

    # Convert KILT-NQ to pandas
    df_kilt = pd.DataFrame(kilt_nq)

    # df_kilt has columns: ["id", "input", "meta", "output"]
    # each "output" is a list of {answer, meta, provenance}
    def unpack_output(outputs):
        # collect non-empty answer strings
        answers = [o.get("answer", "") for o in outputs if o.get("answer")]
        prov = None
        for o in outputs:
            for p in o.get("provenance", []):
                if p.get("wikipedia_id"):
                    prov = p
                    break
            if prov is not None:
                break

        if prov is None:
            return pd.Series(
                {
                    "answers": answers,
                    "wikipedia_id": None,
                    "title": None,
                    "start_paragraph_id": None,
                    "end_paragraph_id": None,
                }
            )

        return pd.Series(
            {
                "answers": answers,
                "wikipedia_id": prov.get("wikipedia_id"),
                "title": prov.get("title"),
                "start_paragraph_id": prov.get("start_paragraph_id"),
                "end_paragraph_id": prov.get("end_paragraph_id"),
            }
        )

    df_meta = df_kilt["output"].apply(unpack_output)

    df_nq = pd.concat(
        [
            df_kilt[["id", "input"]],
            df_meta,
        ],
        axis=1,
    )

    # Rename to match process_qa_data expectations
    df_nq = df_nq.rename(columns={"input": "query"})
    # Drop rows with missing wiki ids or titles
    df_nq = df_nq.dropna(subset=["wikipedia_id", "title"])
    # Drop rows with empty answers
    df_nq = df_nq[df_nq["answers"].apply(lambda x: x is not None and len(x) > 0)]

    if n_rows > 0:
        df_nq = df_nq.head(n_rows)

    logger.info(f"KILT-NQ: {len(df_nq)} rows after filtering.")

    # Load KILT Wikipedia - directly from JSON, fuck HuggingFace's processing
    logger.info("Loading KILT Wikipedia from raw JSON...")

    import json
    import os
    from pathlib import Path

    # Find the downloaded JSON file
    kilt_json_path = None

    # Build possible paths based on configuration
    hf_home = Path(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    )
    file_rel = "downloads/kilt_knowledgesource.json"
    possible_paths = [
        hf_home / "datasets" / file_rel,  # HF_HOME environment variable
        settings.path_data_datasets / file_rel,  # Custom cache dir from settings
    ]

    for path in possible_paths:
        path_str = str(path)
        if os.path.exists(path_str):
            kilt_json_path = path_str
            logger.info(f"Found KILT Wikipedia JSON: {path_str}")
            break

    if kilt_json_path is None:
        raise FileNotFoundError(
            f"KILT Wikipedia JSON not found. Tried: {[str(p) for p in possible_paths]}\n"
            f"Download it with: wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json -P {hf_home}/datasets/downloads/"
        )

    # Get the Wikipedia IDs we actually need (from the NQ data)
    needed_ids = set(df_nq["wikipedia_id"].dropna().astype(str).unique())
    logger.info(f"Need {len(needed_ids)} unique Wikipedia articles")

    # Read JSONL by line and filter
    wiki_data = []
    processed = 0

    logger.info("Reading and filtering Wikipedia articles...")
    with open(kilt_json_path, "r", encoding="utf-8") as f:
        for line in f:
            processed += 1
            if processed % 100000 == 0:
                logger.info(
                    f"Processed {processed:,} articles, found {len(wiki_data)}/{len(needed_ids)}"
                )

            article = json.loads(line)
            wiki_id = str(article.get("wikipedia_id", ""))

            if wiki_id in needed_ids:
                wiki_data.append(
                    {
                        "wikipedia_id": wiki_id,
                        "wikipedia_title": article.get("wikipedia_title", ""),
                        "paragraphs": article.get("text", []),
                    }
                )

                # Early exit if we found everything
                if len(wiki_data) >= len(needed_ids):
                    logger.info(
                        f"Found all {len(needed_ids)} articles after {processed:,} lines"
                    )
                    break

    logger.info(
        f"Loaded {len(wiki_data)}/{len(needed_ids)} Wikipedia articles from JSON"
    )
    df_wiki = pd.DataFrame(wiki_data)

    # Merge question-level info with wiki paragraphs
    df_merged = pd.merge(
        df_nq,
        df_wiki,
        on="wikipedia_id",
        how="left",
        validate="many_to_one",
    )

    def build_evidence_text(row):
        paras = row["paragraphs"]
        if paras is None:
            return None
        s = row["start_paragraph_id"]
        e = row["end_paragraph_id"]
        if pd.isna(s):
            return None
        s = int(s)
        e = int(e) if not pd.isna(e) else s
        s = max(0, s)
        e = max(s, e)
        e = min(e, len(paras) - 1)
        return "\n\n".join(paras[s : e + 1])

    df_merged["text"] = df_merged.apply(build_evidence_text, axis=1)
    df_merged["title"] = df_merged["wikipedia_title"]

    # Final DF for process_qa_data
    df_final = df_merged[["title", "text", "query", "answers"]].dropna(
        subset=["text", "query"]
    )

    logger.info(
        f"Prepared {len(df_final)} KILT-NQ QA rows with evidence paragraphs. "
        "Running process_qa_data..."
    )

    if return_df:
        return df_final

    await process_qa_data(
        df_final,
        settings.NATURAL_QUESTIONS + "-kilt",
        n_rows=n_rows,
        n_commited=n_commited,
    )


async def etl_nq_to_sql(
    n_rows: int = settings.rag_max_rows,
    n_commited: int = settings.rag_max_rows // 10,
    max_loaded_rows: int = 20_000_000,
    use_kilt: bool = True,
):
    if use_kilt:
        return await etl_kilt_nq_to_sql(n_rows, n_commited)

    # Load NQ dataset
    dataset_name = "BeIR/nq"
    logger.info(f"Loading dataset: {dataset_name}")
    corpus = load_dataset(
        dataset_name,
        "corpus",
        split="corpus",
        cache_dir=settings.path_data_datasets,
    )
    queries = load_dataset(
        dataset_name, "queries", split="queries", cache_dir=settings.path_data_datasets
    )
    qrels = load_dataset(
        f"{dataset_name}-qrels",
        "default",
        split="test",
        cache_dir=settings.path_data_datasets,
    )
    logger.info(
        f"Loaded {len(corpus)} corpus rows, "
        f"{len(queries)} queries rows and "
        f"{len(qrels)} qrels rows."
    )

    # Create dataframes
    logger.info("Processing dataframes")
    df_corpus = pd.DataFrame(corpus[:max_loaded_rows])
    df_queries = pd.DataFrame(queries)
    df_qrels = pd.DataFrame(qrels)

    # Join dataframes
    df_queries_ans = pd.merge(
        df_queries,
        df_qrels,
        left_on="_id",
        right_on="query-id",
    ).drop(columns=["_id", "title"])

    df_corpus_ans = (
        pd.merge(
            df_corpus,
            df_queries_ans,
            left_on="_id",
            right_on="corpus-id",
            suffixes=["_corpus", "_query"],
            how="left",
        )
        .drop(columns=["_id"])
        .rename(
            columns={
                "text_corpus": "text",
                "text_query": "query",
            }
        )
        .dropna(subset=["text", "query"])
        .query("score > 0")
    )

    # IMPORTANT: Set answers to the text of the corpus
    df_corpus_ans["answers"] = df_corpus_ans["text"]

    await process_qa_data(
        df_corpus_ans,
        settings.NATURAL_QUESTIONS,
        n_rows,
        n_commited,
    )


async def etl_squad_to_sql(
    n_rows: int = settings.rag_max_rows,
    n_commited: int = settings.rag_max_rows // 10,
):
    # Load SQuAD dataset
    dataset_name = "rajpurkar/SQuAD"
    logger.info(f"Loading dataset: {dataset_name}")
    data = load_dataset(
        dataset_name,
        cache_dir=str(settings.path_data_datasets),
        split=f"train[:{n_rows}]" if n_rows > 0 else "train",
    )
    df_data = pd.DataFrame(data)
    df_data = df_data.rename(
        columns={
            "context": "text",
            "question": "query",
        }
    )
    df_data["answers"] = df_data["answers"].apply(
        lambda x: x["text"] if isinstance(x, dict) else x
    )
    await process_qa_data(
        df_data,
        settings.SQUAD,
        n_rows,
        n_commited,
    )


async def etl_techqa_to_sql(
    n_rows: int = settings.rag_max_rows,
    n_commited: int = settings.rag_max_rows // 10,
):
    # Load TechQA dataset
    dataset_name = "TechQA"
    logger.info(f"Loading dataset: {dataset_name}")
    df_data = pd.read_csv(
        f"{str(settings.path_data_datasets)}/techqa/techqa_train_dev.csv"
    )
    await process_qa_data(
        df_data,
        settings.TECH_QA,
        n_rows,
        n_commited,
    )


async def etl_treccovid_to_sql(
    n_rows: int = settings.rag_max_rows,
    n_commited: int = settings.rag_max_rows // 10,
):
    # Load TREC-COVID generated dataset
    dataset_name = "BeIR/trec-covid"
    logger.info(f"Loading dataset: {dataset_name}")

    data = load_dataset(
        f"{dataset_name}-generated-queries",
        cache_dir=str(settings.path_data_datasets),
        split=f"train[:{n_rows}]" if n_rows > 0 else "train",
    )
    df_data = pd.DataFrame(data)

    # Drop identic texts with different titles
    df_data = df_data.query(
        '~(`text`.str.contains("This article is one of ten reviews selected from the Annual Update in Intensive Care and Emergency Medicine 2020") or `text` == "[Image: see text]" or `text` == "[Figure: see text]")'
    )

    await process_qa_data(
        df_data,
        settings.TREC_COVID,
        n_rows,
        n_commited,
    )


async def etl_multihop_to_sql(
    n_rows: int = settings.rag_max_rows,
    n_commited: int = settings.rag_max_rows // 10,
):
    from datasets import Features, Value

    dataset_name = "yixuantt/MultiHopRAG"
    query_dataset = load_dataset(
        dataset_name,
        "MultiHopRAG",
        cache_dir=str(settings.path_data_datasets),
    )
    # Timezone error workaraund by o3
    features = Features(
        {
            "title": Value("string"),
            "author": Value("string"),
            "source": Value("string"),
            "published_at": Value("timestamp[ns]"),
            "category": Value("string"),
            "url": Value("string"),
            "body": Value("string"),
            "timestamp": Value("timestamp[ns]"),
            "text": Value("string"),
            "id": Value("string"),
        }
    )

    corpus_dataset = load_dataset(
        dataset_name,
        "corpus",
        features=features,
        cache_dir=str(settings.path_data_datasets),
    )

    df_queries = pd.DataFrame(query_dataset["train"])  # pyright: ignore
    df_corpus = pd.DataFrame(corpus_dataset["train"])  # pyright: ignore

    # We will limit the number of queries
    # And based on that we will use the number of documents
    # The number of queries per document is ~3
    if n_rows > 0:
        df_queries = df_queries.head(n_rows)

    required_titles = [
        evidence["title"]
        for evidence_list in df_queries["evidence_list"]
        for evidence in evidence_list
    ]
    df_corpus = df_corpus.loc[df_corpus["title"].isin(required_titles)]

    logger.info(
        f"Loaded {len(df_queries)} queries and {len(df_corpus)} corpus documents from MultiHopRAG dataset."
    )

    def evidence_list_to_text(evidence_list):
        """Convert evidence list to formatted text.

        Example of evidence_list:
        [
            {
                "author": "Elizabeth Lopatto",
                "category": "technology",
                "fact": "Before his fall, Bankman-Fried made himself out to be the Good Boy of crypto \u2014 the trustworthy face of a sometimes-shady industry.",
                "published_at": "2023-09-28T12:00:00+00:00",
                "source": "The Verge",
                "title": "The FTX trial is bigger than Sam Bankman-Fried",
                "url": "https://www.theverge.com/2023/9/28/23893269/ftx-sam-bankman-fried-trial-evidence-crypto"
            },
            ...
        ]
        """
        text_elems = []
        for evidence in evidence_list:  # NOTE: Make sure this format is valid
            text = "\n".join(
                [
                    f"# {evidence['document_title'] or 'Document'}",
                    "",
                    f"Source: {evidence['source']}",
                    f"Published at: {str(evidence['published_at'])}",
                    f"Author: {evidence['author']}",
                    f"Category: {evidence['category']}",
                    f"Relevant Fact: {evidence['fact']}",
                    # f"Content:\n\n{evidence['document_text']}",
                ]
            )
            text_elems.append(text)
        return "\n\n".join(text_elems)

    async with get_db() as session:
        # First, process corpus documents
        doc_title_to_id = {}
        for idx, (hf_id, row) in enumerate(df_corpus.iterrows()):
            doc, _ = await get_or_create_item(
                session,
                Doc,
                dataset=shorten_hf_id(dataset_name),
                title=row["title"],
                text=row["body"],
                hf_id=str(hf_id),
                meta={
                    "author": row["author"],
                    "category": row["category"],
                    "source": row["source"],
                    "published_at": str(row["published_at"]),
                    "url": row["url"],
                },
            )

            # Store mapping for later use
            doc_id = doc.id
            doc_title_to_id[row["title"]] = {
                "id": doc_id,
                "title": row["title"],
                "text": row["body"],
            }

            chunks = str(row["body"]).split("\n\n")
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_text = chunk_text.strip()
                if chunk_text:  # Only create chunk if not empty
                    await get_or_create_item(
                        session,
                        Chunk,
                        doc_id=doc_id,
                        text=chunk_text,
                        meta={
                            "chunk_id": chunk_idx,
                        },
                    )

            if (idx + 1) % n_commited == 0:
                await session.commit()
                logger.info(f"Committed {idx + 1} corpus documents.")

        # Second, process queries
        for idx, (hf_id, row) in enumerate(df_queries.iterrows()):
            evidence_list = row["evidence_list"].copy()
            for i, evidence in enumerate(evidence_list):
                document_info = doc_title_to_id[evidence["title"]]
                evidence_list[i]["document_title"] = document_info["title"]
                evidence_list[i]["document_text"] = document_info["text"]
            evidence_text = evidence_list_to_text(evidence_list)

            # Get doc IDs immediately to avoid lazy loading issues
            doc_ids = []
            for evidence in evidence_list:
                doc_ids.append(doc_title_to_id[evidence["title"]]["id"])
            doc, _ = await get_or_create_item(
                session,
                Doc,
                dataset=shorten_hf_id(dataset_name),
                title=None,
                text=evidence_text,
                is_virtual=True,
                meta={
                    "doc_ids": doc_ids,
                },
            )

            virtual_doc_id = doc.id
            query, _ = await get_or_create_item(
                session,
                Query,
                force_create=True,
                doc_id=virtual_doc_id,
                text=row["query"],
                llm=None,
                prompt_id=None,
                hf_id=str(hf_id),
            )
            query_id = int(query.id)

            for i, evidence in enumerate(row["evidence_list"]):
                chunk_text = evidence["fact"]
                await get_or_create_item(
                    session,
                    Chunk,
                    doc_id=doc_ids[i],
                    text=chunk_text,
                    meta={
                        "chunk_id": i,
                        "chunk_type": "evidence",
                    },
                )
            await session.flush()
            answer = Answer(
                query_id=query_id,
                text=row["answer"],
            )
            session.add(answer)
            await session.flush()

            if (idx + 1) % n_commited == 0:
                await session.commit()
                logger.info(f"Committed {idx + 1} queries.")

        await session.commit()
    logger.info(
        "ETL complete: MultiHopRAG dataset processed with docs, chunks, queries, and answers."
    )
