import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..config.settings import settings
from ..store_rel.entry import get_db
from ..store_rel.schema import Answer, Chunk, Doc, Prompt, Query, get_or_create_item

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)


async def load_csv_queries_to_db(
    csv_path: str,
    dataset_name: str = settings.rag_dataset,
    n_rows: int = settings.rag_max_rows,
    n_committed: int = settings.rag_max_rows // 10,
    skip_existing: bool = True,
    prompt_name: str = "ragas_queries",
    prompt_text: Optional[str] = None,
    llm: Optional[str] = None,
):
    """
    Load queries from CSV files into the database.

    Expected CSV format:
    - original_row_idx: Maps to hf_id in chunks table
    - user_input: The query text
    - reference: The reference answer
    - response: The model response (optional)
    - multi_responses: Multiple responses (optional)
    - reference_contexts: Reference contexts for the query
    - retrieved_contexts: Retrieved contexts for the query
    - rubrics: Evaluation rubrics (optional)
    - persona_name: Name of the persona
    - persona_role_description: Description of the persona role
    - node_id: Node identifier (optional)
    - source_chunk_idx: Source chunk index (optional)

        Args:
        csv_path: Path to the CSV file
        dataset_name: Name of the dataset
        n_rows: Maximum number of rows to process (0 for all)
        n_committed: Commit every N items
        skip_existing: Skip queries that already exist in the database
        prompt_name: Name of the prompt to associate with CSV queries
        prompt_text: Text/description of the prompt for CSV queries (default: prompt_name)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    prompt_text = prompt_text or prompt_name

    logger.info(f"Loading queries from {csv_path} for dataset: {dataset_name}")

    # Load CSV data
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from CSV")

    # Limit rows if specified
    if n_rows > 0:
        df = df.head(n_rows)
        logger.info(f"Processing {len(df)} rows (limited by n_rows)")

    # Validate required columns
    required_columns = ["original_row_idx", "user_input"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    async with get_db() as session:
        # Create or get prompt for CSV queries
        prompt, created = await get_or_create_item(
            session,
            Prompt,
            name=prompt_name,
            text=prompt_text,
        )
        prompt_id = prompt.id
        if created:
            logger.info(f"Created new prompt '{prompt_name}' with ID {prompt_id}")
        else:
            logger.info(f"Using existing prompt '{prompt_name}' with ID {prompt_id}")

        # Get existing queries to avoid duplicates
        existing_queries = set()
        if skip_existing:
            existing_result = await session.execute(
                select(Query.hf_id)
                .join(Doc)
                .where(
                    Doc.dataset == dataset_name,
                    Query.hf_id.isnot(None),
                    Query.prompt_id == prompt_id,  # Only check queries with our prompt
                )
            )
            existing_queries = {hf_id for (hf_id,) in existing_result.all()}
            logger.info(f"Found {len(existing_queries)} existing CSV queries to skip")

        # Create a mapping from hf_id to chunk_id
        chunk_mapping_result = await session.execute(
            select(Chunk.hf_id, Chunk.id, Chunk.doc_id)
            .join(Doc)
            .where(Doc.dataset == dataset_name)
        )
        chunk_mapping = {
            str(row["hf_id"]): {
                "chunk_id": row["id"],
                "doc_id": row["doc_id"],
            }
            for row in chunk_mapping_result.mappings().all()
            if row["hf_id"] is not None
        }
        logger.debug(f"Chunk mapping: {chunk_mapping}")
        logger.info(f"Found {len(chunk_mapping)} chunks for mapping")

        processed_count = 0
        skipped_count = 0
        error_count = 0

        for idx, row in df.iterrows():
            try:
                original_row_idx = str(row["source_chunk_idx"])
                question = str(row["user_input"]).strip()

                # Skip if query already exists
                if skip_existing and original_row_idx in existing_queries:
                    skipped_count += 1
                    continue

                # Find corresponding chunk
                if original_row_idx not in chunk_mapping:
                    logger.warning(
                        f"No chunk found for\n"
                        f"{original_row_idx=}\n"
                        f"{row['source_chunk_idx']=}\n"
                        f"{question=}\n"
                        f"{row['reference_contexts']=}\n"
                        "skipping"
                    )
                    error_count += 1
                    continue

                chunk_info = chunk_mapping[original_row_idx]
                chunk_id = chunk_info["chunk_id"]
                doc_id = chunk_info["doc_id"]

                # Ensure query ends with question mark
                if not question.endswith("?"):
                    question = question + "?"

                # Prepare metadata
                meta = {}

                # Add persona information if available
                if "persona_name" in row and pd.notna(row["persona_name"]):
                    meta["persona_name"] = str(row["persona_name"])

                if "persona_role_description" in row and pd.notna(
                    row["persona_role_description"]
                ):
                    meta["persona_role_description"] = str(
                        row["persona_role_description"]
                    )

                # Add rubrics if available
                if "rubrics" in row and pd.notna(row["rubrics"]):
                    meta["rubrics"] = str(row["rubrics"])

                # Add node_id if available
                if "node_id" in row and pd.notna(row["node_id"]):
                    meta["node_id"] = str(row["node_id"])

                # Create query
                query, created = await get_or_create_item(
                    session,
                    Query,
                    force_create=True,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=question,
                    llm=llm,
                    prompt_id=prompt_id,
                    hf_id=original_row_idx,
                    meta=meta if meta else None,
                )

                query_id = query.id

                # Create answer if reference is available
                if "reference" in row and pd.notna(row["reference"]):
                    reference_answer = str(row["reference"]).strip()
                    if reference_answer:
                        answer = Answer(
                            query_id=query_id,
                            text=reference_answer,
                        )
                        session.add(answer)

                processed_count += 1

                # Commit periodically
                if processed_count % n_committed == 0:
                    await session.commit()
                    logger.info(f"Committed {processed_count} queries")

            except Exception as e:
                error_count += 1
                logger.error(f"Error processing row {idx}: {e}", exc_info=True)
                continue

        # Final commit
        await session.commit()

        logger.info(
            f"ETL complete for {csv_path}: "
            f"processed={processed_count}, skipped={skipped_count}, errors={error_count}"
        )


async def load_queries_csv_dir(
    csv_directory: str,
    dataset_name: str = settings.rag_dataset,
    n_rows: int = settings.rag_max_rows,
    n_committed: int = settings.rag_max_rows // 10,
    skip_existing: bool = True,
    prompt_name: str = "ragas_queries",
    prompt_text: Optional[str] = None,
    llm: str = settings.question_generation_model,
):
    """
    Load queries from multiple CSV files in a directory.

    Args:
        csv_directory: Directory containing CSV files
        dataset_name: Name of the dataset
        n_rows: Maximum number of rows to process per file (0 for all)
        n_committed: Commit every N items
        skip_existing: Skip queries that already exist in the database
        prompt_name: Name of the prompt to associate with CSV queries
        prompt_text: Text/description of the prompt for CSV queries (default: prompt_name)
    """
    prompt_text = prompt_text or prompt_name

    csv_dir = Path(csv_directory)
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    dataset_name = dataset_name or csv_dir.name

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_dir}")
        return

    logger.info(f"Found {len(csv_files)} CSV files to process")

    for csv_file in csv_files:
        logger.info(f"Processing {csv_file}")
        await load_csv_queries_to_db(
            csv_path=str(csv_file),
            dataset_name=dataset_name,
            n_rows=n_rows,
            n_committed=n_committed,
            skip_existing=skip_existing,
            prompt_name=prompt_name,
            prompt_text=prompt_text,
            llm=llm,
        )

    logger.info(f"All CSV files processed for dataset: {dataset_name}")


async def etl_csv_queries(
    csv_paths: List[str],
    dataset_name: str = settings.rag_dataset,
    n_rows: int = settings.rag_max_rows,
    n_committed: int = settings.rag_max_rows // 10,
    prompt_name: str = "ragas_queries",
    prompt_text: Optional[str] = None,
    llm: str = settings.question_generation_model,
):
    """
    Main ETL function for loading CSV queries from single or multiple paths.

    Args:
        csv_paths: Path(s) to CSV file(s) or directory(ies) - can be:
                  - str: single file or directory path
                  - List[str]: multiple file or directory paths
        dataset_name: Dataset name
        n_rows: Maximum rows to process per file
        n_committed: Commit frequency
        prompt_name: Name of the prompt to associate with CSV queries
        prompt_text: Text/description of the prompt for CSV queries (default: prompt_name)
        llm: LLM model to associate with CSV queries
    """
    prompt_text = prompt_text or prompt_name

    logger.info(f"Processing {len(csv_paths)} CSV path(s)")

    total_processed = 0

    for i, csv_path in enumerate(csv_paths, 1):
        logger.info(f"Processing path {i}/{len(csv_paths)}: {csv_path}")

        try:
            path = settings.find_file(csv_path, settings.path_data_ragas_queries)
            if path is None:
                logger.error(f"CSV file not found: {csv_path}")
                continue

            if path.is_file():
                await load_csv_queries_to_db(
                    csv_path=str(path),
                    dataset_name=dataset_name,
                    n_rows=n_rows,
                    n_committed=n_committed,
                    prompt_name=prompt_name,
                    prompt_text=prompt_text,
                    llm=llm,
                )
                total_processed += 1

            elif path.is_dir():
                await load_queries_csv_dir(
                    csv_directory=str(path),
                    dataset_name=dataset_name,
                    n_rows=n_rows,
                    n_committed=n_committed,
                    prompt_name=prompt_name,
                    prompt_text=prompt_text,
                )
                total_processed += 1

            else:
                logger.error(f"Invalid path (not file or directory): {csv_path}")
                continue

            logger.info(f"Successfully processed path {i}/{len(csv_paths)}")

        except Exception as e:
            logger.error(f"Failed to process path {csv_path}: {e}", exc_info=True)
            continue

    logger.info(
        f"Completed processing CSV paths: {total_processed}/{len(csv_paths)} successful"
    )
