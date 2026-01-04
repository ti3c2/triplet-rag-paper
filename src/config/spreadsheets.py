import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import gspread
import pandas as pd
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

from ..config.settings import settings
from ..rag.models import EvaluationResult

logger = logging.getLogger(__name__)


def _number_to_column_letter(n):
    """Convert a number to Excel column letter (1=A, 2=B, ..., 26=Z, 27=AA, etc.)."""
    result = ""
    while n > 0:
        n -= 1  # Adjust for 0-based indexing
        result = chr(ord("A") + n % 26) + result
        n //= 26
    return result


def _get_gspread_client():
    """Get authenticated gspread client."""
    cred_file = settings.path_config / "google_config.json"

    if not cred_file.exists():
        raise FileNotFoundError(
            f"Google credentials file not found at: {cred_file}\n"
            "Please ensure the google_config.json file exists in the data directory."
        )

    logger.info(f"Using Google credentials from: {cred_file}")

    # Define the scopes needed
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]

    # Authenticate and return client
    credentials = Credentials.from_service_account_file(str(cred_file), scopes=scopes)
    return gspread.authorize(credentials)


def load_spreadsheet(
    sheet_id: Optional[str] = settings.spreadsheet_id,
    gid: Optional[Union[str, int]] = settings.spreadsheet_gid,
) -> pd.DataFrame:
    """Load data from Google Spreadsheet.

    Args:
        sheet_id: Spreadsheet ID. If None, loads from BENCHMARK_SPREADSHEET_ID env var
        gid: Sheet identifier. Can be either:
            - Sheet ID (numeric)
            - Sheet name (string)
            If None, loads the first sheet

    Returns:
        DataFrame with loaded data
    """
    if sheet_id is None:
        load_dotenv()
        sheet_id = os.environ.get("BENCHMARK_SPREADSHEET_ID")
        if not sheet_id:
            raise ValueError("No spreadsheet ID provided")

    logger.info(f"Loading questions from spreadsheet ({sheet_id[:15]}...)/{gid}")
    # Check if gid is numeric (sheet ID) or string (sheet name)
    if gid is None or str(gid).isdigit():
        # Use CSV export URL for numeric gid
        csv_load_url = (
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        )
        if gid is not None:
            csv_load_url = f"{csv_load_url}&gid={gid}"
        df = pd.read_csv(csv_load_url)
    else:
        # Load by sheet name using gspread directly
        gc = _get_gspread_client()
        spreadsheet = gc.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet(str(gid))
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

    return df


def evaluation_result_to_dataframe(
    evaluation_result: EvaluationResult,
    experiment_name: str,
    retrieval_sources: Optional[str] = None,
) -> pd.DataFrame:
    """Transform EvaluationResult into the spreadsheet DataFrame format.

    Args:
        evaluation_result: EvaluationResult object from evaluate.py
        experiment_name: Name of the experiment (e.g., results_dirname from evaluate.py)
        retrieval_sources: Retrieval sources (e.g., "query", "query+answer"), if None will extract from retrieval_sources

    Returns:
        DataFrame formatted for spreadsheet writing
    """
    if not isinstance(evaluation_result, EvaluationResult):
        raise ValueError("evaluation_result must be an EvaluationResult instance")

    if not evaluation_result.rag_responses:
        raise ValueError("evaluation_result must contain rag_responses")

    # Extract metadata from the first response
    first_response = evaluation_result.rag_responses[0]
    retrieval_result = first_response.retrieval_result

    # Get embedding and RAG model names
    emb_model = evaluation_result.get_retres_attr("emb_model")
    qgen_model = evaluation_result.get_retres_attr("qgen_model")
    rag_model = first_response.model or ""

    # Get dataset name
    dataset = retrieval_result.dataset or ""

    # Determine qa_type and make it readable for the spreadsheet
    qa_type = evaluation_result.qa_type
    # qa_type = {
    #     "c": "chunk",
    #     "q": "query",
    #     "qa": "query+answer",
    #     "qc": "query+chunk",
    # }.get(qa_type)
    if qa_type is None:
        logger.error(f"Unknown qa_type: {qa_type}")
        qa_type = "unk"

    # Build rows for each metric
    rows = []
    for i, metric in enumerate(evaluation_result.metrics):
        row = {  # NOTE: Placing it explicitly to match the structure of the spreadsheet
            "experiment_name": experiment_name,
            "dataset": dataset,
            "qa_size": evaluation_result.qa_size,
            "qa_type": qa_type,
            "embedding_model": emb_model,
            "qgen_model": qgen_model,
            "qgen_prompt_name": evaluation_result.qgen_prompt_name,
            "rag_model": rag_model,
            "deduplication_factor": metric.deduplication_factor,
            "k": metric.k,
            "context_accuracy": metric.context_accuracy,
            "title_accuracy": metric.title_accuracy,
            "mrr": metric.mrr,
            "ndcg": metric.ndcg,
            "queries_ratio": metric.queries_ratio,
            # Ragas metrics (using ragas naming conventions) and eval models
            "eval_llm": evaluation_result.eval_llm,
            "eval_embedding_model": evaluation_result.eval_embedding_model,
            "ragas_n_evals": metric.ragas_n_evals,
            # Generation metrics
            "ragas_faithfulness": metric.ragas_faithfulness,
            "ragas_answer_relevancy": metric.ragas_answer_relevancy,
            "ragas_nv_accuracy": metric.ragas_nv_accuracy,
            "ragas_nv_context_relevance": metric.ragas_nv_context_relevance,
            "ragas_nv_response_groundedness": metric.ragas_nv_response_groundedness,
            "ragas_summary_score": metric.ragas_summary_score,
            # Classical metrics for generation
            "ragas_rouge_score": metric.ragas_rouge_score,
            "ragas_bleu_score": metric.ragas_bleu_score,
            "ragas_non_llm_string_similarity": metric.ragas_non_llm_string_similarity,
            "ragas_string_present": metric.ragas_string_present,
            "ragas_exact_match": metric.ragas_exact_match,
            "ragas_semantic_similarity": metric.ragas_semantic_similarity,
            "ragas_factual_correctness": metric.ragas_factual_correctness,
            # Retrieval metrics
            "ragas_context_precision": metric.ragas_context_precision,
            "ragas_context_recall": metric.ragas_context_recall,
            "ragas_context_entity_recall": metric.ragas_context_entity_recall,
            "ragas_noise_sensitivity": metric.ragas_noise_sensitivity,
        }

        # Handle different metric types for different datasets
        if (
            hasattr(metric, "full_match_accuracy")
            and metric.full_match_accuracy is not None
        ):
            row["full_match_accuracy"] = metric.full_match_accuracy
        if (
            hasattr(metric, "partial_match_accuracy")
            and metric.partial_match_accuracy is not None
        ):
            row["partial_match_accuracy"] = metric.partial_match_accuracy

        rows.append(row)

    return pd.DataFrame(rows)


def write_evaluation_to_spreadsheet(
    evaluation_result,
    experiment_name: str,
    sheet_id: Optional[str] = None,
    sheet_name: Optional[str] = None,
    retrieval_sources: Optional[str] = None,
) -> None:
    """Write evaluation results to Google Spreadsheet.

    Args:
        evaluation_result: EvaluationResult object from evaluate.py
        experiment_name: Name of the experiment
        sheet_id: Google Spreadsheet ID, if None uses settings.spreadsheet_id
        sheet_name: Sheet name to write to, if None uses settings.spreadsheet_gid
        dataset: Dataset name (e.g., "natural-questions")
        retrieval_sources: Retrieval sources (e.g., "query", "query+answer")
    """
    if sheet_id is None:
        sheet_id = settings.spreadsheet_id
        if not sheet_id:
            load_dotenv()
            sheet_id = os.environ.get("BENCHMARK_SPREADSHEET_ID")
            if not sheet_id:
                raise ValueError("No spreadsheet ID provided")

    if sheet_name is None:
        sheet_name = settings.spreadsheet_gid
        if not sheet_name:
            raise ValueError("No sheet name provided")

    # Transform evaluation result to DataFrame
    df = evaluation_result_to_dataframe(
        evaluation_result=evaluation_result,
        experiment_name=experiment_name,
        retrieval_sources=retrieval_sources,
    )

    logger.info(
        f"Writing evaluation results to spreadsheet ({sheet_id[:17]}...)/{sheet_name}"
    )

    # Setup gspread client
    gc = _get_gspread_client()
    spreadsheet = gc.open_by_key(sheet_id)

    # Find the worksheet
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        raise ValueError(f"Sheet '{sheet_name}' not found in spreadsheet")

    # Get existing data to determine where to append
    existing_data = worksheet.get_all_values()
    start_row = len(existing_data) + 1

    # Convert DataFrame to list of lists for gspread
    data_to_write = df.values.tolist()

    # Write data to spreadsheet
    if data_to_write:
        # Calculate the range to write to
        end_col_letter = _number_to_column_letter(len(df.columns))
        range_to_write = (
            f"A{start_row}:{end_col_letter}{start_row + len(data_to_write) - 1}"
        )

        worksheet.update(range_to_write, data_to_write)

    logger.info(
        f"Successfully wrote {len(df)} rows to spreadsheet starting at row {start_row}"
    )


def update_spreadsheet_with_ragas_metrics(
    evaluation_result: EvaluationResult,
    experiment_name: str,
    sheet_id: Optional[str] = None,
    sheet_name: Optional[str] = None,
    retrieval_sources: Optional[str] = None,
) -> None:
    """Update existing spreadsheet rows with ragas metrics.

    This function finds existing rows by experiment_name and updates them with ragas metrics.
    If no existing rows are found, it appends new row(s) for the experiment and writes the
    available metrics into the new rows (aligned to the sheet header).

    Args:
        evaluation_result: EvaluationResult object with ragas metrics
        experiment_name: Name of the experiment to find and update
        sheet_id: Google Spreadsheet ID, if None uses settings.spreadsheet_id
        sheet_name: Sheet name to write to, if None uses settings.spreadsheet_gid
        retrieval_sources: Retrieval sources (for validation)
    """
    if sheet_id is None:
        sheet_id = settings.spreadsheet_id
        if not sheet_id:
            load_dotenv()
            sheet_id = os.environ.get("BENCHMARK_SPREADSHEET_ID")
            if not sheet_id:
                raise ValueError("No spreadsheet ID provided")

    if sheet_name is None:
        sheet_name = settings.spreadsheet_gid
        if not sheet_name:
            raise ValueError("No sheet name provided")

    logger.info(
        f"Updating spreadsheet with ragas metrics for experiment: {experiment_name}"
    )

    # Setup gspread client
    gc = _get_gspread_client()
    spreadsheet = gc.open_by_key(sheet_id)

    # Find the worksheet
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        raise ValueError(f"Sheet '{sheet_name}' not found in spreadsheet")

    # Get all existing data
    all_values = worksheet.get_all_values()
    if not all_values:
        raise ValueError("Spreadsheet is empty - no rows to update")

    # Find header row and experiment rows
    headers = all_values[0]

    def _norm_header(s: str) -> str:
        return "".join(ch.lower() for ch in str(s).strip() if ch.isalnum() or ch == "_")

    def _find_col_idx(headers_: List[str], candidates: List[str]) -> Optional[int]:
        # Exact match first
        for c in candidates:
            try:
                return headers_.index(c)
            except ValueError:
                continue
        # Normalized match
        norm_to_idx = {_norm_header(h): i for i, h in enumerate(headers_)}
        for c in candidates:
            if (idx := norm_to_idx.get(_norm_header(c))) is not None:
                return idx
        return None

    # Find the experiment_name column (support common header variants)
    exp_name_col_idx = _find_col_idx(
        headers,
        [
            "Experiment",
            "experiment",
            "experiment_name",
            "Experiment name",
            "Folder",
            "folder",
        ],
    )
    if exp_name_col_idx is None:
        raise ValueError(
            "experiment_name column not found in spreadsheet (expected one of: "
            "Experiment/experiment_name/Folder)"
        )

    # Find all rows that match the experiment name
    matching_row_indices = []
    for i, row in enumerate(all_values[1:], start=2):  # Start from row 2 (1-indexed)
        if len(row) > exp_name_col_idx and row[exp_name_col_idx] == experiment_name:
            matching_row_indices.append(i)

    # Create DataFrame with ragas metrics for each k value
    df_ragas = evaluation_result_to_dataframe(
        evaluation_result=evaluation_result,
        experiment_name=experiment_name,
        retrieval_sources=retrieval_sources,
    )

    if df_ragas.empty:
        logger.warning("No metrics available to write to spreadsheet")
        return

    if not matching_row_indices:
        # No existing experiment rows: append new rows aligned to header.
        logger.warning(f"No existing rows found for experiment: {experiment_name}")
        logger.info(
            "Appending new rows for this experiment (folder) instead of updating"
        )

        # Build a mapping from sheet headers -> dataframe columns
        df_cols = list(df_ragas.columns)
        norm_df_cols = {_norm_header(c): c for c in df_cols}
        header_aliases = {
            "experiment": "experiment_name",
            "experimentname": "experiment_name",
            "folder": "experiment_name",
        }

        rows_to_append: List[List[object]] = []
        for _, r in df_ragas.iterrows():
            row_values: List[object] = []
            for h in headers:
                h_norm = _norm_header(h)
                df_col = norm_df_cols.get(h_norm) or norm_df_cols.get(
                    _norm_header(header_aliases.get(h_norm, ""))
                )
                if df_col and df_col in r and pd.notna(r[df_col]):
                    row_values.append(r[df_col])
                else:
                    row_values.append("")
            rows_to_append.append(row_values)

        start_row = len(all_values) + 1
        end_col_letter = _number_to_column_letter(len(headers))
        range_to_write = (
            f"A{start_row}:{end_col_letter}{start_row + len(rows_to_append) - 1}"
        )
        worksheet.update(range_to_write, rows_to_append)
        logger.info(
            f"Successfully appended {len(rows_to_append)} row(s) for experiment: {experiment_name}"
        )
        return

    logger.info(
        f"Found {len(matching_row_indices)} rows to update for experiment: {experiment_name}"
    )

    # Map ragas metrics by k value for easy lookup
    ragas_metrics_by_k = {}
    for _, row in df_ragas.iterrows():
        k_value = row["k"]
        ragas_metrics_by_k[k_value] = row

    # Find ragas metric columns in the spreadsheet
    ragas_columns = {}
    for col_name in headers:
        if col_name.startswith("ragas_"):
            try:
                col_idx = headers.index(col_name)
                ragas_columns[col_name] = col_idx
            except ValueError:
                continue

    if not ragas_columns:
        logger.warning("No ragas metric columns found in spreadsheet")
        logger.info("Available columns: " + ", ".join(headers))
        return

    logger.info(
        f"Found {len(ragas_columns)} ragas columns to update: {list(ragas_columns.keys())}"
    )

    # Update each matching row
    updates = []
    for row_idx in matching_row_indices:
        row_data = all_values[row_idx - 1]  # Convert to 0-indexed

        # Get the k value for this row
        try:
            k_col_idx = headers.index("k")
            k_value = int(row_data[k_col_idx])
        except (ValueError, IndexError):
            logger.warning(f"Could not determine k value for row {row_idx}")
            continue

        # Find corresponding ragas metrics for this k value
        if k_value not in ragas_metrics_by_k:
            logger.warning(f"No ragas metrics found for k={k_value}")
            continue

        ragas_row = ragas_metrics_by_k[k_value]

        # Prepare updates for this row
        for col_name, col_idx in ragas_columns.items():
            if col_name in ragas_row and pd.notna(ragas_row[col_name]):
                cell_address = f"{_number_to_column_letter(col_idx + 1)}{row_idx}"
                updates.append(
                    {"range": cell_address, "values": [[ragas_row[col_name]]]}
                )

    if not updates:
        logger.warning("No valid updates to perform")
        return

    # Batch update all cells
    logger.info(f"Performing {len(updates)} cell updates")
    worksheet.batch_update(updates)

    logger.info(
        f"Successfully updated {len(matching_row_indices)} rows with ragas metrics"
    )
    logger.info(f"Updated experiment: {experiment_name}")

    # Log which metrics were updated
    updated_metrics = set()
    for update in updates:
        # Extract column from range (e.g., "AB5" -> "AB")
        col_letter = "".join(c for c in update["range"] if c.isalpha())
        col_idx = (
            sum(
                (ord(c) - ord("A") + 1) * (26**i)
                for i, c in enumerate(reversed(col_letter))
            )
            - 1
        )
        if col_idx < len(headers):
            updated_metrics.add(headers[col_idx])

    logger.info(f"Updated metrics: {sorted(updated_metrics)}")
