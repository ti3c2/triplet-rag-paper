import copy
import json
import logging
import warnings
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import ragas.metrics as ragas_metrics
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from ..config.settings import settings
from ..rag.models import (
    EvaluationMetrics,
    EvaluationResult,
    GroundTruthItem,
    RagResponse,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.log_level)


def create_ragas_evaluators(
    metrics_to_use: List[str] = settings.ragas_metrics,
) -> Dict[str, object]:
    """Create ragas metric evaluators using configured LLM and embedding models."""

    logger.info(f"Creating ragas evaluators for metrics: {metrics_to_use}")

    # Create LLM and embeddings wrappers
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.eval_llm,
            api_key=settings.openai_api_key,
            base_url=settings.eval_llm_api_base,
            max_retries=settings.ragas_max_retries,
            timeout=settings.ragas_timeout,
            extra_body=(
                {
                    "guided_json": None,
                    "guided_choice": None,
                }
                if settings.eval_llm_api_base
                else {}
            ),
            temperature=0.0,  # Does not matter because ragas sets it?
        ),
        attempt_structured_output=False,
    )

    # Create embeddings for evaluation
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.eval_embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.eval_embedding_model_api_base,
        )
    )

    # Initialize available metrics using normalized naming conventions
    # Note: The keys should match the names we expect in our settings and models
    available_metrics = {
        # Generation metrics
        "faithfulness": ragas_metrics.Faithfulness(llm=llm),
        "answer_relevancy": ragas_metrics.AnswerRelevancy(llm=llm, embeddings=emb),
        "nv_accuracy": ragas_metrics.AnswerAccuracy(llm=llm),
        "nv_context_relevance": ragas_metrics.ContextRelevance(llm=llm),
        "summary_score": ragas_metrics.SummarizationScore(llm=llm),
        "nv_response_groundedness": ragas_metrics.ResponseGroundedness(llm=llm),
        # Classical metrics for generation
        "rouge_score": ragas_metrics.RougeScore(),  # outputs "rouge_score(mode=fmeasure)"
        "bleu_score": ragas_metrics.BleuScore(),
        "non_llm_string_similarity": ragas_metrics.NonLLMStringSimilarity(),
        "string_present": ragas_metrics.StringPresence(),
        "exact_match": ragas_metrics.ExactMatch(),
        "semantic_similarity": ragas_metrics.SemanticSimilarity(embeddings=emb),
        # Retrieval metrics
        "factual_correctness": ragas_metrics.FactualCorrectness(llm=llm),
        # outputs "factual_correctness(mode=f1)"
        "context_precision": ragas_metrics.LLMContextPrecisionWithoutReference(llm=llm),
        # outputs "llm_context_precision_without_reference"
        "context_recall": ragas_metrics.ContextRecall(llm=llm),
        "context_entity_recall": ragas_metrics.ContextEntityRecall(llm=llm),
        "noise_sensitivity": ragas_metrics.NoiseSensitivity(llm=llm),
        # outputs "noise_sensitivity(mode=relevant)"
    }

    # Return only requested metrics
    selected_metrics = {}
    for name in metrics_to_use:
        if name in available_metrics:
            selected_metrics[name] = available_metrics[name]
            logger.debug(f"Added metric: {name}")
        else:
            logger.warning(
                f"Requested metric '{name}' not available. Available metrics: {list(available_metrics.keys())}"
            )

    if not selected_metrics:
        raise ValueError(
            f"No valid metrics found from requested: {metrics_to_use}. Available: {list(available_metrics.keys())}"
        )

    logger.info(
        f"Created {len(selected_metrics)} ragas evaluators: {list(selected_metrics.keys())}"
    )
    return selected_metrics


def rag_response_to_ragas_sample(
    rag_response: RagResponse,
    ground_truth: GroundTruthItem,
    k: int,
    ignore_retrieval_sources: List[str] = settings.ragas_ignore_retrieval_sources,
) -> Optional[SingleTurnSample]:
    """Convert RagResponse and GroundTruthItem to ragas SingleTurnSample format."""
    # Validate required fields
    if not rag_response.query or not rag_response.query.strip():
        logger.warning("Skipping sample with empty query")
        return None

    # Clean and validate answer
    answer = (rag_response.answer or "").strip()
    if len(answer) < 5:
        logger.debug(
            f"Short answer ({len(answer)} chars) for query: {rag_response.query[:50]}..."
        )

    # Get contexts from retrieval result
    retrieval_items = rag_response.retrieval_result.items
    contexts = []
    for item in retrieval_items:
        metadata = item.metadata
        if metadata.type == "chunk" and metadata.type not in ignore_retrieval_sources:
            contexts.append(item.text)
        elif metadata.type == "query" and metadata.type not in ignore_retrieval_sources:
            query = metadata.query.split(settings.qa_separator)[0]
            gt_answer = ground_truth.answer.strip() if ground_truth.answer else None
            if settings.rag_add_chunk_to_query:
                context = f"C: {metadata.chunk_text}\n" f"Q: {query}"
            else:
                context = f"Q: {query}"

            if gt_answer:
                context += f"\nA: {gt_answer}"
            contexts.append(context)
        else:
            logger.warning(f"Unknown item type: {metadata.type}")

    if not contexts:
        logger.warning(f"No contexts available for query: {rag_response.query[:50]}...")

    logger.debug(f"Using {k} contexts out of {len(contexts)} available contexts")
    contexts = contexts[:k]
    return SingleTurnSample(
        user_input=rag_response.query,
        response=answer,
        retrieved_contexts=contexts,
        reference_contexts=ground_truth.chunk_texts,
        reference=ground_truth.answer.strip() if ground_truth.answer else None,
    )


async def evaluate_with_ragas(
    rag_responses: List[RagResponse],
    ground_truths: List[List[GroundTruthItem]],
    metrics_to_use: List[str] = settings.ragas_metrics,
    k: int = max(settings.eval_context_ks),
) -> Dict[str, float]:
    """Evaluate RAG responses using ragas metrics."""
    logger.info(f"Starting ragas evaluation with metrics: {metrics_to_use}")

    # Convert to ragas format with validation
    ragas_samples = []
    skipped_samples = 0

    logger.debug(f"Parsing ragas samples: {len(rag_responses)=}, {len(ground_truths)=}")
    for rag_response, ground_truth_list in zip(rag_responses, ground_truths):
        # Use first ground truth item
        if ground_truth_list:
            ground_truth = ground_truth_list[0]
        else:
            ground_truth = GroundTruthItem(
                doc_ids=[], doc_titles=[], query=rag_response.query
            )
            logger.warning(f"No ground truth found for {rag_response.query[:50]}...")

        sample = rag_response_to_ragas_sample(rag_response, ground_truth, k)
        if sample is not None:
            ragas_samples.append(sample)
            logger.debug(f"Added sample: {sample.model_dump_json(indent=2)}")
        else:
            skipped_samples += 1

    if skipped_samples > 0:
        logger.warning(f"Skipped {skipped_samples} invalid samples")

    if not ragas_samples:
        logger.error("No valid samples for ragas evaluation")
        return {}

    logger.info(f"Converted {len(ragas_samples)} valid samples for ragas evaluation")

    # Create evaluators - only for the requested metrics
    evaluators = create_ragas_evaluators(
        metrics_to_use=metrics_to_use,
    )
    selected_evaluators = list(evaluators.values())
    run_config = RunConfig(
        max_workers=settings.ragas_max_workers,
        max_retries=settings.ragas_max_retries,
        timeout=settings.ragas_timeout,
        log_tenacity=True,
    )

    if not selected_evaluators:
        logger.warning("No valid ragas evaluators found")
        return {}

    logger.info(
        f"Using {len(selected_evaluators)} ragas evaluators for metrics: {list(evaluators.keys())}"
    )

    # Run evaluation with better error handling
    try:
        logger.info(
            f"Running ragas evaluation with {len(selected_evaluators)} evaluators"
        )
        ragas_eval_dataset = EvaluationDataset(samples=ragas_samples)

        # Add debug logging for the dataset
        logger.debug(f"Dataset created with {len(ragas_eval_dataset)} samples")
        logger.debug(
            f"Sample data types: {[type(s) for s in ragas_eval_dataset.samples[:2]]}"
        )

        result = evaluate(
            metrics=selected_evaluators,
            dataset=ragas_eval_dataset,
            run_config=run_config,
            batch_size=settings.ragas_batch_size,
        )

        # Debug the result object
        logger.debug(f"Result type: {type(result)}")
        logger.debug(f"Result attributes: {dir(result)}")

        df = result.to_pandas()
        logger.debug(f"Ragas DataFrame: {df.shape} with columns {df.columns.tolist()}")
        logger.debug(f"First few rows of results:\n{df.head()}")

        # Extract ragas metrics directly using ragas column names (no mapping needed)
        ragas_results = {}

        logger.debug(f"Available DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"Requested metrics: {metrics_to_use}")

        # Extract metrics from DataFrame columns - normalize column names to match our schema
        metadata_columns = {
            "user_input",
            "retrieved_contexts",
            "reference_contexts",
            "response",
            "reference",
        }

        def normalize_column_name(col_name: str) -> str:
            """Normalize ragas column names to match our expected metric names."""
            # Remove parentheses and their contents (e.g., "rouge_score(mode=fmeasure)" -> "rouge_score")
            normalized = col_name.split("(")[0]

            # Handle specific ragas naming variations
            column_mappings = {
                "llm_context_precision_without_reference": "context_precision",
                # Add more mappings as needed
            }

            return column_mappings.get(normalized, normalized)

        for col in df.columns:
            # Skip metadata columns
            if col in metadata_columns:
                continue

            # Normalize the column name
            normalized_col = normalize_column_name(col)

            # Only extract if this normalized metric was actually requested
            if normalized_col in metrics_to_use:
                if not pd.isna(df[col]).all():  # Check if column has valid data
                    mean_value = float(df[col].mean())
                    ragas_results[f"ragas_{normalized_col}"] = mean_value
                    logger.info(
                        f"Extracted {normalized_col} (from column '{col}'): {mean_value:.4f}"
                    )
                else:
                    logger.warning(f"Column '{col}' contains only NaN values")
            else:
                # Log unmatched columns for debugging
                logger.debug(
                    f"Column '{col}' -> '{normalized_col}' not in requested metrics: {metrics_to_use}"
                )

        # Warn about missing metrics
        extracted_metrics = {k.replace("ragas_", "") for k in ragas_results.keys()}
        missing_metrics = set(metrics_to_use) - extracted_metrics
        if missing_metrics:
            logger.warning(
                f"Could not extract the following requested metrics: {missing_metrics}"
            )
            logger.warning(
                f"Available columns were: {[col for col in df.columns if col not in metadata_columns]}"
            )

        # Always include bookkeeping of how many samples were evaluated
        ragas_results["ragas_n_evals"] = len(ragas_samples)

        if ragas_results:
            logger.info(
                f"Ragas evaluation completed successfully. Extracted {len(ragas_results)} metrics: {list(ragas_results.keys())}"
            )
        else:
            logger.error("No ragas metrics were successfully extracted!")
            logger.error(f"DataFrame columns: {df.columns.tolist()}")
            logger.error(f"Requested metrics: {metrics_to_use}")

        return ragas_results

    except Exception as e:
        logger.error(f"Error during ragas evaluation: {str(e)}", exc_info=True)
        return {}


def merge_ragas_metrics(
    classical_metrics: List[EvaluationMetrics],
    ragas_results: Dict[str, float],
    k: int,
) -> List[EvaluationMetrics]:
    """
    Merge ragas results into classical metrics for given k value.
    Takes classical metrics and adds ragas results to them.
    """
    updated_metrics = []
    k_found = False

    for metric in classical_metrics:
        if metric.k == k:
            k_found = True
            metric_dict = metric.model_dump()
            # Update with ragas results
            metric_dict.update(ragas_results)
            updated_metrics.append(EvaluationMetrics(**metric_dict))
            logger.debug(
                f"Updated metrics for k={k} with ragas results: {list(ragas_results.keys())}"
            )
        else:
            updated_metrics.append(metric)

    if not k_found:
        logger.warning(
            f"No existing metrics found for k={k}. Available k values: {[m.k for m in classical_metrics]}"
        )

    return updated_metrics


async def run_ragas_evaluation(
    eval_res: Optional[EvaluationResult] = None,
    json_path: Optional[str] = None,
    metrics_to_use: List[str] = settings.ragas_metrics,
    ks: List[int] = settings.eval_context_ks,
    max_queries: Optional[int] = settings.eval_max_queries_ragas,
    update_existing: bool = False,
    save_results: bool = True,
    experiment_name: Optional[str] = None,
    update_spreadsheet: bool = settings.rag_write_spreadsheet,
) -> EvaluationResult:
    """
    Run ragas evaluation on existing retrieval evaluation results.

    Args:
        eval_res: Optional[EvaluationResult] - Existing evaluation results to run ragas evaluation on
        json_path: Path to JSON file containing evaluation results (optional if eval_res is provided)
        metrics_to_use: List of ragas metrics to compute (defaults to settings)
        ks: List of K values to evaluate (defaults to settings)
        update_existing: Whether to update the original JSON file
        save_results: Whether to save updated retrieval results to files
        experiment_name: Name of the experiment for spreadsheet updates (auto-detected if None)
        update_spreadsheet: Whether to update Google Spreadsheet with ragas results
    """
    if eval_res is None:
        if json_path is None:
            raise ValueError("Either eval_res or json_path must be provided")
        logger.info(f"Loading evaluation results from {json_path}")
        json_file = Path(json_path)
        if not json_file.exists():
            json_file = settings.path_eval / json_file
        if json_file.is_dir():
            json_files = list(json_file.glob("*.json"))
            json_files = [file for file in json_files if "triplet" not in file.name]
            json_file = json_files[0]
        if not json_file.exists():
            raise FileNotFoundError(f"Evaluation file not found: {json_path}")
        eval_res = EvaluationResult(**json.loads(json_file.read_text(encoding="utf-8")))
        logger.info(
            f"Loaded {len(eval_res.rag_responses)} RAG responses for ragas evaluation"
        )
    elif json_path is not None or eval_res.experiment_name is not None:
        experiment_name = experiment_name or eval_res.experiment_name
        json_path = settings.path_eval / experiment_name
        json_files = list(json_path.glob("*.json"))
        json_files = [file for file in json_files if "triplet" not in file.name]
        json_path = json_files[0]
        if not json_path:
            logger.warning(f"Evaluation file not found: {json_path}")
        logger.info(f"Using provided evaluation results from {json_path}")
    else:
        logger.warning("Unable to specify json path so falling back to using new one")

    # Run ragas evaluation for each k value
    updated_eval_res = copy.deepcopy(eval_res)  # Deep copy to avoid modifying original
    updated_eval_res.eval_llm = settings.eval_llm
    updated_eval_res.eval_embedding_model = settings.eval_embedding_model

    # Normalize max queries value
    max_queries_limit = (
        None if (max_queries is None or max_queries == -1) else max_queries
    )

    for k in ks:
        logger.info(f"Running ragas evaluation for k={k}")

        # Select responses and aligned ground truths for the specific k value
        try:
            selected_indices = [
                idx for idx, r in enumerate(eval_res.rag_responses) if r.n_context == k
            ]
            if not selected_indices:
                raise ValueError("No responses for k")
        except ValueError as e:
            logger.warning(f"No k-specific responses found for k={k}: {e}")
            logger.info(
                f"Falling back to max k responses and slicing contexts to k={k}"
            )
            selected_indices = list(range(len(eval_res.rag_responses)))

        if max_queries_limit is not None:
            selected_indices = selected_indices[:max_queries_limit]

        k_responses = [eval_res.rag_responses[i] for i in selected_indices]
        k_ground_truths = [
            eval_res.ground_truth[i] if i < len(eval_res.ground_truth) else []
            for i in selected_indices
        ]
        logger.info(
            f"Using {len(k_responses)} responses specific to k={k} (aligned with ground truths)"
        )

        ragas_results = await evaluate_with_ragas(
            rag_responses=k_responses,
            ground_truths=k_ground_truths,
            metrics_to_use=metrics_to_use,
            k=k,
        )

        if not ragas_results:
            logger.warning(
                f"No ragas results obtained for k={k}, skipping this k value"
            )
            continue

        # Merge results for this specific k value
        updated_eval_res.metrics = merge_ragas_metrics(
            classical_metrics=updated_eval_res.metrics,
            ragas_results=ragas_results,
            k=k,
        )
        logger.info(
            f"Successfully merged ragas results for k={k}: {list(ragas_results.keys())}"
        )
        logger.critical(
            f"Updated metrics for k={k}:\n%s",
            updated_eval_res.model_dump_json(indent=2, include={"metrics"}),
        )

        if save_results:
            # Save updated results
            datetime_str = dt.now().strftime("%Y%m%d_%H%M%S")

            # Handle case where json_path might be None
            if json_path:
                json_file = Path(json_path)
                if not json_file.exists():
                    json_file = settings.path_eval / json_file
                if json_file.is_dir():
                    json_file = next(json_file.glob("*.json"))
                base_path = json_file.parent
            else:
                # Use default eval path if no json_path provided
                base_path = settings.path_eval / (
                    updated_eval_res.experiment_name or datetime_str
                )
                base_path.mkdir(parents=True, exist_ok=True)

            # Update original JSON if requested
            if json_path and update_existing:
                json_file_new = json_file.with_name(f"{datetime_str}_eval-ragas.json")
                json_file_new.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Saving updated JSON file: {json_file_new}")
                with open(json_file_new, "w", encoding="utf-8") as f:
                    json.dump(
                        updated_eval_res.model_dump(), f, indent=2, ensure_ascii=False
                    )

            model_str = (
                f"{updated_eval_res.eval_llm.split('/')[-1]}_"
                f"{updated_eval_res.eval_embedding_model.split('/')[-1]}"
            )

            # Save updated text summary
            if update_existing:
                summary_path = base_path / f"{datetime_str}_ragas_{model_str}.txt"
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(updated_eval_res.format_str())
                logger.info(f"Updated summary saved to {summary_path}")

            # Save updated metrics CSV
            metrics_path = base_path / f"{datetime_str}_ragas_{model_str}.csv"
            df_metrics = pd.DataFrame(
                [m.model_dump(exclude_none=True) for m in updated_eval_res.metrics]
            )
            df_metrics.to_csv(metrics_path, index=False)
            logger.info(f"Updated metrics saved to {metrics_path}")

        # Update spreadsheet with ragas results if requested
        if update_spreadsheet:
            try:
                # Auto-detect experiment name if not provided
                if experiment_name is None:
                    # Try to get experiment name from evaluation result
                    if (
                        hasattr(updated_eval_res, "experiment_name")
                        and updated_eval_res.experiment_name
                    ):
                        experiment_name = updated_eval_res.experiment_name
                        logger.info(
                            f"Using experiment name from evaluation result: {experiment_name}"
                        )
                    elif json_path:
                        # Extract experiment name from json file path
                        json_file = Path(json_path)
                        if not json_file.exists():
                            json_file = settings.path_eval / json_file
                        if json_file.is_dir():
                            json_file = next(json_file.glob("*.json"))

                        # Extract experiment name from directory name or file name
                        if json_file.parent.name != "eval":  # If in a subdirectory
                            experiment_name = json_file.parent.name
                        else:
                            # Extract from filename (remove datetime and extension)
                            filename = json_file.stem
                            # Remove datetime prefix if present (format: YYYYMMDD_HHMMSS_...)
                            parts = filename.split("_")
                            if (
                                len(parts) > 2
                                and parts[0].isdigit()
                                and len(parts[0]) == 8
                            ):
                                experiment_name = "_".join(parts[2:])
                            else:
                                experiment_name = filename

                        logger.info(
                            f"Auto-detected experiment name from path: {experiment_name}"
                        )
                    else:
                        logger.warning(
                            "Cannot auto-detect experiment name without json_path or experiment_name in result"
                        )
                        experiment_name = None

                if experiment_name:
                    from ..config.spreadsheets import (
                        update_spreadsheet_with_ragas_metrics,
                    )

                    logger.info(
                        f"Updating spreadsheet for experiment: {experiment_name}"
                    )
                    update_spreadsheet_with_ragas_metrics(
                        evaluation_result=updated_eval_res,
                        experiment_name=experiment_name,
                    )
                    logger.info("Successfully updated spreadsheet with ragas metrics")
                else:
                    logger.warning(
                        "No experiment name available - skipping spreadsheet update"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to update spreadsheet with ragas metrics: {str(e)}",
                    exc_info=True,
                )
                logger.warning("Continuing without spreadsheet update...")

    return updated_eval_res
