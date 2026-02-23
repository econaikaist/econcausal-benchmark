"""
Step 4: Critic evaluation of extracted causal triplets.

This step evaluates each causal triplet on five criteria (or six with --with-context-eval):
1. Variable extraction quality
2. Direction correctness (T -> O)
3. Sign accuracy
4. Causality (vs correlation/description)
5. Main claim (core vs peripheral finding)
6. Context appropriateness (only with --with-context-eval)
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import (
    OpenAIClient,
    MultiLLMJudge,
    save_json,
    load_json,
    load_csv,
    setup_logging,
    clean_sign,
    DEFAULT_MODEL,
    DEFAULT_MAX_WORKERS,
)
from common.prompts import STEP4_PROMPT, STEP4_context_PROMPT
from common.schemas import STEP4_SCHEMA, STEP4_context_SCHEMA


# Default paths
DEFAULT_OUTPUT_DIR = "/home/donggyu/econ_causality/new_data/real_data_step4"

DEFAULT_METADATA_CSV = "/home/donggyu/econ_causality/new_data/nber_paper/metadata_published_8journals_with_JEL.csv"
DEFAULT_PDF_DIR = "/home/donggyu/econ_causality/new_data/nber_paper_wo_aer_pp_30pages"


def load_abstracts(csv_path: str) -> dict[str, str]:
    """
    Load abstracts from metadata CSV.

    Args:
        csv_path: Path to metadata CSV file

    Returns:
        Dictionary mapping paper number to abstract
    """
    df = load_csv(csv_path)
    abstracts = {}

    for _, row in df.iterrows():
        paper_num = str(row.get("nber_working_paper_number", ""))
        abstract = row.get("abstract", "")
        if paper_num and abstract:
            abstracts[paper_num] = abstract

    return abstracts


def format_evidence(evidence_list: list[str]) -> str:
    """Format evidence paragraphs for prompt."""
    if not evidence_list:
        return "(No evidence provided)"
    return "\n\n".join(f"[{i+1}] {ev}" for i, ev in enumerate(evidence_list))


def format_context_selected(context: list[str]) -> str:
    """Format selected context for prompt."""
    if not context:
        return "(No context selected)"
    return "\n".join(f"- {c}" for c in context)


def run_step4(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    metadata_csv: str = DEFAULT_METADATA_CSV,
    pdf_dir: str = DEFAULT_PDF_DIR,
    model: str = DEFAULT_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    use_batch: bool = False,
    paper_ids: Optional[list[str]] = None,
    step1_results_path: Optional[str] = None,
    step2_results_path: Optional[str] = None,
    step3_results_path: Optional[str] = None,
    use_multi_llm: bool = True,
    with_context_eval: bool = False,
) -> dict:
    """
    Run Step 4: Critic evaluation of causal triplets.

    Args:
        output_dir: Directory to save output JSON files
        metadata_csv: Path to metadata CSV with abstracts
        pdf_dir: Directory containing PDF files
        model: OpenAI model to use (ignored if use_multi_llm=True)
        max_workers: Maximum parallel workers
        use_batch: Whether to use batch API instead of realtime (ignored if use_multi_llm=True)
        paper_ids: Optional list of specific paper IDs to process
        step1_results_path: Path to step 1 combined results (for file_ids)
        step2_results_path: Path to step 2 combined results (for fallback global context)
        step3_results_path: Path to step 3 combined results
        use_multi_llm: Whether to use multi-LLM judge (Gemini, Grok, Qwen) for scoring
        with_context_eval: Whether to include context_appropriateness as 6th evaluation criterion

    Returns:
        Dictionary with results:
        {
            "results": {paper_id: [triplet_with_scores, ...], ...},
            "summary": {paper_id: {avg_scores, ...}, ...},
            "errors": {paper_id: [...], ...}
        }

        When use_multi_llm=True, each triplet contains:
        - per_model_scores: {gemini: {...}, grok: {...}, qwen: {...}}
        - per_model_reasons: {gemini: {...}, grok: {...}, qwen: {...}}
        - scores: averaged scores across models
        - successful_models: list of models that returned valid scores
    """
    logger = setup_logging()
    logger.info("Starting Step 4: Critic evaluation")
    if use_multi_llm:
        logger.info("Using Multi-LLM Judge mode (Gemini, Grok, Qwen)")
    if with_context_eval:
        logger.info("Context evaluation enabled (6 criteria including context_appropriateness)")

    # Select prompt and schema based on with_context_eval flag
    selected_prompt = STEP4_context_PROMPT if with_context_eval else STEP4_PROMPT
    selected_schema = STEP4_context_SCHEMA if with_context_eval else STEP4_SCHEMA

    output_path = Path(output_dir)

    # Load step 2 results (for fallback global context/id_method and paper type filtering)
    step2_path = step2_results_path or (output_path / "step2_combined.json")
    step2_data = load_json(step2_path)
    step2_results = step2_data.get("results", {})

    # Load step 3 results
    step3_path = step3_results_path or (output_path / "step3_combined.json")
    step3_data = load_json(step3_path)
    step3_results = step3_data.get("results", {})

    # Filter for empirical papers only (skip theoretical papers)
    empirical_papers = set()
    for paper_id, metadata in step2_results.items():
        paper_type = metadata.get("paper_metadata", {}).get("paper_type", "")
        if paper_type == "empirical":
            empirical_papers.add(paper_id)

    step3_results = {k: v for k, v in step3_results.items() if k in empirical_papers}
    logger.info(f"Filtered to {len(step3_results)} empirical papers")

    # Filter papers if specific IDs provided
    if paper_ids:
        step3_results = {k: v for k, v in step3_results.items() if k in paper_ids}

    # Prepare tasks for each triplet
    tasks = []
    task_metadata = []
    pdf_dir_path = Path(pdf_dir)

    for paper_id, triplets in step3_results.items():
        # Get global context/id_method from step 2 for fallback
        step2_metadata = step2_results.get(paper_id, {})
        global_context = step2_metadata.get("context", "")

        # Get PDF path for this paper
        pdf_path = pdf_dir_path / f"{paper_id}.pdf"
        pdf_path_str = str(pdf_path) if pdf_path.exists() else None

        for idx, triplet in enumerate(triplets):
            treatment = triplet.get("treatment", "")
            outcome = triplet.get("outcome", "")
            sign = clean_sign(triplet.get("sign", ""))
            evidence = triplet.get("supporting_evidence", [])
            selection = triplet.get("selection", {})
            context_selected = selection.get("context_selected", [])

            # If step 3 returned empty lists, use global context/id_method from step 2
            if not context_selected and global_context:
                context_selected = [global_context]

            # Format prompt
            prompt = selected_prompt.format(
                treatment=treatment,
                outcome=outcome,
                sign=sign,
                evidence_paragraphs=format_evidence(evidence),
                context_selected=format_context_selected(context_selected),
            )

            tasks.append({
                "user_prompt": prompt,
                "response_schema": selected_schema,
                "paper_id": f"{paper_id}_{idx}",
                "pdf_path": pdf_path_str,
            })

            task_metadata.append({
                "paper_id": paper_id,
                "triplet_idx": idx,
                "triplet": triplet,
            })

    output_path.mkdir(parents=True, exist_ok=True)

    if use_multi_llm:
        # Use Multi-LLM Judge for scoring
        return _run_multi_llm_scoring(
            tasks=tasks,
            task_metadata=task_metadata,
            output_path=output_path,
            max_workers=max_workers,
            logger=logger,
            with_context_eval=with_context_eval,
        )
    else:
        # Use single OpenAI model (legacy mode)
        # Load step 1 results (for file_ids) - only needed for OpenAI PDF mode
        step1_path = step1_results_path or (output_path / "step1_combined.json")
        step1_data = load_json(step1_path)
        file_ids = step1_data.get("file_ids", {})

        return _run_single_model_scoring(
            tasks=tasks,
            task_metadata=task_metadata,
            file_ids=file_ids,
            step3_results=step3_results,
            output_path=output_path,
            model=model,
            max_workers=max_workers,
            use_batch=use_batch,
            logger=logger,
        )


def _get_missing_model_tasks(
    checkpoint_path: Path,
    tasks: list[dict],
    logger: logging.Logger,
) -> list[dict]:
    """
    Check checkpoint for missing model scores and create tasks to fill them.

    Returns:
        List of tasks with missing_models field indicating which models to retry
    """
    if not checkpoint_path.exists():
        return []

    try:
        checkpoint_data = load_json(checkpoint_path)
        results = checkpoint_data.get("results", {})
    except Exception as e:
        logger.warning(f"Failed to load checkpoint for missing model check: {e}")
        return []

    tasks_to_fill = []
    all_models = ["gemini", "grok", "qwen"]

    for task in tasks:
        paper_id = task.get("paper_id")
        if paper_id not in results:
            continue

        result = results[paper_id]
        per_model_scores = result.get("per_model_scores", {})
        successful_models = result.get("successful_models", [])

        # Check which models are missing
        missing_models = [m for m in all_models if m not in successful_models or per_model_scores.get(m) is None]

        if missing_models:
            tasks_to_fill.append({
                **task,
                "missing_models": missing_models,
                "existing_result": result,
            })

    if tasks_to_fill:
        logger.info(f"Found {len(tasks_to_fill)} tasks with missing model scores")
        # Log summary of missing models
        missing_summary = {"gemini": 0, "grok": 0, "qwen": 0}
        for task in tasks_to_fill:
            for model in task["missing_models"]:
                missing_summary[model] += 1
        logger.info(f"Missing model counts: {missing_summary}")

    return tasks_to_fill


def _fill_missing_model_scores(
    judge: 'MultiLLMJudge',
    tasks_to_fill: list[dict],
    checkpoint_path: Path,
    logger: logging.Logger,
    with_context_eval: bool = False,
) -> None:
    """
    Fill in missing model scores for tasks with incomplete results.
    Updates the checkpoint file with the new scores.
    """
    # Load existing checkpoint data
    checkpoint_data = load_json(checkpoint_path)
    results = checkpoint_data.get("results", {})

    logger.info(f"Filling missing model scores for {len(tasks_to_fill)} tasks...")

    # Create tasks for each missing model
    model_tasks = []  # (paper_id, model_name, task)
    for task in tasks_to_fill:
        paper_id = task["paper_id"]
        for model_name in task["missing_models"]:
            model_tasks.append((paper_id, model_name, task))

    logger.info(f"Total {len(model_tasks)} model API calls needed")

    completed_count = 0

    with ThreadPoolExecutor(max_workers=min(judge.max_workers, 256)) as executor:
        futures = {
            executor.submit(
                judge.call_single_model,
                model_name,
                task["user_prompt"],
                task["response_schema"],
                paper_id,
                task.get("pdf_path"),
            ): (paper_id, model_name)
            for paper_id, model_name, task in model_tasks
        }

        with tqdm(total=len(futures), desc="Filling missing scores") as pbar:
            for future in as_completed(futures):
                paper_id, model_name = futures[future]
                try:
                    response = future.result()
                    if response.success and response.data:
                        scores = response.data.get("scores", {})
                        reasons = response.data.get("reasons", {})

                        # Update the result in memory
                        if paper_id in results:
                            if "per_model_scores" not in results[paper_id]:
                                results[paper_id]["per_model_scores"] = {}
                            if "per_model_reasons" not in results[paper_id]:
                                results[paper_id]["per_model_reasons"] = {}

                            results[paper_id]["per_model_scores"][model_name] = scores
                            results[paper_id]["per_model_reasons"][model_name] = reasons

                            # Update successful_models
                            if "successful_models" not in results[paper_id]:
                                results[paper_id]["successful_models"] = []
                            if model_name not in results[paper_id]["successful_models"]:
                                results[paper_id]["successful_models"].append(model_name)

                            # Recalculate average scores
                            _recalculate_average_scores(results[paper_id], with_context_eval)

                        logger.debug(f"Filled {model_name} score for {paper_id}")
                    else:
                        logger.warning(f"Failed to fill {model_name} for {paper_id}: {response.error}")
                except Exception as e:
                    logger.error(f"Error filling {model_name} for {paper_id}: {e}")

                pbar.update(1)
                completed_count += 1

                # Save checkpoint periodically
                if completed_count % 20 == 0:
                    save_json({"results": results}, checkpoint_path)

    # Final checkpoint save
    save_json({"results": results}, checkpoint_path)
    logger.info(f"Completed filling missing model scores. Updated checkpoint saved.")


def _calculate_average_scores(
    per_model_scores: dict,
    successful_models: list,
    with_context_eval: bool = False
) -> dict:
    """
    Calculate average scores from per_model_scores.

    Args:
        per_model_scores: Dict mapping model name to scores dict
        successful_models: List of model names that succeeded
        with_context_eval: Whether to include context_appropriateness

    Returns:
        Dict of averaged scores
    """
    if not successful_models:
        return {}

    score_keys = ["variable_extraction", "direction", "sign", "causality", "main_claim"]
    if with_context_eval:
        score_keys.append("context_appropriateness")
    average_scores = {}

    for key in score_keys:
        values = [
            per_model_scores[model][key]
            for model in successful_models
            if per_model_scores.get(model) and key in per_model_scores[model]
        ]
        if values:
            average_scores[key] = sum(values) / len(values)

    return average_scores


def _recalculate_average_scores(result: dict, with_context_eval: bool = False) -> None:
    """Recalculate average scores based on per_model_scores (in-place update)."""
    per_model_scores = result.get("per_model_scores", {})
    successful_models = result.get("successful_models", [])
    result["average_scores"] = _calculate_average_scores(
        per_model_scores, successful_models, with_context_eval
    )


def _run_multi_llm_scoring(
    tasks: list[dict],
    task_metadata: list[dict],
    output_path: Path,
    max_workers: int,
    logger: logging.Logger,
    with_context_eval: bool = False,
) -> dict:
    """
    Run scoring using Multi-LLM Judge (Gemini, Grok, Qwen).

    Returns results with per-model scores and averaged scores.
    Also fills in missing model scores from incomplete checkpoint data.
    """
    # Initialize Multi-LLM Judge with max 256 workers
    effective_max_workers = min(max_workers, 256)
    judge = MultiLLMJudge(max_workers=effective_max_workers)

    checkpoint_path = output_path / "step4_multi_llm_checkpoint.json"

    # Check for missing model scores and create fill tasks
    tasks_to_fill = _get_missing_model_tasks(checkpoint_path, tasks, logger)

    if tasks_to_fill:
        logger.info(f"Found {len(tasks_to_fill)} tasks with missing model scores to fill...")
        _fill_missing_model_scores(judge, tasks_to_fill, checkpoint_path, logger, with_context_eval)

    logger.info(f"Evaluating {len(tasks)} triplets with 3 LLM judges...")
    multi_results = judge.process_tasks_parallel(
        tasks,
        desc="Step 4: Multi-LLM Evaluation",
        checkpoint_path=checkpoint_path,
        checkpoint_interval=20,
    )

    # Create result map
    result_map = {r["paper_id"]: r for r in multi_results}

    # Collect results by paper
    results = {}
    errors = {}
    summaries = {}

    for meta in task_metadata:
        paper_id = meta["paper_id"]
        triplet_idx = meta["triplet_idx"]
        task_id = f"{paper_id}_{triplet_idx}"

        if paper_id not in results:
            results[paper_id] = []

        multi_result = result_map.get(task_id, {})

        if multi_result.get("successful_models"):
            # Always recalculate average scores from per_model_scores to ensure
            # all score types (including context_appropriateness) are included
            per_model_scores = multi_result.get("per_model_scores", {})
            successful_models = multi_result.get("successful_models", [])
            scores = _calculate_average_scores(
                per_model_scores, successful_models, with_context_eval
            )
            triplet_with_scores = {
                **meta["triplet"],
                "per_model_scores": per_model_scores,
                "per_model_reasons": multi_result.get("per_model_reasons", {}),
                "scores": scores,
                "successful_models": successful_models,
            }
            results[paper_id].append(triplet_with_scores)
        else:
            error_msg = multi_result.get("error", "All models failed")
            if paper_id not in errors:
                errors[paper_id] = []
            errors[paper_id].append({
                "triplet_idx": triplet_idx,
                "error": error_msg,
                "per_model_scores": multi_result.get("per_model_scores", {}),
            })
            # Include triplet with partial results
            results[paper_id].append({
                **meta["triplet"],
                "per_model_scores": multi_result.get("per_model_scores", {}),
                "per_model_reasons": multi_result.get("per_model_reasons", {}),
                "scores": None,
                "successful_models": [],
                "error": error_msg,
            })

    # Calculate summaries for each paper (including per-model summaries)
    score_keys = ["variable_extraction", "direction", "sign", "causality", "main_claim"]
    if with_context_eval:
        score_keys.append("context_appropriateness")

    for paper_id, triplets in results.items():
        valid_triplets = [t for t in triplets if t.get("scores")]
        if valid_triplets:
            # Calculate average across triplets
            avg_scores = {}
            for key in score_keys:
                values = [t["scores"].get(key, 0) for t in valid_triplets if t.get("scores")]
                if values:
                    avg_scores[key] = sum(values) / len(values)
            avg_scores["overall"] = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0

            # Calculate per-model averages
            per_model_avg = {}
            for model_name in ["gemini", "grok", "qwen"]:
                model_scores = [
                    t["per_model_scores"].get(model_name)
                    for t in valid_triplets
                    if t.get("per_model_scores", {}).get(model_name)
                ]
                if model_scores:
                    per_model_avg[model_name] = {}
                    for key in score_keys:
                        values = [s.get(key, 0) for s in model_scores]
                        if values:
                            per_model_avg[model_name][key] = sum(values) / len(values)
                    per_model_avg[model_name]["overall"] = (
                        sum(per_model_avg[model_name].values()) / len(per_model_avg[model_name])
                        if per_model_avg[model_name] else 0
                    )

            summaries[paper_id] = {
                "num_triplets": len(triplets),
                "num_evaluated": len(valid_triplets),
                "avg_scores": avg_scores,
                "per_model_avg_scores": per_model_avg,
            }

    # Save individual results
    for paper_id, triplets in results.items():
        save_json(
            {
                "paper_id": paper_id,
                "triplets": triplets,
                "summary": summaries.get(paper_id, {}),
                "scoring_mode": "multi_llm",
                "models_used": ["gemini", "grok", "qwen"],
            },
            output_path / f"step4_{paper_id}.json"
        )

    # Save combined results
    combined_output = {
        "results": results,
        "summaries": summaries,
        "errors": errors,
        "scoring_mode": "multi_llm",
        "models_used": ["gemini", "grok", "qwen"],
        "with_context_eval": with_context_eval,
    }
    save_json(combined_output, output_path / "step4_combined.json")

    # Save API errors to separate JSON file
    judge.save_errors_json(output_path / "step4_api_errors.json")

    # Calculate and log overall statistics
    _log_overall_statistics(results, logger, multi_llm=True, with_context_eval=with_context_eval)

    total_triplets = sum(len(v) for v in results.values())
    total_errors = sum(len(v) for v in errors.values()) if errors else 0

    logger.info(f"Step 4 complete. Results saved to {output_path}")
    logger.info(f"Evaluated {total_triplets} triplets, {total_errors} errors")

    return combined_output


def _run_single_model_scoring(
    tasks: list[dict],
    task_metadata: list[dict],
    file_ids: dict[str, str],
    step3_results: dict,
    output_path: Path,
    model: str,
    max_workers: int,
    use_batch: bool,
    logger: logging.Logger,
) -> dict:
    """
    Run scoring using single OpenAI model (legacy mode).
    """
    # Add file_ids to tasks
    for task in tasks:
        paper_id = task["paper_id"].rsplit("_", 1)[0]
        task["file_id"] = file_ids.get(paper_id)

    # Filter tasks without file_ids
    valid_tasks = [t for t in tasks if t.get("file_id")]
    skipped = len(tasks) - len(valid_tasks)
    if skipped:
        logger.warning(f"Skipping {skipped} tasks without file_ids")

    # Initialize client
    client = OpenAIClient(
        model=model,
        max_workers=max_workers,
        use_batch=use_batch,
    )

    checkpoint_path = output_path / "step4_checkpoint.json"

    logger.info(f"Evaluating {len(valid_tasks)} triplets with {model}...")
    responses = client.process_tasks(
        valid_tasks,
        use_pdf=True,
        desc="Step 4: Evaluating triplets",
        checkpoint_path=checkpoint_path,
        checkpoint_interval=50,
    )

    # Collect results by paper
    results = {}
    errors = {}
    summaries = {}

    # Map responses back to papers
    response_map = {r.paper_id: r for r in responses}

    for meta in task_metadata:
        paper_id = meta["paper_id"]
        triplet_idx = meta["triplet_idx"]
        task_id = f"{paper_id}_{triplet_idx}"

        if paper_id not in results:
            results[paper_id] = []

        response = response_map.get(task_id)
        if response and response.success:
            scores = response.data.get("scores", {})
            reasons = response.data.get("reasons", {})
            triplet_with_scores = {
                **meta["triplet"],
                "scores": scores,
                "reasons": reasons,
            }
            results[paper_id].append(triplet_with_scores)
        else:
            error_msg = response.error if response else "Response not found"
            if paper_id not in errors:
                errors[paper_id] = []
            errors[paper_id].append({
                "triplet_idx": triplet_idx,
                "error": error_msg,
            })
            # Include triplet without scores
            results[paper_id].append({
                **meta["triplet"],
                "scores": None,
                "error": error_msg,
            })

    # Calculate summaries for each paper
    for paper_id, triplets in results.items():
        valid_scores = [t["scores"] for t in triplets if t.get("scores")]
        if valid_scores:
            avg_scores = {
                "variable_extraction": sum(s.get("variable_extraction", 0) for s in valid_scores) / len(valid_scores),
                "direction": sum(s.get("direction", 0) for s in valid_scores) / len(valid_scores),
                "sign": sum(s.get("sign", 0) for s in valid_scores) / len(valid_scores),
                "causality": sum(s.get("causality", 0) for s in valid_scores) / len(valid_scores),
                "main_claim": sum(s.get("main_claim", 0) for s in valid_scores) / len(valid_scores),
            }
            avg_scores["overall"] = sum(avg_scores.values()) / 5
            summaries[paper_id] = {
                "num_triplets": len(triplets),
                "num_evaluated": len(valid_scores),
                "avg_scores": avg_scores,
            }

    # Save individual results
    for paper_id, triplets in results.items():
        save_json(
            {
                "paper_id": paper_id,
                "triplets": triplets,
                "summary": summaries.get(paper_id, {}),
                "scoring_mode": "single_model",
                "model_used": model,
            },
            output_path / f"step4_{paper_id}.json"
        )

    # Save combined results
    combined_output = {
        "results": results,
        "summaries": summaries,
        "errors": errors,
        "scoring_mode": "single_model",
        "model_used": model,
    }
    save_json(combined_output, output_path / "step4_combined.json")

    # Save API errors to separate JSON file
    client.save_errors_json(output_path / "step4_api_errors.json")

    # Calculate and log overall statistics
    _log_overall_statistics(results, logger, multi_llm=False)

    total_triplets = sum(len(v) for v in results.values())
    total_errors = sum(len(v) for v in errors.values()) if errors else 0

    logger.info(f"Step 4 complete. Results saved to {output_path}")
    logger.info(f"Evaluated {total_triplets} triplets, {total_errors} errors")

    return combined_output


def _log_overall_statistics(results: dict, logger: logging.Logger, multi_llm: bool = False, with_context_eval: bool = False) -> None:
    """Log overall statistics for the evaluation."""
    all_scores = []
    for triplets in results.values():
        for t in triplets:
            if t.get("scores"):
                all_scores.append(t["scores"])

    score_keys = ["variable_extraction", "direction", "sign", "causality", "main_claim"]
    if with_context_eval:
        score_keys.append("context_appropriateness")

    if all_scores:
        overall_avg = {}
        for key in score_keys:
            overall_avg[key] = sum(s.get(key, 0) for s in all_scores) / len(all_scores)
        logger.info(f"Overall average scores across {len(all_scores)} triplets:")
        for dim, score in overall_avg.items():
            logger.info(f"  {dim}: {score:.2f}")

    if multi_llm:
        # Log per-model statistics
        for model_name in ["gemini", "grok", "qwen"]:
            model_scores = []
            for triplets in results.values():
                for t in triplets:
                    if t.get("per_model_scores", {}).get(model_name):
                        model_scores.append(t["per_model_scores"][model_name])

            if model_scores:
                model_avg = {}
                for key in score_keys:
                    model_avg[key] = sum(s.get(key, 0) for s in model_scores) / len(model_scores)
                logger.info(f"\n{model_name.upper()} average scores ({len(model_scores)} triplets):")
                for dim, score in model_avg.items():
                    logger.info(f"  {dim}: {score:.2f}")


def main():
    """Command-line interface for Step 4."""
    parser = argparse.ArgumentParser(
        description="Step 4: Critic evaluation of causal triplets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=DEFAULT_METADATA_CSV,
        help="Path to metadata CSV with abstracts"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=DEFAULT_PDF_DIR,
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum parallel workers"
    )
    parser.add_argument(
        "--use-batch",
        action="store_true",
        help="Use batch API instead of realtime"
    )
    parser.add_argument(
        "--paper-ids",
        type=str,
        nargs="+",
        help="Specific paper IDs to process"
    )
    parser.add_argument(
        "--step1-results",
        type=str,
        help="Path to step1_combined.json (for file_ids)"
    )
    parser.add_argument(
        "--step2-results",
        type=str,
        help="Path to step2_combined.json (for fallback global context)"
    )
    parser.add_argument(
        "--step3-results",
        type=str,
        help="Path to step3_combined.json"
    )
    parser.add_argument(
        "--use-multi-llm",
        action="store_true",
        default=True,
        help="Use Multi-LLM Judge (Gemini, Grok, Qwen) for scoring (default: True)"
    )
    parser.add_argument(
        "--no-multi-llm",
        action="store_true",
        help="Disable Multi-LLM Judge and use single OpenAI model"
    )
    parser.add_argument(
        "--with-context-eval",
        action="store_true",
        help="Include context_appropriateness as 6th evaluation criterion"
    )

    args = parser.parse_args()

    # Handle --no-multi-llm flag
    use_multi_llm = args.use_multi_llm and not args.no_multi_llm

    run_step4(
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        pdf_dir=args.pdf_dir,
        model=args.model,
        max_workers=args.max_workers,
        use_batch=args.use_batch,
        paper_ids=args.paper_ids,
        step1_results_path=args.step1_results,
        step2_results_path=args.step2_results,
        step3_results_path=args.step3_results,
        use_multi_llm=use_multi_llm,
        with_context_eval=args.with_context_eval,
    )


if __name__ == "__main__":
    main()
