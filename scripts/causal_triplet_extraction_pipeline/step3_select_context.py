"""
Step 3: Select applicable context and identification methods for each triplet.

This step takes the causal relations from Step 1 and the global context/methods
from Step 2, and selects which context and methods apply to each specific triplet.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import (
    OpenAIClient,
    save_json,
    load_json,
    load_csv,
    setup_logging,
    clean_sign,
    DEFAULT_MODEL,
    DEFAULT_MAX_WORKERS,
)
from common.prompts import STEP3_PROMPT
from common.schemas import STEP3_SCHEMA


# Default paths
DEFAULT_INPUT_DIR = "/home/donggyu/econ_causality/new_data/nber_paper_wo_aer_pp_30pages"
DEFAULT_OUTPUT_DIR = "/home/donggyu/econ_causality/new_data/real_data"

DEFAULT_METADATA_CSV = "/home/donggyu/econ_causality/new_data/nber_paper/metadata_published_8journals_with_JEL.csv"


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


def run_step3(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    metadata_csv: str = DEFAULT_METADATA_CSV,
    model: str = DEFAULT_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    use_batch: bool = False,
    paper_ids: Optional[list[str]] = None,
    step1_results_path: Optional[str] = None,
    step2_results_path: Optional[str] = None,
) -> dict:
    """
    Run Step 3: Select context for each causal triplet.

    Args:
        output_dir: Directory to save output JSON files
        metadata_csv: Path to metadata CSV with abstracts
        model: OpenAI model to use
        max_workers: Maximum parallel workers
        use_batch: Whether to use batch API instead of realtime
        paper_ids: Optional list of specific paper IDs to process
        step1_results_path: Path to step 1 combined results
        step2_results_path: Path to step 2 combined results

    Returns:
        Dictionary with results:
        {
            "results": {paper_id: [triplet_with_selection, ...], ...},
            "errors": {paper_id: error_message, ...}
        }
    """
    logger = setup_logging()
    logger.info("Starting Step 3: Select context for triplets")

    output_path = Path(output_dir)

    # Load step 1 and step 2 results
    step1_path = step1_results_path or (output_path / "step1_combined.json")
    step2_path = step2_results_path or (output_path / "step2_combined.json")

    step1_data = load_json(step1_path)
    step2_data = load_json(step2_path)

    step1_results = step1_data.get("results", {})
    step2_results = step2_data.get("results", {})
    file_ids = step1_data.get("file_ids", {})

    # Filter for empirical papers only (skip theoretical papers)
    empirical_papers = set()
    for paper_id, metadata in step2_results.items():
        paper_type = metadata.get("paper_metadata", {}).get("paper_type", "")
        if paper_type == "empirical":
            empirical_papers.add(paper_id)

    step1_results = {k: v for k, v in step1_results.items() if k in empirical_papers}
    logger.info(f"Filtered to {len(step1_results)} empirical papers (skipped {len(step2_results) - len(empirical_papers)} theoretical papers)")

    # Load abstracts
    logger.info(f"Loading abstracts from {metadata_csv}")
    abstracts = load_abstracts(metadata_csv)

    # Initialize client
    client = OpenAIClient(
        model=model,
        max_workers=max_workers,
        use_batch=use_batch,
    )

    # Filter papers if specific IDs provided
    if paper_ids:
        step1_results = {k: v for k, v in step1_results.items() if k in paper_ids}

    # Prepare tasks for each triplet
    tasks = []
    task_metadata = []  # Track which paper/triplet each task belongs to

    for paper_id, causal_relations in step1_results.items():
        # Get metadata from step 2
        metadata = step2_results.get(paper_id, {})
        global_context = metadata.get("context", "")
        global_methods = metadata.get("identification_methods", [])

        # Get file_id for this paper
        file_id = file_ids.get(paper_id)
        if not file_id:
            logger.warning(f"No file_id found for {paper_id}, skipping...")
            continue

        # Get abstract
        abstract = abstracts.get(paper_id, "(Abstract not available)")

        # Create task for each triplet
        for idx, triplet in enumerate(causal_relations):
            treatment = triplet.get("treatment", "")
            outcome = triplet.get("outcome", "")
            sign = clean_sign(triplet.get("sign", ""))
            evidence = triplet.get("supporting_evidence", [])

            # Format prompt
            prompt = STEP3_PROMPT.format(
                treatment=treatment,
                outcome=outcome,
                sign=sign,
                evidence_paragraphs=format_evidence(evidence),
                global_context=global_context,
                global_identification_method=json.dumps(global_methods),
            )
            # print(prompt)
            # assert(0)
            tasks.append({
                "system_prompt": "",
                "user_prompt": prompt,
                "response_schema": STEP3_SCHEMA,
                "paper_id": f"{paper_id}_{idx}",
                "file_id": file_id,
            })

            task_metadata.append({
                "paper_id": paper_id,
                "triplet_idx": idx,
                "triplet": triplet,
            })

    # Process tasks with checkpoint
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "step3_checkpoint.json"

    logger.info(f"Processing {len(tasks)} triplets from {len(step1_results)} papers...")
    responses = client.process_tasks(
        tasks,
        use_pdf=True,
        desc="Step 3: Selecting context",
        checkpoint_path=checkpoint_path,
        checkpoint_interval=50,
    )

    # Collect results by paper
    results = {}
    errors = {}

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
            # Merge original triplet with selection
            triplet_with_selection = {
                **meta["triplet"],
                "selection": response.data.get("selection", {}),
            }
            results[paper_id].append(triplet_with_selection)
        else:
            error_msg = response.error if response else "Response not found"
            if paper_id not in errors:
                errors[paper_id] = []
            errors[paper_id].append({
                "triplet_idx": triplet_idx,
                "error": error_msg,
            })
            # Still include triplet without selection
            results[paper_id].append({
                **meta["triplet"],
                "selection": {"context_selected": [], "id_method_selected": []},
                "error": error_msg,
            })

    # Save individual results
    for paper_id, triplets in results.items():
        save_json(
            {"paper_id": paper_id, "triplets": triplets},
            output_path / f"step3_{paper_id}.json"
        )

    # Save combined results
    combined_output = {
        "results": results,
        "errors": errors,
    }
    save_json(combined_output, output_path / "step3_combined.json")

    # Save API errors to separate JSON file
    client.save_errors_json(output_path / "step3_api_errors.json")

    total_triplets = sum(len(v) for v in results.values())
    total_errors = sum(len(v) for v in errors.values()) if errors else 0

    logger.info(f"Step 3 complete. Results saved to {output_path}")
    logger.info(f"Processed {total_triplets} triplets, {total_errors} errors")

    return combined_output


def main():
    """Command-line interface for Step 3."""
    parser = argparse.ArgumentParser(
        description="Step 3: Select context for each causal triplet"
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
        help="Path to step1_combined.json"
    )
    parser.add_argument(
        "--step2-results",
        type=str,
        help="Path to step2_combined.json"
    )

    args = parser.parse_args()

    run_step3(
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        model=args.model,
        max_workers=args.max_workers,
        use_batch=args.use_batch,
        paper_ids=args.paper_ids,
        step1_results_path=args.step1_results,
        step2_results_path=args.step2_results,
    )


if __name__ == "__main__":
    main()
