"""
Step 1: Extract causal relations from economics papers.

This step uploads PDFs to OpenAI and extracts causal relation pairs
(treatment, outcome, sign) with supporting evidence.

Optionally supports consensus mode: runs the extraction multiple times
and only keeps triplets that appear consistently across runs.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import (
    OpenAIClient,
    get_pdf_files,
    save_json,
    load_json,
    setup_logging,
    cosine_similarity,
    clean_causal_relations,
    DEFAULT_MODEL,
    DEFAULT_MAX_WORKERS,
)
from common.prompts import STEP1_PROMPT
from common.schemas import STEP1_SCHEMA


# Default consensus parameters
DEFAULT_CONSENSUS_RUNS = 3
DEFAULT_CONSENSUS_THRESHOLD = 2
DEFAULT_SIMILARITY_THRESHOLD = 0.8


# Default paths
DEFAULT_INPUT_DIR = "/home/donggyu/econ_causality/new_data/test"
DEFAULT_OUTPUT_DIR = "/home/donggyu/econ_causality/new_data/test_data"


def filter_by_consensus(
    all_runs_results: list[list[dict]],
    client: OpenAIClient,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    consensus_threshold: int = DEFAULT_CONSENSUS_THRESHOLD,
    logger: logging.Logger = None,
) -> list[dict]:
    """
    Filter triplets by consensus across multiple runs using embedding similarity.

    Args:
        all_runs_results: List of results from multiple API runs
        client: OpenAI client for embedding
        similarity_threshold: Cosine similarity threshold to consider triplets identical
        consensus_threshold: Minimum number of runs a triplet must appear in
        logger: Logger instance

    Returns:
        List of triplets that appear in at least consensus_threshold runs
    """
    if not all_runs_results or not any(all_runs_results):
        return []

    # Collect all unique triplets with their embeddings
    # Key: (treatment, outcome) -> list of (run_idx, triplet, treatment_emb, outcome_emb)
    all_triplets = []
    for run_idx, run_results in enumerate(all_runs_results):
        for triplet in run_results:
            treatment = triplet.get("treatment", "")
            outcome = triplet.get("outcome", "")
            if treatment and outcome:
                all_triplets.append({
                    "run_idx": run_idx,
                    "triplet": triplet,
                    "treatment": treatment,
                    "outcome": outcome,
                })

    if not all_triplets:
        return []

    # Get embeddings for all treatments and outcomes
    all_texts = []
    for t in all_triplets:
        all_texts.append(t["treatment"])
        all_texts.append(t["outcome"])

    if logger:
        logger.info(f"Getting embeddings for {len(all_texts)} texts...")

    embeddings = client.get_embeddings_batch(all_texts)

    # Assign embeddings back to triplets
    for i, t in enumerate(all_triplets):
        t["treatment_emb"] = embeddings[i * 2]
        t["outcome_emb"] = embeddings[i * 2 + 1]

    # Cluster triplets by similarity
    # A triplet matches another if both treatment and outcome have similarity >= threshold
    clusters = []  # List of lists of triplet indices
    clustered = set()

    for i, t1 in enumerate(all_triplets):
        if i in clustered:
            continue

        cluster = [i]
        clustered.add(i)

        for j, t2 in enumerate(all_triplets):
            if j in clustered:
                continue

            # Check if treatment and outcome are similar
            treatment_sim = cosine_similarity(t1["treatment_emb"], t2["treatment_emb"])
            outcome_sim = cosine_similarity(t1["outcome_emb"], t2["outcome_emb"])

            if treatment_sim >= similarity_threshold and outcome_sim >= similarity_threshold:
                cluster.append(j)
                clustered.add(j)

        clusters.append(cluster)

    # Filter clusters by consensus threshold
    # Count unique runs in each cluster
    consensus_triplets = []
    for cluster in clusters:
        run_indices = set(all_triplets[i]["run_idx"] for i in cluster)
        if len(run_indices) >= consensus_threshold:
            # Pick the triplet from the first run as representative
            representative_idx = min(cluster, key=lambda x: all_triplets[x]["run_idx"])
            representative = all_triplets[representative_idx]["triplet"]

            # Majority voting for sign
            sign_counts = defaultdict(int)
            for idx in cluster:
                sign = all_triplets[idx]["triplet"].get("sign") or ""
                if sign:
                    sign_counts[sign] += 1

            # Select the most common sign
            if sign_counts:
                voted_sign = max(sign_counts.keys(), key=lambda s: sign_counts[s])
            else:
                voted_sign = representative.get("sign") or ""

            # Merge supporting evidence from all triplets in cluster
            all_evidence = []
            seen_evidence = set()
            for idx in cluster:
                for ev in all_triplets[idx]["triplet"].get("supporting_evidence", []):
                    if ev not in seen_evidence:
                        all_evidence.append(ev)
                        seen_evidence.add(ev)

            # Create merged triplet
            merged_triplet = {
                "treatment": representative.get("treatment", ""),
                "outcome": representative.get("outcome", ""),
                "sign": voted_sign,
                "supporting_evidence": all_evidence[:3],  # Keep max 3 evidence
                "_consensus_count": len(run_indices),
                "_sign_votes": dict(sign_counts),  # Store vote distribution for debugging
            }
            consensus_triplets.append(merged_triplet)

    if logger:
        logger.info(
            f"Consensus filtering: {sum(len(r) for r in all_runs_results)} total -> "
            f"{len(consensus_triplets)} after filtering (threshold={consensus_threshold}/{len(all_runs_results)})"
        )

    return consensus_triplets


def run_step1(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model: str = DEFAULT_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    use_batch: bool = False,
    paper_ids: Optional[list[str]] = None,
    consensus_mode: bool = False,
    consensus_runs: int = DEFAULT_CONSENSUS_RUNS,
    consensus_threshold: int = DEFAULT_CONSENSUS_THRESHOLD,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict:
    """
    Run Step 1: Extract causal relations from papers.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save output JSON files
        model: OpenAI model to use
        max_workers: Maximum parallel workers
        use_batch: Whether to use batch API instead of realtime
        paper_ids: Optional list of specific paper IDs to process
        consensus_mode: Enable consensus mode (run multiple times and filter)
        consensus_runs: Number of API runs per paper in consensus mode
        consensus_threshold: Minimum runs a triplet must appear in to be kept
        similarity_threshold: Cosine similarity threshold for matching triplets

    Returns:
        Dictionary with results and file mappings:
        {
            "file_ids": {paper_id: file_id, ...},
            "results": {paper_id: causal_relations, ...},
            "errors": {paper_id: error_message, ...}
        }
    """
    logger = setup_logging()
    logger.info("Starting Step 1: Extract causal relations")

    if consensus_mode:
        logger.info(f"Consensus mode enabled: {consensus_runs} runs, threshold={consensus_threshold}, similarity={similarity_threshold}")

    # Initialize client
    client = OpenAIClient(
        model=model,
        max_workers=max_workers,
        use_batch=use_batch,
    )

    # Get PDF files
    input_path = Path(input_dir)
    pdf_files = get_pdf_files(input_path)

    if paper_ids:
        pdf_files = [p for p in pdf_files if p.stem in paper_ids]

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    if not pdf_files:
        logger.warning("No PDF files found")
        return {"file_ids": {}, "results": {}, "errors": {}}

    # Upload PDFs in parallel
    logger.info("Uploading PDFs to OpenAI...")
    file_ids = client.upload_pdfs_parallel(pdf_files, desc="Step 1: Uploading PDFs")

    # Filter out failed uploads
    valid_uploads = {pid: fid for pid, fid in file_ids.items() if fid is not None}
    failed_uploads = {pid for pid, fid in file_ids.items() if fid is None}

    logger.info(f"Successfully uploaded {len(valid_uploads)} files")
    if failed_uploads:
        logger.warning(f"Failed to upload: {failed_uploads}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}
    errors = {}

    if consensus_mode:
        # Consensus mode: run multiple times and filter by agreement
        logger.info(
            "Consensus mode enabled: "
            f"{consensus_runs} runs, threshold={consensus_threshold}, similarity={similarity_threshold}"
        )

        # (A) raw 저장용 디렉토리/컨테이너
        raw_runs_dir = output_path / "step1_runs"
        raw_runs_dir.mkdir(parents=True, exist_ok=True)

        raw_runs_results = {}  # paper_id -> {run_idx: [triplets]}
        raw_runs_errors = {}   # paper_id -> {run_idx: error_msg}

        # Create ALL tasks for ALL papers at once (fully parallel)
        tasks = []
        task_paper_map = []  # Track which paper/run each task belongs to

        for paper_id, file_id in valid_uploads.items():
            for run_idx in range(consensus_runs):
                tasks.append({
                    "file_id": file_id,
                    "system_prompt": "",
                    "user_prompt": STEP1_PROMPT,
                    "response_schema": STEP1_SCHEMA,
                    "paper_id": f"{paper_id}_run{run_idx}",
                })
                task_paper_map.append({
                    "paper_id": paper_id,
                    "run_idx": run_idx,
                })

        total_tasks = len(tasks)
        logger.info(
            f"Consensus mode: {len(valid_uploads)} papers × {consensus_runs} runs = "
            f"{total_tasks} total API calls (all parallel)"
        )

        # Process ALL tasks in parallel
        checkpoint_path = output_path / "step1_consensus_checkpoint.json"
        responses = client.process_tasks(
            tasks,
            use_pdf=True,
            desc="Step 1: Extracting causal relations (consensus)",
            checkpoint_path=checkpoint_path,
            checkpoint_interval=50,
        )

        # Group responses by paper_id
        paper_runs = {}  # paper_id -> list of (run_idx, response)
        response_map = {r.paper_id: r for r in responses}

        for meta in task_paper_map:
            paper_id = meta["paper_id"]
            run_idx = meta["run_idx"]
            task_id = f"{paper_id}_run{run_idx}"
            response = response_map.get(task_id)

            if paper_id not in paper_runs:
                paper_runs[paper_id] = []
            paper_runs[paper_id].append((run_idx, response))

        # Process each paper's results - collect data for parallel consensus filtering
        papers_for_consensus = []  # (paper_id, all_runs_results) tuples

        for paper_id, runs in paper_runs.items():
            # Collect successful runs
            all_runs_results = []
            successful_run_indices = []

            raw_runs_results.setdefault(paper_id, {})
            raw_runs_errors.setdefault(paper_id, {})

            for run_idx, response in sorted(runs, key=lambda x: x[0]):
                if response and response.success:
                    run_triplets = clean_causal_relations(response.data.get("causal_relations", []))

                    # (B-1) 메모리에 raw 저장
                    raw_runs_results[paper_id][run_idx] = run_triplets

                    # (B-2) 파일로 raw 저장 (paper/run 단위)
                    save_json(
                        {
                            "paper_id": paper_id,
                            "run_idx": run_idx,
                            "causal_relations": run_triplets,
                        },
                        raw_runs_dir / f"step1_{paper_id}_run{run_idx}.json",
                    )

                    # consensus 계산용
                    all_runs_results.append(run_triplets)
                    successful_run_indices.append(run_idx)
                else:
                    error_msg = response.error if response else "Response not found"
                    raw_runs_errors[paper_id][run_idx] = error_msg

            # (B-3) (선택이지만 유용) paper별 raw runs 묶음 파일 저장
            save_json(
                {
                    "paper_id": paper_id,
                    "successful_runs": successful_run_indices,
                    "raw_runs": raw_runs_results.get(paper_id, {}),
                    "raw_run_errors": raw_runs_errors.get(paper_id, {}),
                },
                raw_runs_dir / f"step1_{paper_id}_runs_raw.json",
            )

            if len(all_runs_results) < consensus_threshold:
                # Not enough successful runs
                errors[paper_id] = f"Only {len(all_runs_results)}/{consensus_runs} runs succeeded"
                logger.error(f"Paper {paper_id}: {errors[paper_id]}")
                continue

            # Collect for parallel consensus filtering
            papers_for_consensus.append((paper_id, all_runs_results))

        # Parallel consensus filtering across papers
        if papers_for_consensus:
            logger.info(f"Running parallel consensus filtering for {len(papers_for_consensus)} papers...")

            from concurrent.futures import ThreadPoolExecutor, as_completed

            def process_paper_consensus(args):
                paper_id, all_runs_results = args
                consensus_results = filter_by_consensus(
                    all_runs_results,
                    client,
                    similarity_threshold=similarity_threshold,
                    consensus_threshold=consensus_threshold,
                    logger=None,  # Suppress per-paper logging
                )
                return paper_id, all_runs_results, consensus_results

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_paper_consensus, args): args[0]
                    for args in papers_for_consensus
                }

                with tqdm(total=len(futures), desc="Consensus filtering") as pbar:
                    for future in as_completed(futures):
                        try:
                            paper_id, all_runs_results, consensus_results = future.result()
                            results[paper_id] = consensus_results
                            logger.info(
                                f"Paper {paper_id}: {sum(len(r) for r in all_runs_results)} raw triplets -> "
                                f"{len(consensus_results)} after consensus filtering"
                            )
                        except Exception as e:
                            paper_id = futures[future]
                            errors[paper_id] = f"Consensus filtering failed: {str(e)}"
                            logger.error(f"Paper {paper_id}: {errors[paper_id]}")
                        pbar.update(1)

    else:
        # Standard mode: single run per paper
        # Prepare API tasks
        tasks = []
        for paper_id, file_id in valid_uploads.items():
            tasks.append({
                "file_id": file_id,
                "system_prompt": "",
                "user_prompt": STEP1_PROMPT,
                "response_schema": STEP1_SCHEMA,
                "paper_id": paper_id,
            })

        # Process tasks with checkpoint
        checkpoint_path = output_path / "step1_checkpoint.json"

        logger.info(f"Processing {len(tasks)} papers...")
        responses = client.process_tasks(
            tasks,
            use_pdf=True,
            desc="Step 1: Extracting causal relations",
            checkpoint_path=checkpoint_path,
            checkpoint_interval=10,
        )

        # Collect results
        for response in responses:
            paper_id = response.paper_id
            if response.success:
                results[paper_id] = clean_causal_relations(response.data.get("causal_relations", []))
                logger.info(
                    f"Paper {paper_id}: extracted {len(results[paper_id])} causal relations"
                )
            else:
                errors[paper_id] = response.error
                logger.error(f"Paper {paper_id}: {response.error}")

    # Add upload failures to errors
    for paper_id in failed_uploads:
        errors[paper_id] = "Failed to upload PDF"

    # Save individual results
    for paper_id, causal_relations in results.items():
        save_json(
            {"paper_id": paper_id, "causal_relations": causal_relations},
            output_path / f"step1_{paper_id}.json"
        )

    # Save combined results
    combined_output = {
        "file_ids": valid_uploads,
        "results": results,
        "errors": errors,
    }
    save_json(combined_output, output_path / "step1_combined.json")

    # Save API errors to separate JSON file
    client.save_errors_json(output_path / "step1_api_errors.json")

    logger.info(f"Step 1 complete. Results saved to {output_path}")
    logger.info(f"Successful: {len(results)}, Errors: {len(errors)}")

    return combined_output


def main():
    """Command-line interface for Step 1."""
    parser = argparse.ArgumentParser(
        description="Step 1: Extract causal relations from economics papers"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files"
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
        "--consensus-mode",
        action="store_true",
        help="Enable consensus mode: run multiple times and filter by agreement"
    )
    parser.add_argument(
        "--consensus-runs",
        type=int,
        default=DEFAULT_CONSENSUS_RUNS,
        help=f"Number of API runs per paper in consensus mode (default: {DEFAULT_CONSENSUS_RUNS})"
    )
    parser.add_argument(
        "--consensus-threshold",
        type=int,
        default=DEFAULT_CONSENSUS_THRESHOLD,
        help=f"Minimum runs a triplet must appear in to be kept (default: {DEFAULT_CONSENSUS_THRESHOLD})"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold for matching triplets (default: {DEFAULT_SIMILARITY_THRESHOLD})"
    )

    args = parser.parse_args()

    run_step1(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        max_workers=args.max_workers,
        use_batch=args.use_batch,
        paper_ids=args.paper_ids,
        consensus_mode=args.consensus_mode,
        consensus_runs=args.consensus_runs,
        consensus_threshold=args.consensus_threshold,
        similarity_threshold=args.similarity_threshold,
    )


if __name__ == "__main__":
    main()
