"""
Main pipeline runner for causal relation extraction.

This script orchestrates all 5 steps of the pipeline:
1. Extract causal relations from papers
2. Extract paper metadata and context
3. Select context for each triplet
4. Critic evaluation
5. Generate final Excel results

The pipeline can be run end-to-end or from any specific step.
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import (
    OpenAIClient,
    get_pdf_files,
    save_json,
    load_json,
    setup_logging,
    DEFAULT_MODEL,
    DEFAULT_MAX_WORKERS,
)

from step1_extract_causal_relations import run_step1
from step2_extract_metadata import run_step2
from step3_select_context import run_step3
from step4_critic_evaluation import run_step4
from step5_generate_results import run_step5


# Default paths
DEFAULT_INPUT_DIR = "/home/donggyu/econ_causality/new_data/nber_paper_wo_aer_pp_new_30pages"
DEFAULT_OUTPUT_DIR = "/home/donggyu/econ_causality/new_data/test_data2"
DEFAULT_METADATA_CSV = "/home/donggyu/econ_causality/new_data/nber_paper/metadata_published_8journals_with_JEL.csv"
DEFAULT_NBER_METADATA_DIR = "/home/donggyu/econ_causality/econ_eval/nber_metadata"
DEFAULT_PDF_DIR = "/home/donggyu/econ_causality/new_data/nber_paper_wo_aer_pp_30pages"


def run_pipeline(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    metadata_csv: str = DEFAULT_METADATA_CSV,
    nber_metadata_dir: str = DEFAULT_NBER_METADATA_DIR,
    pdf_dir: str = DEFAULT_PDF_DIR,
    model: str = DEFAULT_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    use_batch: bool = False,
    paper_ids: Optional[list[str]] = None,
    start_step: int = 1,
    end_step: int = 5,
    consensus_mode: bool = False,
    consensus_runs: int = 3,
    consensus_threshold: int = 2,
    similarity_threshold: float = 0.8,
    exclude_dir: Optional[str] = None,
) -> dict:
    """
    Run the full causal relation extraction pipeline.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save output files
        metadata_csv: Path to metadata CSV with abstracts
        nber_metadata_dir: Directory containing NBER metadata CSVs
        pdf_dir: Directory containing journal subfolders with PDFs (for venue mapping)
        model: OpenAI model to use
        max_workers: Maximum parallel workers
        use_batch: Whether to use batch API instead of realtime
        paper_ids: Optional list of specific paper IDs to process
        start_step: Step to start from (1-5)
        end_step: Step to end at (1-5)
        consensus_mode: Enable consensus mode for step 1
        consensus_runs: Number of API runs per paper in consensus mode
        consensus_threshold: Minimum runs a triplet must appear in
        similarity_threshold: Cosine similarity threshold for matching triplets
        exclude_dir: Directory whose PDFs should be excluded from processing

    Returns:
        Dictionary with results from all completed steps
    """
    logger = setup_logging()

    # If exclude_dir is given, compute new-only paper_ids
    if exclude_dir:
        input_pdfs = set(p.stem for p in Path(input_dir).rglob("*.pdf"))
        exclude_pdfs = set(p.stem for p in Path(exclude_dir).rglob("*.pdf"))
        new_only = sorted(input_pdfs - exclude_pdfs)
        logger.info(f"Exclude dir: {exclude_dir}")
        logger.info(f"Total PDFs in input: {len(input_pdfs)}, in exclude: {len(exclude_pdfs)}, new only: {len(new_only)}")
        if paper_ids:
            # Intersect with user-specified paper_ids
            paper_ids = sorted(set(paper_ids) & set(new_only))
        else:
            paper_ids = new_only

    logger.info("=" * 60)
    logger.info("Starting Causal Relation Extraction Pipeline")
    logger.info(f"Steps: {start_step} -> {end_step}")
    logger.info(f"Model: {model}")
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"Use batch API: {use_batch}")
    if paper_ids:
        logger.info(f"Processing {len(paper_ids)} papers")
    logger.info("=" * 60)

    results = {}
    start_time = time.time()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract causal relations
    if start_step <= 1 <= end_step:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: Extract Causal Relations")
        logger.info("=" * 40)

        step1_start = time.time()
        results["step1"] = run_step1(
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            max_workers=max_workers,
            use_batch=use_batch,
            paper_ids=paper_ids,
            consensus_mode=consensus_mode,
            consensus_runs=consensus_runs,
            consensus_threshold=consensus_threshold,
            similarity_threshold=similarity_threshold,
        )
        step1_time = time.time() - step1_start
        logger.info(f"Step 1 completed in {step1_time:.1f}s")

    # Step 2: Extract paper metadata
    if start_step <= 2 <= end_step:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: Extract Paper Metadata")
        logger.info("=" * 40)

        step2_start = time.time()

        # Use file IDs from step 1 if available
        file_ids = None
        if "step1" in results:
            file_ids = results["step1"].get("file_ids")
        elif start_step > 1:
            # Load from saved results
            step1_path = output_path / "step1_combined.json"
            if step1_path.exists():
                step1_data = load_json(step1_path)
                file_ids = step1_data.get("file_ids")

        results["step2"] = run_step2(
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            max_workers=max_workers,
            use_batch=use_batch,
            paper_ids=paper_ids,
            file_ids=file_ids,
        )
        step2_time = time.time() - step2_start
        logger.info(f"Step 2 completed in {step2_time:.1f}s")

    # Step 3: Select context for triplets
    if start_step <= 3 <= end_step:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: Select Context for Triplets")
        logger.info("=" * 40)

        step3_start = time.time()
        results["step3"] = run_step3(
            output_dir=output_dir,
            metadata_csv=metadata_csv,
            model=model,
            max_workers=max_workers,
            use_batch=use_batch,
            paper_ids=paper_ids,
        )
        step3_time = time.time() - step3_start
        logger.info(f"Step 3 completed in {step3_time:.1f}s")

    # Step 4: Critic evaluation
    if start_step <= 4 <= end_step:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 4: Critic Evaluation")
        logger.info("=" * 40)

        step4_start = time.time()
        results["step4"] = run_step4(
            output_dir=output_dir,
            metadata_csv=metadata_csv,
            model=model,
            max_workers=max_workers,
            use_batch=use_batch,
            paper_ids=paper_ids,
        )
        step4_time = time.time() - step4_start
        logger.info(f"Step 4 completed in {step4_time:.1f}s")

    # Step 5: Generate final results
    if start_step <= 5 <= end_step:
        logger.info("\n" + "=" * 40)
        logger.info("STEP 5: Generate Final Results")
        logger.info("=" * 40)

        step5_start = time.time()
        results["step5"] = run_step5(
            output_dir=output_dir,
            nber_metadata_dir=nber_metadata_dir,
            pdf_dir=pdf_dir,
        )
        step5_time = time.time() - step5_start
        logger.info(f"Step 5 completed in {step5_time:.1f}s")

    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    # Save pipeline run metadata
    pipeline_meta = {
        "start_step": start_step,
        "end_step": end_step,
        "model": model,
        "max_workers": max_workers,
        "use_batch": use_batch,
        "total_time_seconds": total_time,
        "paper_ids": paper_ids,
    }
    save_json(pipeline_meta, output_path / "pipeline_metadata.json")

    return results


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Causal Relation Extraction Pipeline"
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
        "--metadata-csv",
        type=str,
        default=DEFAULT_METADATA_CSV,
        help="Path to metadata CSV with abstracts"
    )
    parser.add_argument(
        "--nber-metadata-dir",
        type=str,
        default=DEFAULT_NBER_METADATA_DIR,
        help="Directory containing NBER metadata CSVs (for step 5)"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=DEFAULT_PDF_DIR,
        help="Directory containing journal subfolders with PDFs (for venue mapping)"
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
        "--start-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Step to start from (1-5)"
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Step to end at (1-5)"
    )
    parser.add_argument(
        "--consensus-mode",
        action="store_true",
        help="Enable consensus mode for step 1: run multiple times and filter by agreement"
    )
    parser.add_argument(
        "--consensus-runs",
        type=int,
        default=3,
        help="Number of API runs per paper in consensus mode (default: 5)"
    )
    parser.add_argument(
        "--consensus-threshold",
        type=int,
        default=2,
        help="Minimum runs a triplet must appear in to be kept (default: 4)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for matching triplets (default: 0.9)"
    )
    parser.add_argument(
        "--exclude-dir",
        type=str,
        default=None,
        help="Directory whose PDFs should be excluded (process only new PDFs not in this dir)"
    )

    args = parser.parse_args()

    if args.start_step > args.end_step:
        parser.error("start-step cannot be greater than end-step")

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        nber_metadata_dir=args.nber_metadata_dir,
        pdf_dir=args.pdf_dir,
        model=args.model,
        max_workers=args.max_workers,
        use_batch=args.use_batch,
        paper_ids=args.paper_ids,
        start_step=args.start_step,
        end_step=args.end_step,
        consensus_mode=args.consensus_mode,
        consensus_runs=args.consensus_runs,
        consensus_threshold=args.consensus_threshold,
        similarity_threshold=args.similarity_threshold,
        exclude_dir=args.exclude_dir,
    )


if __name__ == "__main__":
    main()
