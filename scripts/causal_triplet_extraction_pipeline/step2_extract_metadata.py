"""
Step 2: Extract paper metadata, context, and identification methods.

This step extracts paper-level information including:
- Paper type (empirical/theoretical)
- Global context (when, where, who, background)
- Identification methods used
"""

import argparse
import logging
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
from common.prompts import STEP2_PROMPT
from common.schemas import STEP2_SCHEMA


# Default paths
DEFAULT_INPUT_DIR = "/home/donggyu/econ_causality/new_data/nber_paper_wo_aer_pp_30pages"
DEFAULT_OUTPUT_DIR = "/home/donggyu/econ_causality/new_data/real_data"


def run_step2(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model: str = DEFAULT_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
    use_batch: bool = False,
    paper_ids: Optional[list[str]] = None,
    file_ids: Optional[dict[str, str]] = None,
    step1_results_path: Optional[str] = None,
) -> dict:
    """
    Run Step 2: Extract paper metadata and context.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save output JSON files
        model: OpenAI model to use
        max_workers: Maximum parallel workers
        use_batch: Whether to use batch API instead of realtime
        paper_ids: Optional list of specific paper IDs to process
        file_ids: Optional pre-uploaded file IDs from Step 1
        step1_results_path: Path to step1_combined.json (alternative to file_ids)

    Returns:
        Dictionary with results:
        {
            "file_ids": {paper_id: file_id, ...},
            "results": {paper_id: metadata, ...},
            "errors": {paper_id: error_message, ...}
        }
    """
    logger = setup_logging()
    logger.info("Starting Step 2: Extract paper metadata")

    output_path = Path(output_dir)

    # Initialize client
    client = OpenAIClient(
        model=model,
        max_workers=max_workers,
        use_batch=use_batch,
    )

    # Get file_ids: priority is file_ids param > step1_results_path > default path
    valid_uploads = None

    if file_ids:
        valid_uploads = file_ids
        logger.info(f"Using {len(valid_uploads)} pre-uploaded files")
    else:
        # Try to load from step1 results
        step1_path = step1_results_path or (output_path / "step1_combined.json")
        if Path(step1_path).exists():
            step1_data = load_json(step1_path)
            loaded_file_ids = step1_data.get("file_ids", {})
            if loaded_file_ids:
                valid_uploads = loaded_file_ids
                logger.info(f"Loaded {len(valid_uploads)} file IDs from {step1_path}")

    if not valid_uploads:
        # Upload PDFs
        input_path = Path(input_dir)
        pdf_files = get_pdf_files(input_path)

        if paper_ids:
            pdf_files = [p for p in pdf_files if p.stem in paper_ids]

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        if not pdf_files:
            logger.warning("No PDF files found")
            return {"file_ids": {}, "results": {}, "errors": {}}

        logger.info("Uploading PDFs to OpenAI...")
        uploaded = client.upload_pdfs_parallel(pdf_files, desc="Step 2: Uploading PDFs")
        valid_uploads = {pid: fid for pid, fid in uploaded.items() if fid is not None}

    # Prepare API tasks
    tasks = []
    for paper_id, file_id in valid_uploads.items():
        tasks.append({
            "file_id": file_id,
            "system_prompt": "",
            "user_prompt": STEP2_PROMPT,
            "response_schema": STEP2_SCHEMA,
            "paper_id": paper_id,
        })

    # Process tasks with checkpoint
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "step2_checkpoint.json"

    logger.info(f"Processing {len(tasks)} papers...")
    responses = client.process_tasks(
        tasks,
        use_pdf=True,
        desc="Step 2: Extracting metadata",
        checkpoint_path=checkpoint_path,
        checkpoint_interval=10,
    )

    # Collect results
    results = {}
    errors = {}

    for response in responses:
        paper_id = response.paper_id
        if response.success:
            results[paper_id] = response.data
        else:
            errors[paper_id] = response.error
            logger.error(f"Paper {paper_id}: {response.error}")

    # Save individual results
    for paper_id, metadata in results.items():
        save_json(
            {"paper_id": paper_id, **metadata},
            output_path / f"step2_{paper_id}.json"
        )

    # Save combined results
    combined_output = {
        "file_ids": valid_uploads,
        "results": results,
        "errors": errors,
    }
    save_json(combined_output, output_path / "step2_combined.json")

    # Save API errors to separate JSON file
    client.save_errors_json(output_path / "step2_api_errors.json")

    logger.info(f"Step 2 complete. Results saved to {output_path}")
    logger.info(f"Successful: {len(results)}, Errors: {len(errors)}")

    return combined_output


def main():
    """Command-line interface for Step 2."""
    parser = argparse.ArgumentParser(
        description="Step 2: Extract paper metadata and context"
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
        "--step1-results",
        type=str,
        help="Path to step1_combined.json to reuse file IDs"
    )

    args = parser.parse_args()

    run_step2(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        max_workers=args.max_workers,
        use_batch=args.use_batch,
        paper_ids=args.paper_ids,
        step1_results_path=args.step1_results,
    )


if __name__ == "__main__":
    main()
