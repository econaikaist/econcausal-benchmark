#!/usr/bin/env python3
"""CLI entry point for LLM causal reasoning evaluation."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import EvaluationConfig, TASK_TYPES, SUPPORTED_MODELS
from evaluation.evaluator import EvaluationOrchestrator
from evaluation.metrics import MetricsComputer


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Causal Reasoning Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tasks with default models (auto-resumes, retries errors)
  python run_evaluation.py

  # Run specific tasks
  python run_evaluation.py --tasks task1 task2

  # Run with specific models
  python run_evaluation.py --models openai gemini

  # Run with limited samples for testing
  python run_evaluation.py --max-samples 10

  # Run task3 with 3 examples instead of 1
  python run_evaluation.py --tasks task3 --task3-examples 3

  # Only compute metrics (no new evaluation)
  python run_evaluation.py --metrics-only

Note: Auto-resume is enabled by default. Completed results are preserved,
      and errored/empty cases are automatically retried.
        """
    )

    # Task configuration
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_TYPES + ["all"],
        default=["all"],
        help="Tasks to run (default: all)"
    )

    # Model configuration
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(SUPPORTED_MODELS.keys()) + ["all"],
        default=["openai", "gemini", "grok", "qwen", 'llama'],
        help="Models to evaluate (default: openai gemini grok qwen)"
    )

    # Data configuration
    parser.add_argument(
        "--journal-type",
        type=str,
        choices=["econ", "finance", "all"],
        default="econ",
        help="Journal type: econ (5 econ journals), finance (3 finance journals), or all"
    )
    parser.add_argument(
        "--step2-sample-size",
        type=int,
        choices=[20, 30, 40, 50, 60, 70, 80, 90, 100, 0],
        default=30,
        help="For step2: samples per year (20-100). 0 means use full data."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override: direct path to evaluation JSON (ignores journal-type/sample-size)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for results"
    )

    # Sampling configuration
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per task (for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    # Task-specific configuration
    parser.add_argument(
        "--task2-examples",
        type=int,
        default=1,
        help="Number of examples for task 2 (default: 1)"
    )
    parser.add_argument(
        "--task3-examples",
        type=int,
        default=1,
        help="Number of examples for task 3 (default: 1)"
    )
    parser.add_argument(
        "--task1-no-context",
        action="store_true",
        help="Exclude context from task1 prompts (for ablation study)"
    )
    parser.add_argument(
        "--task1-unknown-option",
        action="store_true",
        help="Add 'unknown' to task1 answer choices (for ablation study)"
    )

    # Processing configuration
    parser.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="Maximum parallel workers"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N samples"
    )

    # Metrics only mode
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only compute metrics from existing results"
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    # Handle "all" options
    task_types = TASK_TYPES if "all" in args.tasks else args.tasks
    models = list(SUPPORTED_MODELS.keys()) if "all" in args.models else args.models

    # Determine data path
    base_dir = Path("")

    if args.data_path:
        # Direct path override
        data_path = args.data_path
    elif args.step2_sample_size > 0:
        # Step2 sampled data
        data_path = str(base_dir / "step2" / f"{args.journal_type}_step2_n{args.step2_sample_size}.json")
    elif args.journal_type == "all":
        # All journals combined
        data_path = str(base_dir / "evaluation_data.json")
    else:
        # Econ or Finance full data
        data_path = str(base_dir / f"{args.journal_type}_journals.json")

    logger.info("=" * 60)
    logger.info("LLM Causal Reasoning Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Tasks: {task_types}")
    logger.info(f"Models: {models}")
    logger.info(f"Journal type: {args.journal_type}")
    logger.info(f"Gemini API: {'FINANCE key' if args.journal_type == 'finance' else 'ECON key'}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Task2 examples: {args.task2_examples}")
    logger.info(f"Task3 examples: {args.task3_examples}")
    logger.info(f"Task1 no context: {args.task1_no_context}")
    logger.info(f"Task1 unknown option: {args.task1_unknown_option}")
    logger.info(f"Output directory: {args.output_dir}")

    # Metrics-only mode
    if args.metrics_only:
        logger.info("\nComputing metrics from existing results...")
        metrics_computer = MetricsComputer(args.output_dir)
        metrics_computer.print_summary()
        return

    # Create configuration
    config = EvaluationConfig(
        task_types=task_types,
        models=models,
        data_path=data_path,
        output_dir=args.output_dir,
        journal_type=args.journal_type,
        max_samples_per_task=args.max_samples,
        random_seed=args.seed,
        task2_num_examples=args.task2_examples,
        task3_num_examples=args.task3_examples,
        task1_no_context=args.task1_no_context,
        task1_unknown_option=args.task1_unknown_option,
        max_workers=args.max_workers,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Run evaluation
    orchestrator = EvaluationOrchestrator(config)
    orchestrator.run_all_tasks()

    # Compute and print metrics
    logger.info("\nComputing metrics...")
    metrics_computer = MetricsComputer(args.output_dir)
    metrics_computer.print_summary()

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
