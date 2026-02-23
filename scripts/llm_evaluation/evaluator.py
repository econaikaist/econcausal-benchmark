"""Evaluation orchestrator for running LLM evaluations."""

import time
import logging
import threading
from pathlib import Path
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import (
    OpenAIClient,
    GeminiClient,
    GrokClient,
    OpenRouterClient,
    APIResponse,
    load_json,
    save_json,
    GEMINI_API_KEY_ECON,
    GEMINI_API_KEY_FINANCE,
    GEMINI_API_KEY_PLUS
)

from .config import EvaluationConfig, SUPPORTED_MODELS, RATE_LIMITED_MODELS, RATE_LIMITED_MAX_WORKERS
from .data_generator import TestCaseGenerator
from .tasks import (
    BaseTask,
    Task1Verification,
    Task2SignPrediction,
    Task3ContextTOFixed,
    Task4ContextFixed,
)
from .tasks.base import TaskResult


# Task registry
TASK_REGISTRY = {
    "task1": Task1Verification,
    "task2": Task2SignPrediction,
    "task3": Task3ContextTOFixed,
    "task4": Task4ContextFixed,
}


class EvaluationOrchestrator:
    """Orchestrates evaluation across multiple LLMs and tasks."""

    def __init__(self, config: EvaluationConfig):
        """
        Initialize orchestrator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Lock for thread-safe file operations
        self._file_lock = threading.Lock()

        # Initialize tasks
        self.tasks = {}
        for name, cls in TASK_REGISTRY.items():
            if name == "task2":
                self.tasks[name] = cls(
                    no_context=config.task2_no_context,
                    unknown_option=config.task2_unknown_option,
                )
            else:
                self.tasks[name] = cls()

        # Initialize data generator
        self.data_generator = TestCaseGenerator(
            data_path=config.data_path,
            seed=config.random_seed,
        )

        # Create checkpoint directory
        checkpoint_dir = Path(config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_client(self, family: str, model_name: str) -> Any:
        """Create a client for a specific model."""
        try:
            if family == "openai":
                return OpenAIClient(
                    model=model_name,
                    max_workers=self.config.max_workers,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
            elif family == "gemini":
                # Use different API keys for econ vs finance
                # gemini_api_key = GEMINI_API_KEY_FINANCE if self.config.journal_type == "finance" else GEMINI_API_KEY_ECON
                gemini_api_key =GEMINI_API_KEY_PLUS
                return GeminiClient(
                    api_key=gemini_api_key,
                    model=model_name,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
            elif family == "grok":
                return GrokClient(
                    model=model_name,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
            elif family in ["qwen", "llama", "deepseek"]:
                supports_logprobs = SUPPORTED_MODELS[family].get("supports_logprobs", False)
                return OpenRouterClient(
                    model=model_name,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                    return_logprobs=supports_logprobs,
                )
            else:
                self.logger.error(f"Unknown family: {family}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to create client for {family}/{model_name}: {e}")
            return None

    def _get_checkpoint_path(self, task_type: str, family: str, model_name: str) -> Path:
        """Get checkpoint path for a specific task/family/model combination."""
        # Sanitize model name for filename (replace / with _)
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        return Path(self.config.output_dir) / "checkpoints" / f"{task_type}_{family}_{safe_model_name}_checkpoint.json"

    def _get_results_path(self, task_type: str, family: str, model_name: str) -> Path:
        """Get results path for a specific task/family/model combination."""
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        return Path(self.config.output_dir) / "results" / f"{task_type}_{family}_{safe_model_name}_results.json"

    def _load_checkpoint(self, checkpoint_path: Path) -> tuple[set, list]:
        """Load checkpoint data."""
        if checkpoint_path.exists():
            try:
                with self._file_lock:
                    data = load_json(checkpoint_path)
                completed_ids = set(data.get("completed_ids", []))
                results = data.get("results", [])
                return completed_ids, results
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        return set(), []

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        completed_ids: set,
        results: list,
        model_name: str = None,
    ) -> None:
        """Save checkpoint data (thread-safe)."""
        try:
            with self._file_lock:
                save_json({
                    "model": model_name,
                    "completed_ids": list(completed_ids),
                    "results": results,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, checkpoint_path)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def _save_model_results(
        self,
        results_path: Path,
        task_type: str,
        family: str,
        model_name: str,
        results: list,
        n_test_cases: int,
    ) -> None:
        """Save results for a specific model (thread-safe)."""
        results_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._file_lock:
                save_json({
                    "task": task_type,
                    "family": family,
                    "model": model_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "n_test_cases": n_test_cases,
                    "n_completed": len(results),
                    "results": results,
                }, results_path)
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def _call_api(
        self,
        client: Any,
        prompt: str,
        schema: dict,
        case_id: str,
    ) -> tuple[APIResponse, float]:
        """Call API and measure latency."""
        start_time = time.time()
        response = client.call_api(
            user_prompt=prompt,
            response_schema=schema,
            paper_id=case_id,
        )
        latency_ms = (time.time() - start_time) * 1000
        return response, latency_ms

    def run_single_model(
        self,
        task_type: str,
        family: str,
        model_name: str,
        test_cases: list,
    ) -> list:
        """
        Run a single task for a single model.

        Args:
            task_type: Task type (task1, task2, etc.)
            family: Model family (openai, gemini, etc.)
            model_name: Specific model name (gpt-4o-mini, gemini-2.0-flash, etc.)
            test_cases: List of test cases

        Returns:
            List of result dictionaries
        """
        task = self.tasks.get(task_type)
        if not task:
            self.logger.error(f"Unknown task type: {task_type}")
            return []

        # Create client for this specific model
        client = self._create_client(family, model_name)
        if not client:
            self.logger.error(f"Failed to create client for {family}/{model_name}")
            return []

        self.logger.info(f"Starting {task_type}/{family}/{model_name}")

        # Load checkpoint (always try to load existing results for smart resume)
        checkpoint_path = self._get_checkpoint_path(task_type, family, model_name)
        completed_ids, results = self._load_checkpoint(checkpoint_path)

        if completed_ids:
            self.logger.info(f"Found existing results for {task_type}/{family}/{model_name}: {len(completed_ids)} completed")

        # Auto-retry errored/empty cases - remove them from completed_ids
        # But skip parse_failed cases (raw response saved, no point retrying)
        if results:
            def _should_retry(r):
                od = r.get("output_data")
                if isinstance(od, dict) and od.get("parse_failed"):
                    return False
                return r.get("error") or not r.get("predicted")
            error_ids = {r["case_id"] for r in results if _should_retry(r)}
            if error_ids:
                completed_ids -= error_ids
                # Remove errored/empty results so they can be replaced
                results = [r for r in results if not _should_retry(r)]
                self.logger.info(f"Will retry {len(error_ids)} errored/empty cases for {task_type}/{family}/{model_name}")

        # Filter remaining cases
        remaining_cases = [c for c in test_cases if c.case_id not in completed_ids]

        if not remaining_cases:
            self.logger.info(f"All cases completed for {task_type}/{family}/{model_name}")
            return results

        self.logger.info(f"Processing {len(remaining_cases)} cases for {task_type}/{family}/{model_name}")

        # Process cases in parallel
        completed_count = 0

        # Determine max_workers (rate-limited models have a cap)
        max_workers = self.config.max_workers
        if model_name in RATE_LIMITED_MODELS:
            max_workers = min(max_workers, RATE_LIMITED_MAX_WORKERS)
            self.logger.info(f"Rate-limited model {model_name}: using max_workers={max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for case in remaining_cases:
                prompt = task.format_prompt(case)
                future = executor.submit(
                    self._call_api,
                    client,
                    prompt,
                    task.schema,
                    case.case_id,
                )
                futures[future] = case

            with tqdm(total=len(futures), desc=f"{task_type}/{family}/{model_name}") as pbar:
                for future in as_completed(futures):
                    case = futures[future]

                    try:
                        response, latency_ms = future.result()
                        result = task.evaluate_single(case, response, latency_ms)
                    except Exception as e:
                        self.logger.error(f"Error processing {case.case_id}: {e}")
                        result = TaskResult(
                            case_id=case.case_id,
                            input_data=task._case_to_dict(case),
                            output_data=None,
                            expected=task.get_expected(case),
                            predicted=None,
                            correct=False,
                            error=str(e),
                        )

                    results.append(asdict(result))
                    completed_ids.add(case.case_id)
                    completed_count += 1
                    pbar.update(1)

                    # Save checkpoint periodically
                    if completed_count % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(checkpoint_path, completed_ids, results, model_name)

        # Final checkpoint save
        self._save_checkpoint(checkpoint_path, completed_ids, results, model_name)

        # Save model-specific results
        results_path = self._get_results_path(task_type, family, model_name)
        self._save_model_results(results_path, task_type, family, model_name, results, len(test_cases))

        self.logger.info(f"Completed {task_type}/{family}/{model_name}: {len(results)} results")

        return results

    def run_family_sequential(
        self,
        task_type: str,
        family: str,
        test_cases: list,
    ) -> dict:
        """
        Run all models in a family sequentially.

        Args:
            task_type: Task type
            family: Model family
            test_cases: List of test cases

        Returns:
            Dictionary mapping model_name -> results
        """
        family_info = SUPPORTED_MODELS.get(family)
        if not family_info:
            self.logger.error(f"Unknown family: {family}")
            return {}

        models = family_info["models"]
        self.logger.info(f"Running {family} family: {models}")

        family_results = {}
        for model_name in models:
            results = self.run_single_model(task_type, family, model_name, test_cases)
            family_results[model_name] = results

        return family_results

    def run_task(self, task_type: str) -> dict:
        """
        Run a task across all configured models.
        Families run in parallel, models within each family run sequentially.

        Args:
            task_type: Task type to run

        Returns:
            Dictionary with results for all models
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running {task_type}")
        self.logger.info(f"{'='*60}")

        # Generate test cases
        if task_type == "task1":
            test_cases = self.data_generator.generate_task1_cases(self.config.max_samples_per_task)
        elif task_type == "task2":
            test_cases = self.data_generator.generate_task2_cases(self.config.max_samples_per_task)
        elif task_type == "task3":
            test_cases = self.data_generator.generate_task3_cases(
                self.config.max_samples_per_task,
                num_examples=self.config.task3_num_examples,
            )
        elif task_type == "task4":
            test_cases = self.data_generator.generate_task4_cases(
                self.config.max_samples_per_task,
                num_examples=self.config.task4_num_examples,
            )
        else:
            self.logger.error(f"Unknown task type: {task_type}")
            return {"error": f"Unknown task type: {task_type}"}

        self.logger.info(f"Generated {len(test_cases)} test cases")

        if not test_cases:
            self.logger.warning(f"No test cases generated for {task_type}")
            return {"error": "No test cases generated"}

        # Run all families in parallel, each family processes its models sequentially
        results_by_family = {}

        with ThreadPoolExecutor(max_workers=len(self.config.models)) as executor:
            futures = {}
            for family in self.config.models:
                if family not in SUPPORTED_MODELS:
                    self.logger.warning(f"Unknown family: {family}")
                    continue

                future = executor.submit(
                    self.run_family_sequential,
                    task_type,
                    family,
                    test_cases,
                )
                futures[future] = family

            for future in as_completed(futures):
                family = futures[future]
                try:
                    family_results = future.result()
                    results_by_family[family] = family_results
                except Exception as e:
                    self.logger.error(f"Error running family {family}: {e}")
                    results_by_family[family] = {"error": str(e)}

        # Save consolidated task results
        results_path = self.config.get_results_path(task_type)
        with self._file_lock:
            save_json({
                "task": task_type,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "models": self.config.models,
                    "max_samples": self.config.max_samples_per_task,
                    "seed": self.config.random_seed,
                },
                "n_test_cases": len(test_cases),
                "results_by_family": results_by_family,
            }, results_path)

        self.logger.info(f"Results saved to {results_path}")

        return results_by_family

    def run_all_tasks(self) -> dict:
        """
        Run all configured tasks.

        Returns:
            Dictionary with results for all tasks and models
        """
        self.logger.info("Starting evaluation pipeline")
        self.logger.info(f"Tasks: {self.config.task_types}")
        self.logger.info(f"Families: {self.config.models}")

        # Log all models to be run
        for family in self.config.models:
            if family in SUPPORTED_MODELS:
                models = SUPPORTED_MODELS[family]["models"]
                self.logger.info(f"  {family}: {models}")

        # Get data statistics
        stats = self.data_generator.get_statistics()
        self.logger.info(f"Data statistics: {stats}")

        all_results = {}

        for task_type in self.config.task_types:
            task_results = self.run_task(task_type)
            all_results[task_type] = task_results

        # Collect actual models used
        models_used = {}
        for family in self.config.models:
            if family in SUPPORTED_MODELS:
                models_used[family] = SUPPORTED_MODELS[family]["models"]

        # Compute model statistics from results
        model_statistics = {}
        for task_type, results_by_family in all_results.items():
            task_stats = {}
            for family, models_dict in results_by_family.items():
                if isinstance(models_dict, dict) and "error" not in models_dict:
                    for model_name, results in models_dict.items():
                        if isinstance(results, list):
                            n_total = len(results)
                            n_correct = sum(1 for r in results if r.get("correct", False))
                            n_errors = sum(1 for r in results if r.get("error"))
                            n_empty = sum(1 for r in results if not r.get("predicted") and not r.get("error"))
                            accuracy = n_correct / n_total if n_total > 0 else 0.0
                            task_stats[model_name] = {
                                "n_total": n_total,
                                "n_correct": n_correct,
                                "n_errors": n_errors,
                                "n_empty": n_empty,
                                "accuracy": round(accuracy, 4),
                            }
            model_statistics[task_type] = task_stats

        # Save summary
        summary_path = Path(self.config.output_dir) / "evaluation_summary.json"
        with self._file_lock:
            save_json({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else vars(self.config),
                "models_used": models_used,
                "data_statistics": stats,
                "model_statistics": model_statistics,
                "tasks_completed": list(all_results.keys()),
            }, summary_path)

        self.logger.info(f"\nEvaluation complete. Summary saved to {summary_path}")

        return all_results
