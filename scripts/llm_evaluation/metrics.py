"""Metrics computation for evaluation results."""

import json
from pathlib import Path
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.utils import load_json, save_json


@dataclass
class TaskMetrics:
    """Metrics for a single task/model combination."""
    task: str
    model: str
    n_samples: int = 0
    n_correct: int = 0
    n_errors: int = 0
    accuracy: float = 0.0
    precision_by_class: dict = field(default_factory=dict)
    recall_by_class: dict = field(default_factory=dict)
    f1_by_class: dict = field(default_factory=dict)
    macro_f1: float = 0.0
    confusion_matrix: dict = field(default_factory=dict)
    avg_latency_ms: float = 0.0


class MetricsComputer:
    """Compute evaluation metrics from results."""

    def __init__(self, output_dir: str):
        """
        Initialize metrics computer.

        Args:
            output_dir: Directory containing evaluation results
        """
        self.output_dir = Path(output_dir)

    def compute_task_metrics(
        self,
        task: str,
        model: str,
        results: list[dict],
    ) -> TaskMetrics:
        """
        Compute metrics for a single task/model combination.

        Args:
            task: Task name
            model: Model name
            results: List of result dictionaries

        Returns:
            TaskMetrics object
        """
        if not results:
            return TaskMetrics(task=task, model=model)

        n_samples = len(results)
        n_correct = sum(1 for r in results if r.get("correct", False))
        n_errors = sum(1 for r in results if r.get("error"))

        # Accuracy
        accuracy = n_correct / n_samples if n_samples > 0 else 0.0

        # Get all classes
        all_expected = [r.get("expected") for r in results if r.get("expected") is not None]
        all_predicted = [r.get("predicted") for r in results if r.get("predicted") is not None]
        classes = sorted(set(all_expected) | set(all_predicted))

        # Confusion matrix
        confusion = {c1: {c2: 0 for c2 in classes} for c1 in classes}
        for r in results:
            expected = r.get("expected")
            predicted = r.get("predicted")
            if expected is not None and predicted is not None:
                if expected in confusion and predicted in confusion[expected]:
                    confusion[expected][predicted] += 1

        # Precision, Recall, F1 per class
        precision_by_class = {}
        recall_by_class = {}
        f1_by_class = {}

        for cls in classes:
            # True positives: predicted == expected == cls
            tp = sum(1 for r in results
                    if r.get("predicted") == cls and r.get("expected") == cls)

            # False positives: predicted == cls but expected != cls
            fp = sum(1 for r in results
                    if r.get("predicted") == cls and r.get("expected") != cls)

            # False negatives: expected == cls but predicted != cls
            fn = sum(1 for r in results
                    if r.get("expected") == cls and r.get("predicted") != cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precision_by_class[str(cls)] = round(precision, 4)
            recall_by_class[str(cls)] = round(recall, 4)
            f1_by_class[str(cls)] = round(f1, 4)

        # Macro F1
        macro_f1 = sum(f1_by_class.values()) / len(f1_by_class) if f1_by_class else 0.0

        # Average latency
        latencies = [r.get("latency_ms") for r in results if r.get("latency_ms")]
        avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

        return TaskMetrics(
            task=task,
            model=model,
            n_samples=n_samples,
            n_correct=n_correct,
            n_errors=n_errors,
            accuracy=round(accuracy, 4),
            precision_by_class=precision_by_class,
            recall_by_class=recall_by_class,
            f1_by_class=f1_by_class,
            macro_f1=round(macro_f1, 4),
            confusion_matrix={str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in confusion.items()},
            avg_latency_ms=round(avg_latency_ms, 2),
        )

    def _extract_results_from_data(self, data: dict) -> dict[str, list]:
        """Extract model -> results mapping from task result file data."""
        model_results = {}

        # Handle both old (results_by_model) and new (results_by_family) formats
        results_by_model = data.get("results_by_model", {})
        results_by_family = data.get("results_by_family", {})

        for model, results in results_by_model.items():
            model_results[model] = results

        for family, models_dict in results_by_family.items():
            if not isinstance(models_dict, dict) or "error" in models_dict:
                continue
            for model_name, results in models_dict.items():
                if isinstance(results, list):
                    model_results[model_name] = results

        return model_results

    def compute_all_metrics(self) -> dict:
        """
        Compute metrics for all tasks and models.

        For task3 and task4, also computes separate metrics for sign_differs cases.

        Returns:
            Dictionary with metrics for all task/model combinations
        """
        all_metrics = {}

        # Find all result files
        for task_file in self.output_dir.glob("task*_results.json"):
            task_name = task_file.stem.replace("_results", "")

            try:
                data = load_json(task_file)
                model_results = self._extract_results_from_data(data)

                task_metrics = {}
                for model_name, results in model_results.items():
                    # Overall metrics
                    metrics = self.compute_task_metrics(task_name, model_name, results)
                    task_metrics[model_name] = asdict(metrics)

                    # For task3/task4: compute sign_differs filtered metrics
                    if task_name in ("task3", "task4"):
                        sign_differs_results = [
                            r for r in results
                            if r.get("input_data", {}).get("sign_differs", False)
                        ]
                        if sign_differs_results:
                            sd_metrics = self.compute_task_metrics(
                                f"{task_name}_sign_differs", model_name, sign_differs_results,
                            )
                            task_metrics[model_name]["sign_differs_metrics"] = asdict(sd_metrics)

                all_metrics[task_name] = task_metrics

            except Exception as e:
                print(f"Error processing {task_file}: {e}")

        return all_metrics

    def generate_summary_report(self) -> dict:
        """
        Generate summary report across all tasks and models.

        Returns:
            Summary report dictionary
        """
        all_metrics = self.compute_all_metrics()

        if not all_metrics:
            return {"error": "No metrics computed"}

        # Collect all models
        all_models = set()
        for task_metrics in all_metrics.values():
            all_models.update(task_metrics.keys())

        # Compute rankings
        rankings_by_task = {}
        overall_scores = defaultdict(list)

        for task, task_metrics in all_metrics.items():
            # Rank by accuracy
            model_accuracies = [
                (model, metrics.get("accuracy", 0))
                for model, metrics in task_metrics.items()
            ]
            model_accuracies.sort(key=lambda x: x[1], reverse=True)
            rankings_by_task[task] = [m[0] for m in model_accuracies]

            # Collect for overall
            for model, acc in model_accuracies:
                overall_scores[model].append(acc)

        # Overall ranking by average accuracy
        overall_avg = {
            model: sum(scores) / len(scores)
            for model, scores in overall_scores.items()
        }
        overall_ranking = sorted(overall_avg.keys(), key=lambda x: overall_avg[x], reverse=True)

        # Collect sign_differs accuracy for task3/task4
        sign_differs_accuracy = {}
        for task_name in ("task3", "task4"):
            if task_name not in all_metrics:
                continue
            task_sd = {}
            for model, metrics in all_metrics[task_name].items():
                sd = metrics.get("sign_differs_metrics", {})
                if sd:
                    task_sd[model] = sd.get("accuracy", 0.0)
            if task_sd:
                sign_differs_accuracy[task_name] = {
                    model: acc
                    for model, acc in sorted(task_sd.items(), key=lambda x: x[1], reverse=True)
                }

        # Generate report
        report = {
            "summary": {
                "total_tasks": len(all_metrics),
                "models_evaluated": list(all_models),
            },
            "rankings": {
                "overall_by_avg_accuracy": overall_ranking,
                "by_task": rankings_by_task,
            },
            "overall_accuracy": {
                model: round(avg, 4)
                for model, avg in sorted(overall_avg.items(), key=lambda x: x[1], reverse=True)
            },
            "sign_differs_accuracy": sign_differs_accuracy,
            "detailed_metrics": all_metrics,
        }

        # Save report
        report_path = self.output_dir / "metrics_report.json"
        save_json(report, report_path)

        return report

    def print_summary(self) -> None:
        """Print summary to console."""
        report = self.generate_summary_report()

        if "error" in report:
            print(f"Error: {report['error']}")
            return

        print("\n" + "=" * 60)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 60)

        print(f"\nTasks evaluated: {report['summary']['total_tasks']}")
        print(f"Models evaluated: {', '.join(report['summary']['models_evaluated'])}")

        print("\n--- Overall Accuracy ---")
        for model, acc in report["overall_accuracy"].items():
            print(f"  {model}: {acc:.2%}")

        print("\n--- Rankings by Task ---")
        for task, ranking in report["rankings"]["by_task"].items():
            print(f"  {task}: {' > '.join(ranking)}")

        print("\n--- Overall Ranking ---")
        ranking = report["rankings"]["overall_by_avg_accuracy"]
        print(f"  {' > '.join(ranking)}")

        # Print sign_differs accuracy (top-level summary)
        sign_differs = report.get("sign_differs_accuracy", {})
        for task_name, task_sd in sign_differs.items():
            print(f"\n--- {task_name} sign_differs Accuracy ---")
            for model, acc in task_sd.items():
                print(f"  {model}: {acc:.2%}")

        print("\n" + "=" * 60)
