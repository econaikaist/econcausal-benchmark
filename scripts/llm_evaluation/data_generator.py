"""Test case generator for evaluation tasks."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Task1Case:
    """Test case for Task 1: Sign Prediction."""
    case_id: str
    context: str
    treatment: str
    outcome: str
    expected_sign: str  # "+", "-", "None", "mixed"
    paper_id: str


@dataclass
class Task2Case:
    """Test case for Task 2: Context-Dependent Sign Prediction (similar T/O, different contexts).

    Uses embedding cosine similarity (avg of treatment + outcome >= 0.8)
    to find similar T/O pairs from different papers.
    """
    case_id: str
    treatment: str              # target treatment
    outcome: str                # target outcome
    examples: list[dict]        # List of {treatment, outcome, context, sign}
    test_context: str           # target context
    expected_sign: str          # true sign of target
    paper_ids: list[str]
    avg_similarity: float = 0.0
    sign_differs: bool = False  # True if any example sign != target expected_sign


@dataclass
class Task3Case:
    """Test case for Task 3: Misinformation-Robust Sign Prediction (similar T/O, different contexts, reverted signs).

    Same pairing as Task 2, but example signs are intentionally reverted
    to test LLM robustness against misleading information.
    """
    case_id: str
    treatment: str              # target treatment
    outcome: str                # target outcome
    examples: list[dict]        # List of {treatment, outcome, context, sign, original_sign}
    test_context: str           # target context
    expected_sign: str          # true sign of target
    paper_ids: list[str]
    avg_similarity: float = 0.0
    sign_differs: bool = False  # True if any example (reverted) sign != target expected_sign


class TestCaseGenerator:
    """Generate test cases from evaluation_data.json."""

    def __init__(
        self,
        data_path: str,
        step2_path: str = None,  # Deprecated, kept for backward compatibility
        seed: int = 42,
    ):
        """
        Initialize test case generator.

        Args:
            data_path: Path to evaluation_data.json (single combined JSON)
            step2_path: Deprecated, ignored
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.data = load_json(data_path)
        self.seed = seed
        random.seed(seed)

        # Extract results - supports both old and new format
        self.triplets = self.data.get("results", {})

    def _get_context(self, paper_id: str, triplet: dict) -> Optional[str]:
        """Get context for a triplet."""
        # New format: context directly in triplet
        if "context" in triplet and triplet["context"]:
            return triplet["context"]

        # Old format: context in selection.context_selected
        selection = triplet.get("selection", {})
        context_selected = selection.get("context_selected", [])
        if context_selected and len(context_selected) > 0:
            return context_selected[0]

        return None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()

    def generate_task1_cases(self, max_samples: Optional[int] = None) -> list[Task1Case]:
        """
        Generate sign prediction task cases.

        Args:
            max_samples: Maximum number of cases to generate

        Returns:
            List of Task1Case objects
        """
        cases = []
        case_id = 0

        for paper_id, triplet_list in self.triplets.items():
            for triplet in triplet_list:
                context = self._get_context(paper_id, triplet)
                if not context:
                    continue

                treatment = triplet.get("treatment", "")
                outcome = triplet.get("outcome", "")
                sign = triplet.get("sign", "")

                if not treatment or not outcome or not sign:
                    continue

                cases.append(Task1Case(
                    case_id=f"t1_{case_id}",
                    context=context,
                    treatment=treatment,
                    outcome=outcome,
                    expected_sign=sign,
                    paper_id=paper_id,
                ))
                case_id += 1

        # Shuffle and limit
        random.shuffle(cases)
        if max_samples:
            cases = cases[:max_samples]

        return cases

    TASK_INPUT_DIR = Path("")

    def _load_task_input(self, task_name: str) -> Optional[dict]:
        """Load pre-computed task input JSON (task2_input.json or task3_input.json).

        Always looks in TASK_INPUT_DIR (real_data_1991/) regardless of data_path.
        """
        input_path = self.TASK_INPUT_DIR / f"{task_name}_input.json"
        if input_path.exists():
            return load_json(str(input_path))
        return None

    def generate_task2_cases(
        self,
        max_samples: Optional[int] = None,
        num_examples: int = 1,
    ) -> list[Task2Case]:
        """
        Generate context-dependent sign prediction cases.

        Uses pre-computed similarity data (task3_input.json) to find similar
        T/O pairs (avg cosine similarity >= 0.8) from different papers.
        Examples have correct signs.

        Args:
            max_samples: Maximum number of cases to generate
            num_examples: Number of examples to include (default: 1)

        Returns:
            List of Task2Case objects
        """
        task2_data = self._load_task_input("task3")
        if task2_data is None:
            raise FileNotFoundError(
                f"task3_input.json not found in {self.TASK_INPUT_DIR}. "
                "Run the generation script first."
            )

        cases = []
        case_id = 0

        for case_data in task2_data["cases"]:
            target = case_data["target"]
            similar_examples = case_data["similar_examples"]

            # Need enough similar examples
            if len(similar_examples) < num_examples:
                continue

            # Sample num_examples from available similar examples
            selected = similar_examples[:num_examples]

            avg_sim = sum(ex["avg_similarity"] for ex in selected) / len(selected)

            # Check if any example sign differs from target sign
            target_sign = target["sign"]
            sign_differs = any(ex["sign"] != target_sign for ex in selected)

            cases.append(Task2Case(
                case_id=f"t2_{case_id}",
                treatment=target["treatment"],
                outcome=target["outcome"],
                examples=[
                    {
                        "treatment": ex["treatment"],
                        "outcome": ex["outcome"],
                        "context": ex["context"],
                        "sign": ex["sign"],
                    }
                    for ex in selected
                ],
                test_context=target["context"],
                expected_sign=target_sign,
                paper_ids=[ex["paper_id"] for ex in selected] + [target["paper_id"]],
                avg_similarity=round(avg_sim, 4),
                sign_differs=sign_differs,
            ))
            case_id += 1

        # Shuffle and limit
        random.shuffle(cases)
        if max_samples:
            cases = cases[:max_samples]

        return cases

    def generate_task3_cases(
        self,
        max_samples: Optional[int] = None,
        num_examples: int = 1,
    ) -> list[Task3Case]:
        """
        Generate misinformation-robust sign prediction cases.

        Uses pre-computed similarity data (task4_input.json) to find similar
        T/O pairs (avg cosine similarity >= 0.8) from different papers.
        Example signs are intentionally REVERTED to test robustness.

        Args:
            max_samples: Maximum number of cases to generate
            num_examples: Number of examples to include (default: 1)

        Returns:
            List of Task3Case objects
        """
        task3_data = self._load_task_input("task4")
        if task3_data is None:
            raise FileNotFoundError(
                f"task4_input.json not found in {self.TASK_INPUT_DIR}. "
                "Run the generation script first."
            )

        cases = []
        case_id = 0

        for case_data in task3_data["cases"]:
            target = case_data["target"]
            similar_examples = case_data["similar_examples"]

            # Need enough similar examples
            if len(similar_examples) < num_examples:
                continue

            # Sample num_examples from available similar examples
            selected = similar_examples[:num_examples]

            avg_sim = sum(ex["avg_similarity"] for ex in selected) / len(selected)

            # Check if any example's reverted sign differs from target sign
            target_sign = target["sign"]
            sign_differs = any(ex["sign"] != target_sign for ex in selected)

            cases.append(Task3Case(
                case_id=f"t3_{case_id}",
                treatment=target["treatment"],
                outcome=target["outcome"],
                examples=[
                    {
                        "treatment": ex["treatment"],
                        "outcome": ex["outcome"],
                        "context": ex["context"],
                        "sign": ex["sign"],  # reverted sign
                        "original_sign": ex.get("original_sign", ex["sign"]),
                    }
                    for ex in selected
                ],
                test_context=target["context"],
                expected_sign=target_sign,  # true sign
                paper_ids=[ex["paper_id"] for ex in selected] + [target["paper_id"]],
                avg_similarity=round(avg_sim, 4),
                sign_differs=sign_differs,
            ))
            case_id += 1

        # Shuffle and limit
        random.shuffle(cases)
        if max_samples:
            cases = cases[:max_samples]

        return cases

    def generate_all_cases(
        self,
        max_samples_per_task: Optional[int] = None,
        task2_num_examples: int = 1,
        task3_num_examples: int = 1,
    ) -> dict[str, list]:
        """
        Generate test cases for all tasks.

        Args:
            max_samples_per_task: Maximum samples per task
            task2_num_examples: Number of examples for task 2 (default: 1)
            task3_num_examples: Number of examples for task 3 (1 or 3, default: 1)

        Returns:
            Dictionary mapping task name to list of cases
        """
        return {
            "task1": self.generate_task1_cases(max_samples_per_task),
            "task2": self.generate_task2_cases(max_samples_per_task, num_examples=task2_num_examples),
            "task3": self.generate_task3_cases(max_samples_per_task, num_examples=task3_num_examples),
        }

    def get_statistics(self) -> dict:
        """Get statistics about available data."""
        total_triplets = sum(len(v) for v in self.triplets.values())
        total_papers = len(self.triplets)

        # Count triplets with context
        triplets_with_context = 0
        for paper_id, triplet_list in self.triplets.items():
            for triplet in triplet_list:
                if self._get_context(paper_id, triplet):
                    triplets_with_context += 1

        # Sign distribution
        sign_counts = defaultdict(int)
        for triplet_list in self.triplets.values():
            for triplet in triplet_list:
                sign = triplet.get("sign", "unknown")
                sign_counts[sign] += 1

        return {
            "total_papers": total_papers,
            "total_triplets": total_triplets,
            "triplets_with_context": triplets_with_context,
            "sign_distribution": dict(sign_counts),
        }
