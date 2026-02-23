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
    """Test case for Task 1: Causality Verification."""
    case_id: str
    context: str
    treatment: str
    outcome: str
    sign: str
    expected_answer: str  # "Yes" or "No"
    paper_id: str
    is_flipped: bool = False  # Whether this is a negative case with flipped sign


@dataclass
class Task2Case:
    """Test case for Task 2: Sign Prediction."""
    case_id: str
    context: str
    treatment: str
    outcome: str
    expected_sign: str  # "+", "-", "None", "mixed"
    paper_id: str


@dataclass
class Task3Case:
    """Test case for Task 3: Context-Aware Reasoning (similar T/O, different contexts).

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
class Task4Case:
    """Test case for Task 4: Noise Robustness (similar T/O, different contexts, reverted signs).

    Same pairing as Task 3, but example signs are intentionally reverted
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

    def _flip_sign(self, sign: str) -> Optional[str]:
        """Flip a sign for negative test cases."""
        flip_map = {
            "+": "-",
            "-": "+",
            "None": "+",  # None becomes positive
            "mixed": None,  # Can't flip mixed
        }
        return flip_map.get(sign)

    def _get_sign_description(self, sign: str) -> str:
        """Get human-readable description of sign."""
        descriptions = {
            "+": "increase",
            "-": "decrease",
            "None": "have no significant effect on",
            "mixed": "have a mixed effect on",
        }
        return descriptions.get(sign, "affect")

    def generate_task1_cases(self, max_samples: Optional[int] = None) -> list[Task1Case]:
        """
        Generate verification task cases.

        For each triplet:
        - Create a positive case (expected: Yes)
        - Create a negative case with flipped sign (expected: No)

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

                # Positive case (true claim)
                cases.append(Task1Case(
                    case_id=f"t1_{case_id}",
                    context=context,
                    treatment=treatment,
                    outcome=outcome,
                    sign=sign,
                    expected_answer="Yes",
                    paper_id=paper_id,
                    is_flipped=False,
                ))
                case_id += 1

                # Negative case (flipped sign)
                flipped_sign = self._flip_sign(sign)
                if flipped_sign:
                    cases.append(Task1Case(
                        case_id=f"t1_{case_id}",
                        context=context,
                        treatment=treatment,
                        outcome=outcome,
                        sign=flipped_sign,
                        expected_answer="No",
                        paper_id=paper_id,
                        is_flipped=True,
                    ))
                    case_id += 1

        # Shuffle and limit
        random.shuffle(cases)
        if max_samples:
            cases = cases[:max_samples]

        return cases

    def generate_task2_cases(self, max_samples: Optional[int] = None) -> list[Task2Case]:
        """
        Generate sign prediction task cases.

        Args:
            max_samples: Maximum number of cases to generate

        Returns:
            List of Task2Case objects
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

                cases.append(Task2Case(
                    case_id=f"t2_{case_id}",
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

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()

    TASK_INPUT_DIR = Path("/home/donggyu/econ_causality/new_data/real_data_1991")

    def _load_task_input(self, task_name: str) -> Optional[dict]:
        """Load pre-computed task input JSON (task3_input.json or task4_input.json).

        Always looks in TASK_INPUT_DIR (real_data_1991/) regardless of data_path.
        """
        input_path = self.TASK_INPUT_DIR / f"{task_name}_input.json"
        if input_path.exists():
            return load_json(str(input_path))
        return None

    def generate_task3_cases(
        self,
        max_samples: Optional[int] = None,
        num_examples: int = 1,
    ) -> list[Task3Case]:
        """
        Generate context-aware reasoning cases.

        Uses pre-computed similarity data (task3_input.json) to find similar
        T/O pairs (avg cosine similarity >= 0.8) from different papers.
        Examples have correct signs.

        Args:
            max_samples: Maximum number of cases to generate
            num_examples: Number of examples to include (default: 1)

        Returns:
            List of Task3Case objects
        """
        task3_data = self._load_task_input("task3")
        if task3_data is None:
            raise FileNotFoundError(
                f"task3_input.json not found in {Path(self.data_path).parent}. "
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

            # Check if any example sign differs from target sign
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

    def generate_task4_cases(
        self,
        max_samples: Optional[int] = None,
        num_examples: int = 1,
    ) -> list[Task4Case]:
        """
        Generate noise robustness cases.

        Uses pre-computed similarity data (task4_input.json) to find similar
        T/O pairs (avg cosine similarity >= 0.8) from different papers.
        Example signs are intentionally REVERTED to test robustness.

        Args:
            max_samples: Maximum number of cases to generate
            num_examples: Number of examples to include (default: 1)

        Returns:
            List of Task4Case objects
        """
        task4_data = self._load_task_input("task4")
        if task4_data is None:
            raise FileNotFoundError(
                f"task4_input.json not found in {Path(self.data_path).parent}. "
                "Run the generation script first."
            )

        cases = []
        case_id = 0

        for case_data in task4_data["cases"]:
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

            cases.append(Task4Case(
                case_id=f"t4_{case_id}",
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
        task3_num_examples: int = 1,
        task4_num_examples: int = 1,
    ) -> dict[str, list]:
        """
        Generate test cases for all tasks.

        Args:
            max_samples_per_task: Maximum samples per task
            task3_num_examples: Number of examples for task 3 (default: 1)
            task4_num_examples: Number of examples for task 4 (1 or 3, default: 1)

        Returns:
            Dictionary mapping task name to list of cases
        """
        return {
            "task1": self.generate_task1_cases(max_samples_per_task),
            "task2": self.generate_task2_cases(max_samples_per_task),
            "task3": self.generate_task3_cases(max_samples_per_task, num_examples=task3_num_examples),
            "task4": self.generate_task4_cases(max_samples_per_task, num_examples=task4_num_examples),
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
