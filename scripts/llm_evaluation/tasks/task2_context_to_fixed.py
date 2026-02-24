"""Task 2: Context-Dependent Sign Prediction (Treatment/Outcome Fixed)."""

from typing import Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.schemas import EVAL_TASK2_SCHEMA
from common.prompts import EVAL_TASK2_PROMPT
from .base import BaseTask
from ..data_generator import Task2Case


class Task2ContextTOFixed(BaseTask):
    """Task 2: Context-Dependent Sign Prediction - Predict sign for new context given examples."""

    task_name = "task2"
    task_description = "Context-Dependent (T/O Fixed): Predict sign for new context given examples"

    VALID_SIGNS = {"+", "-", "None", "mixed"}

    def __init__(self):
        """Initialize Task 2."""
        super().__init__(
            schema=EVAL_TASK2_SCHEMA,
            prompt_template=EVAL_TASK2_PROMPT,
        )

    def _format_examples(self, examples: list[dict]) -> str:
        """Format examples for prompt.

        Each example includes treatment, outcome, context, and sign
        since T/O pairs are similar but not necessarily identical.
        """
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"## Example {i}")
            formatted.append(f"Treatment: {ex['treatment']}")
            formatted.append(f"Outcome: {ex['outcome']}")
            formatted.append(f"Context: {ex['context']}")
            formatted.append(f"Sign: {ex['sign']}")
            formatted.append("")
        return "\n".join(formatted)

    def format_prompt(self, test_case: Task2Case) -> str:
        """
        Format prompt for Task 2.

        Args:
            test_case: Task2Case instance

        Returns:
            Formatted prompt string
        """
        examples_str = self._format_examples(test_case.examples)

        return self.prompt_template.format(
            treatment=test_case.treatment,
            outcome=test_case.outcome,
            examples=examples_str,
            context_new=test_case.test_context,
        )

    def extract_prediction(self, response_data: dict) -> str:
        """
        Extract predicted sign from response.

        Args:
            response_data: Parsed JSON response

        Returns:
            Predicted sign
        """
        sign = response_data.get("predicted_sign", "")
        sign = sign.strip()

        sign_mapping = {
            "positive": "+",
            "negative": "-",
            "none": "None",
            "null": "None",
            "no effect": "None",
            "mixed": "mixed",
            "+": "+",
            "-": "-",
        }

        normalized = sign_mapping.get(sign.lower(), sign)
        return normalized if normalized in self.VALID_SIGNS else sign

    def get_expected(self, test_case: Task2Case) -> str:
        """
        Get expected sign.

        Args:
            test_case: Task2Case instance

        Returns:
            Expected sign
        """
        return test_case.expected_sign

    def is_correct(self, predicted: Any, expected: Any) -> bool:
        """
        Check if predicted sign matches expected.

        Args:
            predicted: Predicted sign
            expected: Expected sign

        Returns:
            True if correct
        """
        if predicted is None:
            return False
        return predicted == expected
