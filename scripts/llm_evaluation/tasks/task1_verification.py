"""Task 1: Causality Verification."""

from typing import Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.schemas import EVAL_TASK1_SCHEMA
from common.prompts import EVAL_TASK1_PROMPT
from .base import BaseTask
from ..data_generator import Task1Case


class Task1Verification(BaseTask):
    """Task 1: Causality Verification - Determine if a causal claim is valid."""

    task_name = "task1"
    task_description = "Causality Verification: Determine if a causal claim is valid under given context"

    def __init__(self):
        """Initialize Task 1."""
        super().__init__(
            schema=EVAL_TASK1_SCHEMA,
            prompt_template=EVAL_TASK1_PROMPT,
        )

    def _get_sign_description(self, sign: str) -> str:
        """Get human-readable description of sign."""
        descriptions = {
            "+": "increase",
            "-": "decrease",
            "None": "have no significant effect on",
            "mixed": "have a mixed effect on",
        }
        return descriptions.get(sign, "affect")

    def format_prompt(self, test_case: Task1Case) -> str:
        """
        Format prompt for Task 1.

        Args:
            test_case: Task1Case instance

        Returns:
            Formatted prompt string
        """
        sign_description = self._get_sign_description(test_case.sign)

        return self.prompt_template.format(
            context=test_case.context,
            treatment=test_case.treatment,
            outcome=test_case.outcome,
            sign=test_case.sign,
            sign_description=sign_description,
        )

    def extract_prediction(self, response_data: dict) -> str:
        """
        Extract Yes/No answer from response.

        Args:
            response_data: Parsed JSON response

        Returns:
            "Yes" or "No"
        """
        answer = response_data.get("answer", "")
        # Normalize answer
        answer = answer.strip()
        if answer.lower() in ["yes", "true", "1"]:
            return "Yes"
        elif answer.lower() in ["no", "false", "0"]:
            return "No"
        return answer

    def get_expected(self, test_case: Task1Case) -> str:
        """
        Get expected answer.

        Args:
            test_case: Task1Case instance

        Returns:
            Expected answer ("Yes" or "No")
        """
        return test_case.expected_answer
