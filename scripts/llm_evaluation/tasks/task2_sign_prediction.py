"""Task 2: Sign Prediction."""

from typing import Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.schemas import EVAL_TASK2_SCHEMA, EVAL_TASK2_UNKNOWN_SCHEMA
from common.prompts import (
    EVAL_TASK2_PROMPT, EVAL_TASK2_PROMPT_NO_CONTEXT,
    EVAL_TASK2_UNKNOWN_PROMPT, EVAL_TASK2_UNKNOWN_PROMPT_NO_CONTEXT,
)
from .base import BaseTask
from ..data_generator import Task2Case


class Task2SignPrediction(BaseTask):
    """Task 2: Sign Prediction - Predict the sign of a causal effect."""

    task_name = "task2"
    task_description = "Sign Prediction: Predict the direction of causal effect"

    VALID_SIGNS = {"+", "-", "None", "mixed"}
    VALID_SIGNS_WITH_UNKNOWN = {"+", "-", "None", "mixed", "unknown"}

    def __init__(self, no_context: bool = False, unknown_option: bool = False):
        """Initialize Task 2.

        Args:
            no_context: If True, exclude context from prompts (for ablation study)
            unknown_option: If True, add 'unknown' to the answer choices
        """
        self.no_context = no_context
        self.unknown_option = unknown_option

        if unknown_option:
            if no_context:
                prompt = EVAL_TASK2_UNKNOWN_PROMPT_NO_CONTEXT
            else:
                prompt = EVAL_TASK2_UNKNOWN_PROMPT
            schema = EVAL_TASK2_UNKNOWN_SCHEMA
        else:
            if no_context:
                prompt = EVAL_TASK2_PROMPT_NO_CONTEXT
            else:
                prompt = EVAL_TASK2_PROMPT
            schema = EVAL_TASK2_SCHEMA

        super().__init__(
            schema=schema,
            prompt_template=prompt,
        )

    def format_prompt(self, test_case: Task2Case) -> str:
        """
        Format prompt for Task 2.

        Args:
            test_case: Task2Case instance

        Returns:
            Formatted prompt string
        """
        if self.no_context:
            return self.prompt_template.format(
                treatment=test_case.treatment,
                outcome=test_case.outcome,
            )
        return self.prompt_template.format(
            context=test_case.context,
            treatment=test_case.treatment,
            outcome=test_case.outcome,
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
        # Normalize sign
        sign = sign.strip()

        # Handle common variations
        sign_mapping = {
            "positive": "+",
            "negative": "-",
            "none": "None",
            "null": "None",
            "no effect": "None",
            "mixed": "mixed",
            "unknown": "unknown",
            "+": "+",
            "-": "-",
        }

        valid = self.VALID_SIGNS_WITH_UNKNOWN if self.unknown_option else self.VALID_SIGNS
        normalized = sign_mapping.get(sign.lower(), sign)
        return normalized if normalized in valid else sign

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
