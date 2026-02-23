"""Base class for evaluation tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.dataclasses import APIResponse


@dataclass
class TaskResult:
    """Result from a single task evaluation."""
    case_id: str
    input_data: dict
    output_data: Optional[dict]
    expected: Any
    predicted: Any
    correct: bool
    reasoning: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    avg_logprob: Optional[float] = None  # Average log probability
    logprobs: Optional[list] = None  # Token-level log probabilities


class BaseTask(ABC):
    """Base class for evaluation tasks."""

    task_name: str = "base"
    task_description: str = "Base evaluation task"

    def __init__(self, schema: dict, prompt_template: str):
        """
        Initialize task.

        Args:
            schema: JSON schema for structured output
            prompt_template: Prompt template with placeholders
        """
        self.schema = schema
        self.prompt_template = prompt_template

    @abstractmethod
    def format_prompt(self, test_case: Any) -> str:
        """
        Format prompt for a test case.

        Args:
            test_case: Test case dataclass

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def extract_prediction(self, response_data: dict) -> Any:
        """
        Extract prediction from API response data.

        Args:
            response_data: Parsed JSON from API response

        Returns:
            Extracted prediction value
        """
        pass

    @abstractmethod
    def get_expected(self, test_case: Any) -> Any:
        """
        Get expected answer from test case.

        Args:
            test_case: Test case dataclass

        Returns:
            Expected answer
        """
        pass

    def evaluate_single(
        self,
        test_case: Any,
        api_response: APIResponse,
        latency_ms: Optional[float] = None,
    ) -> TaskResult:
        """
        Evaluate a single API response.

        Args:
            test_case: Test case dataclass
            api_response: API response from LLM
            latency_ms: API call latency in milliseconds

        Returns:
            TaskResult with evaluation details
        """
        expected = self.get_expected(test_case)

        if not api_response.success or api_response.data is None:
            return TaskResult(
                case_id=test_case.case_id,
                input_data=self._case_to_dict(test_case),
                output_data=None,
                expected=expected,
                predicted=None,
                correct=False,
                error=api_response.error,
                latency_ms=latency_ms,
            )

        try:
            predicted = self.extract_prediction(api_response.data)
            correct = self.is_correct(predicted, expected)

            return TaskResult(
                case_id=test_case.case_id,
                input_data=self._case_to_dict(test_case),
                output_data=api_response.data,
                expected=expected,
                predicted=predicted,
                correct=correct,
                reasoning=api_response.data.get("reasoning"),
                latency_ms=latency_ms,
                avg_logprob=api_response.avg_logprob,
                logprobs=api_response.logprobs,
            )
        except Exception as e:
            return TaskResult(
                case_id=test_case.case_id,
                input_data=self._case_to_dict(test_case),
                output_data=api_response.data,
                expected=expected,
                predicted=None,
                correct=False,
                error=str(e),
                latency_ms=latency_ms,
            )

    def is_correct(self, predicted: Any, expected: Any) -> bool:
        """
        Check if prediction matches expected answer.

        Args:
            predicted: Predicted value
            expected: Expected value

        Returns:
            True if correct
        """
        if predicted is None:
            return False
        return str(predicted).strip().lower() == str(expected).strip().lower()

    def _case_to_dict(self, test_case: Any) -> dict:
        """Convert test case to dictionary."""
        if hasattr(test_case, "__dataclass_fields__"):
            return {k: getattr(test_case, k) for k in test_case.__dataclass_fields__}
        return {"case": str(test_case)}
