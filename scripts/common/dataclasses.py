"""Shared dataclasses for evaluation pipeline."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class APIResponse:
    """Response from an LLM API call."""
    success: bool
    data: Any
    error: Optional[str] = None
    paper_id: Optional[str] = None
    logprobs: Optional[list] = None  # Token-level log probabilities if available
    avg_logprob: Optional[float] = None  # Average log probability across tokens
