"""
LLM Causal Reasoning Evaluation Pipeline.

This package provides tools for evaluating LLM performance on causal reasoning tasks
using a dataset of causal triplets extracted from economics papers.

Tasks:
- Task 1: Causality Verification - Determine if a causal claim is valid
- Task 2: Sign Prediction - Predict the sign of a causal effect
- Task 3: Context-Aware (T/O Fixed) - Predict sign given context variation
- Task 4: Context-Aware (Context Fixed) - Predict sign for new T/O pair
"""

# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == "EvaluationConfig":
        from .config import EvaluationConfig
        return EvaluationConfig
    elif name == "SUPPORTED_MODELS":
        from .config import SUPPORTED_MODELS
        return SUPPORTED_MODELS
    elif name == "TestCaseGenerator":
        from .data_generator import TestCaseGenerator
        return TestCaseGenerator
    elif name == "EvaluationOrchestrator":
        from .evaluator import EvaluationOrchestrator
        return EvaluationOrchestrator
    elif name == "MetricsComputer":
        from .metrics import MetricsComputer
        return MetricsComputer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EvaluationConfig",
    "SUPPORTED_MODELS",
    "TestCaseGenerator",
    "EvaluationOrchestrator",
    "MetricsComputer",
]
