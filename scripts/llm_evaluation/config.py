"""Configuration for the evaluation pipeline."""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# Default paths
DEFAULT_DATA_PATH = "/home/donggyu/econ_causality/new_data/real_data_1991/evaluation_data.json"
DEFAULT_OUTPUT_DIR = "/home/donggyu/econ_causality/econ_eval/evaluation_results"

# Supported models and their configurations
SUPPORTED_MODELS = {
    # "openai": {
    #     "client": "OpenAIClient",
    #     "default_model": "gpt-4o",
    #     "models": [ "gpt-4o"],
    #     "supports_logprobs": True,
    # },
    "openai": {
        "client": "OpenAIClient",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5.2"],
        "supports_logprobs": True,
    },
    "gemini": {
        "client": "GeminiClient",
        "default_model": "gemini-2.0-flash",
        "models": ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-pro",  ],
        "supports_logprobs": False,
    },
    "grok": {
        "client": "GrokClient",
        "default_model": "grok-3-mini",
        "models": ["grok-3-mini", "grok-3","grok-4-1-fast-reasoning",  ],
        "supports_logprobs": False,
    },
    "qwen": {
        "client": "OpenRouterClient",
        "default_model": "qwen/qwen3-8b",
        "models": ["qwen/qwen3-8b", "qwen/qwen3-14b", "qwen/qwen3-32b"],
        "supports_logprobs": False,
    },
    "llama": {
        "client": "OpenRouterClient",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
        "models": ["meta-llama/llama-3.2-1b-instruct", "meta-llama/llama-3.2-3b-instruct", "meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.3-70b-instruct"],
        "supports_logprobs": True,
    },
    # "deepseek": {
    #     "client": "OpenRouterClient",
    #     "default_model": "deepseek/deepseek-r1",
    #     "models": ["deepseek/deepseek-r1", "deepseek/deepseek-chat"],
    #     "supports_logprobs": True,
    # },
}

# Task types
TASK_TYPES = ["task1", "task2", "task3", "task4"]

# Models with rate limit restrictions (max_workers capped at 20)
RATE_LIMITED_MODELS = [
    "gemini-2.5-pro",
    "gemini-3-pro-preview"
]
RATE_LIMITED_MAX_WORKERS = 10


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    # Task configuration
    task_types: list[str] = field(default_factory=lambda: TASK_TYPES.copy())

    # Model configuration
    models: list[str] = field(default_factory=lambda: ["openai", "gemini", "grok", "qwen"])
    model_names: Optional[dict[str, str]] = None  # Override default model names

    # Data configuration
    data_path: str = DEFAULT_DATA_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR
    journal_type: str = "econ"  # "econ" or "finance" - used for API key selection

    # Sampling configuration
    max_samples_per_task: Optional[int] = None
    random_seed: int = 42

    # Task-specific configuration
    task3_num_examples: int = 1  # Number of examples for task 3 (default: 1)
    task4_num_examples: int = 1  # Number of examples for task 4 (1 or 3, default: 1)
    task2_no_context: bool = False  # If True, exclude context from task2 prompts
    task2_unknown_option: bool = False  # If True, add 'unknown' to task2 answer choices

    # Processing configuration
    max_workers: int = 128
    checkpoint_interval: int = 10
    timeout: int = 300
    max_retries: int = 3

    def __post_init__(self):
        """Validate configuration."""
        # Validate task types
        for task_type in self.task_types:
            if task_type not in TASK_TYPES:
                raise ValueError(f"Unknown task type: {task_type}. Must be one of {TASK_TYPES}")

        # Validate models
        for model in self.models:
            if model not in SUPPORTED_MODELS:
                raise ValueError(f"Unknown model: {model}. Must be one of {list(SUPPORTED_MODELS.keys())}")

        # Validate num_examples
        if self.task3_num_examples < 1:
            raise ValueError(f"task3_num_examples must be >= 1, got {self.task3_num_examples}")
        if self.task4_num_examples < 1:
            raise ValueError(f"task4_num_examples must be >= 1, got {self.task4_num_examples}")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_model_name(self, provider: str) -> str:
        """Get the model name for a provider."""
        if self.model_names and provider in self.model_names:
            return self.model_names[provider]
        return SUPPORTED_MODELS[provider]["default_model"]

    def get_checkpoint_path(self, task_type: str, model: str) -> Path:
        """Get checkpoint path for a task/model combination."""
        return Path(self.output_dir) / "checkpoints" / f"{task_type}_{model}_checkpoint.json"

    def get_results_path(self, task_type: str) -> Path:
        """Get results path for a task."""
        return Path(self.output_dir) / f"{task_type}_results.json"
