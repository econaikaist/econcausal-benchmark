"""Common utilities for OpenAI API interactions and file operations."""

import os
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Default configuration
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MAX_WORKERS = 512  # Optimized for 32-core server
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_API_KEY = ""


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress HTTP request logs from httpx (used by OpenAI)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def load_json(file_path: str | Path) -> Any:
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_surrogates(obj: Any) -> Any:
    """Remove surrogate characters that can't be encoded in UTF-8."""
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {_sanitize_surrogates(k): _sanitize_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_surrogates(item) for item in obj]
    return obj


def save_json(data: Any, file_path: str | Path, indent: int = 2) -> None:
    """Save data to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    data = _sanitize_surrogates(data)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    return pd.read_csv(file_path)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (0 to 1)
    """
    import math
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def clean_sign(sign: str) -> str:
    """Remove leading/trailing quotes from sign value.
    Returns cleaned sign or empty string (never None)."""
    if sign:
        return sign.strip("'\"")
    return ""


def clean_causal_relations(causal_relations: list[dict]) -> list[dict]:
    """Clean sign values in causal relations list."""
    for triplet in causal_relations:
        if "sign" in triplet:
            triplet["sign"] = clean_sign(triplet["sign"])
    return causal_relations


def get_pdf_files(directory: str | Path) -> list[Path]:
    """Get all PDF files from directory, sorted by numeric filename."""
    dir_path = Path(directory)
    pdf_files = list(dir_path.rglob("*.pdf"))

    # Sort by numeric part of filename (e.g., 1.pdf, 2.pdf, ...)
    def get_num(p: Path) -> int:
        try:
            return int(p.stem)
        except ValueError:
            return float("inf")

    return sorted(pdf_files, key=get_num)


@dataclass
class APIResponse:
    """Container for API response data."""
    success: bool
    data: Any
    error: Optional[str] = None
    paper_id: Optional[str] = None
    logprobs: Optional[list] = None  # Token-level log probabilities if available
    avg_logprob: Optional[float] = None  # Average log probability across tokens


class OpenAIClient:
    """OpenAI API client with support for realtime and batch processing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_workers: int = DEFAULT_MAX_WORKERS,
        use_batch: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to hardcoded key)
            model: Model to use (default: gpt-5-mini)
            max_workers: Maximum parallel workers for concurrent requests
            use_batch: Whether to use batch API instead of realtime
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key or DEFAULT_API_KEY

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_workers = max_workers
        self.use_batch = use_batch
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

        # Track uploaded files
        self._uploaded_files: dict[str, str] = {}  # paper_id -> file_id

        # Track errors for error list JSON
        self._errors: list[dict] = []

    def _is_gpt5_model(self, model: Optional[str] = None) -> bool:
        """Check if the model is a GPT-5 series model."""
        model = model or self.model
        return model.startswith("gpt-5")

    def _supports_logprobs(self, model: Optional[str] = None) -> bool:
        """Check if the model supports logprobs."""
        model = model or self.model
        # Models that don't support logprobs
        no_logprobs_models = ["gpt-5-nano", "gpt-5-mini"]
        return model not in no_logprobs_models

    def _get_token_limit_params(self, max_tokens: Optional[int], model: Optional[str] = None) -> dict:
        """Get the correct token limit parameter based on model type.

        GPT-5 models require 'max_completion_tokens' instead of 'max_tokens'.
        """
        if max_tokens is None:
            return {}
        if self._is_gpt5_model(model):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    def upload_pdf(self, pdf_path: str | Path, paper_id: str) -> str:
        """
        Upload PDF file to OpenAI with retry logic.

        Args:
            pdf_path: Path to PDF file
            paper_id: Identifier for the paper

        Returns:
            OpenAI file ID
        """
        pdf_path = Path(pdf_path)
        last_error = None

        for attempt in range(self.max_retries):
            try:
                with open(pdf_path, "rb") as f:
                    response = self.client.files.create(
                        file=f,
                        purpose="assistants"
                    )

                file_id = response.id
                self._uploaded_files[paper_id] = file_id
                return file_id

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self._record_error("upload_pdf", paper_id, str(e))
                raise

        raise last_error

    def _record_error(self, operation: str, paper_id: str, error_msg: str) -> None:
        """Record an error for later export to JSON."""
        self._errors.append({
            "operation": operation,
            "paper_id": paper_id,
            "error": error_msg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def get_errors(self) -> list[dict]:
        """Get all recorded errors."""
        return self._errors.copy()

    def save_errors_json(self, file_path: str | Path) -> None:
        """Save recorded errors to JSON file."""
        if self._errors:
            save_json({"errors": self._errors, "total_count": len(self._errors)}, file_path)

    def upload_pdfs_parallel(
        self,
        pdf_paths: list[Path],
        paper_ids: Optional[list[str]] = None,
        desc: str = "Uploading PDFs",
    ) -> dict[str, str]:
        """
        Upload multiple PDFs in parallel.

        Args:
            pdf_paths: List of PDF file paths
            paper_ids: List of paper identifiers (defaults to filename stems)
            desc: Description for progress bar

        Returns:
            Dictionary mapping paper_id to file_id
        """
        if paper_ids is None:
            paper_ids = [p.stem for p in pdf_paths]

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paper = {
                executor.submit(self.upload_pdf, path, pid): pid
                for path, pid in zip(pdf_paths, paper_ids)
            }

            with tqdm(total=len(future_to_paper), desc=desc) as pbar:
                for future in as_completed(future_to_paper):
                    paper_id = future_to_paper[future]
                    try:
                        file_id = future.result()
                        results[paper_id] = file_id
                    except Exception as e:
                        self.logger.error(f"Failed to upload {paper_id}: {e}")
                        results[paper_id] = None
                    pbar.update(1)

        return results

    def call_api_with_pdf(
        self,
        file_id: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict,
        paper_id: Optional[str] = None,
    ) -> APIResponse:
        """
        Call OpenAI API with a PDF file and retry logic.

        Args:
            file_id: OpenAI file ID for the PDF
            system_prompt: System prompt
            user_prompt: User prompt (may include placeholders)
            response_schema: JSON schema for structured output
            paper_id: Optional paper identifier for tracking

        Returns:
            APIResponse with parsed JSON data
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "file",
                                    "file": {"file_id": file_id}
                                },
                                {
                                    "type": "text",
                                    "text": user_prompt
                                }
                            ]
                        }
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_schema.get("name", "response"),
                            "strict": True,
                            "schema": response_schema["schema"]
                        }
                    },
                    timeout=self.timeout,
                )

                content = response.choices[0].message.content
                data = json.loads(content)

                return APIResponse(success=True, data=data, paper_id=paper_id)

            except json.JSONDecodeError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.logger.error(f"JSON decode error for {paper_id}: {e}")
                self._record_error("call_api_with_pdf", paper_id, str(e))
                return APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.logger.error(f"API call failed for {paper_id}: {e}")
                self._record_error("call_api_with_pdf", paper_id, str(e))
                return APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)

        self.logger.error(f"API call failed after {self.max_retries} retries for {paper_id}: {last_error}")
        self._record_error("call_api_with_pdf", paper_id, str(last_error))
        return APIResponse(success=False, data=None, error=str(last_error), paper_id=paper_id)

    def call_api_with_text(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict,
        paper_id: Optional[str] = None,
        return_logprobs: bool = True,
    ) -> APIResponse:
        """
        Call OpenAI API with text input only and retry logic.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt with content
            response_schema: JSON schema for structured output
            paper_id: Optional paper identifier for tracking
            return_logprobs: Whether to request log probabilities (default: True)

        Returns:
            APIResponse with parsed JSON data and optional logprobs
        """
        last_error = None

        # Check if model supports logprobs
        use_logprobs = return_logprobs and self._supports_logprobs()

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_schema.get("name", "response"),
                            "strict": True,
                            "schema": response_schema["schema"]
                        }
                    },
                    logprobs=use_logprobs,
                    top_logprobs=5 if use_logprobs else None,
                    timeout=self.timeout,
                )

                content = response.choices[0].message.content
                data = json.loads(content)

                # Extract logprobs if available
                logprobs_data = None
                avg_logprob = None
                if use_logprobs and hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                    logprobs_content = response.choices[0].logprobs.content
                    if logprobs_content:
                        logprobs_data = [
                            {
                                "token": lp.token,
                                "logprob": lp.logprob,
                                "top_logprobs": [{"token": t.token, "logprob": t.logprob} for t in (lp.top_logprobs or [])]
                            }
                            for lp in logprobs_content
                        ]
                        # Calculate average log probability
                        all_logprobs = [lp.logprob for lp in logprobs_content if lp.logprob is not None]
                        if all_logprobs:
                            avg_logprob = sum(all_logprobs) / len(all_logprobs)

                return APIResponse(
                    success=True,
                    data=data,
                    paper_id=paper_id,
                    logprobs=logprobs_data,
                    avg_logprob=avg_logprob,
                )

            except json.JSONDecodeError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.logger.error(f"JSON decode error for {paper_id}: {e}")
                self._record_error("call_api_with_text", paper_id, str(e))
                return APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.logger.error(f"API call failed for {paper_id}: {e}")
                self._record_error("call_api_with_text", paper_id, str(e))
                return APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)

        self.logger.error(f"API call failed after {self.max_retries} retries for {paper_id}: {last_error}")
        self._record_error("call_api_with_text", paper_id, str(last_error))
        return APIResponse(success=False, data=None, error=str(last_error), paper_id=paper_id)

    def call_api(
        self,
        user_prompt: str,
        response_schema: dict,
        paper_id: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> APIResponse:
        """
        Unified API call method compatible with other LLM clients.

        Args:
            user_prompt: User prompt with content
            response_schema: JSON schema for structured output
            paper_id: Optional paper identifier for tracking
            pdf_path: Optional path to PDF (not used for text-only calls)

        Returns:
            APIResponse with parsed JSON data and optional logprobs
        """
        system_prompt = "You are a helpful assistant that responds in JSON format."
        return self.call_api_with_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=response_schema,
            paper_id=paper_id,
            return_logprobs=True,
        )

    def call_api_parallel(
        self,
        tasks: list[dict],
        use_pdf: bool = True,
        desc: str = "Processing API calls",
        checkpoint_path: Optional[str | Path] = None,
        checkpoint_interval: int = 10,
    ) -> list[APIResponse]:
        """
        Execute multiple API calls in parallel with checkpointing.

        Args:
            tasks: List of task dictionaries with keys:
                - file_id (if use_pdf=True) or user_prompt
                - system_prompt
                - user_prompt (for PDF: additional text prompt)
                - response_schema
                - paper_id (optional)
            use_pdf: Whether tasks include PDF files
            desc: Description for progress bar
            checkpoint_path: Path to save incremental results (optional)
            checkpoint_interval: Save checkpoint every N completed tasks

        Returns:
            List of APIResponse objects
        """
        results = []
        results_dict = {}  # paper_id -> result for checkpointing
        completed_ids = set()

        # Load previous results if checkpoint exists
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                try:
                    prev_data = load_json(checkpoint_path)
                    if "results" in prev_data:
                        for paper_id, data in prev_data["results"].items():
                            completed_ids.add(paper_id)
                            results_dict[paper_id] = data
                        self.logger.info(f"Resuming from checkpoint: {len(completed_ids)} already completed")
                except Exception as e:
                    self.logger.warning(f"Failed to load checkpoint: {e}")

        # Filter out already completed tasks
        remaining_tasks = [t for t in tasks if t.get("paper_id") not in completed_ids]

        if not remaining_tasks:
            self.logger.info("All tasks already completed from checkpoint")
            return [
                APIResponse(success=True, data=results_dict[t.get("paper_id")], paper_id=t.get("paper_id"))
                for t in tasks if t.get("paper_id") in results_dict
            ]

        self.logger.info(f"Processing {len(remaining_tasks)} remaining tasks (skipped {len(completed_ids)} completed)")

        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if use_pdf:
                futures = {
                    executor.submit(
                        self.call_api_with_pdf,
                        task["file_id"],
                        task["system_prompt"],
                        task["user_prompt"],
                        task["response_schema"],
                        task.get("paper_id"),
                    ): task.get("paper_id")
                    for task in remaining_tasks
                }
            else:
                futures = {
                    executor.submit(
                        self.call_api_with_text,
                        task["system_prompt"],
                        task["user_prompt"],
                        task["response_schema"],
                        task.get("paper_id"),
                    ): task.get("paper_id")
                    for task in remaining_tasks
                }

            with tqdm(total=len(futures), desc=desc) as pbar:
                for future in as_completed(futures):
                    paper_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result.success:
                            results_dict[paper_id] = result.data
                    except Exception as e:
                        self.logger.error(f"Task failed for {paper_id}: {e}")
                        results.append(
                            APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)
                        )
                    pbar.update(1)
                    completed_count += 1

                    # Save checkpoint periodically
                    if checkpoint_path and completed_count % checkpoint_interval == 0:
                        self._save_checkpoint(checkpoint_path, results_dict)

        # Final checkpoint save
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, results_dict)

        # Include previously completed results
        for paper_id, data in results_dict.items():
            if paper_id in completed_ids:
                results.append(APIResponse(success=True, data=data, paper_id=paper_id))

        return results

    def _save_checkpoint(self, checkpoint_path: Path, results_dict: dict) -> None:
        """Save checkpoint to file."""
        try:
            checkpoint_data = {"results": results_dict}
            save_json(checkpoint_data, checkpoint_path)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    # ==================== Batch API Methods ====================

    def create_batch_request(
        self,
        tasks: list[dict],
        use_pdf: bool = True,
    ) -> str:
        """
        Create a batch request file and submit to OpenAI Batch API.

        Args:
            tasks: List of task dictionaries
            use_pdf: Whether tasks include PDF files

        Returns:
            Batch ID
        """
        import tempfile

        # Create JSONL batch input
        batch_requests = []
        for i, task in enumerate(tasks):
            custom_id = task.get("paper_id", f"request_{i}")

            if use_pdf:
                messages = [
                    {"role": "system", "content": task["system_prompt"]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "file", "file": {"file_id": task["file_id"]}},
                            {"type": "text", "text": task["user_prompt"]}
                        ]
                    }
                ]
            else:
                messages = [
                    {"role": "system", "content": task["system_prompt"]},
                    {"role": "user", "content": task["user_prompt"]}
                ]

            batch_requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": messages,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": task["response_schema"].get("name", "response"),
                            "strict": True,
                            "schema": task["response_schema"]["schema"]
                        }
                    }
                }
            })

        # Write to temp file and upload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for req in batch_requests:
                f.write(json.dumps(req) + "\n")
            temp_path = f.name

        try:
            # Upload batch input file
            with open(temp_path, "rb") as f:
                batch_input_file = self.client.files.create(
                    file=f,
                    purpose="batch"
                )

            # Create batch
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )

            self.logger.info(f"Created batch {batch.id}")
            return batch.id

        finally:
            os.unlink(temp_path)

    def get_batch_status(self, batch_id: str) -> dict:
        """Get status of a batch request."""
        batch = self.client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "request_counts": batch.request_counts,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 60,
        max_wait: int = 86400,  # 24 hours
    ) -> dict:
        """
        Wait for batch to complete.

        Args:
            batch_id: Batch ID to wait for
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait

        Returns:
            Final batch status
        """
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status = self.get_batch_status(batch_id)
            self.logger.info(f"Batch {batch_id} status: {status['status']}")

            if status["status"] in ("completed", "failed", "expired", "cancelled"):
                return status

            time.sleep(poll_interval)

        raise TimeoutError(f"Batch {batch_id} did not complete within {max_wait}s")

    def get_batch_results(self, batch_id: str) -> list[APIResponse]:
        """
        Get results from a completed batch.

        Args:
            batch_id: Batch ID

        Returns:
            List of APIResponse objects
        """
        status = self.get_batch_status(batch_id)

        if status["status"] != "completed":
            raise ValueError(f"Batch {batch_id} is not completed: {status['status']}")

        if not status["output_file_id"]:
            raise ValueError(f"Batch {batch_id} has no output file")

        # Download output file
        output_content = self.client.files.content(status["output_file_id"])

        results = []
        for line in output_content.text.strip().split("\n"):
            result = json.loads(line)
            custom_id = result["custom_id"]

            if result.get("error"):
                results.append(APIResponse(
                    success=False,
                    data=None,
                    error=str(result["error"]),
                    paper_id=custom_id
                ))
            else:
                try:
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    data = json.loads(content)
                    results.append(APIResponse(
                        success=True,
                        data=data,
                        paper_id=custom_id
                    ))
                except (KeyError, json.JSONDecodeError) as e:
                    results.append(APIResponse(
                        success=False,
                        data=None,
                        error=str(e),
                        paper_id=custom_id
                    ))

        return results

    def process_tasks(
        self,
        tasks: list[dict],
        use_pdf: bool = True,
        desc: str = "Processing tasks",
        checkpoint_path: Optional[str | Path] = None,
        checkpoint_interval: int = 10,
    ) -> list[APIResponse]:
        """
        Process tasks using either realtime or batch API based on configuration.

        Args:
            tasks: List of task dictionaries
            use_pdf: Whether tasks include PDF files
            desc: Description for progress bar
            checkpoint_path: Path to save incremental results (optional)
            checkpoint_interval: Save checkpoint every N completed tasks

        Returns:
            List of APIResponse objects
        """
        if self.use_batch:
            batch_id = self.create_batch_request(tasks, use_pdf)
            self.wait_for_batch(batch_id)
            return self.get_batch_results(batch_id)
        else:
            return self.call_api_parallel(
                tasks, use_pdf, desc=desc,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=checkpoint_interval
            )

    # ==================== Embedding Methods ====================

    def get_embedding(
        self,
        text: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> list[float]:
        """
        Get embedding for a single text with retry logic.

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            Embedding vector as list of floats
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=model,
                )
                return response.data[0].embedding
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self._record_error("get_embedding", None, str(e))
                raise

        raise last_error

    def get_embeddings_batch(
        self,
        texts: list[str],
        model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 100,
    ) -> list[list[float]]:
        """
        Get embeddings for multiple texts in batches with retry logic.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            batch_size: Number of texts per batch (max 2048 for OpenAI)

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            last_error = None

            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=model,
                    )
                    # Sort by index to maintain order
                    sorted_data = sorted(response.data, key=lambda x: x.index)
                    all_embeddings.extend([d.embedding for d in sorted_data])
                    break  # Success, move to next batch
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    self._record_error("get_embeddings_batch", f"batch_{i}", str(e))
                    raise

        return all_embeddings


# ==================== Multi-LLM Judge Classes ====================

# API Keys for multi-model scoring
# Different Gemini keys for econ vs finance to avoid rate limits
GEMINI_API_KEY_ECON = ""
GEMINI_API_KEY_FINANCE = ""
GEMINI_API_KEY_PLUS =""
# GEMINI_API_KEY = GEMINI_API_KEY_ECON  # Default
GEMINI_API_KEY = GEMINI_API_KEY_PLUS  # Default
GROK_API_KEY = ""
QWEN_API_KEY = ""


class GeminiClient:
    """Google Gemini API client for LLM-as-a-judge scoring with PDF support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3-flash-preview",
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key
            model: Model to use (default: gemini-3-flash-preview)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        try:
            from google import genai
        except ImportError:
            raise ImportError("Please install google-genai: pip install google-genai")

        self.api_key = api_key or GEMINI_API_KEY
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

        self.client = genai.Client(api_key=self.api_key)
        self._uploaded_files: dict[str, Any] = {}  # Cache uploaded files

    def _convert_schema_for_gemini(self, schema: dict) -> dict:
        """
        Convert JSON schema to Gemini-compatible format.
        Removes integer enums and additionalProperties which are not supported.
        """
        import copy
        schema = copy.deepcopy(schema)

        def clean_schema(obj):
            if isinstance(obj, dict):
                # Remove additionalProperties (not supported by Gemini)
                if "additionalProperties" in obj:
                    del obj["additionalProperties"]
                # If it's an integer type with enum, remove the enum
                if obj.get("type") == "integer" and "enum" in obj:
                    del obj["enum"]
                # Recursively process all dict values
                for key, value in list(obj.items()):
                    clean_schema(value)
            elif isinstance(obj, list):
                for item in obj:
                    clean_schema(item)

        clean_schema(schema)
        return schema

    def _upload_pdf(self, pdf_path: str) -> Any:
        """Upload PDF file to Gemini and cache the result."""
        if pdf_path in self._uploaded_files:
            return self._uploaded_files[pdf_path]

        uploaded_file = self.client.files.upload(file=pdf_path)
        self._uploaded_files[pdf_path] = uploaded_file
        return uploaded_file

    def call_api(
        self,
        user_prompt: str,
        response_schema: dict,
        paper_id: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> APIResponse:
        """
        Call Gemini API with structured JSON output and optional PDF.

        Args:
            user_prompt: User prompt
            response_schema: JSON schema for structured output
            paper_id: Optional paper identifier
            pdf_path: Optional path to PDF file to attach

        Returns:
            APIResponse with parsed JSON data
        """
        from google.genai import types

        gemini_schema = self._convert_schema_for_gemini(response_schema["schema"])

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Build contents with optional PDF
                if pdf_path:
                    uploaded_file = self._upload_pdf(pdf_path)
                    contents = [uploaded_file, user_prompt]
                else:
                    contents = user_prompt

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=gemini_schema,
                    ),
                )

                content = response.text
                data = json.loads(content)

                return APIResponse(success=True, data=data, paper_id=paper_id)

            except json.JSONDecodeError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.logger.error(f"Gemini JSON decode error for {paper_id}: {e}")
                return APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)
            except Exception as e:
                last_error = e
                # Retry on 503 errors or other transient failures
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.logger.error(f"Gemini API call failed for {paper_id}: {e}")
                return APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)

        self.logger.error(f"Gemini API failed after {self.max_retries} retries for {paper_id}: {last_error}")
        return APIResponse(success=False, data=None, error=str(last_error), paper_id=paper_id)


class GrokClient:
    """xAI Grok API client (OpenAI-compatible) for LLM-as-a-judge scoring with PDF support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-4-1-fast-reasoning",
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        """
        Initialize Grok client.

        Args:
            api_key: xAI API key
            model: Model to use (default: grok-4-1-fast-reasoning)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key or GROK_API_KEY
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

        # Use OpenAI client with xAI base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1",
        )
        self._uploaded_files: dict[str, str] = {}  # Cache uploaded file IDs

    def _upload_pdf(self, pdf_path: str) -> str:
        """Upload PDF file to xAI and cache the result."""
        if pdf_path in self._uploaded_files:
            return self._uploaded_files[pdf_path]

        with open(pdf_path, "rb") as f:
            response = self.client.files.create(file=f, purpose="assistants")
        file_id = response.id
        self._uploaded_files[pdf_path] = file_id
        return file_id

    def call_api(
        self,
        user_prompt: str,
        response_schema: dict,
        paper_id: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> APIResponse:
        """
        Call Grok API with structured JSON output and optional PDF.

        Args:
            user_prompt: User prompt
            response_schema: JSON schema for structured output
            paper_id: Optional paper identifier
            pdf_path: Optional path to PDF file to attach

        Returns:
            APIResponse with parsed JSON data
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Build message content with optional PDF
                if pdf_path:
                    file_id = self._upload_pdf(pdf_path)
                    content = [
                        {"type": "file", "file": {"file_id": file_id}},
                        {"type": "text", "text": user_prompt}
                    ]
                else:
                    content = user_prompt

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": content}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_schema.get("name", "response"),
                            "strict": True,
                            "schema": response_schema["schema"]
                        }
                    },
                    timeout=self.timeout,
                )

                content_text = response.choices[0].message.content
                data = json.loads(content_text)

                return APIResponse(success=True, data=data, paper_id=paper_id)

            except json.JSONDecodeError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.logger.error(f"Grok JSON decode error for {paper_id}: {e}")
                return APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.logger.error(f"Grok API call failed for {paper_id}: {e}")
                return APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)

        self.logger.error(f"Grok API failed after {self.max_retries} retries for {paper_id}: {last_error}")
        return APIResponse(success=False, data=None, error=str(last_error), paper_id=paper_id)


class QwenClient:
    """Qwen API client via OpenRouter for LLM-as-a-judge scoring with PDF support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen3-vl-30b-a3b-thinking",
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        """
        Initialize Qwen client via OpenRouter.

        Args:
            api_key: OpenRouter API key
            model: Model to use (default: qwen/qwen3-235b-a22b)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key or QWEN_API_KEY
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

        # Use OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def _get_json_instruction(self, response_schema: dict) -> str:
        """Generate JSON instruction for the prompt."""
        schema_str = json.dumps(response_schema["schema"], indent=2)
        return f"\n\nYou MUST respond with valid JSON matching this schema:\n```json\n{schema_str}\n```\nRespond ONLY with the JSON object, no other text."

    def _encode_pdf_base64(self, pdf_path: str) -> str:
        """Encode PDF file as base64."""
        import base64
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def call_api(
        self,
        user_prompt: str,
        response_schema: dict,
        paper_id: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> APIResponse:
        """
        Call Qwen API with JSON output via prompt instruction and optional PDF.

        Args:
            user_prompt: User prompt
            response_schema: JSON schema for structured output
            paper_id: Optional paper identifier
            pdf_path: Optional path to PDF file to attach

        Returns:
            APIResponse with parsed JSON data
        """
        # Add JSON instruction to user prompt
        json_instruction = self._get_json_instruction(response_schema)
        enhanced_prompt = user_prompt + json_instruction

        last_error = None
        content_text = None  # Initialize for error logging
        for attempt in range(self.max_retries):
            try:
                # Build message content - OpenRouter supports PDF via URL or base64
                if pdf_path:
                    pdf_base64 = self._encode_pdf_base64(pdf_path)
                    content = [
                        {
                            "type": "file",
                            "file": {
                                "filename": Path(pdf_path).name,
                                "file_data": f"data:application/pdf;base64,{pdf_base64}"
                            }
                        },
                        {"type": "text", "text": enhanced_prompt}
                    ]
                else:
                    content = enhanced_prompt

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": content}
                    ],
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                )

                # Handle potential None response with detailed logging
                if not response:
                    self.logger.warning(f"Qwen response is None for {paper_id}")
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Retrying ({attempt + 1}/{self.max_retries})...")
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError("Response is None")

                if not response.choices:
                    # Log detailed response info
                    self.logger.warning(f"Qwen empty choices for {paper_id}. Response: id={getattr(response, 'id', 'N/A')}, "
                                       f"model={getattr(response, 'model', 'N/A')}, "
                                       f"usage={getattr(response, 'usage', 'N/A')}")
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Retrying ({attempt + 1}/{self.max_retries})...")
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError(f"Empty choices in response (id={getattr(response, 'id', 'N/A')})")

                choice = response.choices[0]
                if not choice.message:
                    self.logger.warning(f"Qwen empty message for {paper_id}. "
                                       f"finish_reason={getattr(choice, 'finish_reason', 'N/A')}, "
                                       f"index={getattr(choice, 'index', 'N/A')}")
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Retrying ({attempt + 1}/{self.max_retries})...")
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError(f"Empty message (finish_reason={getattr(choice, 'finish_reason', 'N/A')})")

                content_text = choice.message.content
                finish_reason = getattr(choice, 'finish_reason', 'N/A')
                if not content_text:
                    self.logger.warning(f"Qwen empty content for {paper_id}. "
                                       f"finish_reason={finish_reason}, "
                                       f"refusal={getattr(choice.message, 'refusal', 'N/A')}, "
                                       f"role={getattr(choice.message, 'role', 'N/A')}")
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Retrying ({attempt + 1}/{self.max_retries})...")
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError(f"Empty content (finish_reason={finish_reason})")

                data = json.loads(content_text)
                return APIResponse(success=True, data=data, paper_id=paper_id)

            except json.JSONDecodeError as e:
                last_error = e
                raw_preview = content_text[:500] if content_text else 'None'
                self.logger.warning(f"Qwen JSON decode error for {paper_id}: {e}. Raw content: {raw_preview}...")
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Retrying ({attempt + 1}/{self.max_retries})...")
                    time.sleep(2 ** attempt)
                    continue
                self.logger.error(f"Qwen JSON decode error for {paper_id} after {self.max_retries} retries: {e}")
                return APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                self.logger.warning(f"Qwen {error_type} for {paper_id}: {e}")
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Retrying ({attempt + 1}/{self.max_retries})...")
                    time.sleep(2 ** attempt)
                    continue
                self.logger.error(f"Qwen API call failed for {paper_id} after {self.max_retries} retries: {error_type}: {e}")
                return APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)

        self.logger.error(f"Qwen API failed after {self.max_retries} retries for {paper_id}: {last_error}")
        return APIResponse(success=False, data=None, error=str(last_error), paper_id=paper_id)


class MultiLLMJudge:
    """
    Multi-LLM Judge that uses multiple models for scoring and averages results.

    Uses three different LLM providers:
    - Gemini (Google)
    - Grok (xAI)
    - Qwen (via OpenRouter)
    """

    MODEL_NAMES = ["gemini", "grok", "qwen"]

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        grok_api_key: Optional[str] = None,
        qwen_api_key: Optional[str] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize MultiLLMJudge with all three model clients.

        Args:
            gemini_api_key: Gemini API key
            grok_api_key: Grok API key
            qwen_api_key: Qwen/OpenRouter API key
            max_workers: Maximum parallel workers
            timeout: Request timeout in seconds
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Initialize all three clients
        self.clients = {
            "gemini": GeminiClient(api_key=gemini_api_key, timeout=timeout),
            "grok": GrokClient(api_key=grok_api_key, timeout=timeout),
            "qwen": QwenClient(api_key=qwen_api_key, timeout=timeout),
        }

        # Track errors for error list JSON
        self._errors: list[dict] = []

    def _record_error(self, model: str, paper_id: str, error_msg: str) -> None:
        """Record an error for later export to JSON."""
        self._errors.append({
            "model": model,
            "paper_id": paper_id,
            "error": error_msg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def get_errors(self) -> list[dict]:
        """Get all recorded errors."""
        return self._errors.copy()

    def save_errors_json(self, file_path: str | Path) -> None:
        """Save recorded errors to JSON file."""
        if self._errors:
            save_json({"errors": self._errors, "total_count": len(self._errors)}, file_path)

    def call_single_model(
        self,
        model_name: str,
        user_prompt: str,
        response_schema: dict,
        paper_id: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> APIResponse:
        """Call a single model and return response."""
        client = self.clients[model_name]
        return client.call_api(user_prompt, response_schema, paper_id, pdf_path)

    def call_all_models(
        self,
        user_prompt: str,
        response_schema: dict,
        paper_id: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> dict[str, APIResponse]:
        """
        Call all three models in parallel and return their responses.

        Args:
            user_prompt: User prompt
            response_schema: JSON schema for structured output
            paper_id: Optional paper identifier
            pdf_path: Optional path to PDF file

        Returns:
            Dictionary mapping model name to APIResponse
        """
        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self.call_single_model,
                    model_name,
                    user_prompt,
                    response_schema,
                    paper_id,
                    pdf_path,
                ): model_name
                for model_name in self.MODEL_NAMES
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    results[model_name] = result
                    # Record error if the call failed
                    if not result.success:
                        self._record_error(model_name, paper_id, result.error)
                except Exception as e:
                    self.logger.error(f"Model {model_name} failed for {paper_id}: {e}")
                    self._record_error(model_name, paper_id, str(e))
                    results[model_name] = APIResponse(
                        success=False, data=None, error=str(e), paper_id=paper_id
                    )

        return results

    def score_and_average(
        self,
        user_prompt: str,
        response_schema: dict,
        paper_id: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> dict:
        """
        Get scores from all models and compute average.

        Args:
            user_prompt: User prompt
            response_schema: JSON schema for structured output
            paper_id: Optional paper identifier
            pdf_path: Optional path to PDF file

        Returns:
            Dictionary containing:
            - per_model_scores: Dict of scores from each model
            - per_model_reasons: Dict of reasons from each model
            - average_scores: Averaged scores across models
            - successful_models: List of models that succeeded
        """
        responses = self.call_all_models(user_prompt, response_schema, paper_id, pdf_path)

        per_model_scores = {}
        per_model_reasons = {}
        successful_models = []

        for model_name, response in responses.items():
            if response.success and response.data:
                scores = response.data.get("scores", {})
                reasons = response.data.get("reasons", {})
                per_model_scores[model_name] = scores
                per_model_reasons[model_name] = reasons
                successful_models.append(model_name)
            else:
                per_model_scores[model_name] = None
                per_model_reasons[model_name] = None

        # Calculate average scores
        average_scores = {}
        if successful_models:
            # Dynamically get score keys from the first successful model's scores
            first_scores = per_model_scores[successful_models[0]]
            score_keys = list(first_scores.keys()) if first_scores else []
            for key in score_keys:
                values = [
                    per_model_scores[model][key]
                    for model in successful_models
                    if per_model_scores[model] and key in per_model_scores[model]
                ]
                if values:
                    average_scores[key] = sum(values) / len(values)

        return {
            "per_model_scores": per_model_scores,
            "per_model_reasons": per_model_reasons,
            "average_scores": average_scores,
            "successful_models": successful_models,
        }

    def process_tasks_parallel(
        self,
        tasks: list[dict],
        desc: str = "Multi-LLM scoring",
        checkpoint_path: Optional[str | Path] = None,
        checkpoint_interval: int = 10,
    ) -> list[dict]:
        """
        Process multiple tasks with all three models in parallel.

        Args:
            tasks: List of task dictionaries with keys:
                - user_prompt
                - response_schema
                - paper_id (optional)
                - pdf_path (optional)
            desc: Description for progress bar
            checkpoint_path: Path to save incremental results
            checkpoint_interval: Save checkpoint every N tasks

        Returns:
            List of result dictionaries with per-model and average scores
        """
        results = []
        results_dict = {}
        completed_ids = set()

        # Load checkpoint if exists
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                try:
                    prev_data = load_json(checkpoint_path)
                    if "results" in prev_data:
                        for paper_id, data in prev_data["results"].items():
                            completed_ids.add(paper_id)
                            results_dict[paper_id] = data
                        self.logger.info(f"Resuming from checkpoint: {len(completed_ids)} completed")
                except Exception as e:
                    self.logger.warning(f"Failed to load checkpoint: {e}")

        # Filter remaining tasks
        remaining_tasks = [t for t in tasks if t.get("paper_id") not in completed_ids]

        if not remaining_tasks:
            self.logger.info("All tasks completed from checkpoint")
            return [results_dict[t.get("paper_id")] for t in tasks if t.get("paper_id") in results_dict]

        self.logger.info(f"Processing {len(remaining_tasks)} tasks ({len(completed_ids)} completed)")

        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.score_and_average,
                    task["user_prompt"],
                    task["response_schema"],
                    task.get("paper_id"),
                    task.get("pdf_path"),
                ): task.get("paper_id")
                for task in remaining_tasks
            }

            with tqdm(total=len(futures), desc=desc) as pbar:
                for future in as_completed(futures):
                    paper_id = futures[future]
                    try:
                        result = future.result()
                        result["paper_id"] = paper_id
                        results.append(result)
                        results_dict[paper_id] = result
                    except Exception as e:
                        self.logger.error(f"Task failed for {paper_id}: {e}")
                        error_result = {
                            "paper_id": paper_id,
                            "per_model_scores": {},
                            "per_model_reasons": {},
                            "average_scores": {},
                            "successful_models": [],
                            "error": str(e),
                        }
                        results.append(error_result)
                        results_dict[paper_id] = error_result
                    pbar.update(1)
                    completed_count += 1

                    # Save checkpoint
                    if checkpoint_path and completed_count % checkpoint_interval == 0:
                        self._save_checkpoint(checkpoint_path, results_dict)

        # Final checkpoint save
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, results_dict)

        # Include previously completed results
        for paper_id, data in results_dict.items():
            if paper_id in completed_ids:
                results.append(data)

        return results

    def _save_checkpoint(self, checkpoint_path: Path, results_dict: dict) -> None:
        """Save checkpoint to file."""
        try:
            checkpoint_data = {"results": results_dict}
            save_json(checkpoint_data, checkpoint_path)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")


# OpenRouter API Key (for Llama, Qwen, and other open-source models)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", QWEN_API_KEY)


class OpenRouterClient:
    """OpenRouter API client for Llama, Qwen, and other models with logprobs support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-3.1-70b-instruct",
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        return_logprobs: bool = True,
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key
            model: Model to use (e.g., meta-llama/llama-3.1-70b-instruct, qwen/qwen-2.5-72b-instruct)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            return_logprobs: Whether to request log probabilities
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.return_logprobs = return_logprobs
        self.logger = logging.getLogger(__name__)

        # Use OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def _get_json_instruction(self, response_schema: dict) -> str:
        """Generate JSON instruction for the prompt."""
        schema_str = json.dumps(response_schema["schema"], indent=2)
        return f"\n\nYou MUST respond with valid JSON matching this schema:\n```json\n{schema_str}\n```\nRespond ONLY with the JSON object, no other text."

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from model output."""
        import re

        # First try: direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Second try: find JSON block in markdown
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Third try: find JSON object pattern
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("No valid JSON found in response", text, 0)

    def call_api(
        self,
        user_prompt: str,
        response_schema: dict,
        paper_id: Optional[str] = None,
        pdf_path: Optional[str] = None,
    ) -> APIResponse:
        """
        Call OpenRouter API with JSON output and optional logprobs.

        Args:
            user_prompt: User prompt
            response_schema: JSON schema for structured output
            paper_id: Optional paper identifier
            pdf_path: Not supported (ignored)

        Returns:
            APIResponse with parsed JSON data and logprobs if available
        """
        # Add JSON instruction to user prompt
        json_instruction = self._get_json_instruction(response_schema)
        enhanced_prompt = user_prompt + json_instruction

        last_error = None
        content_text = None

        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                request_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": enhanced_prompt}
                    ],
                    "response_format": {"type": "json_object"},
                    "timeout": self.timeout,
                }

                # Add logprobs if supported
                if self.return_logprobs:
                    request_params["logprobs"] = True
                    request_params["top_logprobs"] = 5

                response = self.client.chat.completions.create(**request_params)

                # Handle potential None response
                if not response or not response.choices:
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Empty response for {paper_id}, retrying...")
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError("Empty response from model")

                choice = response.choices[0]
                content_text = choice.message.content

                if not content_text:
                    if attempt < self.max_retries - 1:
                        self.logger.warning(f"Empty content for {paper_id}, retrying...")
                        time.sleep(2 ** attempt)
                        continue
                    raise ValueError("Empty content from model")

                data = self._extract_json(content_text)

                # Extract logprobs if available
                logprobs_data = None
                avg_logprob = None
                if self.return_logprobs and hasattr(choice, 'logprobs') and choice.logprobs:
                    logprobs_content = choice.logprobs.content if hasattr(choice.logprobs, 'content') else None
                    if logprobs_content:
                        logprobs_data = [
                            {
                                "token": lp.token,
                                "logprob": lp.logprob,
                                "top_logprobs": [{"token": t.token, "logprob": t.logprob} for t in (lp.top_logprobs or [])]
                            }
                            for lp in logprobs_content
                        ]
                        # Calculate average log probability
                        all_logprobs = [lp.logprob for lp in logprobs_content if lp.logprob is not None]
                        if all_logprobs:
                            avg_logprob = sum(all_logprobs) / len(all_logprobs)

                return APIResponse(
                    success=True,
                    data=data,
                    paper_id=paper_id,
                    logprobs=logprobs_data,
                    avg_logprob=avg_logprob,
                )

            except json.JSONDecodeError as e:
                last_error = e
                raw_preview = content_text[:500] if content_text else 'None'
                self.logger.warning(f"OpenRouter JSON decode error for {paper_id}: {e}. Raw: {raw_preview}...")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                # Save raw response and mark as parse_failed instead of error
                self.logger.warning(f"OpenRouter parse failed for {paper_id}, recording as no response with raw text")
                return APIResponse(
                    success=True,
                    data={
                        "parse_failed": True,
                        "raw_response": content_text[:2000] if content_text else None,
                    },
                    paper_id=paper_id,
                )
            except Exception as e:
                last_error = e
                self.logger.warning(f"OpenRouter error for {paper_id}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return APIResponse(success=False, data=None, error=str(e), paper_id=paper_id)

        self.logger.error(f"OpenRouter API failed after {self.max_retries} retries for {paper_id}: {last_error}")
        return APIResponse(success=False, data=None, error=str(last_error), paper_id=paper_id)


# Alias for backward compatibility
HuggingFaceClient = OpenRouterClient
