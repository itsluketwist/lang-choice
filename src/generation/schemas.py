"""Data schemas for the generation pipeline."""

from typing import Any

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for a single model, loaded from config/models.yaml.

    reasoning_format describes how the model exposes its reasoning trace.
    defaults hold per-model sampling params used when inference config sets nulls.
    extra_params are passed as extra_body to the API (e.g. enable_thinking for Qwen3).
    """

    name: str
    model_path: str
    defaults: dict[str, Any] = {}
    extra_params: dict[str, Any] = {}


class InferenceConfig(BaseModel):
    """Sampling parameters for a generation run, loaded from config/inference.yaml.

    temperature and top_p may be None to fall back to per-model defaults.
    seed ensures reproducibility for greedy / low-temperature runs.
    """

    name: str
    temperature: float | None
    top_p: float | None
    samples: dict[str, int]  # {"implementation": N, "recommendation": N}
    seed: int


class ResponseData(BaseModel):
    """Data returned from a single API call.

    Produced by OpenRouterRunner.generate(). generate_responses() aggregates
    multiple ResponseData objects into a single GenerationResult per prompt.
    """

    response: str
    reasoning: str | None = None
    reasoning_tokens: int | None = None
    response_tokens: int | None = None
    warnings: list[str] = []


class GenerationResult(BaseModel):
    """All responses for a single prompt across all samples.

    Stores responses as lists (one entry per sample) so prompt_messages and other
    shared fields are not repeated. The id and responses fields match the format
    expected by codechoicebench.evaluate_benchmark().
    """

    id: str  # BenchmarkPrompt.id — "{project_id}__{prompt_variant}"
    project_id: str  # for joining with analysis results
    task_type: str  # "implementation" | "recommendation"
    context_condition: str
    prompt_messages: list[dict[str, str]]  # stored once — same for all samples
    responses: list[str]  # one per sample
    reasoning: list[str | None]  # one per sample (None if model has no reasoning)
    reasoning_tokens: list[int | None] = []
    response_tokens: list[int | None] = []
    warnings: list[list[str]] = []
