"""Data schemas for the generation pipeline."""

from typing import Any

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for a single model, loaded from config/models.yaml.

    provider maps to an llm_cgr PROVIDER_MAP key (e.g. "anthropic", "deepseek").
    defaults hold per-model sampling params used when inference config sets nulls.
    enable_reasoning controls whether the model's chain-of-thought trace is captured.
    """

    name: str
    provider: str
    model_path: str
    defaults: dict[str, Any] = {}
    enable_reasoning: bool = False


class InferenceConfig(BaseModel):
    """Sampling parameters for a generation run, loaded from config/inference.yaml.

    temperature and top_p may be None to fall back to per-model defaults.
    seed ensures reproducibility for greedy / low-temperature runs.
    max_workers controls how many API calls are in-flight concurrently.
    """

    name: str
    temperature: float | None
    top_p: float | None
    samples: dict[str, int]  # {"implementation": N, "recommendation": N}
    seed: int
    max_workers: int = 10


class ResponseData(BaseModel):
    """Data returned from a single API call.

    Produced by ModelRunner.generate(). generate_responses() aggregates
    multiple ResponseData objects into a single GenerationResult per prompt.
    """

    response: str
    reasoning: str | None = None
    warnings: list[str] = []


class GenerationResult(BaseModel):
    """All responses for a single prompt across all samples.

    Stores responses as lists (one entry per sample) so prompt_messages and other
    shared fields are not repeated. The id and responses fields match the format
    expected by langchoicebench.evaluate_benchmark().
    """

    id: str  # BenchmarkPrompt.id — "{project_id}__{prompt_variant}"
    project_id: str  # for joining with analysis results
    task_type: str  # "implementation" | "recommendation"
    context_condition: str
    prompt_messages: list[dict[str, str]]  # stored once — same for all samples
    responses: list[str]  # one per sample
    reasoning: list[str | None]  # one per sample (None if model has no reasoning)
    warnings: list[list[str]] = []
