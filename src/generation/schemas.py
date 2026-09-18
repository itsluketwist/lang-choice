"""Data schemas for the generation pipeline."""

from typing import Any, Literal

from pydantic import BaseModel


# run modes for the experiment pipeline:
#   default:   generate only if the output file does not exist (else load it as-is)
#   overwrite: ignore any existing results and regenerate everything from scratch
#   update:    top up existing results until each prompt has the required number
#              of valid (non-empty) responses
#   evaluate:  skip generation entirely and run evaluation/analysis on existing files
Mode = Literal["default", "overwrite", "update", "evaluate"]


class ModelConfig(BaseModel):
    """Configuration for a single model, loaded from config/models.yaml."""

    name: str
    provider: str
    model_path: str
    defaults: dict[str, Any] = {}
    enable_reasoning: bool = False


class InferenceConfig(BaseModel):
    """Sampling parameters for a generation run, loaded from config/inference.yaml."""

    name: str
    temperature: float | None
    top_p: float | None
    samples: dict[str, int]  # {"implementation": N, "recommendation": N}
    seed: int
    max_workers: int = 10
    # output file prefix, e.g. "t03" — defaults to the first 3 chars of the name,
    # which is needed when several presets share those chars (temp-0.3, temp-0.6)
    prefix: str | None = None


class ResponseData(BaseModel):
    """Data returned from a single API call."""

    response: str
    reasoning: str | None = None
    warnings: list[str] = []


class GenerationResult(BaseModel):
    """All responses for a single prompt across all samples."""

    id: str  # BenchmarkPrompt.id — "{project_id}__{prompt_variant}"
    project_id: str  # for joining with analysis results
    task_type: str  # "implementation" | "recommendation"

    responses: list[str]  # one per sample
    reasoning: list[str | None]  # one per sample (None if model has no reasoning)

    context_condition: str = "none"
    prompt_messages: list[dict[str, str]] = []  # stored once — same for all samples
    warnings: list[list[str]] = []

    # two-stage runs only: the recommendation prompt whose responses were replayed
    # as the first turn. responses[i] answers recommendation sample i.
    stage_one_id: str | None = None
