"""Generate model responses for a list of benchmark prompts."""

from codechoicebench.schema import BenchmarkPrompt
from tqdm import tqdm

from src.generation.context import build_context_messages
from src.generation.openrouter import OpenRouterRunner
from src.generation.schemas import (
    GenerationResult,
    InferenceConfig,
    ModelConfig,
    ResponseData,
)
from src.utils.log import log


def generate_responses(
    prompts: list[BenchmarkPrompt],
    model_config: ModelConfig,
    inference_config: InferenceConfig,
    n_samples: int,
    task_type: str,
    context_condition: str = "none",
) -> list[GenerationResult]:
    """Generate model responses for all prompts, returning one GenerationResult per prompt.

    Creates an OpenRouterRunner internally. For each prompt, calls the runner once per
    sample and aggregates into a single GenerationResult with list fields, so shared
    data like prompt_messages is stored only once per prompt.

    For greedy decoding (temperature=0.0) n_samples is capped at 1 since outputs
    are deterministic.

    Returns a list of GenerationResults ordered by prompt.
    """
    # greedy decoding is deterministic — generating more than one sample is wasteful
    n_samples = 1 if inference_config.temperature == 0.0 else n_samples

    log(f"    Generating {len(prompts)} prompts × {n_samples} samples")

    runner = OpenRouterRunner()
    results: list[GenerationResult] = []

    for prompt in tqdm(prompts, desc=task_type, unit="prompt"):
        # use the first preferred language for context template substitution
        preferred_language = (
            prompt.preferred_languages[0] if prompt.preferred_languages else "Go"
        )
        context_messages = build_context_messages(
            condition=context_condition,
            preferred_language=preferred_language,
        )
        messages = context_messages + [{"role": "user", "content": prompt.prompt}]

        # collect all samples for this prompt
        calls: list[ResponseData] = []
        for _ in range(n_samples):
            calls.append(
                runner.generate(
                    messages=messages,
                    model_config=model_config,
                    inference_config=inference_config,
                )
            )

        results.append(
            GenerationResult(
                id=prompt.id,
                project_id=prompt.project_id,
                task_type=task_type,
                context_condition=context_condition,
                prompt_messages=messages,
                responses=[c.response for c in calls],
                reasoning=[c.reasoning for c in calls],
                reasoning_tokens=[c.reasoning_tokens for c in calls],
                response_tokens=[c.response_tokens for c in calls],
                warnings=[c.warnings for c in calls],
            )
        )

    return results
