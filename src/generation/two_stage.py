"""Two-stage deliberation: replay a saved recommendation, then ask for the code.

The first turn is a recommendation prompt the model has already answered in the
single-turn run, replayed from its saved responses, so only the follow-up turn is
generated here. Responses stay positional: responses[i] answers recommendation
sample i, which keeps each conversation pair exact.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, NamedTuple, cast

from llm_cgr import get_llm
from tqdm import tqdm

from src.generation.generate import generate_with_retries, resolve_sampling
from src.generation.schemas import GenerationResult, InferenceConfig, ModelConfig
from src.utils.log import log


# the follow-up is the paired implementation prompt with a short conversational
# lead-in, e.g. "Great, now write code for ..."
FOLLOW_UP_PREFIX = "Great, now "


def build_follow_up(implementation_prompt: str) -> str:
    """Build the second turn from the project's implementation prompt.

    Returns the implementation prompt with the conversational lead-in attached.
    """
    return f"{FOLLOW_UP_PREFIX}{implementation_prompt[0].lower()}{implementation_prompt[1:]}"


# each recommendation variant is paired with an implementation variant of the same
# project, which gives the follow-up prompt and the id the answer is scored under
# (so it is scored as code, not by looking for <language> tags)
VARIANT_PAIRS: dict[str, str] = {
    "what_language": "write",
    "best_language": "create",
    "choose_explain": "generate",
}


class TwoStageTask(NamedTuple):
    """One recommendation prompt plus the saved answers to replay as the first turn."""

    recommendation_id: str
    project_id: str
    prompt: str
    # the second turn, built from the paired implementation prompt
    follow_up: str
    # the model's own earlier recommendations, one per sample
    stage_one_responses: list[str]
    # implementation turns generated so far; blank entries still need generating
    responses: list[str]
    reasoning: list[str | None]


def implementation_id(recommendation_id: str) -> str:
    """Map a recommendation prompt id onto the paired implementation prompt id.

    Returns the "{project_id}__{variant}" id used to score the generated code.
    """
    project_id, _, variant = recommendation_id.rpartition("__")
    if variant not in VARIANT_PAIRS:
        raise ValueError(
            f"Recommendation id '{recommendation_id}' has no paired implementation "
            f"variant (known variants: {sorted(VARIANT_PAIRS)})",
        )
    return f"{project_id}__{VARIANT_PAIRS[variant]}"


def generate_two_stage_responses(
    tasks: list[TwoStageTask],
    model_config: ModelConfig,
    inference_config: InferenceConfig,
    on_result: Callable[[GenerationResult], None],
) -> None:
    """Generate the follow-up implementation turn for each saved recommendation sample.

    Calls on_result with each GenerationResult as it completes, so the caller can
    persist results to disk incrementally.
    """
    temperature, top_p, max_tokens = resolve_sampling(model_config, inference_config)

    total_samples = sum(sum(1 for r in task.responses if not r) for task in tasks)
    log(
        f"    Generating {len(tasks)} conversations × {total_samples} total samples"
        f" ({inference_config.max_workers} concurrent)",
    )

    def _conversation(
        prompt: str,
        recommendation: str,
        follow_up: str,
    ) -> tuple[str, str | None]:
        """Replay one recommendation exchange, then ask for the implementation."""
        # a fresh client per conversation: llm_cgr stores chat history on the client,
        # so sharing one across threads would interleave unrelated conversations
        llm = get_llm(
            model=model_config.model_path,
            provider=model_config.provider,
            enable_reasoning=model_config.enable_reasoning,
        )

        def _call() -> tuple[str | None, str | None]:
            """Seed the earlier turns into the history, then send the follow-up."""
            # llm_cgr has no public way to prefill an assistant turn, so the history
            # is seeded directly — _build_message keeps each provider's own format
            # (e.g. google needs the "model" role rather than "assistant")
            client = cast(Any, llm)
            client._history = [
                client._build_message(role="user", content=prompt),
                client._build_message(role="assistant", content=recommendation),
            ]
            generation = client.chat(
                user=follow_up,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            if model_config.enable_reasoning:
                return cast(tuple[str | None, str | None], generation)
            return cast(str | None, generation), None

        return generate_with_retries(_call)

    def _call(task: TwoStageTask) -> GenerationResult:
        """Fill in the missing implementation turns for one recommendation prompt."""
        responses = list(task.responses)
        reasoning = list(task.reasoning)

        for index, recommendation in enumerate(task.stage_one_responses):
            if responses[index]:
                continue  # already generated by an earlier run
            if not recommendation:
                continue  # nothing to replay, leave the slot blank
            response, thinking = _conversation(
                task.prompt,
                recommendation,
                task.follow_up,
            )
            responses[index] = response
            reasoning[index] = thinking

        # warn if any samples came back blank — visible in the terminal and log
        empty = sum(1 for r in responses if not r)
        if empty:
            log(
                f"\n    [WARNING] {task.recommendation_id}: "
                f"{empty}/{len(responses)} samples returned empty"
            )

        return GenerationResult(
            id=implementation_id(task.recommendation_id),
            project_id=task.project_id,
            task_type="implementation",
            context_condition="two_stage",
            stage_one_id=task.recommendation_id,
            # the assistant turn between these two varies by sample, so only the
            # user turns are stored here
            prompt_messages=[
                {"role": "user", "content": task.prompt},
                {"role": "user", "content": task.follow_up},
            ],
            responses=responses,
            reasoning=reasoning,
        )

    with ThreadPoolExecutor(max_workers=inference_config.max_workers) as executor:
        futures = {executor.submit(_call, task): task for task in tasks}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="two-stage",
            unit="conversation",
        ):
            on_result(future.result())
