"""Build the deterministic gold-set sample for hand labelling.

Usage:
    python -m judge.build_gold
"""

import random
from collections import defaultdict
from pathlib import Path

from judge.traces import Trace, list_reasoning_models, load_python_response_traces
from src.utils.io import save_jsonl
from src.utils.log import log


GOLD_UNLABELLED_PATH = Path("judge/data/gold_unlabelled.jsonl")
GOLD_LABELLED_PATH = Path("judge/data/gold_labelled.jsonl")

SEED = 42
TRACES_PER_MODEL = 20

# how many of each model's traces go to "selection" (compare judges, pick a
# winner) vs "validation" (held-out check on the winner)
SELECTION_SPLIT_SIZE = 10


def sample_gold_traces(
    models: list[str] | None = None,
    traces_per_model: int = TRACES_PER_MODEL,
    selection_split_size: int = SELECTION_SPLIT_SIZE,
    seed: int = SEED,
) -> list[dict]:
    """Sample N gold traces per model, spread over distinct prompts.

    Prompts are drawn one at a time in shuffled rounds, so a prompt only
    repeats once every other qualifying prompt has been used. Each model's
    traces are then split into "selection" and "validation" sets. Sampling
    is deterministic for a given seed and set of output files.
    Returns the sampled records (Trace fields plus a "split" key), ordered by
    model then prompt id.
    """
    rng = random.Random(seed)
    if models is None:
        models = list_reasoning_models()

    gold: list[dict] = []
    for model in models:
        by_prompt: dict[str, list[Trace]] = defaultdict(list)
        for trace in load_python_response_traces(model):
            by_prompt[trace.id].append(trace)

        # shuffled prompt order and per-prompt trace order, both deterministic
        prompt_ids = sorted(by_prompt)
        rng.shuffle(prompt_ids)
        remaining = {
            prompt_id: rng.sample(by_prompt[prompt_id], len(by_prompt[prompt_id]))
            for prompt_id in prompt_ids
        }

        picked: list[Trace] = []
        while len(picked) < traces_per_model and any(remaining.values()):
            for prompt_id in prompt_ids:
                if remaining[prompt_id] and len(picked) < traces_per_model:
                    picked.append(remaining[prompt_id].pop())

        # split into selection/validation, independent of prompt order above
        keys = [trace.key for trace in picked]
        rng.shuffle(keys)
        split_by_key = {
            key: "selection" if i < selection_split_size else "validation"
            for i, key in enumerate(keys)
        }

        for trace in sorted(picked, key=lambda t: (t.id, t.sample_index)):
            gold.append({**trace.model_dump(), "split": split_by_key[trace.key]})

    return gold


def main() -> None:
    """Sample the gold set and save it for blind labelling."""
    models = list_reasoning_models()
    n_validation = TRACES_PER_MODEL - SELECTION_SPLIT_SIZE
    log(
        f"Sampling {TRACES_PER_MODEL} traces ({SELECTION_SPLIT_SIZE} selection / "
        f"{n_validation} validation) from each of {len(models)} models..."
    )
    gold = sample_gold_traces(models=models)
    save_jsonl(records=gold, path=GOLD_UNLABELLED_PATH)
    log(f"Saved {len(gold)} traces to {GOLD_UNLABELLED_PATH}")


if __name__ == "__main__":
    main()
