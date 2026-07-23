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
TRACES_PER_MODEL = 15


def sample_gold_traces(
    models: list[str] | None = None,
    traces_per_model: int = TRACES_PER_MODEL,
    seed: int = SEED,
) -> list[Trace]:
    """Sample gold traces: exactly N per model, spread over distinct prompts.

    Traces are taken one per prompt in rounds, so prompts are only revisited
    when a model has fewer than N distinct python-choosing prompts. Sampling
    is fully deterministic for a given seed and set of output files.
    Returns the sampled Traces, ordered by model then prompt id.
    """
    rng = random.Random(seed)
    if models is None:
        models = list_reasoning_models()

    gold = []
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

        gold.extend(sorted(picked, key=lambda t: (t.id, t.sample_index)))
    return gold


def main() -> None:
    """Sample the gold set and save it for blind labelling."""
    models = list_reasoning_models()
    log(f"Sampling {TRACES_PER_MODEL} traces from each of {len(models)} models...")
    gold = sample_gold_traces(models=models)
    save_jsonl(records=gold, path=GOLD_UNLABELLED_PATH)
    log(f"Saved {len(gold)} traces to {GOLD_UNLABELLED_PATH}")


if __name__ == "__main__":
    main()
