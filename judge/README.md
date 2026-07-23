# Why-Python judge

An LLM judge that classifies **why models choose Python**.
Benchmark prompts never specify a language and are strictly single-turn (no prior conversation, no existing code), yet models default to Python constantly.
For every reasoning trace whose final response used Python, the judge reads the trace and assigns one label explaining why.


## Label taxonomy

Defined in [`taxonomy.py`](taxonomy.py), the single source of truth - `prompts.py` builds the judge's system prompt and output schema directly from it, so they can't drift apart.
Priority order (the judge picks the first label that applies):

| Label | Meaning |
| --- | --- |
| `phantom_python_evidence` | Justifies Python with fabricated context: a prior conversation/existing code, or an invented instruction in the current prompt (e.g. "the problem says to use Python"). Every experiment is single-turn with no real prior context, so any such claim is fabricated by construction. |
| `language_mismatch` | The trace settles on a different language and never revisits that decision, but the final response is Python anyway. |
| `python_for_ease` | Python itself — not a specific library — is chosen for being easy, simple, readable, familiar, or quick to prototype, and that's the deciding reason, even if another language was briefly considered. |
| `automatic_python` | Python is assumed with no real language decision: no language ever discussed, no reason given, or only a library-level justification within an already-assumed Python. |
| `unclear_other` | Doesn't fit the above — a garbled trace, or a genuine but different justification. Not used just because a trace skips language discussion (that's `automatic_python`). |


## Scope

Every implementation sample whose final response was Python and which has a complete, raw, reasoning trace.


## Process

Run from the repository root with the virtual environment active.

```shell
# 1. build the frozen gold sample (15 traces/model, deterministic)
#    -> judge/data/gold_unlabelled.jsonl
python -m judge.build_gold

# 2. hand-label it
#    -> judge/data/gold_labelled.jsonl

# 3. pilot both candidate judge models on the gold set
python -m judge.run --judge-model gpt-5-mini --gold-only
python -m judge.run --judge-model gpt-5.4-mini --gold-only

# 4. score both pilots against the gold labels
#    -> judge/data/validation_report.json
python -m judge.validate

# 5. full run with whichever judge scores best
python -m judge.run --judge-model <winner> --model all

# 6. aggregate verdicts -> output/<model>/def-judge-analysis.json
python -m judge.summarise
```

Requests use the OpenAI Batch API, and every step is resumable: `run` skips already-judged traces and picks up in-flight batches.


## Files

```
judge/
  taxonomy.py     — labels, schemas
  traces.py       — load python-choosing traces from output/
  prompts.py      — frozen judge prompt + request assembly
  build_gold.py   — deterministic gold sampling
  run.py          — batch submit / poll / collect
  summarise.py    — per-model aggregation
  validate.py     — scoring vs the gold labels
  data/
    gold_unlabelled.jsonl          — frozen blind sample (committed)
    gold_labelled.jsonl            — hand labels (committed)
    gold_judgements_<judge>.jsonl  — pilot verdicts
    validation_report.json         — precision/recall/F1, kappa, disagreements

output/<model>/
  def-judge-results.jsonl   — one verdict per judged trace
  def-judge-batches.json    — batch ids for resumability
  def-judge-analysis.json   — summary counts, rates, examples
```

`def-judge-analysis.json` reports `python_responses` (samples in judge scope), `judged` (how many have a verdict), and the count per label.
