"""Label taxonomy and data schemas shared by the judge pipeline."""

from typing import Literal

from pydantic import BaseModel


# the experiments are single-turn: one user message that never specifies a
# language, no prior conversation, and no existing code. every trace in scope
# belongs to a sample whose final response was written in Python. the judge
# classifies WHY that happened. order is priority order: prompts.py builds
# the judge prompt and the structured-output schema directly from this.
JudgeLabel = Literal[
    "phantom_python_evidence",
    "language_mismatch",
    "python_for_ease",
    "automatic_python",
    "unclear_other",
]

# canonical definition per label — used for the judge prompt (prompts.py),
# and as button tooltips in the labelling ui (labelling.py). keep each entry
# self-contained: it is the only place a given label is explained.
LABEL_DESCRIPTIONS: dict[str, str] = {
    "phantom_python_evidence": (
        "The trace justifies choosing Python with a fabricated instruction or "
        "context that was never given — either a nonexistent prior "
        "conversation/example/existing code/system prompt (e.g. 'the previous "
        "examples are Python', 'the example in the response uses Python', "
        "'the initial example in the system prompt was Python'), or a "
        "nonexistent requirement invented as part of the CURRENT prompt (e.g. "
        "'the problem says to use Python', 'we are to write code in Python', "
        "'we are restricted to Python') when the real prompt never specified "
        "or restricted the language. There is no system prompt shown to the "
        "model beyond what you see above — a claim about one is fabricated "
        "too. Treating Python as a given constraint counts here even without "
        "the word 'says'."
    ),
    "language_mismatch": (
        "The trace explicitly settles on a language OTHER than Python (e.g. "
        "'We'll write the code in C, which is common for microcontroller "
        "firmware'), but the final response was written in Python anyway — the "
        "reasoning's own decision does not match the output."
    ),
    "python_for_ease": (
        "Python ITSELF (the language, or its ecosystem as a whole) is "
        "explicitly called easy, simple, readable, familiar, quick to write, "
        "or good for prototyping, and that is the DECIDING reason — even if "
        "another language was named and dismissed, and even if the statement "
        "is a single early sentence (e.g. 'I can choose Python due "
        "readability') followed by unrelated hedging later in the trace. "
        "Praising a specific LIBRARY's simplicity does not count on its own. "
        "Calling Python 'common' or 'popular' is not ease either — "
        "commonality is not ease, it's a claim about convention; both fall "
        "under automatic_python."
    ),
    "automatic_python": (
        "No real cross-language consideration, or a language decision never "
        "actually made: the trace never mentions or discusses a programming "
        "language at all; it assumes Python immediately with no stated "
        "reason; it plans multiple languages with no clear reason for the "
        "Python part; it praises a specific Python LIBRARY without ever "
        "framing Python itself as chosen for that reason — a library choice "
        "made inside an already-assumed language is not a language choice; "
        "or it justifies Python only by calling it common/popular, with no "
        "separate ease reason. Discussing Python-specific implementation "
        "details without naming another language is likewise this label."
    ),
    "unclear_other": (
        "None of the other labels fit — the trace is garbled or Python is "
        "justified some other way not covered above. Not for traces that "
        "simply never mention a language (that is automatic_python)."
    ),
}

# the label that counts as phantom evidence
PHANTOM_LABEL = "phantom_python_evidence"


class JudgeVerdict(BaseModel):
    """The judge's classification of a single reasoning trace."""

    label: JudgeLabel
    evidence_quotes: list[str]
    confidence: Literal["high", "medium", "low"]
    rationale: str


class TraceJudgement(BaseModel):
    """A judge verdict tied back to the (model, prompt, sample) it was made on."""

    model: str
    id: str
    project_id: str
    sample_index: int
    judge_model: str
    verdict: JudgeVerdict
