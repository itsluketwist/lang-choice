"""Context-anchor hallucination detector for reasoning traces and final responses.

A context-anchor hallucination occurs when the model's output cites a prior
conversational cue (e.g. "the previous example was in Python") that is absent
from the actual prompt history. This module implements a heuristic detector and
a batch analysis function for collections of generation results.
"""

from typing import Literal

from pydantic import BaseModel

from src.generation.schemas import GenerationResult


# valid output labels for the hallucination detector
AnchorLabel = Literal[
    "no_anchor",
    "real_python_anchor",
    "real_non_python_anchor",
    "phantom_python_anchor",
    "phantom_non_python_anchor",
    "ambiguous_anchor",
]

# keywords suggesting the model is citing prior conversational context
_PRIOR_CONTEXT_TERMS = [
    "previous",
    "earlier",
    "above",
    "initial",
    "already",
    "context",
    "example",
    "given code",
    "existing code",
    "prior",
    "conversation",
    "the last",
    "before",
]

# keywords indicating a Python reference in the model's output
_PYTHON_TERMS = [
    "python",
    ".py",
    "pip ",
    "pandas",
    "flask",
    "django",
    "fastapi",
    "numpy",
    "scipy",
    "asyncio",
    "def ",
    "import os",
    "import sys",
]

# causal/justificatory phrases linking a prior cue to the current language choice
_CAUSAL_TERMS = [
    "so i",
    "therefore",
    "since",
    "because",
    "likely expected",
    "continue with",
    "stick with",
    "use the same",
    "consistent with",
    "following the",
    "as in the",
]


class AnchorResult(BaseModel):
    """The result of running hallucination analysis on a single (prompt, sample) pair."""

    id: str  # BenchmarkPrompt.id
    project_id: str
    task_type: str
    sample_index: int
    context_condition: str
    anchor_label: str
    anchor_rationale: str


class AnalysisSummary(BaseModel):
    """Aggregate counts and rates across all anchor analysis results."""

    total: int
    no_anchor: int
    phantom_python_anchor: int
    phantom_non_python_anchor: int
    real_python_anchor: int
    real_non_python_anchor: int
    ambiguous_anchor: int
    # fraction of responses that showed any phantom anchor behaviour
    phantom_rate: float
    # response and reasoning coverage by task type
    implementation_with_response: int
    implementation_with_reasoning: int
    recommendation_with_response: int
    recommendation_with_reasoning: int


class AnalysisResults(BaseModel):
    """Full output of the hallucination analysis step.

    Mirrors the structure of BenchmarkResults: a summary of aggregate statistics
    and a flat list of per-response results.
    """

    summary: AnalysisSummary
    results: list[AnchorResult]


def analyse_responses(
    results: list[GenerationResult],
) -> AnalysisResults:
    """Run context-anchor hallucination detection on all samples in a list of GenerationResults.

    Iterates over each sample within each GenerationResult and runs detect_context_anchor.
    Returns an AnalysisResults with per-response labels and aggregate summary statistics.
    """
    anchor_results = []
    for result in results:
        for sample_index, response in enumerate(result.responses):
            reasoning = result.reasoning[sample_index] if result.reasoning else None
            label, rationale = detect_context_anchor(
                reasoning=reasoning,
                response=response,
                prior_messages=result.prompt_messages,
            )
            anchor_results.append(
                AnchorResult(
                    id=result.id,
                    project_id=result.project_id,
                    task_type=result.task_type,
                    sample_index=sample_index,
                    context_condition=result.context_condition,
                    anchor_label=label,
                    anchor_rationale=rationale,
                )
            )

    # count each label across all results
    label_counts: dict[str, int] = {
        "no_anchor": 0,
        "phantom_python_anchor": 0,
        "phantom_non_python_anchor": 0,
        "real_python_anchor": 0,
        "real_non_python_anchor": 0,
        "ambiguous_anchor": 0,
    }
    for r in anchor_results:
        if r.anchor_label in label_counts:
            label_counts[r.anchor_label] += 1

    total = len(anchor_results)
    phantom_count = (
        label_counts["phantom_python_anchor"]
        + label_counts["phantom_non_python_anchor"]
    )

    # count valid responses and exposed reasoning traces per task type
    impl_with_response = sum(
        1
        for r in results
        if r.task_type == "implementation"
        for resp in r.responses
        if resp.strip()
    )
    impl_with_reasoning = sum(
        1
        for r in results
        if r.task_type == "implementation"
        for reasoning in r.reasoning
        if reasoning
    )
    rec_with_response = sum(
        1
        for r in results
        if r.task_type == "recommendation"
        for resp in r.responses
        if resp.strip()
    )
    rec_with_reasoning = sum(
        1
        for r in results
        if r.task_type == "recommendation"
        for reasoning in r.reasoning
        if reasoning
    )

    return AnalysisResults(
        summary=AnalysisSummary(
            total=total,
            phantom_rate=phantom_count / total if total else 0.0,
            implementation_with_response=impl_with_response,
            implementation_with_reasoning=impl_with_reasoning,
            recommendation_with_response=rec_with_response,
            recommendation_with_reasoning=rec_with_reasoning,
            **label_counts,
        ),
        results=anchor_results,
    )


def prior_context_contains_language(
    messages: list[dict[str, str]],
    language: str,
) -> bool:
    """Check whether any prior assistant message contains a reference to the given language.

    Searches for the language name (case-insensitive) and common language markers
    in all assistant messages that precede the final user turn.
    Returns True if evidence of the language is found in prior context.
    """
    lang_lower = language.lower()
    # build language-specific markers (for Python: .py, pip, common libs)
    markers = {lang_lower}
    if lang_lower == "python":
        markers.update(["pip", ".py", "import ", "def ", "class "])
    elif lang_lower in ("javascript", "js"):
        markers.update(["npm", "const ", "function ", ".js", "require("])
    elif lang_lower in ("typescript", "ts"):
        markers.update(["npm", "interface ", ": string", ".ts", "tsc"])
    elif lang_lower == "rust":
        markers.update(["cargo", "fn main", "use std::", ".rs"])
    elif lang_lower == "go":
        markers.update(["go.mod", "package main", 'import "'])

    # check assistant messages that precede the last user turn (i.e. prior context)
    prior_assistant_messages = [
        m["content"]
        for m in messages[:-1]  # exclude the final user message
        if m.get("role") == "assistant"
    ]

    for content in prior_assistant_messages:
        content_lower = content.lower()
        if any(marker in content_lower for marker in markers):
            return True

    return False


def detect_context_anchor(
    reasoning: str | None,
    response: str,
    prior_messages: list[dict[str, str]],
) -> tuple[AnchorLabel, str]:
    """Heuristic detector for context-anchor hallucinations.

    Searches the reasoning trace and response for combinations of:
    - prior-context reference terms (e.g. "previous", "earlier")
    - language cue terms (Python-specific or general)
    - causal/justificatory language (e.g. "so", "therefore", "stick with")

    Then cross-checks whether the claimed anchor actually exists in prior_messages.

    Returns (label, rationale) where label is one of the AnchorLabel literals and
    rationale is a human-readable explanation of the classification decision.
    """
    # combine reasoning and response into a single searchable text
    combined = " ".join(filter(None, [reasoning, response])).lower()

    # check for prior-context reference terms
    found_prior_ref = [t for t in _PRIOR_CONTEXT_TERMS if t in combined]

    # check for python-specific terms in output
    found_python_terms = [t for t in _PYTHON_TERMS if t in combined]

    # check for causal/justificatory language
    found_causal = [t for t in _CAUSAL_TERMS if t in combined]

    # no prior-context reference at all — no anchor behaviour to classify
    if not found_prior_ref:
        return "no_anchor", "No prior-context reference terms found in model output."

    # determine if the model is making a language-specific claim
    claims_python = bool(found_python_terms)

    # check what is actually in the prior context
    python_in_context = prior_context_contains_language(prior_messages, "python")
    has_any_prior_context = any(
        m.get("role") == "assistant" for m in prior_messages[:-1]
    )

    if not has_any_prior_context:
        # model claims a prior context that does not exist at all
        if claims_python:
            return (
                "phantom_python_anchor",
                f"Model references prior context (terms: {found_prior_ref}) "
                f"with Python cues (terms: {found_python_terms}) "
                f"but no prior assistant messages exist in prompt history.",
            )
        if found_prior_ref:
            return (
                "ambiguous_anchor",
                f"Model references prior context (terms: {found_prior_ref}) "
                f"but no prior assistant messages exist; language claim unclear.",
            )

    if claims_python and python_in_context:
        return (
            "real_python_anchor",
            f"Model references Python context (terms: {found_python_terms}) "
            f"and Python is present in the actual prior context.",
        )

    if claims_python and not python_in_context:
        causal_note = f" Causal language: {found_causal}." if found_causal else ""
        return (
            "phantom_python_anchor",
            f"Model references Python (terms: {found_python_terms}) citing prior context "
            f"(terms: {found_prior_ref}), but Python is NOT present in prior messages.{causal_note}",
        )

    if found_prior_ref and not claims_python:
        # references prior context but no python-specific terms — could be any language
        return (
            "ambiguous_anchor",
            f"Model references prior context (terms: {found_prior_ref}) "
            f"but without clear Python-specific language cues.",
        )

    return (
        "no_anchor",
        "Prior-context terms found but no language claim could be extracted.",
    )
