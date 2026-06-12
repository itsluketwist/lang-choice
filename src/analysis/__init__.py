"""Analysis modules for the experiment pipeline — not part of the benchmark library."""

from src.analysis.hallucination import (
    AnalysisResults,
    AnalysisSummary,
    AnchorExample,
    AnchorResult,
    analyse_responses,
    detect_context_anchor,
    prior_context_contains_language,
)


__all__ = [
    "AnalysisResults",
    "AnalysisSummary",
    "AnchorExample",
    "AnchorResult",
    "analyse_responses",
    "detect_context_anchor",
    "prior_context_contains_language",
]
