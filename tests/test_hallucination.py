"""Tests for the context-anchor hallucination detector."""

from src.analysis.hallucination import (
    detect_context_anchor,
    prior_context_contains_language,
)


# message lists for different context conditions
_NO_PRIOR_CONTEXT = [{"role": "user", "content": "Write code for a trading platform."}]

_PYTHON_PRIOR_CONTEXT = [
    {"role": "user", "content": "Show me a file reading example."},
    {
        "role": "assistant",
        "content": "```python\nwith open('f.txt') as f:\n    data = f.read()\n```",
    },
    {"role": "user", "content": "Write code for a trading platform."},
]

_NON_PYTHON_PRIOR_CONTEXT = [
    {"role": "user", "content": "Show me a file reading example."},
    {
        "role": "assistant",
        "content": '```rust\nuse std::fs;\nlet data = fs::read_to_string("f.txt").unwrap();\n```',
    },
    {"role": "user", "content": "Write code for a trading platform."},
]

_NEUTRAL_PRIOR_CONTEXT = [
    {"role": "user", "content": "Can you help me set up a dev environment?"},
    {"role": "assistant", "content": "Sure! What kind of project are you working on?"},
    {"role": "user", "content": "Write code for a trading platform."},
]


class TestPriorContextContainsLanguage:
    """Test the prior context language audit function."""

    def test_python_in_python_context(self) -> None:
        """Should detect Python in a context with Python code."""
        assert prior_context_contains_language(_PYTHON_PRIOR_CONTEXT, "python") is True

    def test_python_not_in_rust_context(self) -> None:
        """Should not detect Python when only Rust is in prior context."""
        assert (
            prior_context_contains_language(_NON_PYTHON_PRIOR_CONTEXT, "python")
            is False
        )

    def test_no_prior_context(self) -> None:
        """Should return False when there are no prior assistant messages."""
        assert prior_context_contains_language(_NO_PRIOR_CONTEXT, "python") is False

    def test_rust_in_rust_context(self) -> None:
        """Should detect Rust in a context with Rust code."""
        assert (
            prior_context_contains_language(_NON_PYTHON_PRIOR_CONTEXT, "rust") is True
        )


class TestDetectContextAnchor:
    """Test the full hallucination detector."""

    def test_no_anchor_clean_response(self) -> None:
        """A response with no prior-context references should be labelled no_anchor."""
        label, rationale = detect_context_anchor(
            reasoning="I need to use C++ for this trading platform due to latency.",
            response="```cpp\n// HFT code here\n```",
            prior_messages=_NO_PRIOR_CONTEXT,
        )
        assert label == "no_anchor"

    def test_phantom_python_anchor_no_prior_context(self) -> None:
        """Claiming Python context when none exists should be a phantom anchor."""
        label, rationale = detect_context_anchor(
            reasoning=(
                "The previous example was in Python, so I'll continue with Python. "
                "Since the earlier context was Python-based, Python is probably expected."
            ),
            response="```python\nprint('hello')\n```",
            prior_messages=_NO_PRIOR_CONTEXT,
        )
        assert label == "phantom_python_anchor"
        assert "phantom" in rationale.lower() or "prior" in rationale.lower()

    def test_real_python_anchor(self) -> None:
        """Citing Python context when Python is actually in prior messages is a real anchor."""
        label, rationale = detect_context_anchor(
            reasoning=(
                "The previous example was in Python, so I'll continue with Python "
                "since the earlier context was Python-based."
            ),
            response="```python\nprint('hello')\n```",
            prior_messages=_PYTHON_PRIOR_CONTEXT,
        )
        assert label == "real_python_anchor"

    def test_phantom_anchor_neutral_context(self) -> None:
        """Claiming Python context when prior context is neutral is still a phantom anchor."""
        label, rationale = detect_context_anchor(
            reasoning=(
                "Since the earlier context was Python-based, I'll use Python. "
                "The previous example was in Python."
            ),
            response="```python\nprint('hello')\n```",
            prior_messages=_NEUTRAL_PRIOR_CONTEXT,
        )
        # neutral context has no Python, so claiming Python anchor is phantom
        assert label == "phantom_python_anchor"

    def test_no_anchor_with_python_prior(self) -> None:
        """If the model uses Python but doesn't cite prior context, it's no_anchor."""
        label, _ = detect_context_anchor(
            reasoning="Python is a good general purpose language for this task.",
            response="```python\nprint('result')\n```",
            prior_messages=_PYTHON_PRIOR_CONTEXT,
        )
        # no prior-context reference terms → no_anchor regardless of language
        assert label == "no_anchor"
