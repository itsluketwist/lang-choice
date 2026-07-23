"""Tests for the why-python LLM-judge pipeline."""

import json
from pathlib import Path

from judge.build_gold import sample_gold_traces
from judge.prompts import MAX_TRACE_CHARS, build_request_body, parse_verdict
from judge.summarise import summarise_model
from judge.taxonomy import LABEL_DESCRIPTIONS
from judge.traces import Trace, list_reasoning_models, load_python_response_traces
from judge.validate import binary_scores, cohens_kappa, score_against_gold


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a jsonl file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _impl_record(
    prompt_id: str,
    reasoning: list[str | None],
    uses_python: list[bool] | None = None,
    with_messages: bool = True,
) -> dict:
    """Build a minimal implementation results record with per-sample python flags."""
    record: dict = {
        "id": prompt_id,
        "project_id": prompt_id.rsplit("__", 1)[0],
        "task_type": "implementation",
        "responses": ["some code" for _ in reasoning],
        "reasoning": reasoning,
        "uses_python": uses_python or [True] * len(reasoning),
    }
    if with_messages:
        record["prompt_messages"] = [{"role": "user", "content": "Write the tool."}]
    return record


def _make_output_dir(
    tmp_path: Path,
    model: str,
    records: list[dict],
) -> Path:
    """Create a fake output directory with implementation and evaluation files.

    The evaluation file is written positionally (per record, samples in order)
    to mirror the real pipeline's alignment.
    Returns the output directory root.
    """
    evaluation = [
        {
            "project_id": record["project_id"],
            "sample_index": sample_index,
            "uses_python": record["uses_python"][sample_index],
        }
        for record in records
        for sample_index in range(len(record["responses"]))
    ]
    impl_records = [
        {k: v for k, v in record.items() if k != "uses_python"} for record in records
    ]
    _write_jsonl(tmp_path / model / "def-implementation.jsonl", impl_records)
    (tmp_path / model / "def-evaluation.json").write_text(
        json.dumps({"implementation": evaluation})
    )
    return tmp_path


class TestTraces:
    """Test trace loading from output directories."""

    def test_loads_only_python_response_traces(self, tmp_path: Path) -> None:
        """Should keep only samples that chose python and have a reasoning trace."""
        output_dir = _make_output_dir(
            tmp_path,
            "model-a",
            [
                _impl_record(
                    "proj__write",
                    reasoning=["thinking...", "more thinking...", None],
                    uses_python=[True, False, True],
                )
            ],
        )
        traces = load_python_response_traces("model-a", output_dir=output_dir)
        # sample 1 chose another language, sample 2 has no reasoning trace
        assert [t.sample_index for t in traces] == [0]
        assert traces[0].user_prompt == "Write the tool."

    def test_reconstructs_prompt_for_old_schema(self, tmp_path: Path) -> None:
        """Should fall back to the bundled benchmark prompt when messages are missing."""
        # a real prompt id from the bundled implementation split
        prompt_id = "mobile_native_ios_app__write"
        output_dir = _make_output_dir(
            tmp_path,
            "model-b",
            [_impl_record(prompt_id, reasoning=["thinking"], with_messages=False)],
        )
        traces = load_python_response_traces("model-b", output_dir=output_dir)
        assert len(traces) == 1
        assert len(traces[0].user_prompt) > 0

    def test_lists_reasoning_models_only(self, tmp_path: Path) -> None:
        """Should exclude models without any reasoning traces."""
        _make_output_dir(tmp_path, "with-reasoning", [_impl_record("p__write", ["r"])])
        _make_output_dir(tmp_path, "without", [_impl_record("p__write", [None])])
        _make_output_dir(tmp_path, "debug", [_impl_record("p__write", ["r"])])
        assert list_reasoning_models(output_dir=tmp_path) == ["with-reasoning"]


class TestGoldSampling:
    """Test the deterministic gold-set sampler."""

    def _output_dir(self, tmp_path: Path) -> Path:
        """Build two fake models with python-choosing traces across several prompts."""
        for model in ("model-a", "model-b"):
            records = [
                _impl_record(f"proj{i}__write", ["thinking one", "thinking two"])
                for i in range(5)
            ]
            _make_output_dir(tmp_path, model, records)
        return tmp_path

    def test_distinct_prompts_per_model(self, tmp_path: Path, monkeypatch) -> None:
        """Should sample one trace from each of N distinct prompts per model."""
        self._output_dir(tmp_path)
        monkeypatch.setattr(
            "judge.build_gold.load_python_response_traces",
            lambda model: load_python_response_traces(model, output_dir=tmp_path),
        )
        gold = sample_gold_traces(models=["model-a", "model-b"], traces_per_model=3)
        assert len(gold) == 6
        for model in ("model-a", "model-b"):
            ids = [t.id for t in gold if t.model == model]
            assert len(ids) == len(set(ids)) == 3

    def test_doubles_up_when_prompts_run_out(self, tmp_path: Path, monkeypatch) -> None:
        """Should still sample N traces when a model has fewer than N prompts."""
        records = [
            _impl_record(
                f"proj{i}__write", ["thinking one", "thinking two", "thinking three"]
            )
            for i in range(2)
        ]
        _make_output_dir(tmp_path, "model-a", records)
        monkeypatch.setattr(
            "judge.build_gold.load_python_response_traces",
            lambda model: load_python_response_traces(model, output_dir=tmp_path),
        )
        gold = sample_gold_traces(models=["model-a"], traces_per_model=4)
        assert len(gold) == 4
        # both prompts are covered before any prompt is repeated
        ids = [t.id for t in gold]
        assert set(ids) == {"proj0__write", "proj1__write"}
        assert len(set((t.id, t.sample_index) for t in gold)) == 4

    def test_deterministic(self, tmp_path: Path, monkeypatch) -> None:
        """Should return the same sample for the same seed."""
        self._output_dir(tmp_path)
        monkeypatch.setattr(
            "judge.build_gold.load_python_response_traces",
            lambda model: load_python_response_traces(model, output_dir=tmp_path),
        )
        first = sample_gold_traces(models=["model-a"], traces_per_model=3, seed=7)
        second = sample_gold_traces(models=["model-a"], traces_per_model=3, seed=7)
        assert [t.key for t in first] == [t.key for t in second]


class TestPrompts:
    """Test judge request assembly and verdict parsing."""

    def _trace(self, reasoning: str = "some trace") -> Trace:
        """Build a minimal trace."""
        return Trace(
            model="m",
            id="p__write",
            project_id="p",
            sample_index=3,
            user_prompt="Write the tool.",
            reasoning=reasoning,
        )

    def test_request_body(self) -> None:
        """Should include the system prompt, trace content, and strict schema."""
        body = build_request_body(self._trace(), "gpt-5-mini")
        assert body["model"] == "gpt-5-mini"
        assert "NO prior conversation" in body["messages"][0]["content"]
        assert "in Python" in body["messages"][0]["content"]
        assert "some trace" in body["messages"][1]["content"]
        assert body["response_format"]["json_schema"]["strict"] is True

    def test_prompt_is_built_from_taxonomy(self) -> None:
        """The prompt and structured-output schema must be derived from taxonomy.py.

        Guards against the two drifting apart across independent edits.
        """
        system_prompt = build_request_body(self._trace(), "gpt-5-mini")["messages"][0][
            "content"
        ]
        for label, description in LABEL_DESCRIPTIONS.items():
            assert label in system_prompt
            assert description in system_prompt
        schema_labels = build_request_body(self._trace(), "gpt-5-mini")[
            "response_format"
        ]["json_schema"]["schema"]["properties"]["label"]["enum"]
        assert schema_labels == list(LABEL_DESCRIPTIONS)

    def test_prompt_covers_no_language_mention_and_mismatch(self) -> None:
        """Should explicitly direct the two behaviours the user flagged as gaps."""
        system_prompt = build_request_body(self._trace(), "gpt-5-mini")["messages"][0][
            "content"
        ]
        # traces that never discuss a language at all are automatic_python,
        # not unclear_other
        automatic_python_paragraph = system_prompt.split("automatic_python:")[1].split(
            "\n\n"
        )[0]
        assert "never mentions or discusses" in automatic_python_paragraph
        # a language decided in reasoning but contradicted by the final
        # response is its own label
        assert "language_mismatch" in system_prompt
        assert "We'll write the code in C" in system_prompt

    def test_prompt_covers_fabricated_current_prompt_instruction(self) -> None:
        """A trace claiming the CURRENT prompt specified Python is phantom evidence too."""
        system_prompt = build_request_body(self._trace(), "gpt-5-mini")["messages"][0][
            "content"
        ]
        phantom_paragraph = system_prompt.split("phantom_python_evidence:")[1].split(
            "\n\n"
        )[0]
        assert "the problem says to use Python" in phantom_paragraph
        assert "CURRENT prompt" in phantom_paragraph
        # framing python as a stated requirement/restriction is phantom too,
        # even without the word "says"
        assert "we are to write code in Python" in phantom_paragraph
        assert "we are restricted to Python" in phantom_paragraph

    def test_prompt_excludes_real_quote_plus_unforced_leap(self) -> None:
        """Quoting a real, generic prompt phrase then leaping to Python isn't phantom."""
        system_prompt = build_request_body(self._trace(), "gpt-5-mini")["messages"][0][
            "content"
        ]
        assert "NOT phantom_python_evidence" in system_prompt
        assert "unforced leap to Python" in system_prompt
        assert "not given a specific technology stack" in system_prompt

    def test_prompt_covers_buried_ease_statement(self) -> None:
        """An explicit ease reason stated once early must survive later hedging."""
        system_prompt = build_request_body(self._trace(), "gpt-5-mini")["messages"][0][
            "content"
        ]
        # the "read the whole trace" instruction is now stated once, globally
        assert "Read the WHOLE trace" in system_prompt
        ease_paragraph = system_prompt.split("python_for_ease:")[1].split("\n\n")[0]
        assert "due readability" in ease_paragraph

    def test_prompt_prioritises_ease_over_automatic(self) -> None:
        """Ease-driven reasoning should win over automatic_python when a reason is given."""
        system_prompt = build_request_body(self._trace(), "gpt-5-mini")["messages"][0][
            "content"
        ]
        assert system_prompt.index("python_for_ease:") < system_prompt.index(
            "automatic_python:"
        )
        assert "much easier to write and test" in system_prompt

    def test_prompt_distinguishes_library_praise_from_language_choice(self) -> None:
        """Praising a chosen library isn't the same as choosing Python for ease."""
        system_prompt = build_request_body(self._trace(), "gpt-5-mini")["messages"][0][
            "content"
        ]
        assert "pybullet" in system_prompt
        automatic_python_paragraph = system_prompt.split("automatic_python:")[1].split(
            "\n\n"
        )[0]
        assert "library choice made inside an already-assumed language" in (
            automatic_python_paragraph
        )

    def test_prompt_covers_buried_phantom_reference(self) -> None:
        """A fabricated prior-answer claim buried mid-paragraph must still be caught."""
        system_prompt = build_request_body(self._trace(), "gpt-5-mini")["messages"][0][
            "content"
        ]
        assert "the example in the response uses Python" in system_prompt
        assert "Read the WHOLE trace" in system_prompt

    def test_prompt_covers_fabricated_system_prompt(self) -> None:
        """A fabricated 'system prompt' claim, missed in a real 35k-char trace, is phantom."""
        system_prompt = build_request_body(self._trace(), "gpt-5-mini")["messages"][0][
            "content"
        ]
        assert "the initial example in the system prompt was Python" in system_prompt
        assert "no system prompt shown" in system_prompt.lower()

    def test_prompt_commonality_is_not_ease(self) -> None:
        """Calling Python common/popular is not the same as choosing it for ease."""
        system_prompt = build_request_body(self._trace(), "gpt-5-mini")["messages"][0][
            "content"
        ]
        ease_paragraph = system_prompt.split("python_for_ease:")[1].split("\n\n")[0]
        assert "commonality" in ease_paragraph and "not ease" in ease_paragraph
        automatic_python_paragraph = system_prompt.split("automatic_python:")[1].split(
            "\n\n"
        )[0]
        assert "common/popular" in automatic_python_paragraph

    def test_truncates_long_traces(self) -> None:
        """Should truncate traces beyond the character limit."""
        body = build_request_body(
            self._trace("x" * (MAX_TRACE_CHARS + 100)), "gpt-5-mini"
        )
        assert "[trace truncated]" in body["messages"][1]["content"]

    def test_parse_verdict(self) -> None:
        """Should parse a valid json verdict."""
        verdict = parse_verdict(
            json.dumps(
                {
                    "label": "phantom_python_evidence",
                    "evidence_quotes": ["the previous examples are Python"],
                    "confidence": "high",
                    "rationale": "Fabricates prior examples to justify python.",
                }
            )
        )
        assert verdict.label == "phantom_python_evidence"
        assert verdict.confidence == "high"


class TestSummarise:
    """Test per-model aggregation of judge verdicts."""

    def test_counts_and_scope(self, tmp_path: Path) -> None:
        """Should count labels and report the python-response scope size."""
        model = "model-a"
        _make_output_dir(
            tmp_path,
            model,
            [
                _impl_record("proj__write", ["trace a", "trace b"]),
                _impl_record(
                    "proj__create",
                    reasoning=["trace c", "trace d"],
                    uses_python=[True, False],
                ),
            ],
        )

        def _judgement(prompt_id: str, sample_index: int, label: str) -> dict:
            return {
                "model": model,
                "id": prompt_id,
                "project_id": "proj",
                "sample_index": sample_index,
                "judge_model": "gpt-5-mini",
                "verdict": {
                    "label": label,
                    "evidence_quotes": ["quote"] if label != "automatic_python" else [],
                    "confidence": "high",
                    "rationale": "r",
                },
            }

        _write_jsonl(
            tmp_path / model / "def-judge-results.jsonl",
            [
                _judgement("proj__write", 0, "phantom_python_evidence"),
                _judgement("proj__write", 1, "language_mismatch"),
                _judgement("proj__create", 0, "automatic_python"),
            ],
        )

        analysis = summarise_model(model, output_dir=tmp_path)
        assert analysis.summary.total == 4
        # one of the four samples chose another language
        assert analysis.summary.python_responses == 3
        assert analysis.summary.judged == 3
        assert analysis.summary.phantom_python_evidence == 1
        assert analysis.summary.language_mismatch == 1
        assert analysis.summary.automatic_python == 1
        assert analysis.summary.phantom_rate == 1 / 3
        assert len(analysis.examples["phantom_python_evidence"]) == 1


class TestValidate:
    """Test gold-set scoring metrics."""

    def test_binary_scores(self) -> None:
        """Should compute precision and recall for the positive label."""
        gold = {
            "a": "phantom_python_evidence",
            "b": "automatic_python",
            "c": "phantom_python_evidence",
        }
        predictions = {
            "a": "phantom_python_evidence",
            "b": "phantom_python_evidence",
            "c": "automatic_python",
        }
        scores = binary_scores(gold, predictions, "phantom_python_evidence")
        assert scores["true_positives"] == 1
        assert scores["false_positives"] == 1
        assert scores["false_negatives"] == 1
        assert scores["precision"] == 0.5
        assert scores["recall"] == 0.5

    def test_kappa_perfect_agreement(self) -> None:
        """Should return 1.0 for identical labels."""
        labels = {"a": "x", "b": "y", "c": "x"}
        assert cohens_kappa(labels, dict(labels)) == 1.0

    def test_score_against_gold_confusion(self) -> None:
        """Should build the confusion matrix from gold to predicted labels."""
        gold = {"a": "python_for_ease", "b": "python_for_ease"}
        predictions = {"a": "python_for_ease", "b": "automatic_python"}
        scores = score_against_gold(gold, predictions)
        assert scores["accuracy"] == 0.5
        assert scores["confusion_matrix"]["python_for_ease"] == {
            "python_for_ease": 1,
            "automatic_python": 1,
        }
