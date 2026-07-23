"""Run the LLM judge over reasoning traces via the OpenAI Batch API.

Usage:
    python -m judge.run --judge-model gpt-5-mini --gold-only   # pilot on the gold set
    python -m judge.run --judge-model gpt-5-mini --model all   # full run

State is saved after every collected batch, so the command is safe to
interrupt and re-run: already-judged traces are skipped, and in-flight
batches are resumed rather than resubmitted.
"""

import argparse
import io
import json
import time
from pathlib import Path

from openai import OpenAI

from judge.build_gold import GOLD_UNLABELLED_PATH
from judge.prompts import build_request_body, parse_verdict
from judge.taxonomy import TraceJudgement
from judge.traces import Trace, list_reasoning_models, load_python_response_traces
from src.utils.io import append_jsonl, load_json, load_jsonl, save_json
from src.utils.log import log


# friendly judge name -> openai api model id (dated snapshots for reproducibility)
JUDGE_MODELS = {
    "gpt-5.4-mini": "gpt-5.4-mini-2026-03-17",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
}

# requests per batch file — keeps uploads well under api size limits
CHUNK_SIZE = 2_000
POLL_SECONDS = 60

# batch states that mean no more results are coming
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _load_judged_keys(results_path: Path) -> set[str]:
    """Load the custom_ids of traces that already have a verdict.

    Returns the set of judged custom_ids.
    """
    if not results_path.exists():
        return set()
    return {
        f"{r['model']}|{r['id']}|{r['sample_index']}" for r in load_jsonl(results_path)
    }


def _submit_batch(
    client: OpenAI,
    traces: list[Trace],
    judge_model: str,
) -> str:
    """Upload one chunk of judge requests and start a batch.

    Returns the batch id.
    """
    lines = [
        json.dumps(
            {
                "custom_id": trace.key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": build_request_body(trace, JUDGE_MODELS[judge_model]),
            },
            ensure_ascii=False,
        )
        for trace in traces
    ]
    batch_file = client.files.create(
        file=("judge_requests.jsonl", io.BytesIO("\n".join(lines).encode())),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


def _collect_batch(
    client: OpenAI,
    batch_id: str,
    traces_by_key: dict[str, Trace],
    judge_model: str,
    results_path: Path,
) -> None:
    """Wait for a batch to finish and append its verdicts to the results file."""
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in TERMINAL_STATUSES:
            break
        counts = batch.request_counts
        log(
            f"    batch {batch_id}: {batch.status} "
            f"({counts.completed}/{counts.total} done, {counts.failed} failed)"
        )
        time.sleep(POLL_SECONDS)

    if batch.status != "completed" or batch.output_file_id is None:
        log(f"    batch {batch_id} ended with status '{batch.status}', skipping")
        # surface validation errors (e.g. an unsupported model id) so a failed
        # batch is diagnosable from the run log
        if batch.errors and batch.errors.data:
            first = batch.errors.data[0]
            log(f"    first batch error: [{first.code}] {first.message}")
        if batch.error_file_id:
            first_line = client.files.content(batch.error_file_id).text.splitlines()[0]
            log(f"    first request error: {first_line[:300]}")
        return

    for line in client.files.content(batch.output_file_id).text.splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        key = result["custom_id"]
        response = result.get("response") or {}
        if result.get("error") or response.get("status_code") != 200:
            log(f"    request {key} failed: {result.get('error')}")
            continue
        trace = traces_by_key[key]
        content = response["body"]["choices"][0]["message"]["content"]
        append_jsonl(
            record=TraceJudgement(
                model=trace.model,
                id=trace.id,
                project_id=trace.project_id,
                sample_index=trace.sample_index,
                judge_model=judge_model,
                verdict=parse_verdict(content),
            ),
            path=results_path,
        )


def judge_traces(
    traces: list[Trace],
    judge_model: str,
    results_path: Path,
    batches_path: Path,
) -> None:
    """Judge all given traces, resuming any prior partial run.

    Verdicts are appended to results_path; in-flight batch ids are tracked in
    batches_path so an interrupted run resumes without resubmitting.
    """
    client = OpenAI()
    traces_by_key = {trace.key: trace for trace in traces}

    # first collect any batches submitted by a previous interrupted run
    batches = load_json(batches_path) if batches_path.exists() else []
    for entry in batches:
        if not entry["collected"]:
            log(f"  Resuming batch {entry['batch_id']}")
            _collect_batch(
                client=client,
                batch_id=entry["batch_id"],
                traces_by_key=traces_by_key,
                judge_model=judge_model,
                results_path=results_path,
            )
            entry["collected"] = True
            save_json(batches, batches_path)

    judged = _load_judged_keys(results_path)
    pending = [trace for trace in traces if trace.key not in judged]
    if not pending:
        log(f"  All {len(traces)} traces already judged: {results_path}")
        return

    log(f"  Judging {len(pending)} of {len(traces)} traces with {judge_model}")
    for start in range(0, len(pending), CHUNK_SIZE):
        chunk = pending[start : start + CHUNK_SIZE]
        batch_id = _submit_batch(client=client, traces=chunk, judge_model=judge_model)
        batches.append({"batch_id": batch_id, "collected": False})
        save_json(batches, batches_path)
        log(f"  Submitted batch {batch_id} ({len(chunk)} requests)")
        _collect_batch(
            client=client,
            batch_id=batch_id,
            traces_by_key=traces_by_key,
            judge_model=judge_model,
            results_path=results_path,
        )
        batches[-1]["collected"] = True
        save_json(batches, batches_path)

    judged = _load_judged_keys(results_path)
    log(f"  Done: {len(judged)} of {len(traces)} traces judged -> {results_path}")


def _parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Judge reasoning traces for phantom evidence.",
    )
    parser.add_argument(
        "-j",
        "--judge-model",
        required=True,
        choices=sorted(JUDGE_MODELS),
        help="Judge model to use.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="all",
        help="Model output directory to judge, or 'all' (default: all).",
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Judge only the gold-set traces (pilot mode).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the judge in gold-only (pilot) or per-model (full) mode."""
    args = _parse_args()

    if args.gold_only:
        traces = [Trace(**r) for r in load_jsonl(GOLD_UNLABELLED_PATH)]
        judge_traces(
            traces=traces,
            judge_model=args.judge_model,
            results_path=Path(f"judge/data/gold_judgements_{args.judge_model}.jsonl"),
            batches_path=Path(f"judge/data/gold_batches_{args.judge_model}.json"),
        )
        return

    models = list_reasoning_models() if args.model == "all" else [args.model]
    for model in models:
        log(f"Model: {model}")
        judge_traces(
            traces=load_python_response_traces(model),
            judge_model=args.judge_model,
            results_path=Path(f"output/{model}/def-judge-results.jsonl"),
            batches_path=Path(f"output/{model}/def-judge-batches.json"),
        )


if __name__ == "__main__":
    main()
