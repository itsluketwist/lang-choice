"""I/O helpers for JSON and JSONL files used throughout the experiment pipeline."""

import json
from pathlib import Path
from typing import Any


def save_json(
    data: Any,
    path: str | Path,
) -> None:
    """Save data to a JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")  # ensure trailing newline


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file.

    Returns the parsed object.
    """
    with open(path) as f:
        return json.load(f)


def save_jsonl(
    records: list[Any],
    path: str | Path,
) -> None:
    """Save a list of records to a JSONL file (one JSON object per line).

    Each record is serialised independently; parent directories are created as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            if hasattr(record, "model_dump"):
                # pydantic model — serialise via model_dump for correct enum handling
                line = json.dumps(record.model_dump(), ensure_ascii=False)
            else:
                line = json.dumps(record, ensure_ascii=False)
            f.write(line + "\n")


def append_jsonl(
    record: Any,
    path: str | Path,
) -> None:
    """Append a single record to a JSONL file.

    Creates the file and parent directories if they do not exist.
    Useful for streaming results to disk during long generation runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        if hasattr(record, "model_dump"):
            line = json.dumps(record.model_dump(), ensure_ascii=False)
        else:
            line = json.dumps(record, ensure_ascii=False)
        f.write(line + "\n")


def load_jsonl(path: str | Path) -> list[Any]:
    """Load all records from a JSONL file.

    Returns a list of parsed objects (dicts or primitives depending on content).
    """
    path = Path(path)
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
