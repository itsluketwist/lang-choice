"""Merge developer ratings and reproduce the benchmark's agreement statistics."""

import argparse
import json
from pathlib import Path


# support both direct script execution and python -m benchmark.build.validate.
if __package__:
    from .agreement import analyze_ratings, merge_developers
else:
    from agreement import analyze_ratings, merge_developers


LABELS_PATH = Path(__file__).parent / "labels.json"


def main() -> None:
    """Optionally import developer ratings, then update metrics in labels.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--developer-csv", type=Path)
    args = parser.parse_args()
    data = json.loads(LABELS_PATH.read_text())
    if args.developer_csv:
        merge_developers(data, args.developer_csv)
        data["developer_rating_source"] = args.developer_csv.name
    data["metrics"] = analyze_ratings(data)
    LABELS_PATH.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps(data["metrics"]["all_reviewers"], indent=2))


if __name__ == "__main__":
    main()
