"""Score judge pilots (and the v1 heuristic) against the hand-labelled gold set.

Usage:
    python -m judge.validate
"""

from pathlib import Path

from judge.build_gold import GOLD_LABELLED_PATH
from judge.taxonomy import LABEL_DESCRIPTIONS, PHANTOM_LABEL
from judge.traces import OUTPUT_DIR
from src.utils.io import load_json, load_jsonl, save_json
from src.utils.log import log


DATA_DIR = Path("judge/data")
REPORT_PATH = DATA_DIR / "validation_report.json"


def _key(record: dict) -> str:
    """Unique identifier for a gold/judgement record.

    Returns "{model}|{id}|{sample_index}".
    """
    return f"{record['model']}|{record['id']}|{record['sample_index']}"


def binary_scores(
    gold: dict[str, str],
    predictions: dict[str, str],
    positive_label: str,
) -> dict:
    """Precision/recall/F1 for one label treated as the positive class.

    Returns a dict with counts and the three scores.
    """
    tp = fp = fn = 0
    for key, gold_label in gold.items():
        predicted = predictions.get(key) == positive_label
        actual = gold_label == positive_label
        tp += predicted and actual
        fp += predicted and not actual
        fn += actual and not predicted

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def cohens_kappa(
    gold: dict[str, str],
    predictions: dict[str, str],
) -> float:
    """Cohen's kappa between the gold labels and predictions.

    Returns the kappa value (1.0 = perfect agreement, 0.0 = chance level).
    """
    keys = [key for key in gold if key in predictions]
    n = len(keys)
    if n == 0:
        return 0.0

    observed = sum(gold[key] == predictions[key] for key in keys) / n
    labels = set(gold[key] for key in keys) | set(predictions[key] for key in keys)
    expected = sum(
        (sum(gold[key] == label for key in keys) / n)
        * (sum(predictions[key] == label for key in keys) / n)
        for label in labels
    )
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def score_against_gold(
    gold: dict[str, str],
    predictions: dict[str, str],
) -> dict:
    """Full scoring of one predictor against the gold labels.

    Returns a dict with coverage, accuracy, kappa, per-label scores, and the
    confusion matrix (gold label -> predicted label -> count).
    """
    keys = [key for key in gold if key in predictions]
    confusion: dict[str, dict[str, int]] = {}
    for key in keys:
        row = confusion.setdefault(gold[key], {})
        row[predictions[key]] = row.get(predictions[key], 0) + 1

    return {
        "n_scored": len(keys),
        "n_gold": len(gold),
        "accuracy": round(
            sum(gold[key] == predictions[key] for key in keys) / len(keys), 3
        )
        if keys
        else 0.0,
        "cohens_kappa": round(cohens_kappa(gold, predictions), 3),
        "per_label": {
            label: binary_scores(gold, predictions, label)
            for label in LABEL_DESCRIPTIONS
        },
        "phantom_python_binary": binary_scores(gold, predictions, PHANTOM_LABEL),
        "confusion_matrix": confusion,
    }


def disagreements(
    gold_records: list[dict],
    predictions: dict[str, str],
    evidence: dict[str, list[str]],
) -> list[dict]:
    """Collect gold records where the prediction differs from the hand label.

    Returns a list of dicts for manual adjudication.
    """
    return [
        {
            **{k: record[k] for k in ("model", "id", "sample_index")},
            "gold_label": record["gold_label"],
            "predicted_label": predictions[_key(record)],
            "evidence_quotes": evidence.get(_key(record), []),
            "notes": record.get("notes", ""),
        }
        for record in gold_records
        if _key(record) in predictions
        and predictions[_key(record)] != record["gold_label"]
    ]


def _v1_predictions(gold_records: list[dict]) -> dict[str, str]:
    """Map the v1 heuristic's anchor labels onto the judge taxonomy (binary).

    Only phantom_python_anchor maps to the phantom label; every other v1 label
    is treated as not phantom, since the taxonomies do not align further.
    Returns the key -> label mapping for all gold traces.
    """
    models = {record["model"] for record in gold_records}
    v1_labels = {}
    for model in models:
        analysis = load_json(OUTPUT_DIR / model / "def-analysis.json")
        for result in analysis["results"]:
            key = f"{model}|{result['id']}|{result['sample_index']}"
            v1_labels[key] = (
                PHANTOM_LABEL
                if result["anchor_label"] == "phantom_python_anchor"
                else "not_phantom"
            )
    return {
        _key(record): v1_labels[_key(record)]
        for record in gold_records
        if _key(record) in v1_labels
    }


def main() -> None:
    """Score all judge pilots and the v1 heuristic, and save the report."""
    gold_records = load_jsonl(GOLD_LABELLED_PATH)
    gold = {_key(record): record["gold_label"] for record in gold_records}
    # partial labelling is fine — predictors are scored on the overlap only
    log(f"Gold set: {len(gold)} hand-labelled traces so far")

    report: dict = {"n_gold": len(gold), "predictors": {}}

    for path in sorted(DATA_DIR.glob("gold_judgements_*.jsonl")):
        judge_model = path.stem.removeprefix("gold_judgements_")
        judgements = load_jsonl(path)
        predictions = {_key(r): r["verdict"]["label"] for r in judgements}
        evidence = {_key(r): r["verdict"]["evidence_quotes"] for r in judgements}
        scores = score_against_gold(gold, predictions)
        scores["disagreements"] = disagreements(gold_records, predictions, evidence)
        report["predictors"][judge_model] = scores
        log(
            f"  {judge_model}: accuracy={scores['accuracy']}, "
            f"kappa={scores['cohens_kappa']}, "
            f"phantom-python f1={scores['phantom_python_binary']['f1']}"
        )

    v1 = _v1_predictions(gold_records)
    report["predictors"]["heuristic_v1"] = {
        "note": "binary only — v1 labels do not map onto the full taxonomy",
        "phantom_python_binary": binary_scores(gold, v1, "phantom_python_context"),
    }
    log(
        "  heuristic_v1: phantom-python f1="
        f"{report['predictors']['heuristic_v1']['phantom_python_binary']['f1']}"
    )

    save_json(report, REPORT_PATH)
    log(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
