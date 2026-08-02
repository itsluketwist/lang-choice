"""Compute inter-author agreement statistics for the Python-suitability labels."""

import json
from collections import Counter
from itertools import product
from pathlib import Path


# the labels file lives alongside this script
LABELS_PATH = Path(__file__).parent / "labels.json"


def confusion_matrix(
    labels_a: list[str],
    labels_b: list[str],
    order: list[str],
) -> dict[str, dict[str, int]]:
    """Count how often each (author_one, author_two) label pair occurs.

    Rows are author_one's label, columns are author_two's label.
    Returns a nested dict keyed [author_one_label][author_two_label] -> count.
    """
    counts = Counter(zip(labels_a, labels_b))
    return {row: {col: counts.get((row, col), 0) for col in order} for row in order}


def cohens_kappa(
    labels_a: list[str],
    labels_b: list[str],
    order: list[str],
    weighting: str = "unweighted",
) -> float:
    """Compute Cohen's kappa (optionally ordinal-weighted) between two raters.

    Kappa corrects observed agreement for the agreement expected by chance:
    1 - (weighted observed disagreement / weighted expected disagreement).
    With "linear" or "quadratic" weighting, disagreements between adjacent
    ordinal labels (e.g. Weak vs Acceptable) count less than extreme ones
    (Weak vs Strong); "unweighted" treats every disagreement equally.
    Returns the kappa coefficient as a float.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("both raters must label the same number of items")

    total = len(labels_a)
    size = len(order)
    rank = {label: index for index, label in enumerate(order)}

    # disagreement weight for a pair of labels at ordinal positions i and j
    def weight(i: int, j: int) -> float:
        if weighting == "unweighted":
            return 0.0 if i == j else 1.0
        distance = abs(i - j) / (size - 1)
        return distance if weighting == "linear" else distance**2

    # observed pair proportions and each rater's own label proportions (marginals)
    observed_pairs = Counter(zip(labels_a, labels_b))
    marginal_a = Counter(labels_a)
    marginal_b = Counter(labels_b)

    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for label_i, label_j in product(order, repeat=2):
        w = weight(rank[label_i], rank[label_j])
        # observed: how often this pair actually occurred
        observed_disagreement += w * observed_pairs.get((label_i, label_j), 0) / total
        # expected: probability of this pair if raters labelled independently
        chance = (marginal_a[label_i] / total) * (marginal_b[label_j] / total)
        expected_disagreement += w * chance

    # guard against the degenerate case of zero expected disagreement
    if expected_disagreement == 0.0:
        return 1.0

    return 1 - observed_disagreement / expected_disagreement


def specific_agreement(
    labels_a: list[str],
    labels_b: list[str],
    order: list[str],
) -> dict[str, float]:
    """Compute per-label specific agreement (how reliably each label is applied).

    For a label, this is 2 * (times both authors used it) / (total times either
    author used it) - i.e. the chance the other author agrees when one uses it.
    Returns a dict mapping each label to its specific-agreement fraction.
    """
    result = {}
    for label in order:
        both = sum(1 for a, b in zip(labels_a, labels_b) if a == label and b == label)
        uses = labels_a.count(label) + labels_b.count(label)
        result[label] = round(2 * both / uses, 4) if uses else 0.0
    return result


def main() -> None:
    """Load the combined labels, compute agreement statistics, and write them back."""
    data = json.loads(LABELS_PATH.read_text())
    order = data["label_order"]
    records = data["records"]

    labels_one = [record["author_one"]["label"] for record in records]
    labels_two = [record["author_two"]["label"] for record in records]
    total = len(records)
    rank = {label: index for index, label in enumerate(order)}

    # raw agreement: fraction of items where both authors gave the identical label
    raw_agreement = sum(1 for a, b in zip(labels_one, labels_two) if a == b) / total

    # split the disagreements into adjacent (off by one) vs extreme (Weak vs Strong)
    gaps = [abs(rank[a] - rank[b]) for a, b in zip(labels_one, labels_two)]
    adjacent = sum(1 for gap in gaps if gap == 1)
    extreme = sum(1 for gap in gaps if gap >= 2)

    data["metrics"] = {
        "n_items": total,
        "raw_agreement": round(raw_agreement, 4),
        "cohens_kappa": round(cohens_kappa(labels_one, labels_two, order), 4),
        "weighted_kappa_linear": round(
            cohens_kappa(labels_one, labels_two, order, "linear"),
            4,
        ),
        "weighted_kappa_quadratic": round(
            cohens_kappa(labels_one, labels_two, order, "quadratic"),
            4,
        ),
        "label_counts": {
            "author_one": {label: labels_one.count(label) for label in order},
            "author_two": {label: labels_two.count(label) for label in order},
        },
        # per-label reliability; "Weak" is the key label for the benchmark
        "specific_agreement": specific_agreement(labels_one, labels_two, order),
        "disagreements": {
            "adjacent": adjacent,
            "extreme": extreme,
        },
        "confusion_matrix": confusion_matrix(labels_one, labels_two, order),
    }

    LABELS_PATH.write_text(json.dumps(data, indent=2) + "\n")

    metrics = data["metrics"]
    print(f"items:                    {metrics['n_items']}")
    print(f"raw agreement:            {metrics['raw_agreement']:.2%}")
    print(f"cohen's kappa:            {metrics['cohens_kappa']}")
    print(f"weighted kappa (linear):  {metrics['weighted_kappa_linear']}")
    print(f"weighted kappa (quad):    {metrics['weighted_kappa_quadratic']}")
    print(f"weak agreement:           {metrics['specific_agreement']['Weak']:.2%}")
    print(f"disagreements:            {adjacent} adjacent, {extreme} extreme")


if __name__ == "__main__":
    main()
