"""Measure agreement and Python suitability across benchmark reviewers."""

import csv
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


REVIEWERS = ("author_one", "author_two", "developer_one", "developer_two")


def confusion_matrix(
    labels_a: list[str],
    labels_b: list[str],
    order: list[str],
) -> dict[str, dict[str, int]]:
    """Count how often each pair of reviewer labels occurs.

    Rows are the first reviewer's labels; columns are the second's.
    Returns a nested dict keyed by the first and second reviewer labels.
    """
    counts = Counter(zip(labels_a, labels_b))
    return {row: {col: counts.get((row, col), 0) for col in order} for row in order}


def cohens_kappa(
    labels_a: list[str],
    labels_b: list[str],
    order: list[str],
    weighting: str = "unweighted",
) -> float | None:
    """Compute Cohen's kappa (optionally ordinal-weighted) between two raters.

    Kappa corrects observed agreement for the agreement expected by chance:
    1 - (weighted observed disagreement / weighted expected disagreement).
    With "linear" or "quadratic" weighting, disagreements between adjacent
    ordinal labels (e.g. Weak vs Acceptable) count less than extreme ones
    (Weak vs Strong); "unweighted" treats every disagreement equally.
    Returns the kappa coefficient, or None when chance disagreement is zero.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("both raters must label the same number of items")

    if not labels_a or len(order) < 2 or len(set(order)) != len(order):
        raise ValueError(
            "nonempty ratings and distinct ordered categories are required",
        )
    if weighting not in {"unweighted", "linear", "quadratic"}:
        raise ValueError("unknown weighting")
    if any(label not in order for label in labels_a + labels_b):
        raise ValueError("unknown rating label")
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
        return None

    return 1 - observed_disagreement / expected_disagreement


def specific_agreement(
    labels_a: list[str],
    labels_b: list[str],
    order: list[str],
) -> dict[str, float | None]:
    """Compute per-label specific agreement (how reliably each label is applied).

    For a label, this is 2 * (times both authors used it) / (total times either
    author used it) - i.e. the chance the other author agrees when one uses it.
    Returns per-label agreement, or None for labels used by neither reviewer.
    """
    result = {}
    for label in order:
        both = sum(1 for a, b in zip(labels_a, labels_b) if a == label and b == label)
        uses = labels_a.count(label) + labels_b.count(label)
        result[label] = 2 * both / uses if uses else None
    return result


def merge_developers(
    data: dict[str, Any],
    csv_path: Path,
) -> None:
    """Validate the complete developer export before adding ratings by project id."""
    records = data["records"]
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate project ids in labels.json")
    developers = {"1": "developer_one", "2": "developer_two"}
    ratings: dict[tuple[str, str], str] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            developer = row["developer_id"].strip()
            project = row["project_id"].strip()
            label = row["rating"].strip()
            if developer not in developers or project not in ids:
                raise ValueError(f"unknown developer or project: {row}")
            if label not in data["label_order"]:
                raise ValueError(f"unknown label: {label}")
            key = (project, developer)
            if key in ratings:
                raise ValueError(f"duplicate developer rating: {key}")
            ratings[key] = label
    if set(ratings) != set(product(ids, developers)):
        raise ValueError("each developer must rate every project exactly once")
    for record in records:
        for developer, reviewer in developers.items():
            label = ratings[(record["id"], developer)]
            if reviewer in record and record[reviewer]["label"] != label:
                raise ValueError(
                    f"existing rating conflicts with export: {record['id']}"
                )
    for record in records:
        for developer, reviewer in developers.items():
            # preserve existing annotations; the export contains no justifications.
            record.setdefault(reviewer, {"label": ratings[(record["id"], developer)]})


def agreement_coefficients(
    ratings: NDArray[np.int64],
    n_categories: int,
) -> dict[str, float | None]:
    """Measure exact agreement, Fleiss' kappa, and nominal/ordinal alpha.

    Ordinal alpha uses squared distances between cumulative category midpoints,
    following https://doi.org/10.1016/j.mex.2023.102545 (equation 4).
    Returns coefficients for a complete items-by-reviewers rating matrix.
    """
    if ratings.ndim != 2 or ratings.shape[0] == 0 or ratings.shape[1] < 2:
        raise ValueError("at least one item and two reviewers are required")
    if n_categories < 2 or np.any(ratings < 0) or np.any(ratings >= n_categories):
        raise ValueError("ratings must index the ordered categories")
    if not np.issubdtype(ratings.dtype, np.integer):
        raise ValueError("ratings must be integer category indices")
    n_items, n_reviewers = ratings.shape
    counts = np.stack(
        [(ratings == index).sum(axis=1) for index in range(n_categories)], axis=1
    )
    totals = counts.sum(axis=0)
    n_ratings = int(totals.sum())

    # count ordered within-item pairs, excluding comparisons of a rater to themself.
    observed = counts.T @ counts - np.diag(totals)
    observed = observed / (n_items * n_reviewers * (n_reviewers - 1))
    expected = (np.outer(totals, totals) - np.diag(totals)) / (
        n_ratings * (n_ratings - 1)
    )
    nominal_distance = 1.0 - np.eye(n_categories)
    midpoints = np.cumsum(totals) - totals / 2
    ordinal_distance = (midpoints[:, None] - midpoints[None, :]) ** 2

    def alpha(distance: NDArray[np.float64]) -> float | None:
        """Correct observed disagreement using finite-sample expected disagreement.

        Returns alpha, or None when the expected disagreement is zero.
        """
        expected_disagreement = float((expected * distance).sum())
        if expected_disagreement == 0:
            return None
        return 1 - float((observed * distance).sum()) / expected_disagreement

    raw = float(np.trace(observed))
    chance = float(((totals / n_ratings) ** 2).sum())
    return {
        "mean_pairwise_exact_agreement": raw,
        "unanimous_agreement": float(
            np.mean(np.all(ratings == ratings[:, :1], axis=1))
        ),
        "fleiss_kappa": (raw - chance) / (1 - chance) if chance < 1 else None,
        "krippendorff_alpha_nominal": alpha(nominal_distance),
        "krippendorff_alpha_ordinal": alpha(ordinal_distance),
    }


def bootstrap_intervals(
    ratings: NDArray[np.int64],
    n_categories: int,
    n_resamples: int = 10000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Resample projects with all their ratings intact, holding reviewers fixed.

    Returns percentile 95% intervals and counts of defined bootstrap estimates.
    """
    if n_resamples < 1:
        raise ValueError("at least one bootstrap resample is required")
    samples: dict[str, list[float]] = {
        name: [] for name in agreement_coefficients(ratings, n_categories)
    }
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        indices = rng.integers(0, len(ratings), size=len(ratings))
        for name, value in agreement_coefficients(
            ratings[indices], n_categories
        ).items():
            if value is not None:
                samples[name].append(value)
    return {
        "method": "project-level percentile bootstrap; fixed reviewers; 95% intervals",
        "n_resamples": n_resamples,
        "seed": seed,
        "intervals": {
            name: {
                "lower": float(np.quantile(values, 0.025)) if values else None,
                "upper": float(np.quantile(values, 0.975)) if values else None,
                "n_defined": len(values),
            }
            for name, values in samples.items()
        },
    }


def summarize_ratings(
    records: list[dict[str, Any]],
    order: list[str],
    reviewers: tuple[str, ...],
) -> dict[str, Any]:
    """Distinguish agreement, weak suitability, and not being optimal.

    Returns coefficients, label counts, vote counts, and project-level exceptions.
    """
    ratings = np.array(
        [
            [order.index(record[reviewer]["label"]) for reviewer in reviewers]
            for record in records
        ],
        dtype=np.int64,
    )
    result: dict[str, Any] = {
        "n_items": len(records),
        "reviewers": list(reviewers),
        **agreement_coefficients(ratings, len(order)),
        "label_counts": {
            reviewer: {
                label: int((ratings[:, column] == index).sum())
                for index, label in enumerate(order)
            }
            for column, reviewer in enumerate(reviewers)
        },
    }
    for name, labels in {
        "weak": {"Weak"},
        "not_optimal": {"Weak", "Acceptable"},
    }.items():
        supported = np.isin(ratings, [order.index(label) for label in labels])
        votes = supported.sum(axis=1)
        result[name] = {
            "definition": sorted(labels),
            "n_ratings": int(supported.size),
            "n_supporting_ratings": int(supported.sum()),
            "supporting_fraction": float(supported.mean()),
            "projects_by_supporting_votes": {
                str(count): int((votes == count).sum())
                for count in range(len(reviewers) + 1)
            },
            "unanimous_projects": [
                record["id"]
                for record, count in zip(records, votes)
                if count == len(reviewers)
            ],
            "nonunanimous_projects": [
                record["id"]
                for record, count in zip(records, votes)
                if count < len(reviewers)
            ],
            "binary_agreement": agreement_coefficients(supported.astype(np.int64), 2),
        }
    return result


def analyze_ratings(
    data: dict[str, Any],
    n_resamples: int = 10000,
) -> dict[str, Any]:
    """Calculate four-reviewer reliability, subgroup summaries, and pairwise diagnostics.

    Returns reproducible metrics without changing the supplied ratings.
    """
    records, order = data["records"], data["label_order"]
    all_reviewers = summarize_ratings(records, order, REVIEWERS)
    ratings = np.array(
        [
            [order.index(record[reviewer]["label"]) for reviewer in REVIEWERS]
            for record in records
        ],
        dtype=np.int64,
    )
    all_reviewers["bootstrap"] = bootstrap_intervals(ratings, len(order), n_resamples)
    pairs: dict[str, Any] = {}
    for first, second in combinations(REVIEWERS, 2):
        labels_a = [record[first]["label"] for record in records]
        labels_b = [record[second]["label"] for record in records]
        gaps = [
            abs(order.index(a) - order.index(b)) for a, b in zip(labels_a, labels_b)
        ]
        pairs[f"{first}__{second}"] = {
            "raw_agreement": sum(gap == 0 for gap in gaps) / len(gaps),
            "cohens_kappa": cohens_kappa(labels_a, labels_b, order),
            "weighted_kappa_linear": cohens_kappa(labels_a, labels_b, order, "linear"),
            "weighted_kappa_quadratic": cohens_kappa(
                labels_a, labels_b, order, "quadratic"
            ),
            "specific_agreement": specific_agreement(labels_a, labels_b, order),
            "confusion_matrix": confusion_matrix(labels_a, labels_b, order),
            "disagreements": {
                "adjacent": gaps.count(1),
                "extreme": sum(gap >= 2 for gap in gaps),
            },
        }
    return {
        "all_reviewers": all_reviewers,
        "authors": summarize_ratings(records, order, REVIEWERS[:2]),
        "developers": summarize_ratings(records, order, REVIEWERS[2:]),
        "pairwise": pairs,
    }
