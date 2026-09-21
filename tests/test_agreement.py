"""Check agreement formulas, uncertainty, and lossless developer-rating import."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark.build.agreement import (
    agreement_coefficients,
    bootstrap_intervals,
    cohens_kappa,
    merge_developers,
)


def test_hand_calculated_three_category_example() -> None:
    """Check finite-sample alpha and frequency-based ordinal distances by hand."""
    ratings = np.array([[0, 0, 1], [0, 1, 2]], dtype=np.int64)
    result = agreement_coefficients(ratings, 3)
    assert result["mean_pairwise_exact_agreement"] == pytest.approx(1 / 6)
    assert result["fleiss_kappa"] == pytest.approx(-4 / 11)
    assert result["krippendorff_alpha_nominal"] == pytest.approx(-3 / 22)
    # category totals are (3, 2, 1), so midpoint distances are 2.5, 4, and 1.5.
    assert result["krippendorff_alpha_ordinal"] == pytest.approx(-1 / 36)


def test_perfect_and_degenerate_agreement() -> None:
    """Distinguish perfect reliability from undefined single-category reliability."""
    perfect = agreement_coefficients(np.array([[0, 0], [2, 2]], dtype=np.int64), 3)
    assert all(value == 1 for value in perfect.values())
    degenerate = agreement_coefficients(np.zeros((3, 4), dtype=np.int64), 3)
    assert degenerate["mean_pairwise_exact_agreement"] == 1
    assert degenerate["fleiss_kappa"] is None
    assert degenerate["krippendorff_alpha_nominal"] is None
    assert degenerate["krippendorff_alpha_ordinal"] is None
    assert cohens_kappa(["Weak"], ["Weak"], ["Weak", "Strong"]) is None


def test_bootstrap_preserves_reviewer_groups_and_is_reproducible() -> None:
    """Keep unanimous projects unanimous and report undefined resamples."""
    ratings = np.array([[0, 0], [1, 1]], dtype=np.int64)
    result = bootstrap_intervals(ratings, 2, n_resamples=100)
    assert result == bootstrap_intervals(ratings, 2, n_resamples=100)
    assert result["intervals"]["mean_pairwise_exact_agreement"]["lower"] == 1
    alpha = result["intervals"]["krippendorff_alpha_ordinal"]
    assert alpha["lower"] == alpha["upper"] == 1
    assert 0 < alpha["n_defined"] < 100


@pytest.mark.parametrize(
    "rows",
    [
        "1,a,Weak\n",
        "1,a,Weak\n1,a,Weak\n2,a,Strong\n",
        "1,unknown,Weak\n2,a,Strong\n",
        "1,a,invalid\n2,a,Strong\n",
        "3,a,Weak\n2,a,Strong\n",
    ],
)
def test_invalid_import_does_not_modify_data(tmp_path: Path, rows: str) -> None:
    """Reject incomplete, duplicate, or unknown ratings before changing records."""
    data: dict[str, Any] = {"label_order": ["Weak", "Strong"], "records": [{"id": "a"}]}
    before = deepcopy(data)
    path = tmp_path / "ratings.csv"
    path.write_text("developer_id,project_id,rating\n" + rows)
    with pytest.raises(ValueError):
        merge_developers(data, path)
    assert data == before


def test_import_matches_ids_and_preserves_authors(tmp_path: Path) -> None:
    """Allow reordered exports and repeated imports without losing annotations."""
    data: dict[str, Any] = {
        "label_order": ["Weak", "Strong"],
        "records": [
            {"id": "a", "author_one": {"label": "Weak", "justification": "original"}},
            {"id": "b", "author_one": {"label": "Strong"}},
        ],
    }
    before = deepcopy(data)
    path = tmp_path / "ratings.csv"
    path.write_text(
        "developer_id,project_id,rating\n2,b,Weak\n1,a,Weak\n2,a,Strong\n1,b,Strong\n"
    )
    merge_developers(data, path)
    merged = deepcopy(data)
    merge_developers(data, path)
    assert data == merged
    for original, record in zip(before["records"], data["records"]):
        assert original["author_one"] == record["author_one"]
    assert data["records"][0]["developer_two"] == {"label": "Strong"}
    assert data["records"][1]["developer_one"] == {"label": "Strong"}
    path.write_text(path.read_text().replace("2,a,Strong", "2,a,Weak"))
    with pytest.raises(ValueError, match="conflicts"):
        merge_developers(data, path)
    assert data == merged
