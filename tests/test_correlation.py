"""Tests for the cross-model Spearman correlation."""

import json
import math
from pathlib import Path

import pandas as pd

from src.analysis.correlation import load_overall_metric, spearman_rho


class TestSpearmanRho:
    """Test the rank correlation itself."""

    def test_opposite_order_is_minus_one(self) -> None:
        """Values in exactly reversed order should give rho = -1."""
        rho = spearman_rho(
            x=pd.Series([0.1, 0.2, 0.3, 0.4]),
            y=pd.Series([0.9, 0.7, 0.5, 0.1]),
        )
        assert rho == -1.0

    def test_ties_share_average_rank(self) -> None:
        """Tied values should take their average rank."""
        # x ranks are [1, 2.5, 2.5, 4], which gives rho = 3 / sqrt(10)
        rho = spearman_rho(
            x=pd.Series([1, 2, 2, 3]),
            y=pd.Series([1, 2, 3, 4]),
        )
        assert math.isclose(rho, 3 / math.sqrt(10))


class TestLoadOverallMetric:
    """Test reading per-model averages from evaluation summaries."""

    def test_reads_overall_block(self, tmp_path: Path) -> None:
        """Should return the overall value for each model, indexed by model."""
        for model, rate in [("model-a", 0.25), ("model-b", 0.75)]:
            (tmp_path / model).mkdir()
            (tmp_path / model / "def-evaluation.json").write_text(
                json.dumps({"summary": {"overall": {"suitable_rate": rate}}})
            )
        values = load_overall_metric(
            metric="suitable_rate",
            models=["model-a", "model-b"],
            output_dir=tmp_path,
        )
        assert values.to_dict() == {"model-a": 0.25, "model-b": 0.75}
