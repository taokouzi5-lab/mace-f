"""Tests for the MACE-F decision matrix: thresholds and tier carbon values."""

import statistics as st
from pathlib import Path

import pandas as pd
import pytest

from carbon_benchmark.decision_matrix import TIER_PROFILES, MACEFramework

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
N_REPS = 3

# Method A training models per tier (same grouping as decision_matrix.TIER_PROFILES).
TIER_MODELS = {
    "T1": ["linear_regression", "decision_tree"],
    "T2": ["random_forest", "xgboost", "mobilenet_v2"],
    "T3": ["resnet_50", "lstm"],
    "T4": ["resnet_152", "vit_b_16"],
}


def _rep_totals_g(model: str) -> list[float]:
    """Sum the CodeCarbon `emissions` column (kg → g) for each rep of a model."""
    totals = []
    for rep in range(1, N_REPS + 1):
        path = RESULTS_DIR / f"E_methodA_{model}_rep{rep}_emissions.csv"
        df = pd.read_csv(path)
        totals.append(df["emissions"].sum() * 1000.0)
    return totals


def _expected_tier_median(models: list[str]) -> float:
    """Median of per-model medians (median of 3 reps per model)."""
    per_model_medians = [st.median(_rep_totals_g(m)) for m in models]
    return st.median(per_model_medians)


class TestCarbonClassThresholds:
    """Carbon class bands follow report Table 4: <0.1 / 0.1–1 / 1–100 / 100–1000 / ≥1000."""

    @pytest.mark.parametrize(
        ("total_annual_kg", "expected_letter"),
        [
            (0.05, "A"),
            (0.1, "B"),
            (0.5, "B"),
            (1.0, "C"),
            (50.0, "C"),
            (100.0, "D"),
            (500.0, "D"),
            (1000.0, "E"),
            (5000.0, "E"),
        ],
    )
    def test_carbon_class_thresholds_match_report_table4(
        self, total_annual_kg: float, expected_letter: str
    ) -> None:
        """Boundary values classify into the Table 4 bands A–E."""
        carbon_class, _ = MACEFramework._classify_and_recommend(
            total_annual_kg, "T1", "UK", "edge_device", False
        )
        assert carbon_class.startswith(expected_letter)


class TestTierTrainingCarbon:
    """TIER_PROFILES training carbon matches CSV medians (median of per-model medians)."""

    def test_tier_training_co2_matches_csv_medians(self) -> None:
        """T1–T4 training_co2_g equal the independently computed CSV medians."""
        for tier, models in TIER_MODELS.items():
            expected = _expected_tier_median(models)
            assert TIER_PROFILES[tier]["training_co2_g"] == pytest.approx(
                expected, rel=1e-6
            ), f"{tier} training_co2_g mismatch"

    def test_t5_training_carbon_is_none(self) -> None:
        """T5 has no training measurement (requires cloud/cluster)."""
        assert TIER_PROFILES["T5"]["training_co2_g"] is None
