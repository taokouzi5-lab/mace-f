"""Tests for the MACE-F decision matrix: thresholds and tier carbon values."""

import runpy
import statistics as st
from pathlib import Path

import pandas as pd
import pytest

from carbon_benchmark.decision_matrix import (
    DEPLOYMENT_SCENARIOS,
    LOCATION_FACTORS,
    MACEFramework,
    TIER_PROFILES,
    generate_decision_table,
)

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


class TestTierLookup:
    """get_tier_profile / tier_for_model lookups."""

    def test_get_tier_profile_returns_copy(self) -> None:
        fw = MACEFramework()
        profile = fw.get_tier_profile("T1")
        profile["label"] = "mutated"
        assert fw.get_tier_profile("T1")["label"] == "Ultra-Lightweight ML"

    def test_get_tier_profile_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown tier 'T9'"):
            MACEFramework().get_tier_profile("T9")

    @pytest.mark.parametrize(
        ("model_name", "expected_tier"),
        [
            ("Linear Regression", "T1"),
            ("Random Forest", "T2"),
            ("ResNet-50", "T3"),
            ("ViT-B/16", "T4"),
            ("Stable Diffusion", "T5"),
        ],
    )
    def test_tier_for_model(self, model_name: str, expected_tier: str) -> None:
        assert MACEFramework().tier_for_model(model_name) == expected_tier

    def test_tier_for_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown model"):
            MACEFramework().tier_for_model("NoSuchModel")


class TestMultipliers:
    """Location and infrastructure factor lookups."""

    def test_location_factors(self) -> None:
        fw = MACEFramework()
        assert fw.get_location_factor("UK") == 1.0
        assert fw.get_location_factor("China") == pytest.approx(2.52)

    def test_location_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown country 'Atlantis'"):
            MACEFramework().get_location_factor("Atlantis")

    def test_infrastructure_factors(self) -> None:
        fw = MACEFramework()
        assert fw.get_infrastructure_factor("edge_device") == 1.0
        assert fw.get_infrastructure_factor("factory_floor") == pytest.approx(1.96)

    def test_infrastructure_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown infrastructure 'floppy'"):
            MACEFramework().get_infrastructure_factor("floppy")


class TestAssess:
    """Full assessment arithmetic and branches."""

    def test_basic_assessment_arithmetic(self) -> None:
        fw = MACEFramework()
        r = fw.assess(
            model_tier="T3", deployment_scenario="batch_qc",
            country="UK", infrastructure="on_premise_datacenter",
            use_transfer_learning=False,
        )
        tier = TIER_PROFILES["T3"]
        loc = LOCATION_FACTORS["UK"]["factor"]
        pue_factor = 1.67  # on_premise_datacenter
        expected_training = round(tier["training_co2_g"] * loc * pue_factor, 3)
        expected_single = tier["annual_infer_co2_kg_24x7"] * 1000 / 31_536_000
        annual_n = DEPLOYMENT_SCENARIOS["batch_qc"]["inferences_per_year"]
        expected_annual = round(
            expected_single * annual_n / 1000 * loc * pue_factor, 3
        )
        assert r.training_co2_g == expected_training
        assert r.single_inference_co2_g == round(expected_single, 8)
        assert r.annual_inferences == annual_n
        assert r.annual_co2eq_kg == expected_annual
        assert r.tier_label == "Standard Deep Learning"
        assert r.transfer_learning_saving == 0.0

    def test_t5_training_none_treated_as_zero(self) -> None:
        r = MACEFramework().assess(
            model_tier="T5", deployment_scenario="continuous_24x7",
            country="UK", infrastructure="edge_device",
        )
        assert r.training_co2_g == 0.0
        assert r.carbon_class.startswith("E")
        assert any("T5" in rec for rec in r.recommendations)

    def test_transfer_learning_applied_to_t2_t4(self) -> None:
        r = MACEFramework().assess(
            model_tier="T3", deployment_scenario="batch_qc",
            country="UK", infrastructure="edge_device", use_transfer_learning=True,
        )
        raw = TIER_PROFILES["T3"]["training_co2_g"]
        assert r.training_co2_g == pytest.approx(round(raw * (1 - 0.606), 3))
        assert r.transfer_learning_saving == pytest.approx(0.606)

    def test_transfer_learning_ignored_for_t1(self) -> None:
        r = MACEFramework().assess(
            model_tier="T1", deployment_scenario="batch_qc",
            country="UK", infrastructure="edge_device", use_transfer_learning=True,
        )
        assert r.transfer_learning_saving == 0.0
        assert r.training_co2_g == pytest.approx(
            round(TIER_PROFILES["T1"]["training_co2_g"], 3)
        )

    def test_data_modality_override(self) -> None:
        r = MACEFramework().assess(
            model_tier="T1", data_modality="image",
        )
        assert r.data_modality == "image"

    def test_default_data_modality_joins_tier_modalities(self) -> None:
        r = MACEFramework().assess(model_tier="T1")
        assert r.data_modality == "tabular, time-series"

    def test_invalid_scenario_raises(self) -> None:
        with pytest.raises(KeyError):
            MACEFramework().assess(model_tier="T1", deployment_scenario="nope")

    def test_tier_without_annual_infer_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A tier with no annual-inference data yields zero inference estimates."""
        import carbon_benchmark.decision_matrix as dm

        patched = {k: dict(v) for k, v in TIER_PROFILES.items()}
        patched["T1"] = dict(patched["T1"], annual_infer_co2_kg_24x7=None)
        monkeypatch.setattr(dm, "TIER_PROFILES", patched)
        r = MACEFramework().assess(model_tier="T1")
        assert r.single_inference_co2_g == 0.0
        assert r.annual_co2eq_kg == 0.0

    def test_invalid_tier_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown tier"):
            MACEFramework().assess(model_tier="T9")

    def test_high_grid_country_recommendation(self) -> None:
        r = MACEFramework().assess(
            model_tier="T3", deployment_scenario="continuous_24x7",
            country="China", infrastructure="edge_device",
        )
        assert any("High-carbon grid" in rec for rec in r.recommendations)

    def test_t3_t4_without_tl_recommends_transfer_learning(self) -> None:
        r = MACEFramework().assess(
            model_tier="T4", deployment_scenario="batch_qc",
            country="Sweden", infrastructure="edge_device",
            use_transfer_learning=False,
        )
        assert any("transfer learning" in rec for rec in r.recommendations)


class TestReductionSummary:
    """_reduction_summary branch coverage."""

    def test_t5_lever(self) -> None:
        assert "Replace generative AI" in MACEFramework._reduction_summary(
            "T5", "UK", "edge_device"
        )

    def test_high_grid_and_high_pue(self) -> None:
        assert "Migrate to cloud" in MACEFramework._reduction_summary(
            "T1", "China", "on_premise_datacenter"
        )

    def test_high_grid_only(self) -> None:
        assert "Deploy in low-carbon region" in MACEFramework._reduction_summary(
            "T1", "China", "edge_device"
        )

    def test_t3_t4_lever(self) -> None:
        assert "Transfer learning" in MACEFramework._reduction_summary(
            "T3", "UK", "edge_device"
        )

    def test_low_impact_default(self) -> None:
        assert "already low" in MACEFramework._reduction_summary(
            "T1", "Sweden", "google_cloud"
        )

    def test_unknown_country_and_infra_defaults(self) -> None:
        assert "already low" in MACEFramework._reduction_summary(
            "T1", "Atlantis", "floppy"
        )


class TestClassifyRecommendations:
    """Recommendation branches beyond the class bands."""

    def test_t5_recommendation(self) -> None:
        _, recs = MACEFramework._classify_and_recommend(
            5000.0, "T5", "UK", "edge_device", False
        )
        assert any("T5 generative models dominate" in r for r in recs)

    def test_t3_without_tl_recommendation(self) -> None:
        _, recs = MACEFramework._classify_and_recommend(
            50.0, "T3", "UK", "edge_device", False
        )
        assert any("Apply transfer learning" in r for r in recs)

    def test_t3_with_tl_skips_recommendation(self) -> None:
        _, recs = MACEFramework._classify_and_recommend(
            50.0, "T3", "UK", "edge_device", True
        )
        assert not any("Apply transfer learning" in r for r in recs)

    def test_critical_grid_recommendation(self) -> None:
        _, recs = MACEFramework._classify_and_recommend(
            500.0, "T2", "South Africa", "edge_device", False
        )
        assert any("High-carbon grid" in r for r in recs)

    def test_low_grid_no_grid_recommendation(self) -> None:
        _, recs = MACEFramework._classify_and_recommend(
            500.0, "T2", "Sweden", "edge_device", False
        )
        assert not any("High-carbon grid" in r for r in recs)


class TestGenerateDecisionTable:
    def test_full_cross_product(self) -> None:
        rows = generate_decision_table()
        # 5 tiers × 3 scenarios × 3 countries × 2 infrastructures
        assert len(rows) == 90
        first = rows[0]
        for key in ("tier", "scenario", "country", "infra",
                    "training_co2_g", "annual_co2eq_kg", "carbon_class"):
            assert key in first

    def test_rows_match_assess(self) -> None:
        fw = MACEFramework()
        sample = generate_decision_table()[0]
        r = fw.assess(
            model_tier=sample["tier"],
            deployment_scenario=sample["scenario"],
            country=sample["country"],
            infrastructure=sample["infra"],
            use_transfer_learning=sample["tier"] in ("T2", "T3", "T4"),
        )
        assert sample["training_co2_g"] == r.training_co2_g
        assert sample["annual_co2eq_kg"] == r.annual_co2eq_kg
        assert sample["carbon_class"] == r.carbon_class

    def test_all_tiers_present(self) -> None:
        tiers = {row["tier"] for row in generate_decision_table()}
        assert tiers == {"T1", "T2", "T3", "T4", "T5"}


class TestCliDemo:
    def test_main_demo_runs(self, capsys: pytest.CaptureFixture) -> None:
        """Execute the __main__ demo block via runpy (pure computation)."""
        module_path = REPO_ROOT / "decision_matrix.py"
        runpy.run_path(str(module_path), run_name="__main__")
        out = capsys.readouterr().out
        assert "MACE-F Carbon-Efficiency Decision Framework" in out
        assert "Case: Semiconductor yield prediction" in out
        assert "Case: Generative design tool" in out
        assert "Framework loaded" in out
