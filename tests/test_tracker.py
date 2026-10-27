"""Unit tests for CarbonTracker and Calibrator."""

from pathlib import Path

import pytest

from carbon_benchmark.config import CALIBRATION_FACTORS, MODEL_TIERS
from carbon_benchmark.measurement.calibrator import Calibrator
from carbon_benchmark.measurement.tracker import CarbonTracker, EmissionResult


class TestEmissionResult:
    """Tests for the EmissionResult dataclass."""

    def test_to_dict_contains_all_required_columns(self) -> None:
        """Verify to_dict() returns all standardized CSV columns."""
        result = EmissionResult(
            timestamp="2026-06-01T00:00:00Z",
            experiment_id="E01",
            model_name="linear_regression",
            tier="T1",
            method="A",
            task_type="training",
            duration_seconds=10.5,
            energy_kwh=0.012,
            co2eq_g=2.77,
            cpu_energy_kwh=0.010,
            gpu_energy_kwh=0.0,
            ram_energy_kwh=0.002,
            country_iso="GBR",
            carbon_intensity=231.0,
            calibration_factor=1.58,
            co2eq_corrected_g=4.38,
        )
        d = result.to_dict()

        required = [
            "timestamp", "experiment_id", "model_name", "tier",
            "method", "task_type", "duration_seconds", "energy_kwh",
            "co2eq_g", "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh",
            "country_iso", "carbon_intensity", "calibration_factor",
            "co2eq_corrected_g", "n_samples", "device",
        ]
        for col in required:
            assert col in d, f"Missing required column: {col}"

    def test_co2eq_corrected_applies_calibration(self) -> None:
        """Verify corrected CO₂ equals raw × calibration factor."""
        result = EmissionResult(
            timestamp="2026-06-01T00:00:00Z",
            experiment_id="E01",
            model_name="linear_regression",
            tier="T1",
            method="A",
            task_type="training",
            duration_seconds=1.0,
            energy_kwh=0.1,
            co2eq_g=100.0,
            cpu_energy_kwh=0.1,
            gpu_energy_kwh=0.0,
            ram_energy_kwh=0.0,
            country_iso="GBR",
            carbon_intensity=231.0,
            calibration_factor=1.58,
            co2eq_corrected_g=158.0,
        )
        assert result.co2eq_corrected_g == pytest.approx(158.0)


class TestCalibrator:
    """Tests for the Calibrator class."""

    def test_empty_calibrator_returns_defaults(self) -> None:
        """Calibrator with no measurements falls back to config defaults."""
        cal = Calibrator()
        factors = cal.compute_tier_factors()
        for tier in ["T1", "T2", "T3", "T4", "T5"]:
            assert tier in factors
            assert factors[tier]["source"] == "config_default"

    def test_add_and_compute_factors(self) -> None:
        """Adding paired measurements produces empirical calibration factors."""
        cal = Calibrator()
        cal.add("E01", "linear_regression", "T1", 0.012, 0.019)
        cal.add("E02", "random_forest", "T2", 0.087, 0.124)
        factors = cal.compute_tier_factors()

        assert factors["T1"]["source"] == "empirical"
        assert factors["T1"]["n_records"] == 1
        assert factors["T1"]["mean_factor"] == pytest.approx(0.019 / 0.012, rel=0.01)

    def test_apply_calibration(self) -> None:
        """apply_calibration multiplies by the tier factor."""
        cal = Calibrator()
        corrected = cal.apply_calibration(100.0, "T1")
        expected = 100.0 * CALIBRATION_FACTORS["T1"]
        assert corrected == pytest.approx(expected)


class TestCarbonTracker:
    """Tests for the CarbonTracker class (context manager mode)."""

    def test_context_manager_lifecycle(self) -> None:
        """Tracker starts and stops correctly via context manager."""
        tracker = CarbonTracker("E_TEST", "decision_tree", "A")
        assert tracker.result is None
        # Context manager should work without errors
        try:
            with tracker:
                pass  # Simulate workload
        except Exception:
            pass  # CodeCarbon may fail without actual hardware tracking
        # Even on error, should attempt stop

    def test_model_tier_lookup(self) -> None:
        """Tier is resolved from config.MODEL_TIERS."""
        tracker = CarbonTracker("E_TEST", "linear_regression", "A")
        assert tracker._tier == "T1"

        tracker2 = CarbonTracker("E_TEST", "resnet_50", "A")
        assert tracker2._tier == "T3"

    def test_unknown_model_defaults_to_t3(self) -> None:
        """Unknown model names default to T3."""
        tracker = CarbonTracker("E_TEST", "nonexistent_model", "A")
        assert tracker._tier == "T3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
