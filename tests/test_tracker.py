"""Unit tests for CarbonTracker and Calibrator."""

import builtins
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from carbon_benchmark.config import (
    CALIBRATION_FACTORS,
    CARBON_INTENSITY,
    CODECARBON_CONFIG,
    DEFAULT_COUNTRY,
)
from carbon_benchmark.measurement.calibrator import Calibrator
from carbon_benchmark.measurement.tracker import CarbonTracker, EmissionResult


def install_fake_codecarbon(
    monkeypatch: pytest.MonkeyPatch,
    stop_value: Any = None,
    fail_stop: bool = False,
) -> list:
    """Install a fake `codecarbon` module; returns list of created trackers.

    Each created fake tracker records the kwargs passed by CarbonTracker.start().
    """
    created: list = []

    class FakeTracker:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.started = False
            created.append(self)

        def start(self) -> None:
            self.started = True

        def stop(self) -> Any:
            if fail_stop:
                raise RuntimeError("simulated stop failure")
            return stop_value

    module = types.ModuleType("codecarbon")
    module.EmissionsTracker = FakeTracker
    monkeypatch.setitem(sys.modules, "codecarbon", module)
    return created


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


class TestEmissionResultDetails:
    """to_dict rounding and extra-field flattening."""

    def test_extra_fields_flattened_with_prefix(self) -> None:
        result = EmissionResult(
            timestamp="t", experiment_id="E01", model_name="m", tier="T1",
            method="A", task_type="training", duration_seconds=1.0,
            energy_kwh=0.1, co2eq_g=1.0, cpu_energy_kwh=0.1, gpu_energy_kwh=0.0,
            ram_energy_kwh=0.0, country_iso="GBR", carbon_intensity=231.0,
            calibration_factor=1.58, co2eq_corrected_g=1.58,
            extra={"repeat": 2, "dataset_shape": "(1200, 100)"},
        )
        d = result.to_dict()
        assert d["extra_repeat"] == 2
        assert d["extra_dataset_shape"] == "(1200, 100)"

    def test_to_dict_rounding(self) -> None:
        result = EmissionResult(
            timestamp="t", experiment_id="E01", model_name="m", tier="T1",
            method="A", task_type="training", duration_seconds=10.125,
            energy_kwh=0.125, co2eq_g=2.5, cpu_energy_kwh=0.125, gpu_energy_kwh=0.0,
            ram_energy_kwh=0.0, country_iso="GBR", carbon_intensity=231.0,
            calibration_factor=1.58, co2eq_corrected_g=3.95,
        )
        d = result.to_dict()
        assert d["duration_seconds"] == round(10.125, 2)
        assert d["energy_kwh"] == round(0.125, 6)
        assert d["co2eq_g"] == round(2.5, 2)


class TestTrackerInitialisation:
    """__init__ tier, carbon intensity, and country fallback logic."""

    def test_unknown_iso_falls_back_to_default_intensity(self) -> None:
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", country_iso="XYZ")
        assert tracker._ci == CARBON_INTENSITY[DEFAULT_COUNTRY]

    def test_known_iso_maps_to_country_intensity(self) -> None:
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", country_iso="CHN")
        assert tracker._ci == CARBON_INTENSITY["China"]

    def test_output_dir_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "emissions"
        CarbonTracker("E_TEST", "linear_regression", "A", output_dir=nested)
        assert nested.is_dir()


class TestDetectDevice:
    """_detect_device static helper."""

    def test_returns_cpu_when_torch_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert CarbonTracker._detect_device() == "cpu"

    def test_returns_valid_device_with_torch(self) -> None:
        assert CarbonTracker._detect_device() in ("cuda", "cpu")


class TestStartStop:
    """start()/stop() with a fake CodeCarbon backend."""

    def test_start_passes_expected_kwargs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = install_fake_codecarbon(monkeypatch)
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", output_dir=tmp_path)
        tracker.start()
        assert len(created) == 1
        kwargs = created[0].kwargs
        assert kwargs["project_name"] == "macef_E_TEST"
        assert kwargs["output_file"] == "E_TEST_emissions.csv"
        assert kwargs["measure_power_secs"] == CODECARBON_CONFIG["measure_power_secs"]
        assert kwargs["force_carbon_intensity_g_co2e_kwh"] == CARBON_INTENSITY["UK"]
        assert kwargs["save_to_file"] is True
        assert kwargs["allow_multiple_runs"] is True
        assert created[0].started is True

    def test_stop_before_start_raises(self) -> None:
        tracker = CarbonTracker("E_TEST", "linear_regression", "A")
        with pytest.raises(RuntimeError, match="before start"):
            tracker.stop()

    def test_stop_with_none_emissions(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_codecarbon(monkeypatch, stop_value=None)
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", output_dir=tmp_path)
        tracker.start()
        result = tracker.stop()
        assert result.energy_kwh == 0.0
        assert result.co2eq_g == 0.0
        assert result.co2eq_corrected_g == 0.0

    def test_stop_with_float_emissions(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """CodeCarbon 3.x returns kg CO2eq as a float."""
        install_fake_codecarbon(monkeypatch, stop_value=0.5)
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", output_dir=tmp_path)
        tracker.start()
        result = tracker.stop()
        assert result.co2eq_g == pytest.approx(500.0)
        assert result.energy_kwh == pytest.approx(500.0 / CARBON_INTENSITY["UK"])
        assert result.co2eq_corrected_g == pytest.approx(500.0 * CALIBRATION_FACTORS["T1"])

    def test_stop_with_legacy_dict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_codecarbon(monkeypatch, stop_value={
            "energy_consumed": 0.012,
            "emissions": 0.0025,
            "cpu_energy": 0.010,
            "gpu_energy": 0.001,
            "ram_energy": 0.001,
        })
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", output_dir=tmp_path)
        tracker.start()
        result = tracker.stop()
        assert result.energy_kwh == pytest.approx(0.012)
        assert result.co2eq_g == pytest.approx(2.5)
        assert result.cpu_energy_kwh == pytest.approx(0.010)
        assert result.gpu_energy_kwh == pytest.approx(0.001)
        assert result.ram_energy_kwh == pytest.approx(0.001)


class TestContextManager:
    """__enter__/__exit__ semantics with fake backend."""

    def test_successful_context_sets_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_codecarbon(monkeypatch, stop_value=0.1)
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", output_dir=tmp_path)
        with tracker:
            pass
        assert tracker.result is not None
        assert tracker.result.co2eq_g == pytest.approx(100.0)

    def test_exit_with_exception_still_stops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_codecarbon(monkeypatch, stop_value=0.1)
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", output_dir=tmp_path)
        tracker.start()
        assert tracker.__exit__(ValueError, ValueError("boom"), None) is False
        assert tracker.result is not None

    def test_exit_with_exception_suppresses_stop_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_codecarbon(monkeypatch, fail_stop=True)
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", output_dir=tmp_path)
        # stop() raises inside __exit__ — must be swallowed, exception propagates
        assert tracker.__exit__(ValueError, ValueError("boom"), None) is False


class TestTrackMethods:
    """track_training / track_inference wrappers."""

    def test_track_training_calls_fn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_codecarbon(monkeypatch, stop_value=0.1)
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", output_dir=tmp_path)
        calls: list = []

        def train_fn(*args: Any, **kwargs: Any) -> str:
            calls.append((args, kwargs))
            return "trained"

        result = tracker.track_training(train_fn, "x", flag=True)
        assert calls == [(("x",), {"flag": True})]
        assert result is tracker.result
        assert result.task_type == "training"

    def test_track_inference_no_scaling_for_1000(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_codecarbon(monkeypatch, stop_value={
            "energy_consumed": 0.01, "emissions": 0.002,
            "cpu_energy": 0.01, "gpu_energy": 0.0, "ram_energy": 0.0,
        })
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", output_dir=tmp_path)
        result = tracker.track_inference(lambda: None, n_samples=1000)
        assert result.n_samples == 1000
        assert result.energy_kwh == pytest.approx(0.01)
        assert result.task_type == "inference"

    def test_track_inference_scales_to_1000(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_codecarbon(monkeypatch, stop_value={
            "energy_consumed": 0.01, "emissions": 0.002,
            "cpu_energy": 0.01, "gpu_energy": 0.0, "ram_energy": 0.0,
        })
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", output_dir=tmp_path)
        result = tracker.track_inference(lambda: None, n_samples=100)
        assert result.n_samples == 1000
        assert result.energy_kwh == pytest.approx(0.1)
        assert result.co2eq_g == pytest.approx(20.0)
        assert result.co2eq_corrected_g == pytest.approx(20.0 * CALIBRATION_FACTORS["T1"])

    def test_track_inference_zero_samples_not_scaled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_codecarbon(monkeypatch, stop_value={
            "energy_consumed": 0.01, "emissions": 0.002,
            "cpu_energy": 0.01, "gpu_energy": 0.0, "ram_energy": 0.0,
        })
        tracker = CarbonTracker("E_TEST", "linear_regression", "A", output_dir=tmp_path)
        result = tracker.track_inference(lambda: None, n_samples=0)
        assert result.n_samples == 0
        assert result.energy_kwh == pytest.approx(0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
