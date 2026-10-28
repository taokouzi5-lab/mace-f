"""Smoke tests for the experiment modules.

Import smoke + CLI argument parsing + mocked end-to-end runs of the `run()`
entry points.  Data loading, model factories, and CarbonTracker are all
monkeypatched so no real training, hardware measurement, or network access
happens.  Methods E/F/G are pure computations and run for real (with tmp_path
output directories).
"""

import inspect
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from carbon_benchmark.config import CALIBRATION_FACTORS, CARBON_INTENSITY, MODEL_TIERS
from carbon_benchmark.experiments import methods_bc_efg as bc
from carbon_benchmark.experiments import method_a_baseline_training as ma
from carbon_benchmark.experiments import method_d_baseline_inference as md
from carbon_benchmark.measurement.tracker import EmissionResult


def make_result(**overrides: Any) -> EmissionResult:
    defaults: dict = dict(
        timestamp="2026-06-01T00:00:00Z",
        experiment_id="E_TEST",
        model_name="decision_tree",
        tier="T1",
        method="A",
        task_type="training",
        duration_seconds=1.0,
        energy_kwh=0.05,
        co2eq_g=10.0,
        cpu_energy_kwh=0.05,
        gpu_energy_kwh=0.0,
        ram_energy_kwh=0.0,
        country_iso="GBR",
        carbon_intensity=231.0,
        calibration_factor=1.58,
        co2eq_corrected_g=15.8,
    )
    defaults.update(overrides)
    return EmissionResult(**defaults)


def _tiny_data(n: int = 20, n_features: int = 3, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n, n_features)).astype(np.float32)
    X_test = rng.normal(size=(6, n_features)).astype(np.float32)
    y_train = rng.integers(0, 2, n)
    y_test = rng.integers(0, 2, 6)
    return X_train, X_test, y_train, y_test


def _fake_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"carbon_benchmark.tests.{name}")


# ---------------------------------------------------------------------------
# Import smoke
# ---------------------------------------------------------------------------

class TestImportSmoke:
    def test_experiments_package_exports(self) -> None:
        from carbon_benchmark.experiments import run_method_a, run_method_d

        assert callable(run_method_a)
        assert callable(run_method_d)

    def test_registries_populated(self) -> None:
        assert len(ma.TRAINING_REGISTRY) == 9
        assert len(md.INFERENCE_REGISTRY) == 11
        assert set(ma.TRAINING_REGISTRY) == set(MODEL_TIERS) - {
            "stable_diffusion", "gpt_4o_mini",
        }
        assert set(md.INFERENCE_REGISTRY) == set(MODEL_TIERS)

    def test_run_signatures(self) -> None:
        a_params = inspect.signature(ma.run).parameters
        assert set(a_params) == {
            "model_name", "country_iso", "output_dir", "n_repeats", "logger",
        }
        d_params = inspect.signature(md.run).parameters
        assert set(d_params) == {
            "model_name", "country_iso", "output_dir", "n_samples", "logger",
        }


# ---------------------------------------------------------------------------
# Method A
# ---------------------------------------------------------------------------

class TestMethodARun:
    def test_unknown_model_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="not in training registry"):
            ma.run("not_a_model", logger=_fake_logger("ma_unknown"))

    def test_mocked_full_flow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        X_train, X_test, y_train, y_test = _tiny_data()

        spec = {
            "loader": lambda: (X_train, X_test, y_train, y_test),
            "preprocessor": lambda a, b, c, d: (a, b, c, d),
            "factory": _fake_tree_factory,
            "task": "classification",
            "n_repeats": 3,
        }
        monkeypatch.setitem(ma.TRAINING_REGISTRY, "decision_tree", spec)

        tracker_created: list = []

        class FakeTracker:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                tracker_created.append(kwargs)

            def track_training(self, fn: Any) -> EmissionResult:
                fn()  # exercise the adapted train_fn (incl. TypeError branch)
                return make_result(experiment_id=self.kwargs["experiment_id"])

        monkeypatch.setattr(ma, "CarbonTracker", FakeTracker)

        results = ma.run(
            "decision_tree", country_iso="GBR", output_dir=tmp_path,
            logger=_fake_logger("ma_mocked"),
        )
        assert len(results) == 3
        assert all(r["experiment_id"].startswith("E_methodA_decision_tree_rep") for r in results)
        assert all("extra_repeat" in r for r in results)
        assert all("extra_dataset_shape" in r for r in results)
        assert all("extra_metric_accuracy" in r for r in results)
        assert len(tracker_created) == 3
        summary = tmp_path / "decision_tree_training.csv"
        assert summary.exists()
        assert len(pd.read_csv(summary)) == 3


def _fake_tree_factory() -> tuple:
    def train_fn(*args: Any, **kwargs: Any) -> str:
        if len(args) == 2:
            raise TypeError("needs four positional args")
        return "trained"

    def inference_fn(X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.int64)

    return "model", train_fn, inference_fn


class TestMethodAMain:
    def test_cli_parses_args(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list = []
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "--model", "decision_tree", "--country", "CHN",
             "--output-dir", str(tmp_path), "--repeats", "1"],
        )
        monkeypatch.setattr(ma, "print_hardware_summary", lambda: None)

        def fake_run(**kwargs: Any) -> list:
            calls.append(kwargs)
            return []

        monkeypatch.setattr(ma, "run", fake_run)
        ma.main()
        assert len(calls) == 1
        assert calls[0]["model_name"] == "decision_tree"
        assert calls[0]["country_iso"] == "CHN"
        assert calls[0]["output_dir"] == tmp_path
        assert calls[0]["n_repeats"] == 1

    def test_cli_defaults_output_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carbon_benchmark.config import RESULTS_DIR

        calls: list = []
        monkeypatch.setattr(sys, "argv", ["prog", "--model", "decision_tree"])
        monkeypatch.setattr(ma, "print_hardware_summary", lambda: None)
        monkeypatch.setattr(ma, "run", lambda **kw: calls.append(kw) or [])
        ma.main()
        assert calls[0]["output_dir"] == RESULTS_DIR
        assert calls[0]["n_repeats"] is None


# ---------------------------------------------------------------------------
# Method D
# ---------------------------------------------------------------------------

class TestLoggerDefaulting:
    """run() with logger=None must call setup_logging."""

    def test_method_a_run_defaults_logger(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list = []
        monkeypatch.setattr(
            ma, "setup_logging",
            lambda name: calls.append(name) or _fake_logger("ma_default"),
        )
        with pytest.raises(KeyError):
            ma.run("not_a_model", logger=None)
        assert calls == ["E_methodA_not_a_model"]

    def test_method_d_run_defaults_logger_and_delegates_generative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setup_calls: list = []
        gen_calls: list = []
        monkeypatch.setattr(
            md, "setup_logging",
            lambda name: setup_calls.append(name) or _fake_logger("md_default"),
        )
        monkeypatch.setattr(
            md, "_run_generative_inference",
            lambda *a: gen_calls.append(a) or [],
        )
        md.run("stable_diffusion", output_dir=tmp_path, logger=None)
        assert setup_calls == ["E_methodD_stable_diffusion"]
        assert gen_calls[0][0] == "stable_diffusion"

    def test_method_e_defaults_logger(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list = []
        monkeypatch.setattr(
            bc, "setup_logging",
            lambda name: calls.append(name) or _fake_logger("me_default"),
        )
        bc.run_method_e({"m": 0.01}, {}, output_dir=tmp_path, logger=None)
        assert calls == ["E_methodE_deployment"]

    def test_method_f_defaults_logger(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list = []
        monkeypatch.setattr(
            bc, "setup_logging",
            lambda name: calls.append(name) or _fake_logger("mf_default"),
        )
        bc.run_method_f(0.5, output_dir=tmp_path, logger=None)
        assert calls == ["E_methodF_carbon_intensity"]

    def test_method_g_defaults_logger(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list = []
        monkeypatch.setattr(
            bc, "setup_logging",
            lambda name: calls.append(name) or _fake_logger("mg_default"),
        )
        bc.run_method_g(1.0, output_dir=tmp_path, logger=None)
        assert calls == ["E_methodG_deployment_topology"]


class TestMethodDRun:
    def test_unknown_model_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="not in inference registry"):
            md.run("not_a_model", logger=_fake_logger("md_unknown"))

    def test_mocked_full_flow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        X_train, X_test, y_train, y_test = _tiny_data()

        spec = {
            "loader": lambda: (X_train, X_test, y_train, y_test),
            "preprocessor": lambda a, b, c, d: (a, b, c, d),
            "factory": _fake_tree_factory,
            "n_samples": 1000,
        }
        monkeypatch.setitem(md.INFERENCE_REGISTRY, "decision_tree", spec)

        class FakeTracker:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

            def track_inference(self, fn: Any, n_samples: int) -> EmissionResult:
                fn()
                return make_result(experiment_id=self.kwargs["experiment_id"])

        monkeypatch.setattr(md, "CarbonTracker", FakeTracker)

        results = md.run(
            "decision_tree", country_iso="GBR", output_dir=tmp_path,
            logger=_fake_logger("md_mocked"),
        )
        assert len(results) == 1
        assert results[0]["experiment_id"] == "E_methodD_decision_tree"
        assert (tmp_path / "summary.csv").exists()

    def test_generative_sd_branch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        gen_calls: list = []

        def fake_generate_fn(**kwargs: Any) -> list:
            gen_calls.append(kwargs)
            return []

        monkeypatch.setattr(md, "create_stable_diffusion_pipeline",
                            lambda: (None, fake_generate_fn, None))

        class FakeTracker:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

            def track_inference(self, fn: Any, n_samples: int) -> EmissionResult:
                fn()
                return make_result()

        monkeypatch.setattr(md, "CarbonTracker", FakeTracker)
        results = md._run_generative_inference(
            "stable_diffusion", "GBR", tmp_path, 100, _fake_logger("md_sd"),
        )
        assert len(results) == 1
        # 100 samples → 10 prompts × 10 images each
        assert gen_calls[0]["n_per_prompt"] == 10
        assert len(gen_calls[0]["prompts"]) == 10
        assert (tmp_path / "stable_diffusion_inference.csv").exists()

    def test_generative_gpt_branch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        query_calls: list = []

        def fake_query_fn() -> list:
            query_calls.append(True)
            return []

        monkeypatch.setattr(md, "create_gpt_manufacturing_queries",
                            lambda: (None, fake_query_fn, None))

        class FakeTracker:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

            def track_inference(self, fn: Any, n_samples: int) -> EmissionResult:
                fn()
                return make_result()

        monkeypatch.setattr(md, "CarbonTracker", FakeTracker)
        results = md._run_generative_inference(
            "gpt_4o_mini", "GBR", tmp_path, 3, _fake_logger("md_gpt"),
        )
        assert len(results) == 1
        assert query_calls == [True]
        assert (tmp_path / "gpt_4o_mini_inference.csv").exists()

    def test_generative_unknown_model_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown generative model"):
            md._run_generative_inference(
                "resnet_50", "GBR", tmp_path, 100, _fake_logger("md_gen_unknown"),
            )


class TestMethodDMain:
    def test_cli_parses_args(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list = []
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "--model", "resnet_50", "--country", "GBR",
             "--output-dir", str(tmp_path), "--n-samples", "500"],
        )
        monkeypatch.setattr(md, "print_hardware_summary", lambda: None)
        monkeypatch.setattr(md, "run", lambda **kw: calls.append(kw) or [])
        md.main()
        assert calls[0]["model_name"] == "resnet_50"
        assert calls[0]["output_dir"] == tmp_path
        assert calls[0]["n_samples"] == 500


# ---------------------------------------------------------------------------
# Methods E / F / G — pure computation, run for real
# ---------------------------------------------------------------------------

class TestMethodE:
    def test_annual_simulation_arithmetic(self, tmp_path: Path) -> None:
        kwh_per_1k = {"linear_regression": 0.01}
        training_g = {"linear_regression": 2.77}
        df = bc.run_method_e(
            kwh_per_1k, training_g, output_dir=tmp_path,
            logger=_fake_logger("me"),
        )
        assert len(df) == 3  # three scenarios × one model
        expected_cols = {
            "scenario", "model", "tier", "annual_inferences",
            "annual_energy_kwh", "annual_co2eq_kg", "training_co2eq_kg",
            "inference_pct",
        }
        assert expected_cols <= set(df.columns)
        row = df[df["scenario"] == "continuous_24x7"].iloc[0]
        annual_n = 86_400 * 365
        annual_energy = 0.01 * (annual_n / 1000)
        cf = CALIBRATION_FACTORS["T1"]
        expected_co2 = (annual_energy * CARBON_INTENSITY["UK"] * cf) / 1000.0
        assert row["annual_inferences"] == annual_n
        assert row["annual_energy_kwh"] == pytest.approx(round(annual_energy, 3))
        assert row["annual_co2eq_kg"] == pytest.approx(round(expected_co2, 3))
        assert row["tier"] == "T1"
        assert (tmp_path / "method_e_deployment_simulation.csv").exists()

    def test_unknown_model_tier_fallback(self, tmp_path: Path) -> None:
        df = bc.run_method_e(
            {"mystery_model": 0.01}, {}, output_dir=tmp_path,
            logger=_fake_logger("me_tier"),
        )
        assert set(df["tier"]) == {"T3"}
        assert set(df["model"]) == {"mystery_model"}


class TestMethodF:
    def test_sensitivity_arithmetic(self, tmp_path: Path) -> None:
        energy = 0.5
        df = bc.run_method_f(energy, output_dir=tmp_path, logger=_fake_logger("mf"))
        assert len(df) == len(CARBON_INTENSITY)
        assert list(df.columns) == [
            "country", "carbon_intensity_g_per_kwh", "energy_kwh",
            "co2eq_g", "relative_to_best",
        ]
        # Sorted by carbon intensity ascending
        assert df["carbon_intensity_g_per_kwh"].is_monotonic_increasing
        best = df.iloc[0]
        assert best["country"] == "Sweden"
        assert best["relative_to_best"] == 1.0
        cf = CALIBRATION_FACTORS["T3"]
        assert best["co2eq_g"] == round(energy * best["carbon_intensity_g_per_kwh"] * cf, 2)
        assert (tmp_path / "method_f_carbon_intensity_sensitivity.csv").exists()

    def test_unknown_model_uses_t3_factor(self, tmp_path: Path) -> None:
        df = bc.run_method_f(0.5, model_name="mystery", output_dir=tmp_path,
                             logger=_fake_logger("mf_tier"))
        cf = CALIBRATION_FACTORS["T3"]
        row = df[df["country"] == "Sweden"].iloc[0]
        assert row["co2eq_g"] == round(0.5 * 13 * cf, 2)


class TestMethodG:
    def test_topology_arithmetic(self, tmp_path: Path) -> None:
        energy = 1.0
        df = bc.run_method_g(energy, output_dir=tmp_path, logger=_fake_logger("mg"))
        assert len(df) == 8
        assert set(df["pue_source"]) == {
            "factory_floor", "on_premise_datacenter", "aws", "azure",
            "google_cloud", "edge_device",
        }
        edge = df[df["scenario"] == "Edge Device — Any"].iloc[0]
        cf = CALIBRATION_FACTORS["T3"]
        assert edge["total_energy_kwh"] == pytest.approx(round(energy * 1.02, 6))
        assert edge["co2eq_g"] == round(energy * 1.02 * 13 * cf, 2)
        worst = df[df["scenario"] == "Factory Floor — China"].iloc[0]
        assert worst["co2eq_g"] > edge["co2eq_g"]
        assert (tmp_path / "method_g_deployment_topology.csv").exists()


# ---------------------------------------------------------------------------
# Methods B / C — fully mocked (real code would train ResNet-50)
# ---------------------------------------------------------------------------

class TestMethodBMocked:
    def test_mocked_flow_with_inline_training_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("torch")
        import torch

        from carbon_benchmark.data import loader, preprocessing
        from carbon_benchmark.models import deep_learning

        X_train, X_test, y_train, y_test = _tiny_data(n=10, seed=1)
        monkeypatch.setattr(loader, "load_neu_steel_defects",
                            lambda: (X_train, X_test, y_train, y_test))
        monkeypatch.setattr(preprocessing, "preprocess_images",
                            lambda a, b, c, d: (a, b, c, d))

        class TinyFC(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(3, 6)

            def forward(self, x: Any) -> Any:
                return self.fc(x)

        monkeypatch.setattr(deep_learning, "create_resnet50",
                            lambda num_classes=6: (TinyFC(), None, None))

        failures: list = []

        class FakeTracker:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                self.result = make_result(energy_kwh=0.05)

            def track_training(self, *args: Any, **kwargs: Any) -> EmissionResult:
                if not failures:
                    failures.append(True)  # first combo triggers TypeError fallback
                    raise TypeError("inline training required")
                return self.result

            def __enter__(self) -> "FakeTracker":
                return self

            def __exit__(self, *exc: Any) -> bool:
                return False

        monkeypatch.setattr(bc, "CarbonTracker", FakeTracker)
        monkeypatch.setattr(bc, "setup_logging", lambda name: _fake_logger("mb"))
        df = bc.run_method_b(output_dir=tmp_path)
        assert len(df) == 8
        assert failures == [True]
        assert list(df.columns) == [
            "batch_size", "learning_rate", "energy_kwh", "co2eq_g", "duration_s",
        ]
        assert (df["energy_kwh"] > 0).all()
        assert (tmp_path / "method_b_hyperparameter_results.csv").exists()


class TestMethodCMocked:
    def test_mocked_flow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from carbon_benchmark.data import loader, preprocessing
        from carbon_benchmark.models import deep_learning

        X_train, X_test, y_train, y_test = _tiny_data(seed=2)
        monkeypatch.setattr(loader, "load_neu_steel_defects",
                            lambda: (X_train, X_test, y_train, y_test))
        monkeypatch.setattr(preprocessing, "preprocess_images",
                            lambda a, b, c, d: (a, b, c, d))
        monkeypatch.setattr(
            deep_learning, "compare_training_modes",
            lambda *a, **k: {"from_scratch": {}, "frozen_backbone": {}, "full_finetune": {}},
        )
        monkeypatch.setattr(deep_learning, "create_resnet50", lambda **kw: (None, None, None))

        energies = iter([0.1, 0.05, 0.12])

        class FakeTracker:
            def __init__(self, **kwargs: Any) -> None:
                self.result = make_result(
                    energy_kwh=next(energies),
                    experiment_id=kwargs["experiment_id"],
                )

            def track_training(self, *args: Any, **kwargs: Any) -> EmissionResult:
                return self.result

        monkeypatch.setattr(bc, "CarbonTracker", FakeTracker)
        monkeypatch.setattr(bc, "setup_logging", lambda name: _fake_logger("mc"))
        df = bc.run_method_c(output_dir=tmp_path)
        assert len(df) == 3
        assert list(df["mode"]) == ["from_scratch", "frozen_backbone", "full_finetune"]
        assert (tmp_path / "method_c_transfer_learning_results.csv").exists()
