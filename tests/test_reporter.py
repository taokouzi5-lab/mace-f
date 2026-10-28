"""Unit tests for ResultReporter: aggregation, CSV export, summary statistics."""

from pathlib import Path

import pandas as pd
import pytest

from carbon_benchmark.measurement.reporter import ResultReporter
from carbon_benchmark.measurement.tracker import EmissionResult


def make_result(
    experiment_id: str = "E01",
    model_name: str = "linear_regression",
    tier: str = "T1",
    method: str = "A",
    task_type: str = "training",
    duration_seconds: float = 10.0,
    energy_kwh: float = 0.012,
    co2eq_g: float = 2.77,
    co2eq_corrected_g: float = 4.38,
) -> EmissionResult:
    """Build a fully populated EmissionResult for reporter tests."""
    return EmissionResult(
        timestamp="2026-06-01T00:00:00Z",
        experiment_id=experiment_id,
        model_name=model_name,
        tier=tier,
        method=method,
        task_type=task_type,
        duration_seconds=duration_seconds,
        energy_kwh=energy_kwh,
        co2eq_g=co2eq_g,
        cpu_energy_kwh=0.010,
        gpu_energy_kwh=0.0,
        ram_energy_kwh=0.002,
        country_iso="GBR",
        carbon_intensity=231.0,
        calibration_factor=1.58,
        co2eq_corrected_g=co2eq_corrected_g,
    )


class TestRegistration:
    """add / add_batch / result_count."""

    def test_add_and_count(self, tmp_path: Path) -> None:
        reporter = ResultReporter(tmp_path)
        assert reporter.result_count == 0
        reporter.add(make_result())
        reporter.add(make_result(experiment_id="E02"))
        assert reporter.result_count == 2

    def test_add_batch(self, tmp_path: Path) -> None:
        reporter = ResultReporter(tmp_path)
        reporter.add_batch([make_result(), make_result(experiment_id="E02")])
        assert reporter.result_count == 2

    def test_output_dir_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        ResultReporter(nested)
        assert nested.is_dir()


class TestToDataframe:
    """to_dataframe conversions."""

    def test_empty_dataframe(self, tmp_path: Path) -> None:
        df = ResultReporter(tmp_path).to_dataframe()
        assert df.empty

    def test_dataframe_rows_and_columns(self, tmp_path: Path) -> None:
        reporter = ResultReporter(tmp_path)
        reporter.add(make_result())
        df = reporter.to_dataframe()
        assert len(df) == 1
        for col in ResultReporter._column_order():
            assert col in df.columns


class TestSaveSummary:
    """save_summary CSV export."""

    def test_empty_summary_writes_schema_headers(self, tmp_path: Path) -> None:
        path = ResultReporter(tmp_path).save_summary()
        assert path == tmp_path / "summary.csv"
        df = pd.read_csv(path)
        assert len(df) == 0
        assert list(df.columns) == ResultReporter._column_order()

    def test_populated_summary_writes_rows(self, tmp_path: Path) -> None:
        reporter = ResultReporter(tmp_path)
        reporter.add(make_result())
        reporter.add(make_result(experiment_id="E02", model_name="decision_tree"))
        path = reporter.save_summary("custom.csv")
        assert path == tmp_path / "custom.csv"
        df = pd.read_csv(path)
        assert len(df) == 2
        assert set(df["experiment_id"]) == {"E01", "E02"}

    def test_save_per_experiment(self, tmp_path: Path) -> None:
        reporter = ResultReporter(tmp_path)
        reporter.add(make_result(experiment_id="E_methodA_lin_rep1", model_name="linear_regression"))
        reporter.add(make_result(experiment_id="E_methodA_dt_rep1", model_name="decision_tree"))
        paths = reporter.save_per_experiment()
        assert len(paths) == 2
        expected_names = {
            "E_methodA_lin_rep1_linear_regression.csv",
            "E_methodA_dt_rep1_decision_tree.csv",
        }
        assert {p.name for p in paths} == expected_names
        for p in paths:
            assert p.exists()
            assert len(pd.read_csv(p)) == 1


class TestSummaryStats:
    """get_summary_stats aggregation."""

    def test_empty_stats(self, tmp_path: Path) -> None:
        assert ResultReporter(tmp_path).get_summary_stats() == {}

    def test_per_tier_stats(self, tmp_path: Path) -> None:
        reporter = ResultReporter(tmp_path)
        reporter.add(make_result(tier="T1", energy_kwh=0.012, co2eq_g=2.77,
                                 co2eq_corrected_g=4.38, duration_seconds=10.0))
        reporter.add(make_result(experiment_id="E02", tier="T1", energy_kwh=0.02,
                                 co2eq_g=3.23, co2eq_corrected_g=5.1, duration_seconds=20.0))
        reporter.add(make_result(experiment_id="E03", model_name="random_forest", tier="T2",
                                 energy_kwh=0.087, co2eq_g=19.0, co2eq_corrected_g=27.17,
                                 duration_seconds=30.0))
        stats = reporter.get_summary_stats()
        assert set(stats) == {"T1", "T2"}
        t1 = stats["T1"]
        assert t1["count"] == 2
        assert t1["mean_energy_kwh"] == pytest.approx(0.016)
        assert t1["mean_co2eq_g"] == pytest.approx(3.0)
        assert t1["mean_co2eq_corrected_g"] == pytest.approx(4.74)
        assert t1["mean_duration_s"] == pytest.approx(15.0)
        t2 = stats["T2"]
        assert t2["count"] == 1
        assert t2["mean_energy_kwh"] == pytest.approx(0.087)

    def test_tiers_sorted(self, tmp_path: Path) -> None:
        reporter = ResultReporter(tmp_path)
        reporter.add(make_result(tier="T3", model_name="resnet_50"))
        reporter.add(make_result(tier="T1"))
        assert list(reporter.get_summary_stats()) == ["T1", "T3"]


class TestTrainingInferenceRatio:
    """get_training_vs_inference_ratio."""

    def test_empty_returns_none(self, tmp_path: Path) -> None:
        assert ResultReporter(tmp_path).get_training_vs_inference_ratio() is None

    def test_ratio_computed(self, tmp_path: Path) -> None:
        reporter = ResultReporter(tmp_path)
        reporter.add(make_result(co2eq_corrected_g=10.0, task_type="training"))
        reporter.add(make_result(experiment_id="E02", co2eq_corrected_g=30.0, task_type="training"))
        reporter.add(make_result(experiment_id="E03", co2eq_corrected_g=4.0, task_type="inference"))
        assert reporter.get_training_vs_inference_ratio() == pytest.approx(4.0 / 40.0)

    def test_no_training_returns_none(self, tmp_path: Path) -> None:
        reporter = ResultReporter(tmp_path)
        reporter.add(make_result(co2eq_corrected_g=4.0, task_type="inference"))
        assert reporter.get_training_vs_inference_ratio() is None

    def test_inference_only_with_zero_training_sum(self, tmp_path: Path) -> None:
        """Train rows exist but sum to <= 0 — returns None."""
        reporter = ResultReporter(tmp_path)
        reporter.add(make_result(co2eq_corrected_g=0.0, task_type="training"))
        reporter.add(make_result(experiment_id="E02", co2eq_corrected_g=2.0, task_type="inference"))
        assert reporter.get_training_vs_inference_ratio() is None

    def test_zero_inference_gives_zero_ratio(self, tmp_path: Path) -> None:
        reporter = ResultReporter(tmp_path)
        reporter.add(make_result(co2eq_corrected_g=10.0, task_type="training"))
        reporter.add(make_result(experiment_id="E02", co2eq_corrected_g=0.0, task_type="inference"))
        assert reporter.get_training_vs_inference_ratio() == 0.0


class TestPrintSummary:
    """print_summary stdout output."""

    def test_empty_summary(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        ResultReporter(tmp_path).print_summary()
        out = capsys.readouterr().out
        assert "CARBON BENCHMARK RESULTS SUMMARY" in out
        assert "No results recorded yet." in out

    def test_populated_summary(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        reporter = ResultReporter(tmp_path)
        reporter.add(make_result())
        reporter.add(make_result(experiment_id="E02", task_type="inference", co2eq_corrected_g=1.0))
        reporter.print_summary()
        out = capsys.readouterr().out
        assert "T1" in out
        assert "Inference / Training emission ratio" in out


class TestExportTables:
    """export_tables CSV output."""

    def test_export_rows(self, tmp_path: Path) -> None:
        reporter = ResultReporter(tmp_path)
        reporter.add(make_result(tier="T1"))
        reporter.add(make_result(experiment_id="E02", tier="T3", model_name="resnet_50"))
        path = reporter.export_tables()
        assert path == tmp_path / "tables.csv"
        df = pd.read_csv(path)
        assert list(df["tier"]) == ["T1", "T3"]
        assert "mean_co2eq_corrected_g" in df.columns

    def test_export_empty(self, tmp_path: Path) -> None:
        """Empty stats produce an empty CSV file (no rows to export)."""
        path = ResultReporter(tmp_path).export_tables("empty.csv")
        assert path.exists()
        # Empty DataFrame export produces a whitespace-only file on all platforms
        assert path.read_bytes().strip() == b""


class TestColumnOrder:
    """_column_order schema contract."""

    def test_column_order_stable(self) -> None:
        cols = ResultReporter._column_order()
        assert cols[0] == "timestamp"
        assert cols[-1] == "device"
        assert len(cols) == 18
        assert len(set(cols)) == 18
