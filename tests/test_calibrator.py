"""Unit tests for Calibrator and CalibrationRecord (Fischer 2025 methodology)."""

from pathlib import Path

import pandas as pd
import pytest

from carbon_benchmark.config import CALIBRATION_FACTORS
from carbon_benchmark.measurement.calibrator import CalibrationRecord, Calibrator


class TestCalibrationRecord:
    """CalibrationRecord.factor property."""

    def test_factor_ratio(self) -> None:
        record = CalibrationRecord("E01", "linear_regression", "T1", 0.012, 0.019)
        assert record.factor == pytest.approx(0.019 / 0.012)

    def test_factor_zero_codecarbon_returns_one(self) -> None:
        record = CalibrationRecord("E01", "linear_regression", "T1", 0.0, 0.019)
        assert record.factor == 1.0

    def test_factor_negative_codecarbon_returns_one(self) -> None:
        record = CalibrationRecord("E01", "linear_regression", "T1", -0.5, 0.019)
        assert record.factor == 1.0


class TestAddRecords:
    """add / add_from_result."""

    def test_add_from_result_alias(self) -> None:
        cal = Calibrator()
        cal.add_from_result("E01", "linear_regression", "T1", 0.012, 0.019, notes="alias")
        factors = cal.compute_tier_factors()
        assert factors["T1"]["source"] == "empirical"
        assert factors["T1"]["n_records"] == 1

    def test_add_stores_notes(self) -> None:
        cal = Calibrator()
        cal.add("E01", "linear_regression", "T1", 0.012, 0.019, notes="fluke meter")
        df = cal._to_dataframe()
        assert df.iloc[0]["notes"] == "fluke meter"


class TestComputeTierFactors:
    """compute_tier_factors aggregation and defaults."""

    def test_multi_record_statistics(self) -> None:
        cal = Calibrator()
        # factors: 1.5, 1.5, 2.0
        cal.add("E01", "m", "T1", 0.010, 0.015)
        cal.add("E02", "m", "T1", 0.020, 0.030)
        cal.add("E03", "m", "T1", 0.030, 0.060)
        t1 = cal.compute_tier_factors()["T1"]
        assert t1["source"] == "empirical"
        assert t1["n_records"] == 3
        assert t1["mean_factor"] == pytest.approx(1.667, abs=0.001)
        assert t1["min_factor"] == pytest.approx(1.5)
        assert t1["max_factor"] == pytest.approx(2.0)
        # Sample std (ddof=1) of [1.5, 1.5, 2.0]
        assert t1["std_dev"] == pytest.approx(0.289, abs=0.001)

    def test_unmeasured_tiers_use_config_defaults(self) -> None:
        cal = Calibrator()
        cal.add("E01", "m", "T1", 0.010, 0.015)
        factors = cal.compute_tier_factors()
        for tier in ["T2", "T3", "T4", "T5"]:
            assert factors[tier]["source"] == "config_default"
            assert factors[tier]["n_records"] == 0
            assert factors[tier]["mean_factor"] == CALIBRATION_FACTORS[tier]
            assert factors[tier]["min_factor"] == CALIBRATION_FACTORS[tier]
            assert factors[tier]["max_factor"] == CALIBRATION_FACTORS[tier]

    def test_empty_records_all_defaults(self) -> None:
        factors = Calibrator().compute_tier_factors()
        assert set(factors) == {"T1", "T2", "T3", "T4", "T5"}
        assert all(f["source"] == "config_default" for f in factors.values())


class TestApplyCalibration:
    """apply_calibration behaviour."""

    def test_applies_empirical_factor(self) -> None:
        cal = Calibrator()
        cal.add("E01", "m", "T2", 0.010, 0.020)  # factor 2.0
        assert cal.apply_calibration(100.0, "T2") == pytest.approx(200.0)

    def test_unknown_tier_falls_back_to_1_40(self) -> None:
        cal = Calibrator()
        assert cal.apply_calibration(100.0, "T9") == pytest.approx(140.0)

    def test_zero_input_stays_zero(self) -> None:
        assert Calibrator().apply_calibration(0.0, "T1") == 0.0


class TestExportReport:
    """export_report CSV output."""

    def test_export_report_rows_and_columns(self, tmp_path: Path) -> None:
        cal = Calibrator(tmp_path)
        cal.add("E01", "m", "T1", 0.010, 0.015)
        path = cal.export_report()
        assert path == tmp_path / "calibration_factors.csv"
        df = pd.read_csv(path)
        assert len(df) == 5
        assert list(df["tier"]) == ["T1", "T2", "T3", "T4", "T5"]
        for col in ["mean_factor", "std_dev", "n_records", "min_factor", "max_factor", "source"]:
            assert col in df.columns
        t1_row = df[df["tier"] == "T1"].iloc[0]
        assert t1_row["source"] == "empirical"
        t2_row = df[df["tier"] == "T2"].iloc[0]
        assert t2_row["source"] == "config_default"

    def test_export_custom_filename(self, tmp_path: Path) -> None:
        path = Calibrator(tmp_path).export_report("custom_cal.csv")
        assert path.name == "custom_cal.csv"
        assert path.exists()


class TestPrintSummary:
    """print_summary stdout output."""

    def test_print_summary_defaults(self, capsys: pytest.CaptureFixture) -> None:
        Calibrator().print_summary()
        out = capsys.readouterr().out
        assert "CALIBRATION FACTOR SUMMARY" in out
        assert "T1" in out and "T5" in out
        assert "config_default" in out
        assert "CF = External_Meter_kWh / CodeCarbon_kWh" in out

    def test_print_summary_empirical(self, capsys: pytest.CaptureFixture) -> None:
        cal = Calibrator()
        cal.add("E01", "m", "T1", 0.010, 0.015)
        cal.print_summary()
        out = capsys.readouterr().out
        assert "empirical" in out


class TestToDataframe:
    """_to_dataframe structure."""

    def test_dataframe_columns(self) -> None:
        cal = Calibrator()
        cal.add("E01", "m", "T1", 0.010, 0.015)
        df = cal._to_dataframe()
        assert list(df.columns) == [
            "experiment_id", "model_name", "tier", "codecarbon_kwh",
            "external_meter_kwh", "factor", "notes",
        ]
        assert df.iloc[0]["factor"] == pytest.approx(1.5)
