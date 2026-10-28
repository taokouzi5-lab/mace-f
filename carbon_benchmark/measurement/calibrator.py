"""
Calibration factor computation for correcting CodeCarbon underestimation.

Per Fischer (2025), CodeCarbon systematically underestimates total system
energy by 25-37 % because it measures only CPU+GPU+RAM and ignores:
  - Power supply unit (PSU) losses (~5-15 %)
  - Cooling fans and pump overhead
  - Chipset, storage, and networking
  - Peripheral devices (industrial cameras, PLC interfaces in manufacturing)

This module:
  1. Accepts paired (CodeCarbon, external-meter) measurements.
  2. Computes calibration factors per workload tier.
  3. Applies factors to correct raw CodeCarbon emission estimates.

References:
  Fischer R. "Ground-Truthing AI Energy Consumption." arXiv:2509.22092, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from carbon_benchmark.config import CALIBRATION_FACTORS, RESULTS_DIR


@dataclass
class CalibrationRecord:
    """A single paired measurement for computing calibration factors."""

    experiment_id: str
    model_name: str
    tier: str
    codecarbon_kwh: float
    external_meter_kwh: float
    notes: str = ""

    @property
    def factor(self) -> float:
        """Compute individual calibration factor."""
        if self.codecarbon_kwh <= 0:
            return 1.0
        return self.external_meter_kwh / self.codecarbon_kwh


class Calibrator:
    """Manages calibration factor computation and application.

    Stores paired measurements and computes tier-aggregated factors.
    Can export the calibration table for inclusion in the documentation.

    Usage:
        >>> cal = Calibrator()
        >>> cal.add("E01", "linear_regression", "T1", 0.012, 0.019)
        >>> cal.add("E06", "resnet_50", "T3", 0.423, 0.587)
        >>> factors = cal.compute_tier_factors()
        >>> corrected = cal.apply_calibration(500.0, "T3")
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initialise the calibrator.

        Args:
            output_dir: Directory for calibration report CSV.
        """
        self._records: List[CalibrationRecord] = []
        self._output_dir = Path(output_dir) if output_dir else RESULTS_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        experiment_id: str,
        model_name: str,
        tier: str,
        codecarbon_kwh: float,
        external_meter_kwh: float,
        notes: str = "",
    ) -> None:
        """Record a calibration measurement pair.

        Args:
            experiment_id: Experiment identifier, e.g. 'E01'.
            model_name: Model key matching config.MODEL_TIERS.
            tier: Model tier label ('T1'–'T5').
            codecarbon_kwh: Energy reported by CodeCarbon (kWh).
            external_meter_kwh: Energy measured by external power meter (kWh).
            notes: Optional annotation (e.g. meter model, environmental conditions).
        """
        record = CalibrationRecord(
            experiment_id=experiment_id,
            model_name=model_name,
            tier=tier,
            codecarbon_kwh=codecarbon_kwh,
            external_meter_kwh=external_meter_kwh,
            notes=notes,
        )
        self._records.append(record)

    def add_from_result(
        self,
        experiment_id: str,
        model_name: str,
        tier: str,
        codecarbon_kwh: float,
        external_meter_kwh: float,
        notes: str = "",
    ) -> None:
        """Convenience alias for add()."""
        self.add(experiment_id, model_name, tier, codecarbon_kwh, external_meter_kwh, notes)

    def compute_tier_factors(self) -> Dict[str, Dict[str, float]]:
        """Aggregate records per tier and compute mean calibration factors.

        Returns:
            Dictionary mapping tier → {mean_factor, std_dev, n_records, min, max}.
            Falls back to config defaults for tiers with no measurements.
        """
        if not self._records:
            return self._default_factors()

        df = self._to_dataframe()
        result: Dict[str, Dict[str, float]] = {}

        for tier in ["T1", "T2", "T3", "T4", "T5"]:
            tier_data = df[df["tier"] == tier]["factor"]
            if len(tier_data) == 0:
                result[tier] = {
                    "mean_factor": CALIBRATION_FACTORS.get(tier, 1.40),
                    "std_dev": 0.0,
                    "n_records": 0,
                    "min_factor": CALIBRATION_FACTORS.get(tier, 1.40),
                    "max_factor": CALIBRATION_FACTORS.get(tier, 1.40),
                    "source": "config_default",
                }
            else:
                result[tier] = {
                    "mean_factor": round(float(tier_data.mean()), 3),
                    "std_dev": round(float(tier_data.std()), 3),
                    "n_records": int(len(tier_data)),
                    "min_factor": round(float(tier_data.min()), 3),
                    "max_factor": round(float(tier_data.max()), 3),
                    "source": "empirical",
                }
        return result

    def apply_calibration(self, co2eq_g: float, tier: str) -> float:
        """Apply the appropriate calibration factor to a CodeCarbon emission value.

        Args:
            co2eq_g: Raw CO₂ equivalent in grams (from CodeCarbon).
            tier: Model tier ('T1'–'T5').

        Returns:
            Calibrated CO₂ equivalent in grams.
        """
        factors = self.compute_tier_factors()
        factor = factors.get(tier, {}).get("mean_factor", CALIBRATION_FACTORS.get(tier, 1.40))
        return co2eq_g * factor

    def export_report(self, filename: str = "calibration_factors.csv") -> Path:
        """Write calibration factor summary to CSV.

        Args:
            filename: Output CSV filename.

        Returns:
            Path to the written CSV file.
        """
        factors = self.compute_tier_factors()
        rows = []
        for tier, data in factors.items():
            rows.append({
                "tier": tier,
                **data,
            })
        df = pd.DataFrame(rows)
        output_path = self._output_dir / filename
        df.to_csv(output_path, index=False)
        return output_path

    def print_summary(self) -> None:
        """Print a human-readable calibration summary to stdout."""
        factors = self.compute_tier_factors()
        print("\n" + "=" * 55)
        print("CALIBRATION FACTOR SUMMARY (Fischer 2025 methodology)")
        print("=" * 55)
        print(f"{'Tier':<6} {'Factor':<10} {'StdDev':<10} {'N':<6} {'Source'}")
        print("-" * 55)
        for tier in ["T1", "T2", "T3", "T4", "T5"]:
            f = factors[tier]
            print(
                f"{tier:<6} {f['mean_factor']:<10.3f} "
                f"{f['std_dev']:<10.3f} {f['n_records']:<6} {f['source']}"
            )
        print("=" * 55)
        print("CF = External_Meter_kWh / CodeCarbon_kWh")
        print("Corrected_CO2 = Raw_CO2 × CF\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert stored records to a pandas DataFrame with computed factors."""
        rows = [
            {
                "experiment_id": r.experiment_id,
                "model_name": r.model_name,
                "tier": r.tier,
                "codecarbon_kwh": r.codecarbon_kwh,
                "external_meter_kwh": r.external_meter_kwh,
                "factor": r.factor,
                "notes": r.notes,
            }
            for r in self._records
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _default_factors() -> Dict[str, Dict[str, float]]:
        """Return fallback factors from config when no measurements exist."""
        return {
            tier: {
                "mean_factor": cf,
                "std_dev": 0.0,
                "n_records": 0,
                "min_factor": cf,
                "max_factor": cf,
                "source": "config_default",
            }
            for tier, cf in CALIBRATION_FACTORS.items()
        }
