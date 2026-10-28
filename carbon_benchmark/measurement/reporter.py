"""
Results aggregation, CSV export, and summary statistics.

Consolidates EmissionResult objects from multiple experiments into a single
analysis-ready DataFrame.  Produces the master summary.csv and per-experiment
detail files that feed into the analysis tables.

References:
  Luccioni et al. (2024) — per-1000-inference normalisation
  Geißler et al. (2024) — energy/accuracy trade-off reporting
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from carbon_benchmark.config import RESULTS_DIR
from carbon_benchmark.measurement.tracker import EmissionResult


class ResultReporter:
    """Collects, persists, and summarises carbon measurement results.

    Usage:
        >>> reporter = ResultReporter()
        >>> reporter.add(result)
        >>> reporter.add(another_result)
        >>> reporter.save_summary()
        >>> reporter.print_summary()
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initialise the reporter.

        Args:
            output_dir: Directory for output CSV files.  Defaults to config.
        """
        self._output_dir = Path(output_dir) if output_dir else RESULTS_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._results: List[EmissionResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, result: EmissionResult) -> None:
        """Register a single experiment result.

        Args:
            result: Populated EmissionResult from CarbonTracker.
        """
        self._results.append(result)

    def add_batch(self, results: List[EmissionResult]) -> None:
        """Register multiple results at once.

        Args:
            results: List of EmissionResult objects.
        """
        self._results.extend(results)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all stored results to a pandas DataFrame.

        Returns:
            DataFrame with one row per experiment and standardised columns.
            Empty DataFrame with correct columns if no results stored.
        """
        if not self._results:
            return pd.DataFrame()
        rows = [r.to_dict() for r in self._results]
        return pd.DataFrame(rows)

    def save_summary(self, filename: str = "summary.csv") -> Path:
        """Write the master summary CSV combining all experiment results.

        Args:
            filename: Output filename.

        Returns:
            Path to the written file.
        """
        df = self.to_dataframe()
        if df.empty:
            # Write an empty file with headers so the schema is documented
            empty = pd.DataFrame(columns=self._column_order())
            output_path = self._output_dir / filename
            empty.to_csv(output_path, index=False)
            return output_path
        output_path = self._output_dir / filename
        df.to_csv(output_path, index=False)
        return output_path

    def save_per_experiment(self) -> List[Path]:
        """Write one CSV per experiment for detailed traceability.

        Returns:
            List of paths to written CSV files.
        """
        paths: List[Path] = []
        for result in self._results:
            filename = f"{result.experiment_id}_{result.model_name}.csv"
            output_path = self._output_dir / filename
            pd.DataFrame([result.to_dict()]).to_csv(output_path, index=False)
            paths.append(output_path)
        return paths

    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute per-tier aggregated statistics.

        Returns:
            Nested dictionary: tier → {mean_energy_kwh, mean_co2eq_g, count, ...}.
        """
        df = self.to_dataframe()
        if df.empty:
            return {}

        stats: Dict[str, Dict[str, float]] = {}
        for tier in sorted(df["tier"].unique()):
            tier_df = df[df["tier"] == tier]
            stats[tier] = {
                "count": int(len(tier_df)),
                "mean_energy_kwh": round(float(tier_df["energy_kwh"].mean()), 6),
                "mean_co2eq_g": round(float(tier_df["co2eq_g"].mean()), 2),
                "mean_co2eq_corrected_g": round(
                    float(tier_df["co2eq_corrected_g"].mean()), 2
                ),
                "mean_duration_s": round(float(tier_df["duration_seconds"].mean()), 1),
            }
        return stats

    def get_training_vs_inference_ratio(self) -> Optional[float]:
        """Compute aggregate inference-to-training emission ratio.

        Only includes models where both training and inference were measured.
        Returns None if insufficient data.

        Returns:
            Ratio of total inference CO₂ to total training CO₂ across all models.
        """
        df = self.to_dataframe()
        if df.empty:
            return None

        train = df[df["task_type"] == "training"]["co2eq_corrected_g"].sum()
        infer = df[df["task_type"] == "inference"]["co2eq_corrected_g"].sum()

        if train <= 0:
            return None
        return infer / train

    def print_summary(self) -> None:
        """Print a human-readable results summary to stdout."""
        stats = self.get_summary_stats()
        ratio = self.get_training_vs_inference_ratio()

        print("\n" + "=" * 65)
        print("CARBON BENCHMARK RESULTS SUMMARY")
        print("=" * 65)

        if not stats:
            print("  No results recorded yet.")
            return

        print(f"{'Tier':<6} {'Count':<8} {'Mean Energy (kWh)':<18} "
              f"{'Mean CO₂eq (g)':<16} {'Mean Duration (s)'}")
        print("-" * 65)
        for tier in ["T1", "T2", "T3", "T4", "T5"]:
            if tier in stats:
                s = stats[tier]
                print(
                    f"{tier:<6} {s['count']:<8} {s['mean_energy_kwh']:<18.6f} "
                    f"{s['mean_co2eq_corrected_g']:<16.2f} {s['mean_duration_s']:<.1f}"
                )

        if ratio is not None:
            print("-" * 65)
            print(f"  Inference / Training emission ratio: {ratio:.1f}×")
        print("=" * 65 + "\n")

    def export_tables(self, filename: str = "tables.csv") -> Path:
        """Export aggregated data in a tabular format.

        Produces a CSV with tier-level aggregates and key comparisons suitable
        for direct inclusion in reports and documentation.

        Args:
            filename: Output CSV filename.

        Returns:
            Path to the written file.
        """
        stats = self.get_summary_stats()
        rows = []
        for tier, data in stats.items():
            rows.append({"tier": tier, **data})

        df = pd.DataFrame(rows)
        output_path = self._output_dir / filename
        df.to_csv(output_path, index=False)
        return output_path

    @property
    def result_count(self) -> int:
        """Return the total number of stored results."""
        return len(self._results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _column_order() -> List[str]:
        """Return the canonical column order for the summary CSV."""
        return [
            "timestamp", "experiment_id", "model_name", "tier",
            "method", "task_type", "duration_seconds", "energy_kwh",
            "co2eq_g", "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh",
            "country_iso", "carbon_intensity", "calibration_factor",
            "co2eq_corrected_g", "n_samples", "device",
        ]
