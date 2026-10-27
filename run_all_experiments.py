#!/usr/bin/env python3
"""
Master orchestrator for the Manufacturing AI Carbon Emission Framework (MACE-F).

Executes the complete experiment matrix (23 experiments) or a user-specified
subset.  All results are consolidated into a single summary CSV for analysis.

Usage:
    python run_all_experiments.py --all              # Full experiment matrix
    python run_all_experiments.py --methods A D      # Only methods A and D
    python run_all_experiments.py --models resnet_50 mobilenet_v2
    python run_all_experiments.py --dry-run          # Print plan without executing
    python run_all_experiments.py --country GBR      # Use UK grid intensity

Experiment Matrix (from config.EXPERIMENT_MATRIX):
    E01-E09:  Method A — Baseline training (T1-T4 models)
    E10-E19:  Method D — Baseline inference (T1-T5 models)
    E20:      Method B — Hyperparameter carbon efficiency (ResNet-50)
    E21:      Method C — Transfer learning comparison (ResNet-50)
    E22:      Method F — Carbon intensity sensitivity (14 countries)
    E23:      Method G — Deployment topology comparison (8 scenarios)

References:
    Schmidt et al. (2021) — CodeCarbon architecture
    Fischer (2025) — external meter calibration
    Luccioni et al. (2024) — inference benchmarking methodology
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from carbon_benchmark.config import EXPERIMENT_MATRIX, RESULTS_DIR
from carbon_benchmark.measurement.calibrator import Calibrator
from carbon_benchmark.measurement.reporter import ResultReporter
from carbon_benchmark.utils.hardware import print_hardware_summary
from carbon_benchmark.utils.logging_setup import setup_logging


def main() -> None:
    """Parse CLI args and execute the experiment matrix."""
    parser = argparse.ArgumentParser(
        description="MACE-F: Manufacturing AI Carbon Emission Framework — Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments.py --all
  python run_all_experiments.py --methods A D
  python run_all_experiments.py --models linear_regression random_forest resnet_50
  python run_all_experiments.py --country GBR --dry-run
        """,
    )
    parser.add_argument("--all", action="store_true", help="Run all 23 experiments")
    parser.add_argument(
        "--methods", type=str, nargs="+",
        help="Methods to run: A, B, C, D, E, F, G (space-separated)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        help="Models to run (space-separated keys from config.MODEL_TIERS)",
    )
    parser.add_argument(
        "--country", type=str, default="GBR",
        help="3-letter ISO country code (default: GBR)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: carbon_benchmark/results/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print experiment plan without executing",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(f"macef_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print_hardware_summary(logger)

    # Resolve which experiments to run
    experiments = _filter_experiments(args)
    if not experiments:
        logger.error("No experiments match the filter criteria.")
        logger.error("Use --all, --methods, and/or --models to select experiments.")
        sys.exit(1)

    logger.info(f"\nExperiment Plan ({len(experiments)} experiments):")
    for exp in experiments:
        logger.info(f"  {exp['id']}: Method {exp['method']} — {exp['model']} ({exp['task']})")

    if args.dry_run:
        logger.info("\nDry run — no experiments executed.")
        return

    # Execute
    reporter = ResultReporter(output_dir)
    calibrator = Calibrator(output_dir)
    failures: List[str] = []
    inference_energy: Dict[str, float] = {}
    training_co2eq: Dict[str, float] = {}

    for i, exp in enumerate(experiments):
        eid = exp["id"]
        method = exp["method"]
        model = exp["model"]
        task = exp["task"]

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(experiments)}] {eid}: Method {method} — {model}")
        logger.info(f"{'='*60}")

        try:
            if method == "A":
                from carbon_benchmark.experiments.method_a_baseline_training import run
                results = run(model, args.country, output_dir, logger=logger)
                # Populate training energy (kWh) for Methods E/F/G cross-reference
                if results:
                    training_co2eq[model] = results[0].get("energy_kwh", 0.0)

            elif method == "B":
                from carbon_benchmark.experiments.methods_bc_efg import run_method_b
                run_method_b(args.country, output_dir, logger)

            elif method == "C":
                from carbon_benchmark.experiments.methods_bc_efg import run_method_c
                run_method_c(args.country, output_dir, logger)

            elif method == "D":
                from carbon_benchmark.experiments.method_d_baseline_inference import run
                results = run(model, args.country, output_dir, logger=logger)
                # Populate inference energy for Method E cross-reference
                if results:
                    inference_energy[model] = results[0].get("energy_kwh", 0.0)

            elif method == "F":
                from carbon_benchmark.experiments.methods_bc_efg import run_method_f
                from carbon_benchmark.config import REPRESENTATIVE_ENERGY_KWH
                energy_kwh = training_co2eq.get(model, 0.0)
                if energy_kwh <= 0:
                    energy_kwh = REPRESENTATIVE_ENERGY_KWH
                    logger.info(f"  Using representative energy: {energy_kwh} kWh")
                run_method_f(energy_kwh, model, output_dir, logger)

            elif method == "G":
                from carbon_benchmark.experiments.methods_bc_efg import run_method_g
                from carbon_benchmark.config import REPRESENTATIVE_ENERGY_KWH
                energy_kwh = training_co2eq.get(model, 0.0)
                if energy_kwh <= 0:
                    energy_kwh = REPRESENTATIVE_ENERGY_KWH
                run_method_g(energy_kwh, model, output_dir, logger)

            logger.info(f"  [OK] {eid} completed")

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user.")
            break
        except Exception as exc:
            logger.error(f"  ✗ {eid} FAILED: {exc}", exc_info=True)
            failures.append(f"{eid}: {exc}")

    # Post-experiment: merge all per-experiment CSVs into master summary
    logger.info("\nMerging results...")
    _merge_results(output_dir)

    # Post-experiment: run Method E (deployment simulation) if we have data
    if inference_energy and training_co2eq:
        try:
            from carbon_benchmark.experiments.methods_bc_efg import run_method_e
            run_method_e(inference_energy, training_co2eq, output_dir, logger)
        except Exception as exc:
            logger.error(f"Method E failed: {exc}")

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT RUN COMPLETE")
    logger.info(f"  Successful: {len(experiments) - len(failures)}/{len(experiments)}")
    if failures:
        logger.info(f"  Failures ({len(failures)}):")
        for f in failures:
            logger.info(f"    - {f}")
    logger.info(f"  Results directory: {output_dir}")
    logger.info(f"{'='*60}")


def _merge_results(output_dir: Path) -> None:
    """Merge all per-experiment CSVs into a single summary.csv."""
    import glob

    csv_files = (
        list(Path(output_dir).glob("*_training.csv"))
        + list(Path(output_dir).glob("*_inference.csv"))
        + list(Path(output_dir).glob("method_*.csv"))
    )
    if not csv_files:
        return

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                frames.append(df)
        except Exception:
            pass

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged_path = Path(output_dir) / "summary.csv"
        merged.to_csv(merged_path, index=False)


def _filter_experiments(args) -> List[Dict]:
    """Filter the experiment matrix based on CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Filtered list of experiment dictionaries.
    """
    experiments = EXPERIMENT_MATRIX

    if args.all:
        return list(experiments)

    filtered = list(experiments)

    if args.methods:
        methods = set(args.methods)
        filtered = [e for e in filtered if e["method"] in methods]

    if args.models:
        models = set(args.models)
        filtered = [e for e in filtered if e["model"] in models]

    return filtered


if __name__ == "__main__":
    main()
