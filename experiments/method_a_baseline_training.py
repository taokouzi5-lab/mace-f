"""
Method A: Baseline Training Carbon Measurement.

Measures the carbon footprint of training each model (T1–T4) on its
corresponding manufacturing dataset with default hyperparameters.

Usage:
    python -m carbon_benchmark.experiments.method_a_baseline_training \\
        --model linear_regression --country GBR --output-dir ../results

Each model is trained 3 times for statistical validity.
Results are saved to the project results directory.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from carbon_benchmark.config import (
    DEFAULT_COUNTRY,
    HARDWARE,
    MODEL_TIERS,
    RANDOM_SEED,
    RESULTS_DIR,
)
from carbon_benchmark.data.loader import (
    load_secom,
    load_nasa_cmapss,
    load_neu_steel_defects,
    load_cnc_tool_wear,
)
from carbon_benchmark.data.preprocessing import (
    preprocess_tabular,
    preprocess_images,
    preprocess_time_series,
)
from carbon_benchmark.measurement.reporter import ResultReporter
from carbon_benchmark.measurement.tracker import CarbonTracker
from carbon_benchmark.models.deep_learning import (
    create_lstm_rul,
    create_mobilenet_v2,
    create_resnet50,
    create_resnet152,
    create_vit_b16,
)
from carbon_benchmark.models.traditional_ml import (
    create_decision_tree,
    create_linear_regression,
    create_random_forest,
    create_xgboost,
    compute_metrics,
)
from carbon_benchmark.utils.hardware import print_hardware_summary
from carbon_benchmark.utils.logging_setup import setup_logging

# ---------------------------------------------------------------------------
# Model → (dataset loader, preprocessor, model factory, task type) mapping
# ---------------------------------------------------------------------------

TRAINING_REGISTRY: Dict[str, Dict] = {
    "linear_regression": {
        "loader": load_nasa_cmapss,
        "preprocessor": preprocess_tabular,
        "factory": create_linear_regression,
        "task": "regression",
        "n_repeats": 3,
    },
    "decision_tree": {
        "loader": load_secom,
        "preprocessor": preprocess_tabular,
        "factory": create_decision_tree,
        "task": "classification",
        "n_repeats": 3,
    },
    "random_forest": {
        "loader": load_secom,
        "preprocessor": preprocess_tabular,
        "factory": create_random_forest,
        "task": "classification",
        "n_repeats": 3,
    },
    "xgboost": {
        "loader": load_nasa_cmapss,
        "preprocessor": preprocess_tabular,
        "factory": create_xgboost,
        "task": "regression",
        "n_repeats": 3,
    },
    "mobilenet_v2": {
        "loader": load_neu_steel_defects,
        "preprocessor": preprocess_images,
        "factory": lambda: create_mobilenet_v2(num_classes=6),
        "task": "classification",
        "n_repeats": 3,
    },
    "resnet_50": {
        "loader": load_neu_steel_defects,
        "preprocessor": preprocess_images,
        "factory": lambda: create_resnet50(num_classes=6),
        "task": "classification",
        "n_repeats": 3,
    },
    "lstm": {
        "loader": load_nasa_cmapss,
        "preprocessor": lambda *a: preprocess_time_series(*a, sequence_length=30),
        "factory": create_lstm_rul,
        "task": "regression",
        "n_repeats": 3,
    },
    "resnet_152": {
        "loader": load_neu_steel_defects,
        "preprocessor": preprocess_images,
        "factory": lambda: create_resnet152(num_classes=6),
        "task": "classification",
        "n_repeats": 3,
    },
    "vit_b_16": {
        "loader": load_neu_steel_defects,
        "preprocessor": preprocess_images,
        "factory": lambda: create_vit_b16(num_classes=6),
        "task": "classification",
        "n_repeats": 3,
    },
}

# T5 models are inference-only — not included here


def run(
    model_name: str,
    country_iso: str = "GBR",
    output_dir: Optional[Path] = None,
    n_repeats: Optional[int] = None,
    logger=None,
) -> List:
    """Execute Method A baseline training measurement for a single model.

    Args:
        model_name: Model key matching MODEL_TIERS in config.
        country_iso: 3-letter ISO country code for carbon intensity.
        output_dir: Directory for result CSV files.
        n_repeats: Override default repeat count.
        logger: Logger instance.

    Returns:
        List of EmissionResult dicts (one per repeat).

    Raises:
        KeyError: If model_name is not in TRAINING_REGISTRY.
    """
    if logger is None:
        logger = setup_logging(f"E_methodA_{model_name}")

    if model_name not in TRAINING_REGISTRY:
        valid = ", ".join(TRAINING_REGISTRY.keys())
        raise KeyError(
            f"Model '{model_name}' not in training registry. Valid: {valid}\n"
            f"Note: T5 models (stable_diffusion, gpt_4o_mini) are inference-only."
        )

    spec = TRAINING_REGISTRY[model_name]
    repeats = n_repeats if n_repeats is not None else spec["n_repeats"]
    tier = MODEL_TIERS.get(model_name, "T3")

    logger.info("=" * 60)
    logger.info(f"METHOD A: Baseline Training — {model_name} ({tier})")
    logger.info(f"  Hardware: {HARDWARE['cpu']} + {HARDWARE['gpu']}")
    logger.info(f"  Country:  {country_iso}")
    logger.info(f"  Repeats:  {repeats}")
    logger.info("=" * 60)

    # Load and preprocess data (once — same data for all repeats)
    logger.info("Loading dataset...")
    X_train, X_test, y_train, y_test = spec["loader"]()
    X_train, X_test, y_train, y_test = spec["preprocessor"](
        X_train, X_test, y_train, y_test,
    )
    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    reporter = ResultReporter(output_dir)
    results: list = []

    for rep in range(repeats):
        logger.info(f"\n--- Repeat {rep+1}/{repeats} ---")
        experiment_id = f"E_methodA_{model_name}_rep{rep+1}"

        # Create model, train_fn, and inference_fn from SINGLE factory call
        model, train_fn, inference_fn = spec["factory"]()

        # Measure training carbon
        tracker = CarbonTracker(
            experiment_id=experiment_id,
            model_name=model_name,
            method="A",
            country_iso=country_iso,
            task_type="training",
            output_dir=output_dir,
        )

        # Adapt train_fn: tracker only passes (X_train, y_train) but DL
        # model factories expect (X_train, y_train, X_test, y_test, ...)
        def _adapted_train():
            try:
                return train_fn(X_train, y_train)
            except TypeError:
                return train_fn(X_train, y_train, X_test, y_test)

        result = tracker.track_training(_adapted_train)
        result.extra["repeat"] = rep + 1
        result.extra["dataset_shape"] = str(X_train.shape)

        # Compute accuracy on test set using the SAME trained model
        y_pred = inference_fn(X_test)
        metrics = compute_metrics(y_test, y_pred, spec["task"])
        for k, v in metrics.items():
            result.extra[f"metric_{k}"] = v

        reporter.add(result)
        results.append(result.to_dict())

        logger.info(
            f"  Energy: {result.energy_kwh:.6f} kWh | "
            f"CO2eq: {result.co2eq_g:.2f} g | "
            f"Corrected: {result.co2eq_corrected_g:.2f} g | "
            f"Duration: {result.duration_seconds:.1f}s"
        )
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    # Save results
    saved = reporter.save_summary(filename=f"{model_name}_training.csv")
    logger.info(f"\nResults saved to {saved}")

    return results



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for Method A."""
    parser = argparse.ArgumentParser(
        description="Method A: Baseline Training Carbon Measurement"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help=f"Model key. Valid: {', '.join(TRAINING_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--country", type=str, default="GBR",
        help="3-letter ISO country code (default: GBR for UK grid)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results CSV",
    )
    parser.add_argument(
        "--repeats", type=int, default=None,
        help="Number of repeat measurements (default: model-specific)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    print_hardware_summary()
    results = run(
        model_name=args.model,
        country_iso=args.country,
        output_dir=output_dir,
        n_repeats=args.repeats,
    )
    print(f"\nDone. {len(results)} result(s) generated.")


if __name__ == "__main__":
    main()
