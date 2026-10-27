"""
Method D: Baseline Inference Carbon Measurement.

Measures the carbon footprint of 1,000 inferences for each model (T1–T5),
normalised per Luccioni et al. (2024).  For generative models (T5), the
sample count is adjusted: 100 images for Stable Diffusion, 50 queries for GPT.

Usage:
    python -m carbon_benchmark.experiments.method_d_baseline_inference \\
        --model resnet_50 --country GBR --n-samples 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from carbon_benchmark.config import (
    DEFAULT_COUNTRY,
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
from carbon_benchmark.models.generative import (
    create_stable_diffusion_pipeline,
    create_gpt_manufacturing_queries,
)
from carbon_benchmark.models.traditional_ml import (
    create_decision_tree,
    create_linear_regression,
    create_random_forest,
    create_xgboost,
)
from carbon_benchmark.utils.hardware import print_hardware_summary
from carbon_benchmark.utils.logging_setup import setup_logging

# ---------------------------------------------------------------------------
# Inference registry
# ---------------------------------------------------------------------------

INFERENCE_REGISTRY: Dict[str, Dict] = {
    "linear_regression": {
        "loader": load_nasa_cmapss,
        "preprocessor": preprocess_tabular,
        "factory": create_linear_regression,
        "n_samples": 1000,
    },
    "decision_tree": {
        "loader": load_secom,
        "preprocessor": preprocess_tabular,
        "factory": create_decision_tree,
        "n_samples": 1000,
    },
    "random_forest": {
        "loader": load_secom,
        "preprocessor": preprocess_tabular,
        "factory": create_random_forest,
        "n_samples": 1000,
    },
    "xgboost": {
        "loader": load_nasa_cmapss,
        "preprocessor": preprocess_tabular,
        "factory": create_xgboost,
        "n_samples": 1000,
    },
    "mobilenet_v2": {
        "loader": load_neu_steel_defects,
        "preprocessor": preprocess_images,
        "factory": lambda: create_mobilenet_v2(num_classes=6),
        "n_samples": 1000,
    },
    "resnet_50": {
        "loader": load_neu_steel_defects,
        "preprocessor": preprocess_images,
        "factory": lambda: create_resnet50(num_classes=6),
        "n_samples": 1000,
    },
    "lstm": {
        "loader": load_nasa_cmapss,
        "preprocessor": lambda *a: preprocess_time_series(*a, sequence_length=30),
        "factory": create_lstm_rul,
        "n_samples": 1000,
    },
    "resnet_152": {
        "loader": load_neu_steel_defects,
        "preprocessor": preprocess_images,
        "factory": lambda: create_resnet152(num_classes=6),
        "n_samples": 1000,
    },
    "vit_b_16": {
        "loader": load_neu_steel_defects,
        "preprocessor": preprocess_images,
        "factory": lambda: create_vit_b16(num_classes=6),
        "n_samples": 1000,
    },
    "stable_diffusion": {
        "loader": None,
        "preprocessor": None,
        "factory": create_stable_diffusion_pipeline,
        "n_samples": 100,  # 10 prompts × 10 images each
    },
    "gpt_4o_mini": {
        "loader": None,
        "preprocessor": None,
        "factory": create_gpt_manufacturing_queries,
        "n_samples": 3,  # 3 QA pairs (placeholder; expand to 50 with real mfg documentation)
    },
}


def run(
    model_name: str,
    country_iso: str = "GBR",
    output_dir: Optional[Path] = None,
    n_samples: Optional[int] = None,
    logger=None,
) -> List:
    """Execute Method D baseline inference measurement for a single model.

    Args:
        model_name: Model key matching INFERENCE_REGISTRY.
        country_iso: 3-letter ISO country code.
        output_dir: Directory for result CSV files.
        n_samples: Override default sample count.
        logger: Logger instance.

    Returns:
        List containing a single EmissionResult dict.
    """
    if logger is None:
        logger = setup_logging(f"E_methodD_{model_name}")

    if model_name not in INFERENCE_REGISTRY:
        valid = ", ".join(INFERENCE_REGISTRY.keys())
        raise KeyError(f"Model '{model_name}' not in inference registry. Valid: {valid}")

    spec = INFERENCE_REGISTRY[model_name]
    samples = n_samples if n_samples is not None else spec["n_samples"]
    tier = MODEL_TIERS.get(model_name, "T3")

    logger.info("=" * 60)
    logger.info(f"METHOD D: Baseline Inference — {model_name} ({tier})")
    logger.info(f"  N samples: {samples}")
    logger.info(f"  Country:   {country_iso}")
    logger.info("=" * 60)

    # Load and preprocess data (for non-generative models)
    if model_name in ("stable_diffusion", "gpt_4o_mini"):
        return _run_generative_inference(model_name, country_iso, output_dir, samples, logger)

    logger.info("Loading dataset...")
    X_train, X_test, y_train, y_test = spec["loader"]()
    X_train, X_test, y_train, y_test = spec["preprocessor"](
        X_train, X_test, y_train, y_test,
    )

    # Use test set for inference benchmark
    X_infer = X_test[:samples]
    logger.info(f"  Inference subset: {X_infer.shape}")

    # Quick train (minimal — just need a trained model for inference)
    logger.info("Training model (minimal, for inference benchmark)...")
    model, train_fn, inference_fn = spec["factory"]()
    try:
        train_fn(X_train, y_train)
    except TypeError:
        train_fn(X_train, y_train, X_test, y_test)

    # Measure inference carbon
    experiment_id = f"E_methodD_{model_name}"
    tracker = CarbonTracker(
        experiment_id=experiment_id,
        model_name=model_name,
        method="D",
        country_iso=country_iso,
        task_type="inference",
        output_dir=output_dir,
        n_samples=samples,
    )

    result = tracker.track_inference(
        lambda: inference_fn(X_infer),
        n_samples=samples,
    )

    reporter = ResultReporter(output_dir)
    reporter.add(result)

    summary_path = reporter.save_summary()
    logger.info(f"\nResults saved to {summary_path}")
    logger.info(
        f"  Per-1000-inference energy: {result.energy_kwh:.6f} kWh | "
        f"CO₂eq: {result.co2eq_corrected_g:.2f} g"
    )

    return [result.to_dict()]


def _run_generative_inference(
    model_name: str,
    country_iso: str,
    output_dir: Optional[Path],
    n_samples: int,
    logger,
) -> List:
    """Handle T5 generative model inference with appropriate sample counts."""
    from carbon_benchmark.config import MANUFACTURING_DESIGN_PROMPTS

    experiment_id = f"E_methodD_{model_name}"

    if model_name == "stable_diffusion":
        _, generate_fn, _ = create_stable_diffusion_pipeline()
        n_prompts = min(10, n_samples // 10) if n_samples <= 100 else 10
        prompts = MANUFACTURING_DESIGN_PROMPTS[:n_prompts]
        n_per = max(1, n_samples // len(prompts))

        tracker = CarbonTracker(
            experiment_id=experiment_id,
            model_name=model_name,
            method="D",
            country_iso=country_iso,
            task_type="inference",
            output_dir=output_dir,
            n_samples=n_samples,
        )
        result = tracker.track_inference(
            lambda: generate_fn(prompts=prompts, n_per_prompt=n_per),
            n_samples=len(prompts) * n_per,
        )

    elif model_name == "gpt_4o_mini":
        _, query_fn, _ = create_gpt_manufacturing_queries()

        tracker = CarbonTracker(
            experiment_id=experiment_id,
            model_name=model_name,
            method="D",
            country_iso=country_iso,
            task_type="inference",
            output_dir=output_dir,
            n_samples=n_samples,
        )
        result = tracker.track_inference(
            lambda: query_fn(),
            n_samples=n_samples,
        )

    else:
        raise ValueError(f"Unknown generative model: {model_name}")

    reporter = ResultReporter(output_dir)
    reporter.add(result)
    reporter.save_summary(filename=f"{model_name}_inference.csv")

    logger.info(
        f"  Energy: {result.energy_kwh:.6f} kWh | "
        f"CO₂eq (corrected): {result.co2eq_corrected_g:.2f} g"
    )
    return [result.to_dict()]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for Method D."""
    parser = argparse.ArgumentParser(
        description="Method D: Baseline Inference Carbon Measurement"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help=f"Model key. Valid: {', '.join(INFERENCE_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--country", type=str, default="GBR",
        help="3-letter ISO country code (default: GBR)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results CSV",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Number of inference samples (default: model-specific)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    print_hardware_summary()
    results = run(
        model_name=args.model,
        country_iso=args.country,
        output_dir=output_dir,
        n_samples=args.n_samples,
    )
    print(f"\nDone. {len(results)} result(s) generated.")


if __name__ == "__main__":
    main()
