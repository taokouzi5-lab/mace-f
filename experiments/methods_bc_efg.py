"""
Methods B, C, E, F, G — supplementary experiments.

  Method B: Hyperparameter carbon efficiency (Geißler et al., 2024)
  Method C: Transfer learning comparison (Geißler et al., 2024)
  Method E: Long-term deployment simulation (Luccioni et al., 2024)
  Method F: Carbon intensity sensitivity (Lacoste et al., 2019)
  Method G: Deployment topology comparison (Patterson et al., 2022)

All methods are imported and executed via run_all_experiments.py or called
programmatically from notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from carbon_benchmark.config import (
    CALIBRATION_FACTORS,
    CARBON_INTENSITY,
    DEFAULT_COUNTRY,
    DEPLOYMENT_SCENARIOS,
    HYPERPARAM_GRID,
    MODEL_TIERS,
    PUE,
    RESULTS_DIR,
)
from carbon_benchmark.measurement.tracker import CarbonTracker
from carbon_benchmark.utils.logging_setup import setup_logging


# ======================================================================
# Method B: Hyperparameter Carbon Efficiency
# ======================================================================

def run_method_b(
    country_iso: str = "GBR",
    output_dir: Optional[Path] = None,
    logger=None,
) -> pd.DataFrame:
    """Compare carbon emissions across hyperparameter combinations for ResNet-50.

    Tests 8 representative (batch_size, learning_rate) combinations.
    Reference: Geißler et al. (2024) — hyperparameters cause up to 5× energy variation.

    Args:
        country_iso: 3-letter ISO country code.
        output_dir: Output directory.
        logger: Logger instance.

    Returns:
        DataFrame with columns: batch_size, lr, energy_kwh, co2eq_g.
    """
    if logger is None:
        logger = setup_logging("E_methodB_hyperparam")

    logger.info("=" * 60)
    logger.info("METHOD B: Hyperparameter Carbon Efficiency (ResNet-50)")
    logger.info("=" * 60)

    from carbon_benchmark.data.loader import load_neu_steel_defects
    from carbon_benchmark.data.preprocessing import preprocess_images
    from carbon_benchmark.models.deep_learning import create_resnet50

    X_train, X_test, y_train, y_test = load_neu_steel_defects()
    X_train, X_test, y_train, y_test = preprocess_images(
        X_train, X_test, y_train, y_test,
    )

    results: List[Dict[str, Any]] = []

    for batch_size, lr in HYPERPARAM_GRID["combinations"]:
        combo_label = f"bs{batch_size}_lr{lr:.0e}"
        logger.info(f"\n--- {combo_label} ---")

        model, train_fn, _ = create_resnet50(num_classes=6)

        tracker = CarbonTracker(
            experiment_id=f"E_methodB_{combo_label}",
            model_name="resnet_50",
            method="B",
            country_iso=country_iso,
            task_type="training",
            output_dir=output_dir,
        )

        # Pass batch_size through kwargs — training loop picks it up
        try:
            result = tracker.track_training(
                train_fn, X_train, y_train, X_test, y_test,
                epochs=HYPERPARAM_GRID["max_epochs"],
                batch_size=batch_size,
            )
        except TypeError:
            # If train_fn doesn't accept kwargs, train inline
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            device = "cuda" if torch.cuda.is_available() else "cpu"
            train_ds = TensorDataset(
                torch.from_numpy(X_train), torch.from_numpy(y_train).long(),
            )
            loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
            model = model.to(device)

            with tracker:
                for epoch in range(HYPERPARAM_GRID["max_epochs"]):
                    model.train()
                    for bx, by in loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        loss = criterion(model(bx), by)
                        loss.backward()
                        optimizer.step()
            result = tracker.result

        results.append({
            "batch_size": batch_size,
            "learning_rate": lr,
            "energy_kwh": result.energy_kwh if result else 0.0,
            "co2eq_g": result.co2eq_corrected_g if result else 0.0,
            "duration_s": result.duration_seconds if result else 0.0,
        })

        logger.info(f"  Energy: {results[-1]['energy_kwh']:.6f} kWh")

    df = pd.DataFrame(results)
    if df["energy_kwh"].min() > 0:
        ratio = df["energy_kwh"].max() / df["energy_kwh"].min()
        logger.info(f"\nHyperparameter energy range: {ratio:.1f}× (worst / best)")

    # Save
    out = Path(output_dir) if output_dir else RESULTS_DIR
    df.to_csv(out / "method_b_hyperparameter_results.csv", index=False)
    return df


# ======================================================================
# Method C: Transfer Learning Comparison
# ======================================================================

def run_method_c(
    country_iso: str = "GBR",
    output_dir: Optional[Path] = None,
    logger=None,
) -> pd.DataFrame:
    """Compare carbon emissions across three training modes for ResNet-50.

    Modes:
      1. From scratch (random init)
      2. Frozen backbone (transfer learning — classifier only)
      3. Full fine-tuning (pretrained, all weights trainable)

    Reference: Geißler et al. (2024) — transfer learning saves 40–62 % energy.

    Args:
        country_iso: 3-letter ISO country code.
        output_dir: Output directory.
        logger: Logger instance.

    Returns:
        DataFrame with mode, energy_kwh, co2eq_g columns.
    """
    if logger is None:
        logger = setup_logging("E_methodC_transfer")

    logger.info("=" * 60)
    logger.info("METHOD C: Transfer Learning Carbon Comparison (ResNet-50)")
    logger.info("=" * 60)

    from carbon_benchmark.data.loader import load_neu_steel_defects
    from carbon_benchmark.data.preprocessing import preprocess_images
    from carbon_benchmark.models.deep_learning import compare_training_modes

    X_train, X_test, y_train, y_test = load_neu_steel_defects()
    X_train, X_test, y_train, y_test = preprocess_images(
        X_train, X_test, y_train, y_test,
    )

    modes = compare_training_modes("resnet_50", X_train, y_train, X_test, y_test)
    results: List[Dict[str, Any]] = []

    for mode_name in ["from_scratch", "frozen_backbone", "full_finetune"]:
        logger.info(f"\n--- Mode: {mode_name} ---")

        if mode_name == "from_scratch":
            from carbon_benchmark.models.deep_learning import create_resnet50
            model, train_fn, _ = create_resnet50(
                num_classes=6, freeze_backbone=False, pretrained=False,
            )
        elif mode_name == "frozen_backbone":
            from carbon_benchmark.models.deep_learning import create_resnet50
            model, train_fn, _ = create_resnet50(
                num_classes=6, freeze_backbone=True, pretrained=True,
            )
        else:
            from carbon_benchmark.models.deep_learning import create_resnet50
            model, train_fn, _ = create_resnet50(
                num_classes=6, freeze_backbone=False, pretrained=True,
            )

        tracker = CarbonTracker(
            experiment_id=f"E_methodC_{mode_name}",
            model_name="resnet_50",
            method="C",
            country_iso=country_iso,
            task_type="training",
            output_dir=output_dir,
        )

        result = tracker.track_training(
            train_fn, X_train, y_train, X_test, y_test,
        )

        results.append({
            "mode": mode_name,
            "energy_kwh": result.energy_kwh,
            "co2eq_g": result.co2eq_corrected_g,
            "duration_s": result.duration_seconds,
        })
        logger.info(f"  Energy: {result.energy_kwh:.6f} kWh")

    df = pd.DataFrame(results)
    if len(df) >= 2 and df["energy_kwh"].iloc[1] > 0:
        saving = (1 - df["energy_kwh"].iloc[1] / df["energy_kwh"].iloc[0]) * 100
        logger.info(f"\nTransfer learning energy saving: {saving:.1f}%")

    out = Path(output_dir) if output_dir else RESULTS_DIR
    df.to_csv(out / "method_c_transfer_learning_results.csv", index=False)
    return df


# ======================================================================
# Method E: Long-term Deployment Simulation
# ======================================================================

def run_method_e(
    inference_energy_kwh_per_1000: Dict[str, float],
    training_co2eq_g: Dict[str, float],
    output_dir: Optional[Path] = None,
    logger=None,
) -> pd.DataFrame:
    """Simulate annual carbon emissions under three deployment scenarios.

    Reference: Luccioni et al. (2024) — inference dominates for high-throughput apps.

    Args:
        inference_energy_kwh_per_1000: Dict of model_name → kWh per 1000 inferences.
        training_co2eq_g: Dict of model_name → training CO₂eq in grams.
        output_dir: Output directory.
        logger: Logger instance.

    Returns:
        DataFrame with scenario, model, annual_co2eq_kg, inference_pct columns.
    """
    if logger is None:
        logger = setup_logging("E_methodE_deployment")

    logger.info("=" * 60)
    logger.info("METHOD E: Long-term Deployment Simulation")
    logger.info("=" * 60)

    rows: List[Dict[str, Any]] = []

    for scenario_name, scenario in DEPLOYMENT_SCENARIOS.items():
        annual_inferences = (
            scenario["inferences_per_day"] * scenario["days_per_year"]
        )
        logger.info(f"\nScenario: {scenario_name}")
        logger.info(f"  Annual inferences: {annual_inferences:,}")

        for model_name, kwh_per_1k in inference_energy_kwh_per_1000.items():
            annual_energy = kwh_per_1k * (annual_inferences / 1000)
            # Use UK carbon intensity as default for simulation
            ci = CARBON_INTENSITY["UK"]
            cf = CALIBRATION_FACTORS.get(MODEL_TIERS.get(model_name, "T3"), 1.40)
            annual_co2eq_kg = (annual_energy * ci * cf) / 1000.0

            train_co2eq_kg = training_co2eq_g.get(model_name, 0.0) / 1000.0
            total = annual_co2eq_kg + train_co2eq_kg
            inference_pct = (annual_co2eq_kg / total * 100) if total > 0 else 0.0

            rows.append({
                "scenario": scenario_name,
                "model": model_name,
                "tier": MODEL_TIERS.get(model_name, "T3"),
                "annual_inferences": annual_inferences,
                "annual_energy_kwh": round(annual_energy, 3),
                "annual_co2eq_kg": round(annual_co2eq_kg, 3),
                "training_co2eq_kg": round(train_co2eq_kg, 3),
                "inference_pct": round(inference_pct, 1),
            })

            logger.info(
                f"  {model_name}: {annual_co2eq_kg:.2f} kgCO₂eq/yr "
                f"({inference_pct:.0f}% from inference)"
            )

    df = pd.DataFrame(rows)
    out = Path(output_dir) if output_dir else RESULTS_DIR
    df.to_csv(out / "method_e_deployment_simulation.csv", index=False)
    return df


# ======================================================================
# Method F: Carbon Intensity Sensitivity
# ======================================================================

def run_method_f(
    model_energy_kwh: float,
    model_name: str = "resnet_50",
    output_dir: Optional[Path] = None,
    logger=None,
) -> pd.DataFrame:
    """Compute CO₂eq for the same workload across 14 manufacturing countries.

    Reference: Lacoste et al. (2019) — 63× variation in carbon intensity.

    Args:
        model_energy_kwh: Measured energy consumption of the workload (kWh).
        model_name: Model identifier for metadata.
        output_dir: Output directory.
        logger: Logger instance.

    Returns:
        DataFrame with country, carbon_intensity, co2eq_g, relative_to_best.
    """
    if logger is None:
        logger = setup_logging("E_methodF_carbon_intensity")

    logger.info("=" * 60)
    logger.info("METHOD F: Carbon Intensity Sensitivity Analysis")
    logger.info(f"  Model: {model_name}, Energy: {model_energy_kwh:.6f} kWh")
    logger.info("=" * 60)

    tier = MODEL_TIERS.get(model_name, "T3")
    cf = CALIBRATION_FACTORS.get(tier, 1.40)

    best_ci = min(CARBON_INTENSITY.values())
    rows: List[Dict[str, Any]] = []

    for country, ci in sorted(CARBON_INTENSITY.items(), key=lambda x: x[1]):
        co2eq_g = model_energy_kwh * ci * cf  # kWh * gCO2/kWh * unitless = gCO2
        rows.append({
            "country": country,
            "carbon_intensity_g_per_kwh": ci,
            "energy_kwh": model_energy_kwh,
            "co2eq_g": round(co2eq_g, 2),
            "relative_to_best": round(ci / best_ci, 1),
        })
        logger.info(f"  {country:<20} CI={ci:4.0f} → {co2eq_g:8.1f} gCO₂eq "
                     f"({ci/best_ci:.0f}× best)")

    df = pd.DataFrame(rows)
    out = Path(output_dir) if output_dir else RESULTS_DIR
    df.to_csv(out / "method_f_carbon_intensity_sensitivity.csv", index=False)
    return df


# ======================================================================
# Method G: Deployment Topology Comparison
# ======================================================================

def run_method_g(
    model_energy_kwh: float,
    model_name: str = "resnet_50",
    output_dir: Optional[Path] = None,
    logger=None,
) -> pd.DataFrame:
    """Compare CO₂eq across deployment topologies for the same workload.

    Combines PUE and regional carbon intensity to reveal the combined effect
    of deployment topology and location.

    Reference: Patterson et al. (2022) — Mechanization + Map = 7–20× reduction.

    Args:
        model_energy_kwh: IT equipment energy consumption (kWh).
        model_name: Model identifier.
        output_dir: Output directory.
        logger: Logger instance.

    Returns:
        DataFrame with topology, location, pue, ci, co2eq_g columns.
    """
    if logger is None:
        logger = setup_logging("E_methodG_deployment_topology")

    logger.info("=" * 60)
    logger.info("METHOD G: Deployment Topology Comparison")
    logger.info(f"  Model: {model_name}, IT Energy: {model_energy_kwh:.6f} kWh")
    logger.info("=" * 60)

    scenarios = [
        ("Factory Floor — China", "factory_floor", "China"),
        ("Factory Floor — Germany", "factory_floor", "Germany"),
        ("On-Premise DC — UK", "on_premise_datacenter", "UK"),
        ("AWS — Ireland", "aws", "Ireland"),
        ("AWS — California", "aws", "USA_California"),
        ("Azure — Netherlands", "azure", "Netherlands"),
        ("Google Cloud — Finland", "google_cloud", "Finland"),
        ("Edge Device — Any", "edge_device", "Sweden"),
    ]

    tier = MODEL_TIERS.get(model_name, "T3")
    cf = CALIBRATION_FACTORS.get(tier, 1.40)
    rows: List[Dict[str, Any]] = []

    for label, pue_key, ci_key in scenarios:
        pue_val = PUE[pue_key]
        ci_val = CARBON_INTENSITY[ci_key]
        total_energy = model_energy_kwh * pue_val
        co2eq_g = total_energy * ci_val * cf  # kWh * gCO2/kWh * unitless = gCO2

        rows.append({
            "scenario": label,
            "pue": pue_val,
            "pue_source": pue_key,
            "carbon_intensity": ci_val,
            "location": ci_key,
            "total_energy_kwh": round(total_energy, 6),
            "co2eq_g": round(co2eq_g, 2),
        })
        logger.info(f"  {label:<30} PUE={pue_val:.2f} CI={ci_val:4.0f} "
                     f"→ {co2eq_g:8.1f} gCO₂eq")

    df = pd.DataFrame(rows)
    if len(df) > 0:
        best = df["co2eq_g"].min()
        worst = df["co2eq_g"].max()
        if best > 0:
            logger.info(f"\n  Deployment ratio (worst/best): {worst/best:.0f}×")

    out = Path(output_dir) if output_dir else RESULTS_DIR
    df.to_csv(out / "method_g_deployment_topology.csv", index=False)
    return df
