"""
Generate publication-quality figures for the MACE-F framework.

Reads experiment results from results/ and produces 8 figures saved to figures/.
All charts use a consistent colour-blind-friendly palette, no overlapping text,
and clean publication formatting.

Usage:  python scripts/generate_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Script lives in scripts/; data and figure outputs are at the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Colour-blind-friendly palette (Okabe-Ito inspired + tier mapping)
TIER_COLORS: dict = {
    "T1": "#0072B2",   # blue
    "T2": "#009E73",   # bluish green
    "T3": "#F0E442",   # yellow
    "T4": "#E69F00",   # orange
    "T5": "#D55E00",   # vermillion
}
# Secondary palette for non-tier charts
PALETTE = ["#0072B2", "#009E73", "#F0E442", "#E69F00", "#D55E00",
           "#CC79A7", "#56B4E9", "#999999"]

DPI = 200
STYLE = "seaborn-v0_8-whitegrid"
plt.style.use(STYLE)

# Global rcParams for clean output
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial", "Helvetica"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 9,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ---------------------------------------------------------------------------
# Model display names (compact, no newlines for most uses)
# ---------------------------------------------------------------------------
MODEL_DISPLAY: dict = {
    "linear_regression": "Linear Reg.",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mobilenet_v2": "MobileNet-v2",
    "resnet_50": "ResNet-50",
    "lstm": "LSTM",
    "resnet_152": "ResNet-152",
    "vit_b_16": "ViT-B/16",
    "stable_diffusion": "Stable Diff.",
    "gpt_4o_mini": "GPT-4o-mini",
}

TIER_ORDER = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T5": 4}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_training_data() -> pd.DataFrame:
    """Extract Method A training results (T1-T4) from summary.csv."""
    df = pd.read_csv(RESULTS_DIR / "summary.csv")
    training = df[df["method"] == "A"].copy()
    grouped = training.groupby("model_name").agg(
        tier=("tier", "first"),
        energy_kwh=("energy_kwh", "median"),
        co2eq_g=("co2eq_corrected_g", "median"),
        duration_s=("duration_seconds", "median"),
        accuracy=("extra_metric_accuracy", "median"),
        f1=("extra_metric_f1", "median"),
        r2=("extra_metric_r2", "median"),
    ).reset_index()
    grouped["display"] = grouped["model_name"].map(MODEL_DISPLAY)
    grouped["tier_rank"] = grouped["tier"].map(TIER_ORDER)
    return grouped.sort_values(["tier_rank", "co2eq_g"]).reset_index(drop=True)


# ============================================================================
# Figure 1 — Training Carbon by Model Tier
# ============================================================================
def fig1_training_carbon() -> None:
    data = _load_training_data()
    # Sort: by tier then by CO2 ascending
    data = data.sort_values(["tier_rank", "co2eq_g"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # --- Main chart: all models ---
    x = range(len(data))
    bars = ax.bar(
        x, data["co2eq_g"],
        color=[TIER_COLORS[t] for t in data["tier"]],
        edgecolor="white", linewidth=0.5, width=0.65,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(data["display"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Calibrated CO₂ Emissions (g)", fontsize=12)
    ax.set_title("Training Carbon Footprint by AI Model Tier", fontsize=14, fontweight="bold")

    # Value labels — offset above bar
    for bar, val in zip(bars, data["co2eq_g"]):
        y_offset = bar.get_height() * 0.02 + 0.15
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    # Expand y-limit to fit labels
    ymax = data["co2eq_g"].max() * 1.25
    ax.set_ylim(0, ymax)

    # Legend — compact labels, no title
    legend_elements = [Patch(facecolor=c, label=i)
                       for i, c in TIER_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    fig.tight_layout(pad=1.2)
    fig.savefig(FIGURES_DIR / "fig1_training_carbon.png")
    plt.close(fig)
    print("  OK  fig1_training_carbon.png")


# ============================================================================
# Figure 2 — Carbon Efficiency Scatter
# ============================================================================
def fig2_carbon_efficiency() -> None:
    data = _load_training_data()
    # Keep classification models with valid accuracy
    cls = data[data["accuracy"].notna() & (data["accuracy"] > 0)].copy()

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for _, row in cls.iterrows():
        tier = row["tier"]
        sz = max(60, row["duration_s"] * 0.06)
        ax.scatter(
            row["co2eq_g"], row["accuracy"] * 100,
            s=sz, c=TIER_COLORS[tier],
            edgecolors="#333333", linewidth=0.6, alpha=0.9, zorder=4,
        )

    # Annotate with collision-avoidance
    annotated: list = []
    for _, row in cls.iterrows():
        cx, cy = row["co2eq_g"], row["accuracy"] * 100
        name = row["display"]

        # Find a clear offset direction
        best_offset = (12, 8)
        min_overlap = float("inf")
        for dx, dy in [(12, 8), (-50, 6), (15, -10), (-45, -10),
                        (20, 2), (-55, -8), (8, 12), (-40, 4)]:
            tx, ty = cx + dx * (cx / 60 if cx > 0 else 1), cy + dy
            overlap = sum(
                1 / max(0.01, abs(tx - ax_) + abs(ty - ay_))
                for ax_, ay_ in annotated
            )
            if overlap < min_overlap:
                min_overlap = overlap
                best_offset = (dx, dy)

        ax.annotate(
            name, (cx, cy),
            textcoords="offset points", xytext=best_offset, fontsize=8.5,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "lw": 0.6, "color": "#666666", "shrinkA": 6},
        )
        annotated.append((cx + best_offset[0], cy + best_offset[1]))

    ax.set_xlabel("Calibrated CO₂ Emissions (g)", fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title("Carbon Efficiency: Accuracy vs. Carbon Cost", fontsize=14, fontweight="bold")

    # Ensure x=0 is visible
    ax.set_xlim(left=-0.5, right=cls["co2eq_g"].max() * 1.4)
    ax.set_ylim(bottom=cls["accuracy"].min() * 100 - 3, top=103)

    legend_elements = [Patch(facecolor=c, label=i)
                       for i, c in TIER_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Annotation: bubble = duration
    ax.text(0.97, 0.04, "Bubble area ∝ training duration",
            transform=ax.transAxes, fontsize=8, color="#888888",
            ha="right", fontstyle="italic")

    fig.tight_layout(pad=1.2)
    fig.savefig(FIGURES_DIR / "fig2_carbon_efficiency.png")
    plt.close(fig)
    print("  OK  fig2_carbon_efficiency.png")


# ============================================================================
# Figure 3 — Carbon Intensity Sensitivity
# ============================================================================
def fig3_carbon_intensity() -> None:
    df = pd.read_csv(RESULTS_DIR / "method_f_carbon_intensity_sensitivity.csv")
    df = df.sort_values("co2eq_g", ascending=True)

    # Clean country names
    df["label"] = df["country"].str.replace("_", " ")

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Color by relative intensity: green→yellow→red
    colors = []
    for rel in df["relative_to_best"]:
        if rel < 5:
            colors.append("#0072B2")
        elif rel < 20:
            colors.append("#F0E442")
        elif rel < 40:
            colors.append("#E69F00")
        else:
            colors.append("#D55E00")

    bars = ax.barh(df["label"], df["co2eq_g"], color=colors,
                   edgecolor="white", linewidth=0.5, height=0.65)

    # Labels beside bars — ensure they don't overflow
    xmax = df["co2eq_g"].max()
    for bar, val, rel in zip(bars, df["co2eq_g"], df["relative_to_best"]):
        ax.text(
            bar.get_width() + xmax * 0.015, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} g   ({rel:.0f}×)", va="center", fontsize=7.5,
        )

    ax.set_xlabel("CO₂ Emissions per Training Run (g)", fontsize=12)
    ax.set_title("Carbon Intensity Sensitivity: Same Model, 20 Countries",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, xmax * 1.42)
    ax.invert_yaxis()

    # Legend for color bands
    from matplotlib.patches import Patch
    leg = [
        Patch(facecolor="#0072B2", label="Low CI  (<5×)"),
        Patch(facecolor="#F0E442", label="Medium CI (5–20×)"),
        Patch(facecolor="#E69F00", label="High CI (20–40×)"),
        Patch(facecolor="#D55E00", label="Very High CI (>40×)"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=8)

    fig.tight_layout(pad=1.2)
    fig.savefig(FIGURES_DIR / "fig3_carbon_intensity.png")
    plt.close(fig)
    print("  OK  fig3_carbon_intensity.png")


# ============================================================================
# Figure 4 — Deployment Topology
# ============================================================================
def fig4_deployment_topology() -> None:
    df = pd.read_csv(RESULTS_DIR / "method_g_deployment_topology.csv")
    df = df.sort_values("co2eq_g", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Color by infra type
    infra_colors = {
        "factory_floor": "#D55E00",
        "on_premise_datacenter": "#E69F00",
        "aws": "#F0E442",
        "azure": "#0072B2",
        "google_cloud": "#009E73",
        "edge_device": "#56B4E9",
    }

    bars = ax.barh(
        df["scenario"], df["co2eq_g"],
        color=[infra_colors.get(s, "#999999") for s in df["pue_source"]],
        edgecolor="white", linewidth=0.5, height=0.6,
    )

    xmax = df["co2eq_g"].max()
    for bar, val, pue, src in zip(bars, df["co2eq_g"], df["pue"], df["pue_source"]):
        infra_label = src.replace("_", " ").title()
        ax.text(
            bar.get_width() + xmax * 0.015, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} g   PUE={pue:.2f}  [{infra_label}]",
            va="center", fontsize=8,
        )

    ax.set_xlabel("CO₂ Emissions per Training Run (g)", fontsize=12)
    ax.set_title("Deployment Topology: PUE × Location Carbon Intensity",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, xmax * 1.55)
    ax.invert_yaxis()

    # Legend
    leg = [
        Patch(facecolor="#D55E00", label="Factory Floor (PUE=2.0)"),
        Patch(facecolor="#E69F00", label="On-Premise DC (PUE=1.7)"),
        Patch(facecolor="#F0E442", label="AWS (PUE=1.15)"),
        Patch(facecolor="#0072B2", label="Azure (PUE=1.2)"),
        Patch(facecolor="#009E73", label="Google Cloud (PUE=1.1)"),
        Patch(facecolor="#56B4E9", label="Edge Device (PUE=1.02)"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=8)

    fig.tight_layout(pad=1.2)
    fig.savefig(FIGURES_DIR / "fig4_deployment_topology.png")
    plt.close(fig)
    print("  OK  fig4_deployment_topology.png")


# ============================================================================
# Figure 5 — Transfer Learning Comparison
# ============================================================================
def fig5_transfer_learning() -> None:
    df = pd.read_csv(RESULTS_DIR / "method_c_transfer_learning_results.csv")

    fig, ax = plt.subplots(figsize=(8, 5.5))

    modes = ["from_scratch", "frozen_backbone", "full_finetune"]
    labels = [
        "From Scratch\n(random init)",
        "Transfer Learning\n(frozen backbone)",
        "Full Fine-tuning\n(all weights)",
    ]
    colors = ["#D55E00", "#009E73", "#E69F00"]

    values = []
    for mode in modes:
        row = df[df["mode"] == mode]
        val = row["energy_kwh"].values[0] if len(row) > 0 else 0
        values.append(val)

    x = range(3)
    bars = ax.bar(x, values, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.55)

    # Value labels on top
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0004,
                f"{val:.4f} kWh", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Energy Consumption (kWh)", fontsize=12)
    ax.set_title("Transfer Learning: 60.6% Energy Savings\n(ResNet-50, NEU Steel Defects)",
                 fontsize=14, fontweight="bold")

    # Arrow + savings label between from-scratch and frozen
    y_top = max(values) * 1.18
    ax.set_ylim(0, y_top)

    from_s = values[0]
    frozen = values[1]
    savings_pct = (1 - frozen / from_s) * 100
    mid_y = (from_s + frozen) / 2
    # Down-arrow from from_scratch to frozen
    ax.annotate(
        "", xy=(1, frozen), xytext=(0, from_s),
        arrowprops={"arrowstyle": "<->", "lw": 1.8, "color": "#333333"},
    )
    ax.text(0.5, mid_y + 0.0008, f"−{savings_pct:.1f}%",
            ha="center", fontsize=13, fontweight="bold", color="#CC0000")

    # Note: full_finetune ≈ from_scratch
    ax.annotate(
        "Full fine-tuning\n≈ from scratch",
        xy=(2, values[2]), xytext=(2.4, values[2] * 0.85),
        fontsize=8.5, color="#888888", fontstyle="italic",
        arrowprops={"arrowstyle": "->", "lw": 0.5, "color": "#888888"},
    )

    fig.tight_layout(pad=1.2)
    fig.savefig(FIGURES_DIR / "fig5_transfer_learning.png")
    plt.close(fig)
    print("  OK  fig5_transfer_learning.png")


# ============================================================================
# Figure 6 — Hyperparameter Carbon Efficiency
# ============================================================================
def fig6_hyperparameter() -> None:
    df = pd.read_csv(RESULTS_DIR / "method_b_hyperparameter_results.csv")

    fig, ax = plt.subplots(figsize=(11, 5))

    # Build readable labels
    labels = [
        f"bs={int(r['batch_size'])}\nlr={r['learning_rate']:.0e}"
        for _, r in df.iterrows()
    ]

    # Color by batch size
    bs_colors = {
        8:  "#D55E00",
        16: "#E69F00",
        32: "#0072B2",
        64: "#009E73",
    }

    x = range(len(df))
    energy_mwh = df["energy_kwh"] * 1000  # convert to mWh

    bars = ax.bar(
        x, energy_mwh,
        color=[bs_colors.get(int(r["batch_size"]), "#999999")
               for _, r in df.iterrows()],
        edgecolor="white", linewidth=0.5, width=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Energy Consumption (mWh)", fontsize=12)
    ax.set_title("Hyperparameter Carbon Efficiency: ResNet-50 on NEU Steel Defects",
                 fontsize=14, fontweight="bold")

    # Best line
    best = energy_mwh.min()
    ax.axhline(y=best, color="#009E73", linestyle="--", alpha=0.6, lw=1.2,
               label=f"Best: {best:.1f} mWh")

    # Value labels above bars (horizontal, not rotated)
    ymax = energy_mwh.max()
    for bar, val in zip(bars, energy_mwh):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.008,
            f"{val:.1f}", ha="center", fontsize=8, fontweight="bold",
        )

    worst = energy_mwh.max()
    ax.set_ylim(0, ymax * 1.16)

    # Legend
    leg = [
        Patch(facecolor="#D55E00", label="bs=8"),
        Patch(facecolor="#E69F00", label="bs=16"),
        Patch(facecolor="#0072B2", label="bs=32"),
        Patch(facecolor="#009E73", label="bs=64"),
    ]
    ax.legend(handles=leg + [plt.Line2D([0], [0], color="#009E73",
                                         linestyle="--", lw=1.2, label=f"Best ({best:.1f} mWh)")],
              title="Batch Size", loc="upper right", fontsize=8, title_fontsize=9)

    # Context note
    ax.text(0.5, 0.97,
            f"Observed range: {worst/best:.2f}×  |  Literature: 2–5× for full-scale training",
            transform=ax.transAxes, ha="center", fontsize=9,
            fontstyle="italic", color="#555555", va="top")

    fig.tight_layout(pad=1.3)
    fig.savefig(FIGURES_DIR / "fig6_hyperparameter.png")
    plt.close(fig)
    print("  OK  fig6_hyperparameter.png")


# ============================================================================
# Figure 7 — Annual Deployment Simulation
# ============================================================================
def fig7_annual_deployment() -> None:
    df = pd.read_csv(RESULTS_DIR / "method_e_deployment_simulation.csv")

    scenario_order = ["on_demand", "batch_qc", "continuous_24x7"]
    scenario_labels = ["On-Demand\n(25k/year)", "Batch QC\n(2.5M/year)", "24/7 Continuous\n(31.5M/year)"]
    scenario_colors = ["#0072B2", "#F0E442", "#D55E00"]

    # Model display order (by tier)
    model_order = [
        "linear_regression", "random_forest", "xgboost",
        "mobilenet_v2", "resnet_50", "lstm",
        "resnet_152", "stable_diffusion", "gpt_4o_mini",
    ]
    model_labels = [MODEL_DISPLAY.get(m, m) for m in model_order]

    # Pivot
    piv = df.pivot_table(
        index="model", columns="scenario",
        values="annual_co2eq_kg", aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(model_order))
    width = 0.25

    for i, (scenario, label, color) in enumerate(
        zip(scenario_order, scenario_labels, scenario_colors)
    ):
        vals = []
        for m in model_order:
            if m in piv.index and scenario in piv.columns:
                v = piv.loc[m, scenario]
            else:
                v = np.nan
            vals.append(v)
        ax.bar(x + i * width, vals, width, label=label,
               color=color, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Annual CO₂ Emissions (kg)", fontsize=12)
    ax.set_title("Annual Carbon Emissions: Deployment Scenario × Model Tier",
                 fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper left", title="Deployment Scenario")
    ax.grid(axis="y", alpha=0.25)

    # Add tier labels above model names
    tier_labels = ["T1", "T2", "T2", "T2", "T3", "T3", "T4", "T5", "T5"]
    for xi, tier in zip(x, tier_labels):
        ax.text(xi + width, -0.08, tier, transform=ax.get_xaxis_transform(),
                ha="center", fontsize=7, fontweight="bold",
                color=TIER_COLORS.get(tier, "#000000"),
                clip_on=False)

    # Handle zero/near-zero values on log scale — add tiny epsilon line
    ylims = ax.get_ylim()
    ax.set_ylim(bottom=max(ylims[0], 5e-5))

    fig.tight_layout(pad=1.5)
    fig.savefig(FIGURES_DIR / "fig7_annual_deployment.png")
    plt.close(fig)
    print("  OK  fig7_annual_deployment.png")


# ============================================================================
# Figure 8 — Training vs. Annual Inference
# ============================================================================
def fig8_training_vs_inference() -> None:
    """Training once vs. one year of 24/7 inference for each model."""
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    deploy = pd.read_csv(RESULTS_DIR / "method_e_deployment_simulation.csv")

    # Training data
    train = summary[summary["method"] == "A"].groupby("model_name").agg(
        train_co2_g=("co2eq_corrected_g", "median"),
        tier=("tier", "first"),
    ).reset_index()

    # Inference data: 24/7 only
    infer = deploy[deploy["scenario"] == "continuous_24x7"][
        ["model", "annual_co2eq_kg"]
    ].rename(columns={"model": "model_name"})
    infer["annual_co2eq_g"] = infer["annual_co2eq_kg"] * 1000

    merged = train.merge(infer, on="model_name", how="inner")

    # Order
    model_order = [
        "linear_regression", "random_forest", "xgboost",
        "mobilenet_v2", "resnet_50", "lstm",
        "resnet_152", "stable_diffusion", "gpt_4o_mini",
    ]
    merged["sort_key"] = merged["model_name"].apply(
        lambda m: model_order.index(m) if m in model_order else 99
    )
    merged = merged.sort_values("sort_key").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(merged))
    width = 0.32

    ax.bar(
        x - width / 2, merged["train_co2_g"], width,
        label="One-Time Training", color="#0072B2",
        edgecolor="white", linewidth=0.3,
    )
    ax.bar(
        x + width / 2, merged["annual_co2eq_g"], width,
        label="Annual Inference (24/7)", color="#D55E00",
        edgecolor="white", linewidth=0.3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(m, m) for m in merged["model_name"]],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_ylabel("CO₂ Emissions (g)", fontsize=12)
    ax.set_title("Training vs. Annual Inference Carbon: The Deployment Multiplier",
                 fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    # Ratio labels above inference bars
    for i, (_, row) in enumerate(merged.iterrows()):
        if row["train_co2_g"] > 0 and row["annual_co2eq_g"] > 0:
            ratio = row["annual_co2eq_g"] / row["train_co2_g"]
            y_pos = row["annual_co2eq_g"] * 2.0  # place above bar on log scale
            ax.text(
                i + width / 2, y_pos,
                f"{ratio:.0f}×", ha="center", fontsize=7.5,
                fontweight="bold", color="#CC0000",
            )

    # Tier labels
    for i, (_, row) in enumerate(merged.iterrows()):
        tier = row["tier"]
        ax.text(i, -0.09, tier if isinstance(tier, str) else "",
                transform=ax.get_xaxis_transform(),
                ha="center", fontsize=7, fontweight="bold",
                color=TIER_COLORS.get(str(tier), "#000000"),
                clip_on=False)

    # Adjust bottom for tier labels
    fig.tight_layout(pad=1.8)
    fig.savefig(FIGURES_DIR / "fig8_training_vs_inference.png")
    plt.close(fig)
    print("  OK  fig8_training_vs_inference.png")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("MACE-F Figures Generator")
    print(f"  Output: {FIGURES_DIR}/")
    print()

    try:
        fig1_training_carbon()
        fig2_carbon_efficiency()
        fig3_carbon_intensity()
        fig4_deployment_topology()
        fig5_transfer_learning()
        fig6_hyperparameter()
        fig7_annual_deployment()
        fig8_training_vs_inference()
    except Exception as exc:  # noqa: BLE001 — top-level guard prints and exits
        print(f"\nERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nDone — 8 figures saved to {FIGURES_DIR}/")
