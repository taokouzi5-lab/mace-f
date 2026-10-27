"""
Central configuration for the Manufacturing AI Carbon Benchmark project.

All experiment parameters, hardware specs, carbon intensity data, and model tier
mappings are defined here. No hardcoded values should appear in other modules.

References:
  - Carbon intensity data sourced from CO2 Signal API and IPCC regional averages
  - PUE values from Lacoste et al. (2019), Patterson et al. (2022)
  - Calibration factors derived from Fischer (2025) validation methodology
"""

from pathlib import Path
from typing import Dict, Final

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).parent.resolve()
DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "cache"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
LOG_DIR: Final[Path] = PROJECT_ROOT / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hardware Specification
# ---------------------------------------------------------------------------
HARDWARE: Final[Dict[str, str]] = {
    "cpu": "Intel i9-13900K",
    "gpu": "NVIDIA RTX 4090",
    "vram_gb": "24",
    "ram_gb": "64",
    "os": "Windows 11",
    "python": "3.11",
}

# ---------------------------------------------------------------------------
# Random Seed (Reproducibility)
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# Grid Carbon Intensity by Country/Region (gCO₂eq/kWh)
# Sourced from CO2 Signal API, IPCC, and national grid operators.
# References: Lacoste et al. (2019), Patterson et al. (2022)
# ---------------------------------------------------------------------------
CARBON_INTENSITY: Final[Dict[str, float]] = {
    "Switzerland": 16.0,
    "Norway": 20.0,
    "France": 56.0,
    "Sweden": 13.0,
    "Finland": 54.0,
    "UK": 231.0,
    "Germany": 338.0,
    "Japan": 472.0,
    "South_Korea": 416.0,
    "USA_California": 204.0,
    "USA_Texas": 427.0,
    "USA_Iowa": 737.0,
    "China": 582.0,
    "India": 708.0,
    "Vietnam": 432.0,
    "Mexico": 431.0,
    "Serbia": 671.0,
    "South_Africa": 1009.0,
    "Ireland": 331.0,
    "Netherlands": 397.0,
}

# ISO 3166-1 alpha-3 → config country key mapping
ISO_TO_COUNTRY: Final[Dict[str, str]] = {
    "GBR": "UK", "DEU": "Germany", "FRA": "France", "JPN": "Japan",
    "KOR": "South_Korea", "CHN": "China", "IND": "India", "VNM": "Vietnam",
    "MEX": "Mexico", "SRB": "Serbia", "ZAF": "South_Africa", "IRL": "Ireland",
    "NLD": "Netherlands", "NOR": "Norway", "SWE": "Sweden", "FIN": "Finland",
    "CHE": "Switzerland", "USA": "USA_California",
}

# Representative ResNet-50 training energy (kWh) — fallback for Methods F, G
REPRESENTATIVE_ENERGY_KWH: Final[float] = 0.423

# Default country for measurements
DEFAULT_COUNTRY: Final[str] = "UK"

# ---------------------------------------------------------------------------
# Power Usage Effectiveness (PUE) by Deployment Topology
# References: Lacoste et al. (2019), Patterson et al. (2022)
# ---------------------------------------------------------------------------
PUE: Final[Dict[str, float]] = {
    "google_cloud": 1.10,
    "aws": 1.15,
    "azure": 1.20,
    "colocation": 1.40,
    "on_premise_datacenter": 1.70,
    "factory_floor": 2.00,
    "edge_device": 1.02,
}

# ---------------------------------------------------------------------------
# Calibration Factors per Tier
# Accounts for CodeCarbon's 25-37% systematic underestimation.
# Derived from Fischer (2025) external power meter validation methodology.
# These are INITIAL ESTIMATES; replace with your own calibration measurements.
# ---------------------------------------------------------------------------
CALIBRATION_FACTORS: Final[Dict[str, float]] = {
    "T1": 1.58,  # Lightweight CPU workloads (largest relative overhead)
    "T2": 1.43,
    "T3": 1.39,
    "T4": 1.34,
    "T5": 1.25,  # Full-system GPU load (smallest relative overhead)
}

# ---------------------------------------------------------------------------
# CodeCarbon Configuration
# ---------------------------------------------------------------------------
CODECARBON_CONFIG: Final[Dict] = {
    "measure_power_secs": 15,       # Sampling interval (seconds)
    "offline_mode": True,           # Keep data local (manufacturing OT policy)
    "output_dir": str(RESULTS_DIR),
    "country_iso_code": "GBR",      # Default: UK
    "save_to_file": True,
    "logging_level": "INFO",
}

# ---------------------------------------------------------------------------
# Model Tier Mapping
# T1: Ultra-lightweight (<1K params)     → Linear Regression, Decision Tree
# T2: Lightweight (1K-10M params)         → Random Forest, XGBoost, MobileNet-v2
# T3: Standard DL (10M-50M params)        → ResNet-50, LSTM
# T4: Large DL (50M-500M params)          → ResNet-152, ViT-B/16
# T5: Generative (>500M params)           → Stable Diffusion, GPT-4o-mini
# ---------------------------------------------------------------------------
MODEL_TIERS: Final[Dict[str, str]] = {
    "linear_regression": "T1",
    "decision_tree": "T1",
    "random_forest": "T2",
    "xgboost": "T2",
    "mobilenet_v2": "T2",
    "resnet_50": "T3",
    "lstm": "T3",
    "resnet_152": "T4",
    "vit_b_16": "T4",
    "stable_diffusion": "T5",
    "gpt_4o_mini": "T5",
}

# ---------------------------------------------------------------------------
# Experiment Matrix
# Maps experiment IDs to (method, model, task_type)
# ---------------------------------------------------------------------------
EXPERIMENT_MATRIX: Final[list] = [
    # Method A: Baseline Training
    {"id": "E01", "method": "A", "model": "linear_regression", "task": "training"},
    {"id": "E02", "method": "A", "model": "decision_tree", "task": "training"},
    {"id": "E03", "method": "A", "model": "random_forest", "task": "training"},
    {"id": "E04", "method": "A", "model": "xgboost", "task": "training"},
    {"id": "E05", "method": "A", "model": "mobilenet_v2", "task": "training"},
    {"id": "E06", "method": "A", "model": "resnet_50", "task": "training"},
    {"id": "E07", "method": "A", "model": "lstm", "task": "training"},
    {"id": "E08", "method": "A", "model": "resnet_152", "task": "training"},
    {"id": "E09", "method": "A", "model": "vit_b_16", "task": "training"},
    # Method D: Baseline Inference (all models, including T5)
    {"id": "E10", "method": "D", "model": "linear_regression", "task": "inference"},
    {"id": "E11", "method": "D", "model": "random_forest", "task": "inference"},
    {"id": "E12", "method": "D", "model": "xgboost", "task": "inference"},
    {"id": "E13", "method": "D", "model": "mobilenet_v2", "task": "inference"},
    {"id": "E14", "method": "D", "model": "resnet_50", "task": "inference"},
    {"id": "E15", "method": "D", "model": "lstm", "task": "inference"},
    {"id": "E16", "method": "D", "model": "resnet_152", "task": "inference"},
    {"id": "E17", "method": "D", "model": "vit_b_16", "task": "inference"},
    {"id": "E18", "method": "D", "model": "stable_diffusion", "task": "inference"},
    {"id": "E19", "method": "D", "model": "gpt_4o_mini", "task": "inference"},
    # Method B: Hyperparameter Carbon Efficiency (ResNet-50 only)
    {"id": "E20", "method": "B", "model": "resnet_50", "task": "training"},
    # Method C: Transfer Learning Comparison (ResNet-50 only)
    {"id": "E21", "method": "C", "model": "resnet_50", "task": "training"},
    # Method F: Carbon Intensity Sensitivity (ResNet-50 representative)
    {"id": "E22", "method": "F", "model": "resnet_50", "task": "analysis"},
    # Method G: Deployment Topology Comparison (ResNet-50 representative)
    {"id": "E23", "method": "G", "model": "resnet_50", "task": "analysis"},
]

# ---------------------------------------------------------------------------
# Dataset URLs and Descriptions
# ---------------------------------------------------------------------------
DATASETS: Final[Dict[str, Dict[str, str]]] = {
    "secom": {
        "name": "SECOM Semiconductor Manufacturing",
        "url": "https://archive.ics.uci.edu/dataset/179/secom",
        "task": "binary_classification",
        "samples": "1567",
        "features": "591",
    },
    "nasa_cmapss": {
        "name": "NASA CMAPSS Turbofan Engine Degradation",
        "url": "https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6",
        "task": "regression",
        "samples": "~20,000 (time series across 21 engines)",
        "features": "26",
    },
    "neu_steel": {
        "name": "NEU Steel Surface Defects",
        "url": "https://www.kaggle.com/datasets/kaustubhdhikshit/neu-surface-defect-database",
        "task": "multi_class_classification",
        "samples": "1800 images",
        "classes": "6",
    },
    "cnc_tool_wear": {
        "name": "CNC Tool Wear Prediction",
        "url": "https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill",
        "task": "regression",
        "samples": "~2,500",
        "features": "~7 sensor channels",
    },
}

# ---------------------------------------------------------------------------
# Hyperparameter Grid for Method B
# ---------------------------------------------------------------------------
HYPERPARAM_GRID: Final[Dict] = {
    "batch_sizes": [8, 16, 32, 64],
    "learning_rates": [1e-3, 5e-4, 1e-4, 5e-5],
    # Simplified: test 8 representative combinations
    "combinations": [
        (8, 1e-3), (16, 1e-3), (32, 1e-3), (64, 1e-3),
        (32, 5e-4), (32, 1e-4), (32, 5e-5), (16, 1e-4),
    ],
    "target_f1": 0.93,
    "max_epochs": 50,
}

# ---------------------------------------------------------------------------
# Inference Scenarios for Method E (Long-term Deployment Simulation)
# ---------------------------------------------------------------------------
DEPLOYMENT_SCENARIOS: Final[Dict] = {
    "continuous_24x7": {
        "inferences_per_day": 86_400,   # 1/sec, 24h
        "days_per_year": 365,
        "description": "Continuous real-time monitoring (1 inference/sec, 24/7)",
    },
    "batch_qc": {
        "inferences_per_day": 10_000,   # Batch quality control
        "days_per_year": 250,
        "description": "Batch quality inspection (10,000/day, working days)",
    },
    "on_demand": {
        "inferences_per_day": 100,
        "days_per_year": 250,
        "description": "On-demand design tool (100/day, working days)",
    },
}

# ---------------------------------------------------------------------------
# Stable Diffusion Prompts for Manufacturing Design (T5)
# ---------------------------------------------------------------------------
MANUFACTURING_DESIGN_PROMPTS: Final[list] = [
    "A lightweight bracket for automotive suspension, isometric view, technical drawing style",
    "An ergonomic handle design for a CNC machine operator panel",
    "A heat sink design for power electronics cooling, with optimized fin geometry",
    "A robotic gripper jaw for picking cylindrical parts, mechanical engineering drawing",
    "A flange coupling design for industrial pipeline, cross-section view",
    "A gear housing for electric motor transmission, exploded view",
    "A mounting plate for industrial sensor array, with dimension annotations",
    "A turbine blade profile for gas compressor, aerodynamic cross-section",
    "A conveyor roller bracket for material handling system, assembly drawing",
    "A protective enclosure for factory floor IoT gateway, sheet metal design",
]
