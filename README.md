# MACE-F: Manufacturing AI Carbon Emission Framework

A modular Python framework for measuring and comparing the carbon footprint of
AI models deployed in manufacturing contexts, using CodeCarbon as the core
measurement instrument.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify your hardware is detected
python -c "from carbon_benchmark.utils.hardware import print_hardware_summary; print_hardware_summary()"

# 3. Run a single quick test
python -m carbon_benchmark.experiments.method_a_baseline_training --model linear_regression

# 4. Run the full experiment matrix
python run_all_experiments.py --all --country GBR
```

## Project Structure

```
carbon_benchmark/
├── config.py                 # Central configuration
├── models/                   # T1–T5 model factories
│   ├── traditional_ml.py     # LinearRegression, DecisionTree, RandomForest, XGBoost
│   ├── deep_learning.py      # MobileNet-v2, ResNet-50/152, ViT-B/16, LSTM
│   └── generative.py         # Stable Diffusion, GPT-4o-mini
├── data/                     # Dataset loaders & preprocessing
│   ├── loader.py             # SECOM, NASA CMAPSS, NEU Steel, CNC Tool Wear
│   └── preprocessing.py      # Tabular, image, time-series pipelines
├── measurement/              # Carbon measurement core
│   ├── tracker.py            # Unified CodeCarbon wrapper
│   ├── calibrator.py         # External meter calibration (Fischer 2025)
│   └── reporter.py           # Results aggregation & CSV export
├── experiments/              # Seven experiment methods (A–G)
│   ├── method_a_baseline_training.py
│   ├── method_d_baseline_inference.py
│   └── methods_bc_efg.py     # Methods B, C, E, F, G
├── utils/                    # Hardware detection & logging
├── tests/                    # Unit tests
├── results/                  # Output CSV files
└── run_all_experiments.py    # Master orchestrator
```

## Experiment Methods

| Method | Description | Key Reference |
|--------|-------------|---------------|
| **A** | Baseline training carbon measurement | Schmidt et al. (2021) |
| **B** | Hyperparameter carbon efficiency (8 combinations) | Geißler et al. (2024) |
| **C** | Transfer learning comparison (3 modes) | Geißler et al. (2024) |
| **D** | Baseline inference carbon measurement (per 1000) | Luccioni et al. (2024) |
| **E** | Long-term deployment simulation (3 scenarios) | Luccioni et al. (2024) |
| **F** | Carbon intensity sensitivity (14 countries) | Lacoste et al. (2019) |
| **G** | Deployment topology comparison (8 scenarios) | Patterson et al. (2022) |

## Supported Models (T1–T5)

| Tier | Models | Manufacturing Application |
|------|--------|--------------------------|
| T1 🟢 | Linear Regression, Decision Tree | Sensor calibration, threshold alerts |
| T2 🟢 | Random Forest, XGBoost, MobileNet-v2 | Predictive maintenance, basic defect detection |
| T3 🟡 | ResNet-50, LSTM | High-res defect detection, process optimisation |
| T4 🟠 | ResNet-152, ViT-B/16 | Multi-sensor fusion, complex inspection |
| T5 🔴 | Stable Diffusion, GPT-4o-mini | Generative design, document Q&A |

## Datasets

| Dataset | Source | Task | Auto-DL |
|---------|--------|------|---------|
| SECOM | [UCI ML Repository](https://archive.ics.uci.edu/dataset/179/secom) | Semiconductor quality classification | ✅ Automatic |
| NASA CMAPSS | [NASA Prognostics Center](https://data.nasa.gov/) | Turbofan engine RUL prediction | ✅ Automatic |
| NEU Steel Defects | [Kaggle](https://www.kaggle.com/datasets/kaustubhdhikshit/neu-surface-defect-database) / [NEU Official](http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm) | Steel surface defect classification | ✅ gdown / kagglehub |
| CNC Tool Wear | [Kaggle](https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill) | CNC tool wear regression | ✅ kagglehub |

**Auto-download**: Each loader automatically attempts to download the dataset on first use:
- SECOM → direct UCI download (`urllib`)
- NASA CMAPSS → dual GitHub mirror
- NEU Steel Defects → Google Drive (`gdown`) → kagglehub → manual prompt
- CNC Tool Wear → kagglehub → direct URL → manual prompt

**If auto-download fails**, all datasets fall back to synthetic data so the pipeline always runs.
See `data/loader.py` for download URLs and fallback logic.

**Optional dependencies for auto-download**: `pip install gdown kagglehub`

## Calibration

Per Fischer (2025), CodeCarbon underestimates total system energy by 25–37 %.
This project includes built-in calibration factors (configurable in `config.py`).

To calibrate with your own external power meter:

```python
from carbon_benchmark.measurement.calibrator import Calibrator

cal = Calibrator()
cal.add("E01", "linear_regression", "T1",
        codecarbon_kwh=0.012, external_meter_kwh=0.019)
cal.add("E06", "resnet_50", "T3",
        codecarbon_kwh=0.423, external_meter_kwh=0.587)

cal.compute_tier_factors()
cal.print_summary()
```

## References

1. Lacoste A, et al. Quantifying the carbon emissions of machine learning. 2019.
2. Schmidt V, et al. CodeCarbon: estimate and track carbon emissions. 2021.
3. Milutinović M, Lukić M. How green is your algorithm? SEMSIE 2025.
4. Fischer R. Ground-truthing AI energy consumption. arXiv:2509.22092, 2025.
5. Strubell E, et al. Energy and policy considerations for deep learning in NLP. ACL 2019.
6. Luccioni AS, et al. Power hungry processing. ACM FAccT 2024.
7. Geißler D, et al. The power of training. ARCS 2024.
8. Patterson D, et al. The carbon footprint of ML training will plateau, then shrink. IEEE Computer, 2022.
