# MACE-F: Manufacturing AI Carbon Emission Framework

**Measure, compare, and reduce the carbon footprint of AI in manufacturing — from a linear regression on a sensor stream to a diffusion model on the factory floor.**

[![CI](https://github.com/taokouzi5-lab/mace-f/actions/workflows/ci.yml/badge.svg)](https://github.com/taokouzi5-lab/mace-f/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

![MACE-F smoke test demo](assets/demo.gif)

## Highlights

| What the benchmark shows | Result |
|---|---|
| Coverage | **26 experiments** (23 experiment IDs plus 3 Method E deployment scenarios) across **10 model families**, all five tiers (T1–T5) and seven methods (A–G) |
| Energy gradient | **387×** from T1 to T4 — growth is non-linear, not a smooth curve |
| Where the energy goes | **>90%** of training energy is consumed by the GPU (91.2% in a ViT-B/16 sampling run) |
| Where the carbon goes | **63×** spread across deployment locations (Switzerland 16 vs South Africa 1009 gCO₂eq/kWh) |
| Measurement stack | CodeCarbon **3.2.7** with tier-specific external-power-meter calibration after Fischer (2025), on an NVIDIA RTX 3070 Ti Laptop 8 GB + Intel i7-12700H |

## What it measures

MACE-F is a modular benchmark and decision framework for the carbon footprint of
manufacturing AI. It combines direct measurement with a lookup model so that a
deployment can be assessed before it is built.

- **Five-tier taxonomy (T1–T5).** Models are grouped by parameter scale and
  computational demand, from linear regression (<1K parameters) to generative
  models (>500M parameters). The tier boundaries are validated against measured
  energy, not assumed.
- **Training and inference.** Method A measures training; Method D measures
  batch inference normalised per 1,000 samples; Method E simulates annual
  deployment under continuous, batch-QC, and on-demand workloads.
- **Carbon-efficiency decision matrix.** `carbon_benchmark/decision_matrix.py`
  maps tier × deployment scenario × grid region × infrastructure profile to an
  estimated annual footprint and an A–E carbon class:

  | Class | Estimated annual CO₂e | Guidance |
  |---|---|---|
  | A | < 0.1 kg | Negligible; below reporting threshold |
  | B | 0.1–1 kg | Low; monitor annually |
  | C | 1–100 kg | Moderate; consider model downgrade or cloud migration |
  | D | 100–1000 kg | High; act before deployment |
  | E | ≥ 1000 kg | Critical; deploy only if essential |

  ```python
  from carbon_benchmark.decision_matrix import MACEFramework

  result = MACEFramework().assess(
      model_tier="T3",
      deployment_scenario="continuous_24x7",
      country="China",
      infrastructure="factory_floor",
      use_transfer_learning=True,
  )
  print(result.carbon_class)      # A-E classification and guidance
  print(result.annual_co2eq_kg)   # estimated annual inference emissions
  ```

- **Grid carbon intensity.** 20 grid regions (17 countries plus three US state
  grids), from Sweden at 13 to South Africa at 1009 gCO₂eq/kWh.
- **Deployment topology.** Seven infrastructure profiles with power usage
  effectiveness (PUE) from 1.02 for an edge device to 2.00 for a factory-floor
  server.
- **Calibration.** CodeCarbon is corrected with tier-specific factors derived
  from the Fischer (2025) external-power-meter methodology. The `Calibrator`
  accepts your own paired measurements and recomputes the factors.
- **Reproducible methods.** Seven experiment methods (A–G) cover baselines,
  hyperparameters, transfer learning, inference, long-term deployment, location,
  and topology.

## Quick Start

Requirements: Python 3.10+, pip. An NVIDIA GPU is optional; CPU-only runs work
with the traditional models.

```bash
# 1. Clone and install (development extras include pytest, ruff and bandit)
git clone https://github.com/taokouzi5-lab/mace-f.git
cd mace-f
pip install -e ".[dev]"

# In mainland China, use the Aliyun mirror:
# pip install -e ".[dev]" --index-url https://mirrors.aliyun.com/pypi/simple/

# 2. Verify that the hardware is detected
python -c "import logging; logging.basicConfig(level=logging.INFO); from carbon_benchmark.utils.hardware import print_hardware_summary; print_hardware_summary()"

# 3. End-to-end smoke test (one training run)
python scripts/smoketest.py

# 4. Full experiment matrix against the UK grid
python scripts/run_all_experiments.py --all --country GBR
```

The smoke test exercises the complete path: hardware detection → dataset loading
→ CodeCarbon measurement → calibration → CSV export. It uses cached or
downloaded data when available and falls back to synthetic data otherwise, so it
always completes. On Windows consoles that still use a legacy code page, set
`PYTHONUTF8=1` before running commands that print CO₂ labels.

Optional extras:

```bash
pip install -e ".[dl]"    # PyTorch models: MobileNet-v2, ResNet-50/152, ViT-B/16, LSTM, Stable Diffusion
pip install -e ".[api]"   # OpenAI model: GPT-4o-mini (requires OPENAI_API_KEY)
pip install -e ".[viz]"   # Chart generation: scripts/generate_figures.py, scripts/generate_flowchart.py
```

## Experiment Methods

| Method | Description | Key reference |
|---|---|---|
| **A** | Baseline training carbon measurement (9 models, 3 repeats) | Schmidt et al. (2021) |
| **B** | Hyperparameter carbon efficiency (8 batch-size / learning-rate combinations) | Geißler et al. (2024) |
| **C** | Transfer learning comparison (from scratch / frozen backbone / full fine-tune) | Geißler et al. (2024) |
| **D** | Baseline inference carbon measurement (normalised per 1,000 samples) | Luccioni et al. (2024) |
| **E** | Long-term deployment simulation (continuous 24/7, batch QC, on-demand) | Luccioni et al. (2024) |
| **F** | Carbon intensity sensitivity across 20 grid regions | Lacoste et al. (2019) |
| **G** | Deployment topology comparison across 7 PUE profiles | Patterson et al. (2022) |

## Supported Models (T1–T5)

| Tier | Models | Tier definition (parameters) | Manufacturing application |
|---|---|---|---|
| T1 | Linear Regression, Decision Tree | < 1K | Sensor calibration, threshold alerts |
| T2 | Random Forest, XGBoost, MobileNet-v2 | 1K–10M | Predictive maintenance, basic defect detection |
| T3 | ResNet-50, LSTM | 10M–50M | High-resolution defect detection, process optimisation |
| T4 | ResNet-152, ViT-B/16 | 50M–500M | Multi-sensor fusion, complex inspection |
| T5 | Stable Diffusion, GPT-4o-mini | > 500M | Generative design, document Q&A |

The ranges are the tier definitions used by `config.py`. The benchmark's LSTM
uses a deliberately compact 128-unit configuration, so its parameter count sits
below the T3 band even though its time-series workload is grouped there. The
measured energy per tier is reported in `results/analysis_summary.md`.

## Datasets

| Dataset | Source | Task | Auto-download |
|---|---|---|---|
| SECOM | [UCI ML Repository](https://archive.ics.uci.edu/dataset/179/secom) | Semiconductor quality classification | Direct download (`urllib`) |
| NASA CMAPSS | [NASA Prognostics Center](https://data.nasa.gov/) | Turbofan engine RUL regression | Mirrored GitHub download |
| NEU Steel Defects | [Kaggle](https://www.kaggle.com/datasets/kaustubhdhikshit/neu-surface-defect-database) / [NEU official](http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm) | Steel surface defect classification | `gdown` / `kagglehub` |
| CNC Tool Wear | [Kaggle](https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill) | CNC tool wear regression | `kagglehub` |

Each loader tries the real dataset first and falls back to synthetic data if the
download or parse fails, so the pipeline always runs:

- **SECOM** — direct UCI download via `urllib`.
- **NASA CMAPSS** — dual GitHub mirror.
- **NEU Steel Defects** — Google Drive via `gdown`, then `kagglehub`, then a
  manual-download prompt.
- **CNC Tool Wear** — `kagglehub`, then a direct URL, then a manual prompt.

For automatic downloads, install the optional helpers:

```bash
pip install gdown kagglehub
```

Datasets are cached under `carbon_benchmark/data/cache/` and are not committed.
Synthetic fallbacks make the benchmark reproducible offline, but accuracy
metrics obtained on them are not meaningful — see the limitations below.

## Calibration

CodeCarbon instruments CPU, GPU and RAM. It does not capture power-supply
losses, cooling fans and pumps, chipset, storage, networking, or peripherals
such as industrial cameras and PLC interfaces. Fischer (2025) reports that total
system energy is therefore **25–37% higher** than CodeCarbon's estimate.

MACE-F stores both the raw and the calibrated result
(`co2eq_g` and `co2eq_corrected_g`) and applies tier-specific factors from
`carbon_benchmark/config.py` (`T1` 1.58 down to `T5` 1.25). These defaults are
initial estimates; replace them with measurements from your own site.

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

The full procedure — meter setup, paired measurements, factor interpretation,
and how to update `CALIBRATION_FACTORS` — is in
**[docs/calibration.md](docs/calibration.md)**.

## Project Structure

```text
carbon_benchmark/                  # repository root
├── carbon_benchmark/              # importable package
│   ├── config.py                  # tiers, grids, PUE, calibration factors
│   ├── decision_matrix.py         # A–E carbon-efficiency decision framework
│   ├── data/
│   │   ├── loader.py              # SECOM, CMAPSS, NEU Steel, CNC Tool Wear
│   │   └── preprocessing.py       # tabular, image and time-series pipelines
│   ├── models/
│   │   ├── traditional_ml.py      # LinearRegression, DecisionTree, RandomForest, XGBoost
│   │   ├── deep_learning.py       # MobileNet-v2, ResNet-50/152, ViT-B/16, LSTM
│   │   └── generative.py          # Stable Diffusion, GPT-4o-mini
│   ├── measurement/
│   │   ├── tracker.py             # CodeCarbon wrapper + calibration
│   │   ├── calibrator.py          # external power-meter factor computation
│   │   └── reporter.py            # result aggregation and CSV export
│   ├── experiments/
│   │   ├── method_a_baseline_training.py
│   │   ├── method_d_baseline_inference.py
│   │   └── methods_bc_efg.py      # Methods B, C, E, F, G
│   └── utils/
│       ├── hardware.py            # GPU/CPU detection
│       └── logging_setup.py       # per-experiment logging
├── scripts/                       # 4 entry points: run_all_experiments, smoketest,
│                                  # generate_figures, generate_flowchart
├── tests/                         # 12 pytest modules
├── results/                       # 79 CSV files + analysis_summary.md + README.md
├── figures/                       # 9 generated PNG figures
├── docs/calibration.md            # external power-meter guide
├── assets/demo.gif                # terminal demo
├── .github/workflows/ci.yml       # lint, security scan, tests, coverage
├── pyproject.toml
└── LICENSE
```

[results/README.md](results/README.md) documents the provenance of every CSV,
including legacy file generations and the known `summary.csv` Method E caveat.
Use [results/analysis_summary.md](results/analysis_summary.md) for the narrative
analysis.

## Results

Measured on the reference platform (RTX 3070 Ti Laptop 8 GB, i7-12700H, UK grid,
CodeCarbon 3.2.7). The tier gradient is the headline result:

| Tier | Median training energy | Relative to T1 |
|---|---|---|
| T1 | 0.000028 kWh | 1× |
| T2 | 0.000054 kWh | 2× |
| T3 | 0.001316 kWh | 47× |
| T4 | 0.010976 kWh | 387× |
| T5 | 0.000489 kWh | 17× (local API overhead only) |

Four findings:

1. **Energy scales non-linearly with model tier.** The jump from T1 to T2 is
   only 2×, but T2 to T3 is about 23× and T3 to T4 about 8×. The T1–T5
   boundaries therefore correspond to real order-of-magnitude differences.
2. **The GPU dominates training energy (>90%).** In the sampled ViT-B/16 run,
   the split was 91.2% GPU, 6.5% RAM and 2.3% CPU. Hardware selection and model
   architecture are the most direct levers.
3. **Deployment location is a first-order lever (63×).** The same ResNet-50
   workload emits 0.06 g CO₂eq on the Swiss grid (16 gCO₂eq/kWh) versus 3.76 g
   on the South African grid (1009 gCO₂eq/kWh).
4. **T5 API emissions are a measured lower bound.** GPT-4o-mini's local energy
   (0.000489 kWh) excludes server-side inference, because CodeCarbon only sees
   the client machine. Literature estimates put the server-side share at
   100–1000× the local measurement. Treat T5 numbers as local overhead, not
   total footprint.

### Scope and limitations

- **Synthetic fallback data.** The current benchmark results include runs where
  a loader fell back to synthetic data because the upstream archive did not
  match the expected format (the CMAPSS loader, for example, can hit a
  column-count mismatch). Accuracy metrics on those runs are not meaningful
  (near random guessing for image models); the energy values characterise
  computational cost, not task quality.
- **Short training runs.** The benchmark uses 2-epoch training. Production
  training at 50–200 epochs scales the absolute emissions accordingly.
- **CPU power estimation.** Without Intel Power Gadget (Windows) or RAPL
  (Linux), CodeCarbon falls back to `cpu_load`, which underestimates CPU power.
- **Single hardware platform.** Results are specific to the reference laptop;
  edge devices and cloud accelerators will differ.
- **Calibration uncertainty.** The default factors correct a known 25–37%
  underestimation but are not a substitute for site measurements.

The complete analysis, including the carbon-intensity table and the
transfer-learning/hyperparameter results, is in
[results/analysis_summary.md](results/analysis_summary.md).

## References

1. Lacoste A, et al. Quantifying the carbon emissions of machine learning. 2019.
2. Schmidt V, et al. CodeCarbon: estimate and track carbon emissions. 2021.
3. Milutinović M, Lukić M. How green is your algorithm? SEMSIE 2025.
4. Fischer R. Ground-truthing AI energy consumption. arXiv:2509.22092, 2025.
5. Strubell E, et al. Energy and policy considerations for deep learning in NLP. ACL 2019.
6. Luccioni AS, et al. Power hungry processing. ACM FAccT 2024.
7. Geißler D, et al. The power of training. ARCS 2024.
8. Patterson D, et al. The carbon footprint of ML training will plateau, then shrink. IEEE Computer, 2022.

## License

MIT License. Copyright (c) 2026 Chunqi Sun. See [LICENSE](LICENSE).
