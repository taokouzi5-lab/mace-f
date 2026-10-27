# results/ — Experiment Output Inventory

All files below are retained. CSV files are raw CodeCarbon v3.2.7 outputs or
reporter-consolidated results from the 23-experiment benchmark (UK grid baseline).

## Method A — baseline training (raw CodeCarbon samples)

- `E_methodA_<model>_rep{1,2,3}_emissions.csv` — 9 models x 3 repeats.
  Column `emissions` is kg CO2eq per sampling interval; summing it per rep and
  multiplying by 1000 gives grams. These files are the canonical source for
  `decision_matrix.TIER_PROFILES` `training_co2_g`
  (median of per-model medians, 3 reps per model).

## Methods B, C, D — raw CodeCarbon samples

- `E_methodB_bs{8,16,32,64}_lr{1e-3,5e-4,1e-4,5e-5}_emissions.csv` (8 hyperparameter runs)
- `E_methodC_{from_scratch,frozen_backbone,full_finetune}_emissions.csv` (3 transfer-learning modes)
- `E_methodD_<model>_emissions.csv` (10 models, baseline inference)

## Consolidated results (reporter output)

- `<model>_training.csv` (9 models) and `gpt_4o_mini_inference.csv`,
  `stable_diffusion_inference.csv` — per-model training/inference totals.
- `method_b_hyperparameter_results.csv`, `method_c_transfer_learning_results.csv`,
  `method_e_deployment_simulation.csv`, `method_f_carbon_intensity_sensitivity.csv`,
  `method_g_deployment_topology.csv` — method-level result tables.
- `summary.csv` — merged multi-method summary (see S8 caveat below).
- `analysis_summary.md` — narrative analysis of the benchmark.

## Legacy Method A files: b_*.csv vs bench_*.csv

Two generations of naming for the same experiment method were found:

| Generation | Files | Models | Rows per file | Timestamps (2026-06-22) |
|------------|-------|--------|---------------|-------------------------|
| `bench_*` (older) | 4 | traditional ML only (decision_tree, linear_regression, random_forest, xgboost) | 1 | ~10:42–10:43 |
| `b_*` (newer) | 9 | same 4 + lstm, mobilenet_v2, resnet_50, resnet_152, vit_b_16 | 3 (traditional ML), 1 (DL) | ~10:44+ |

Determination performed by script comparison:

- Column structure is identical (same 38 CodeCarbon v3.2.7 columns).
- Values are NOT identical: for each of the 4 shared models, the single
  `bench_*` row does not numerically match any `b_*` row (12 numeric columns
  differ, including `emissions` and `duration`). They are independent
  measurement runs, not duplicates.
- Conclusion: `b_*` is a superset only in model coverage, not in values.
  All 13 files are therefore retained.

## Known issue S8 — summary.csv Method E block

The Method E rows of `summary.csv` (rows with the `scenario` column populated)
disagree numerically with `method_e_deployment_simulation.csv`
(e.g. linear_regression, continuous_24x7: 0.161 vs 0.495 kg annual CO2eq).
The authoritative file is `method_e_deployment_simulation.csv`; use it for any
Method E analysis instead of the `summary.csv` Method E block.
