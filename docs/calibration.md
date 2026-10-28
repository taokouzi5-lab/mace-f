# Calibration Guide

MACE-F uses CodeCarbon as its measurement instrument. CodeCarbon tracks CPU,
GPU and RAM power, but it cannot see the rest of the system: power-supply
losses, cooling fans and pumps, chipset, storage, networking, and peripherals
such as industrial cameras or PLC interfaces. Fischer (2025) found that total
system energy is therefore **25–37% higher** than CodeCarbon's estimate.

The framework handles this in two places:

- `carbon_benchmark/measurement/tracker.py` records both the raw value
  (`co2eq_g`) and the calibrated value (`co2eq_corrected_g`) for every run.
- `carbon_benchmark/measurement/calibrator.py` derives tier-specific correction
  factors from paired CodeCarbon / external-power-meter measurements.

The default factors in
[`carbon_benchmark/config.py`](../carbon_benchmark/config.py) are initial
estimates. This guide shows how to replace them with measurements from your own
site.

## The model

For each paired run:

```text
CF = external_meter_kWh / codecarbon_kWh
```

Factors are aggregated per tier as the arithmetic mean:

```text
CF_tier = mean(CF_i)  for all runs i in that tier
Corrected_CO2 = Raw_CO2 × CF_tier
```

The default values are:

| Tier | Factor | CodeCarbon captures | Uncaptured overhead |
|---|---|---|---|
| T1 | 1.58 | ~63% | ~37% |
| T2 | 1.43 | ~70% | ~30% |
| T3 | 1.39 | ~72% | ~28% |
| T4 | 1.34 | ~75% | ~25% |
| T5 | 1.25 | ~80% | ~20% |

The factor decreases from T1 to T5 because the overhead is largely fixed: PSU
losses, fans, chipset and peripherals are a large fraction of a tiny CPU
workload, but a small fraction when the GPU is fully loaded. A factor of 1.58
means the wall meter saw 58% more energy than CodeCarbon reported — a ~37%
underestimate of the total, at the upper end of Fischer's range. A factor of
1.25 means 25% more energy — a ~20% underestimate, slightly below that range
because a GPU-dominated workload dilutes the fixed overhead.

These defaults are useful for comparability, not for regulatory reporting.
Treat them as placeholders until you have measured your own hardware.

## What you need

- An external wall-plug power meter with kWh logging, or a metered PDU/PSU that
  reports cumulative energy for the machine under test.
- The same hardware and workload configuration used by the benchmark. Changing
  the GPU, PSU, cooling setup or peripherals invalidates the factor.
- At least one paired run per tier. Two or three runs per tier are recommended
  so that the reported standard deviation is meaningful.
- The matching CodeCarbon result. The consolidated reporter CSV exposes
  `energy_kwh`; raw CodeCarbon files expose `energy_consumed`.

## Procedure

1. **Connect the meter.** Put it between the wall socket and the complete
   system (not just the GPU). Warm up the machine, then record idle power for a
   few minutes so you can subtract baseline consumption if your meter includes
   idle time.
2. **Run a known experiment.** For example:

   ```bash
   python scripts/run_all_experiments.py --methods A --models resnet_50 \
       --country GBR --output-dir results/calibration
   ```

   The reporter writes `results/calibration/resnet_50_training.csv`; the
   `energy_kwh` column is the CodeCarbon estimate for that run. The raw
   `E_methodA_resnet_50_rep*_emissions.csv` files are written to the same
   directory. Using a separate output directory keeps the canonical benchmark
   results intact.
3. **Read the wall meter.** Record the energy consumed over the measurement
   window and subtract the idle baseline. The result is `external_meter_kwh`.
4. **Record the pair.** Use the same tier that MACE-F assigns to the model
   (`carbon_benchmark.config.MODEL_TIERS`).
5. **Aggregate and inspect.** Compute the factors, print the summary, and export
   the table.
6. **Update the configuration.** Copy the empirical means into
   `CALIBRATION_FACTORS` and restart Python.

## Example

```python
from carbon_benchmark.measurement.calibrator import Calibrator

cal = Calibrator()  # exports default to results/

cal.add("E01", "linear_regression", "T1",
        codecarbon_kwh=0.012, external_meter_kwh=0.019,
        notes="wall meter, 3 min idle baseline subtracted")
cal.add("E06", "resnet_50", "T3",
        codecarbon_kwh=0.423, external_meter_kwh=0.587,
        notes="wall meter, same PSU and cooling as benchmark")

factors = cal.compute_tier_factors()
cal.print_summary()

# Apply the T3 factor to a raw CodeCarbon estimate
corrected_g = cal.apply_calibration(0.86, "T3")
print(f"Corrected: {corrected_g:.2f} g CO2eq")

# Persist the factor table (tier, mean_factor, std_dev, n_records, min, max, source)
print(cal.export_report())
```

Typical output:

```text
=======================================================
CALIBRATION FACTOR SUMMARY (Fischer 2025 methodology)
=======================================================
Tier   Factor     StdDev     N      Source
-------------------------------------------------------
T1     1.583      nan        1      empirical
T2     1.430      0.000      0      config_default
T3     1.388      nan        1      empirical
T4     1.340      0.000      0      config_default
T5     1.250      0.000      0      config_default
=======================================================
CF = External_Meter_kWh / CodeCarbon_kWh
Corrected_CO2 = Raw_CO2 × CF
```

With a single record for a tier, `std_dev` is `nan` (pandas uses the sample
standard deviation with `n - 1`). Collect at least two paired runs per tier if
you need a spread. Tiers with no records keep the configuration defaults and
are marked `config_default`; tiers with records are marked `empirical`.

## API reference

| Member | Behaviour |
|---|---|
| `Calibrator(output_dir=None)` | Stores paired records; exports go to `results/` unless `output_dir` is supplied. |
| `add(experiment_id, model_name, tier, codecarbon_kwh, external_meter_kwh, notes="")` | Adds one paired measurement. |
| `add_from_result(...)` | Alias for `add()` with the same signature. |
| `compute_tier_factors()` | Returns `{tier: {mean_factor, std_dev, n_records, min_factor, max_factor, source}}` for T1–T5. |
| `apply_calibration(co2eq_g, tier)` | Returns `co2eq_g × mean_factor` for the tier. |
| `export_report(filename="calibration_factors.csv")` | Writes the factor table to `output_dir` and returns the path. |
| `print_summary()` | Prints the formatted factor table and the correction formula. |

## Updating the default factors

`Calibrator` computes factors but does not mutate the configuration. To make
experiments use your measurements, edit
`carbon_benchmark/config.py`:

```python
CALIBRATION_FACTORS: Final[dict[str, float]] = {
    "T1": 1.58,  # replace with your empirical mean
    "T2": 1.43,
    "T3": 1.39,
    "T4": 1.34,
    "T5": 1.25,
}
```

Restart the Python process or notebook kernel after editing: `tracker.py` and
`methods_bc_efg.py` read the dictionary at import time. The tracker then writes
the active factor into the `calibration_factor` column and the corrected value
into `co2eq_corrected_g` for every subsequent measurement.

## Practical notes

- **Do not transfer factors across machines.** A factor measured on a desktop
  with a 750 W PSU will not hold on a laptop or an edge device.
- **Keep raw and corrected values.** Both are stored in the result CSVs; report
  the raw measurement alongside the calibrated estimate so the correction is
  auditable.
- **Method E/F/G reuse the same factors.** They are pure calculations over
  measured energy, so they inherit whatever is in `CALIBRATION_FACTORS`.
- **Verify the implementation.** The calibration tests can be run with:

  ```bash
  pytest tests/test_calibrator.py -q
  ```
