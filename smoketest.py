"""Quick smoketest: verify the full pipeline works end-to-end."""
from carbon_benchmark.experiments.method_a_baseline_training import run
from carbon_benchmark.utils.hardware import print_hardware_summary

print_hardware_summary()
print("\nRunning smoketest: Method A — linear_regression (1 repeat)...\n")

results = run("linear_regression", country_iso="GBR", n_repeats=1)
r = results[0]
print(f"\nDone.")
print(f"  Energy:    {r['energy_kwh']:.6f} kWh")
print(f"  CO2eq:     {r['co2eq_g']:.2f} g")
print(f"  Corrected: {r['co2eq_corrected_g']:.2f} g")
print(f"  Duration:  {r['duration_seconds']:.1f} s")
print(f"  Tier:      {r['tier']}")
print(f"  Device:    {r['device']}")
