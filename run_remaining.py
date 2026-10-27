"""Run remaining experiments that were skipped."""
import sys
sys.path.insert(0, "")

from carbon_benchmark.experiments.method_a_baseline_training import run as run_a
from carbon_benchmark.experiments.method_d_baseline_inference import run as run_d
from carbon_benchmark.utils.hardware import print_hardware_summary

print_hardware_summary()

# E09: ViT training
print("\n=== E09: ViT-B/16 Training ===")
try:
    r = run_a("vit_b_16", n_repeats=1)
    print(f"OK: energy={r[0]['energy_kwh']:.6f} kWh")
except Exception as e:
    print(f"FAILED: {e}")

# Inference for deep learning + generative models
models_infer = ["mobilenet_v2", "resnet_50", "lstm", "resnet_152", "vit_b_16"]
for model in models_infer:
    print(f"\n=== {model} Inference ===")
    try:
        r = run_d(model)
        print(f"OK: energy={r[0]['energy_kwh']:.6f} kWh")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

# GPT (already works)
print("\n=== GPT-4o-mini Inference ===")
try:
    r = run_d("gpt_4o_mini")
    print(f"OK: energy={r[0]['energy_kwh']:.6f} kWh")
except Exception as e:
    print(f"FAILED: {e}")

# Stable Diffusion (may fail due to diffusers compat)
print("\n=== Stable Diffusion Inference ===")
try:
    r = run_d("stable_diffusion")
    print(f"OK: energy={r[0]['energy_kwh']:.6f} kWh")
except Exception as e:
    print(f"SKIPPED: {e}")

print("\n=== DONE ===")
