"""Experiment method scripts for the manufacturing AI carbon benchmark.

Each method is implemented as an independently runnable module.
All accept --output-dir, --country, and --model CLI arguments.
"""

from carbon_benchmark.experiments.method_a_baseline_training import run as run_method_a
from carbon_benchmark.experiments.method_d_baseline_inference import run as run_method_d
