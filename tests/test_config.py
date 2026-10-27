"""Tests for central configuration values."""

from carbon_benchmark.config import HARDWARE


def test_hardware_matches_measured() -> None:
    """HARDWARE reflects the values measured on the benchmark machine.

    Source: nvidia-smi (GPU model / VRAM), WMIC (CPU model / RAM), corroborated
    by the cpu_model / gpu_model / ram_total_size columns in the CodeCarbon
    run logs under results/.
    """
    assert "3070 Ti" in HARDWARE["gpu"]
    assert HARDWARE["vram_gb"] == "8"
    assert HARDWARE["ram_gb"] == "16"
    assert "12700H" in HARDWARE["cpu"]
