"""
Hardware detection and monitoring utilities.

Detects available GPU(s), reports CUDA capability, and provides
system information for experiment metadata logging.
"""

import logging
import platform
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_cpu_info() -> Dict[str, str]:
    """
    Return CPU identification information.

    Returns:
        Dictionary with keys: processor, cores (logical), arch, platform.
    """
    import os

    return {
        "processor": platform.processor() or "Unknown",
        "cores_logical": str(os.cpu_count() or "Unknown"),
        "arch": platform.machine(),
        "platform": platform.platform(),
    }


def get_gpu_info() -> List[Dict[str, str]]:
    """
    Return list of detected NVIDIA GPU(s) with key specifications.

    Uses pynvml (NVIDIA Management Library) for accurate GPU data.
    Falls back gracefully if no NVIDIA GPU or pynvml is unavailable.

    Returns:
        List of dictionaries, each containing: name, vram_total_mb, driver_version,
        cuda_available, compute_capability. Empty list if no GPU detected.

    References:
        Uses same pynvml interface as CodeCarbon for consistency
        (Schmidt et al., 2021; Fischer, 2025).
    """
    gpus: List[Dict[str, str]] = []

    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver_ver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_ver, bytes):
                driver_ver = driver_ver.decode("utf-8")

            gpus.append({
                "index": str(i),
                "name": name,
                "vram_total_mb": str(memory_info.total // (1024 * 1024)),
                "driver_version": driver_ver,
            })

        pynvml.nvmlShutdown()

    except ImportError:
        logger.debug("pynvml not installed — GPU info unavailable")
    except Exception as exc:
        logger.debug(f"pynvml error (may be no NVIDIA GPU): {exc}")

    # Augment with CUDA info if PyTorch is available
    cuda_available, cuda_version, compute_cap = _get_cuda_info()
    for gpu in gpus:
        gpu["cuda_available"] = str(cuda_available)
        gpu["cuda_version"] = cuda_version
        gpu["compute_capability"] = compute_cap

    return gpus


def _get_cuda_info() -> Tuple[bool, str, str]:
    """Check CUDA availability via PyTorch. Returns (available, version, compute_cap)."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda or "unknown"
            cap_major = torch.cuda.get_device_capability(0)
            cap = f"{cap_major[0]}.{cap_major[1]}"
            return True, cuda_ver, cap
    except ImportError:
        pass
    except Exception:
        pass
    return False, "N/A", "N/A"


def check_cuda_available() -> bool:
    """
    Check whether CUDA-capable GPU is available for training.

    Returns:
        True if CUDA is available and at least one GPU is detected.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """
    Return the best available PyTorch device string.

    Returns:
        'cuda' if CUDA GPU available, otherwise 'cpu'.
    """
    return "cuda" if check_cuda_available() else "cpu"


def print_hardware_summary(logger_instance: Optional[logging.Logger] = None) -> None:
    """
    Print a human-readable summary of detected hardware to the logger.

    Args:
        logger_instance: Logger to use. Defaults to module-level logger.
    """
    log = logger_instance or logger

    cpu = get_cpu_info()
    gpus = get_gpu_info()

    log.info("=" * 60)
    log.info("HARDWARE SUMMARY")
    log.info(f"  CPU: {cpu['processor']} ({cpu['cores_logical']} logical cores)")
    log.info(f"  Platform: {cpu['platform']}")

    if gpus:
        for gpu in gpus:
            log.info(f"  GPU[{gpu['index']}]: {gpu['name']} "
                     f"({gpu['vram_total_mb']} MB VRAM)")
            log.info(f"    CUDA: {gpu['cuda_available']} "
                     f"(v{gpu['cuda_version']}, compute {gpu['compute_capability']})")
    else:
        log.info("  GPU: None detected — will use CPU fallback")

    log.info("=" * 60)
