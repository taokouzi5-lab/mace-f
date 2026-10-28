"""Unit tests for hardware detection utilities (pynvml/torch mocked)."""

import builtins
import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import carbon_benchmark.utils.hardware as hw

FAKE_NVML = SimpleNamespace(
    nvmlInit=lambda: None,
    nvmlShutdown=lambda: None,
    nvmlDeviceGetCount=lambda: 1,
    nvmlDeviceGetHandleByIndex=lambda i: f"handle-{i}",
    nvmlDeviceGetName=lambda handle: b"Fake RTX 9999",
    nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=8 * 1024**3),
    nvmlSystemGetDriverVersion=lambda: b"555.99",
)


def _patch_torch_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make torch.cuda.is_available() return False and version None."""
    try:
        import torch
    except ImportError:
        return
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "version", SimpleNamespace(cuda=None))


class TestGetCpuInfo:
    def test_returns_expected_keys(self) -> None:
        info = hw.get_cpu_info()
        for key in ("processor", "cores_logical", "arch", "platform"):
            assert key in info
        assert info["cores_logical"].isdigit()


class TestGetGpuInfo:
    def test_detects_fake_gpu_with_byte_strings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", FAKE_NVML)
        _patch_torch_cuda_unavailable(monkeypatch)
        gpus = hw.get_gpu_info()
        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu["name"] == "Fake RTX 9999"
        assert gpu["driver_version"] == "555.99"
        assert gpu["vram_total_mb"] == str(8 * 1024)
        assert gpu["index"] == "0"
        assert gpu["cuda_available"] == "False"
        assert gpu["cuda_version"] == "N/A"
        assert gpu["compute_capability"] == "N/A"

    def test_handles_str_names_and_driver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        str_nvml = SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetCount=lambda: 1,
            nvmlDeviceGetHandleByIndex=lambda i: "h",
            nvmlDeviceGetName=lambda handle: "Fake GPU str",
            nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(total=2 * 1024**3),
            nvmlSystemGetDriverVersion=lambda: "123.45",
        )
        monkeypatch.setitem(sys.modules, "pynvml", str_nvml)
        _patch_torch_cuda_unavailable(monkeypatch)
        gpus = hw.get_gpu_info()
        assert gpus[0]["name"] == "Fake GPU str"
        assert gpus[0]["driver_version"] == "123.45"
        assert gpus[0]["vram_total_mb"] == str(2 * 1024)

    def test_no_gpus_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        no_gpu = SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetCount=lambda: 0,
        )
        monkeypatch.setitem(sys.modules, "pynvml", no_gpu)
        _patch_torch_cuda_unavailable(monkeypatch)
        assert hw.get_gpu_info() == []

    def test_pynvml_missing_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # None in sys.modules forces `import pynvml` to raise ImportError
        monkeypatch.setitem(sys.modules, "pynvml", None)
        _patch_torch_cuda_unavailable(monkeypatch)
        assert hw.get_gpu_info() == []

    def test_pynvml_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        broken = SimpleNamespace(
            nvmlInit=lambda: (_ for _ in ()).throw(RuntimeError("no driver")),
        )
        monkeypatch.setitem(sys.modules, "pynvml", broken)
        _patch_torch_cuda_unavailable(monkeypatch)
        assert hw.get_gpu_info() == []

    def test_cuda_info_augments_gpus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        monkeypatch.setitem(sys.modules, "pynvml", FAKE_NVML)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch, "version", SimpleNamespace(cuda="11.8"))
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (8, 6))
        gpus = hw.get_gpu_info()
        assert gpus[0]["cuda_available"] == "True"
        assert gpus[0]["cuda_version"] == "11.8"
        assert gpus[0]["compute_capability"] == "8.6"


class TestGetCudaInfo:
    def test_cuda_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch, "version", SimpleNamespace(cuda="11.8"))
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (8, 6))
        assert hw._get_cuda_info() == (True, "11.8", "8.6")

    def test_cuda_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert hw._get_cuda_info() == (False, "N/A", "N/A")

    def test_torch_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert hw._get_cuda_info() == (False, "N/A", "N/A")

    def test_cuda_check_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(RuntimeError()))
        assert hw._get_cuda_info() == (False, "N/A", "N/A")


class TestCheckCudaAvailable:
    def test_true_when_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert hw.check_cuda_available() is True

    def test_false_when_no_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert hw.check_cuda_available() is False

    def test_torch_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert hw.check_cuda_available() is False


class TestGetDevice:
    def test_device_follows_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hw, "check_cuda_available", lambda: True)
        assert hw.get_device() == "cuda"
        monkeypatch.setattr(hw, "check_cuda_available", lambda: False)
        assert hw.get_device() == "cpu"


class _RecordingLogger:
    """Minimal logger stand-in capturing info messages."""

    def __init__(self) -> None:
        self.messages: list = []

    def info(self, msg: str, *args: Any) -> None:
        self.messages.append(msg % args if args else msg)


class TestPrintHardwareSummary:
    def test_summary_without_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        logger = _RecordingLogger()
        monkeypatch.setattr(hw, "get_cpu_info", lambda: {
            "processor": "TestCPU", "cores_logical": "8", "platform": "TestOS",
        })
        monkeypatch.setattr(hw, "get_gpu_info", lambda: [])
        hw.print_hardware_summary(logger_instance=logger)  # type: ignore[arg-type]
        joined = "\n".join(logger.messages)
        assert "HARDWARE SUMMARY" in joined
        assert "CPU fallback" in joined

    def test_summary_with_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        logger = _RecordingLogger()
        monkeypatch.setattr(hw, "get_cpu_info", lambda: {
            "processor": "TestCPU", "cores_logical": "8", "platform": "TestOS",
        })
        monkeypatch.setattr(hw, "get_gpu_info", lambda: [{
            "index": "0", "name": "TestGPU", "vram_total_mb": "8192",
            "cuda_available": "True", "cuda_version": "11.8", "compute_capability": "8.6",
        }])
        hw.print_hardware_summary(logger_instance=logger)  # type: ignore[arg-type]
        joined = "\n".join(logger.messages)
        assert "GPU[0]: TestGPU" in joined
        assert "CUDA: True" in joined

    def test_module_logger_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Runs with the module-level logger (info-level messages accepted)."""
        monkeypatch.setattr(hw, "get_gpu_info", lambda: [])
        monkeypatch.setattr(hw, "get_cpu_info", lambda: {
            "processor": "TestCPU", "cores_logical": "8", "platform": "TestOS",
        })
        assert isinstance(hw.logger, logging.Logger)
        hw.print_hardware_summary()  # must not raise
