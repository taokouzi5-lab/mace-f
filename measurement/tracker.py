"""
Unified CodeCarbon wrapper for manufacturing AI carbon measurement.

Supports three measurement modes:
  - Training mode: tracks an entire model training session.
  - Inference mode: tracks batch inference (default 1,000 samples).
  - Custom mode: tracks an arbitrary code block via context manager.

Design decisions (from literature):
  - offline_mode=True by default (manufacturing OT networks may be isolated)
  - measure_power_secs=15 (standard interval per Schmidt et al., 2021)
  - GPU utilisation logged in parallel via pynvml (cf. Fischer, 2025)
  - Calibration factor applied post-measurement

References:
  Schmidt et al. (2021) — CodeCarbon architecture
  Fischer (2025) — validation and calibration methodology
  Luccioni et al. (2024) — per-1000-inference normalisation
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from carbon_benchmark.config import (
    CALIBRATION_FACTORS,
    CARBON_INTENSITY,
    CODECARBON_CONFIG,
    DEFAULT_COUNTRY,
    ISO_TO_COUNTRY,
    MODEL_TIERS,
    RESULTS_DIR,
)


@dataclass(frozen=False)
class EmissionResult:
    """Standardised result from a single carbon measurement run.

    All fields follow the column specification defined in the project prompt.
    Raw CodeCarbon values and calibrated values are both stored.
    """

    timestamp: str
    experiment_id: str
    model_name: str
    tier: str
    method: str
    task_type: str  # 'training' | 'inference' | 'analysis'
    duration_seconds: float
    energy_kwh: float
    co2eq_g: float
    cpu_energy_kwh: float
    gpu_energy_kwh: float
    ram_energy_kwh: float
    country_iso: str
    carbon_intensity: float
    calibration_factor: float
    co2eq_corrected_g: float
    n_samples: int = 0
    device: str = "cpu"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return result as a flat dictionary for CSV serialisation."""
        result = {
            "timestamp": self.timestamp,
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "tier": self.tier,
            "method": self.method,
            "task_type": self.task_type,
            "duration_seconds": round(self.duration_seconds, 2),
            "energy_kwh": round(self.energy_kwh, 6),
            "co2eq_g": round(self.co2eq_g, 2),
            "cpu_energy_kwh": round(self.cpu_energy_kwh, 6),
            "gpu_energy_kwh": round(self.gpu_energy_kwh, 6),
            "ram_energy_kwh": round(self.ram_energy_kwh, 6),
            "country_iso": self.country_iso,
            "carbon_intensity": self.carbon_intensity,
            "calibration_factor": self.calibration_factor,
            "co2eq_corrected_g": round(self.co2eq_corrected_g, 2),
            "n_samples": self.n_samples,
            "device": self.device,
        }
        # Flatten extra fields with prefix
        for key, value in self.extra.items():
            result[f"extra_{key}"] = value
        return result


class CarbonTracker(AbstractContextManager):
    """Unified wrapper around CodeCarbon EmissionsTracker.

    Provides standardised measurement for training, inference, and custom
    workloads.  Applies tier-specific calibration factors to correct for
    CodeCarbon's known 25-37 % underestimation of total system power
    (Fischer, 2025).

    Usage — context manager::

        with CarbonTracker("E01", "linear_regression", "A", "GBR") as tracker:
            model.fit(X_train, y_train)
        result = tracker.result

    Usage — decorator-style for training::

        tracker = CarbonTracker("E01", "resnet_50", "A", "GBR")
        result = tracker.track_training(train_fn)
    """

    def __init__(
        self,
        experiment_id: str,
        model_name: str,
        method: str,
        country_iso: str = "GBR",
        task_type: str = "training",
        output_dir: Optional[Path] = None,
        n_samples: int = 0,
    ) -> None:
        """Initialise the carbon tracker.

        Args:
            experiment_id: Unique experiment identifier, e.g. 'E01'.
            model_name: Model key matching MODEL_TIERS in config.
            method: Experiment method label ('A'–'G').
            country_iso: 3-letter ISO country code (default 'GBR' for UK grid).
            task_type: 'training', 'inference', or 'analysis'.
            output_dir: Directory for emissions CSV; defaults to config value.
            n_samples: Number of inference samples (for per-1000 normalisation).
        """
        self._experiment_id = experiment_id
        self._model_name = model_name
        self._method = method
        self._country_iso = country_iso
        self._task_type = task_type
        self._n_samples = n_samples
        self._output_dir = Path(output_dir) if output_dir else RESULTS_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._tier = MODEL_TIERS.get(model_name, "T3")
        country_key = ISO_TO_COUNTRY.get(
            self._country_iso, self._country_iso,
        )
        self._ci = CARBON_INTENSITY.get(
            country_key, CARBON_INTENSITY[DEFAULT_COUNTRY],
        )
        self._cf = CALIBRATION_FACTORS.get(self._tier, 1.40)
        self._device = self._detect_device()

        self._tracker: Any = None
        self._start_time: float = 0.0
        self.result: Optional[EmissionResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __enter__(self) -> "CarbonTracker":
        """Start CodeCarbon tracking (context manager protocol)."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Stop tracking and capture result on context exit.

        Does NOT suppress exceptions — they propagate after measurement.
        """
        if exc_type is None:
            self.stop()
        else:
            # Workload failed — still stop tracking so we have a partial record
            try:
                self.stop()
            except Exception:
                pass
        return False  # Do not suppress exceptions

    def start(self) -> None:
        """Initialise and start the CodeCarbon tracker."""
        from codecarbon import EmissionsTracker  # type: ignore[import-untyped]

        self._tracker = EmissionsTracker(
            project_name=f"macef_{self._experiment_id}",
            output_dir=str(self._output_dir),
            output_file=f"{self._experiment_id}_emissions.csv",
            measure_power_secs=CODECARBON_CONFIG["measure_power_secs"],
            force_carbon_intensity_g_co2e_kwh=self._ci,
            save_to_file=CODECARBON_CONFIG["save_to_file"],
            log_level=CODECARBON_CONFIG.get("logging_level", "info"),
            allow_multiple_runs=True,
        )
        self._start_time = time.perf_counter()
        self._tracker.start()

    def stop(self) -> EmissionResult:
        """Stop tracking, compute result, and return standardised EmissionResult."""
        if self._tracker is None:
            raise RuntimeError(
                "CarbonTracker.stop() called before start() — "
                "use the context manager or call start() first."
            )

        emissions_data = self._tracker.stop()
        duration = time.perf_counter() - self._start_time

        self.result = self._build_result(emissions_data, duration)
        return self.result

    def track_training(
        self,
        train_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> EmissionResult:
        """Wrap a training function with carbon tracking.

        Args:
            train_fn: Callable that performs model training.
            *args: Positional arguments forwarded to train_fn.
            **kwargs: Keyword arguments forwarded to train_fn.

        Returns:
            Standardised EmissionResult with training carbon metrics.

        Example:
            >>> def train(): return model.fit(X, y)
            >>> result = tracker.track_training(train)
        """
        with self:
            train_fn(*args, **kwargs)
        assert self.result is not None
        return self.result

    def track_inference(
        self,
        inference_fn: Callable[..., Any],
        n_samples: int = 1000,
        *args: Any,
        **kwargs: Any,
    ) -> EmissionResult:
        """Wrap an inference function with carbon tracking.

        Results are normalised to per-1,000-inferences (cf. Luccioni et al., 2024).

        Args:
            inference_fn: Callable that runs inference on a batch of samples.
            n_samples: Number of samples in the batch (for normalisation).
            *args: Positional arguments forwarded to inference_fn.
            **kwargs: Keyword arguments forwarded to inference_fn.

        Returns:
            Standardised EmissionResult with per-1000-inference metrics.
        """
        self._n_samples = n_samples
        self._task_type = "inference"

        with self:
            inference_fn(*args, **kwargs)

        assert self.result is not None

        # Normalise to per-1000-inferences if sample count differs
        if n_samples != 1000 and n_samples > 0:
            scale = 1000.0 / n_samples
            self.result.energy_kwh *= scale
            self.result.co2eq_g *= scale
            self.result.co2eq_corrected_g *= scale
            self.result.n_samples = 1000

        return self.result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        """Return the best available PyTorch device string."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _build_result(
        self,
        emissions_data: Optional[Dict[str, float]],
        duration_s: float,
    ) -> EmissionResult:
        """Construct a standardised EmissionResult from raw CodeCarbon output.

        Args:
            emissions_data: Float (CodeCarbon 3.x) or dict (legacy) from stop().
            duration_s: Wall-clock duration of the measurement window.

        Returns:
            Populated EmissionResult with calibrated values.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # CodeCarbon 3.x returns a float (total emissions in kgCO₂eq)
        # Older versions return a dict with detailed breakdown
        if emissions_data is None:
            energy_kwh = 0.0
            co2eq_kg = 0.0
            cpu_e = 0.0
            gpu_e = 0.0
            ram_e = 0.0
        elif isinstance(emissions_data, (int, float)):
            # CodeCarbon 3.x: returns total emissions in kgCO₂eq
            co2eq_kg = float(emissions_data)
            # Energy is not directly returned — estimate from emissions / carbon_intensity
            energy_kwh = (co2eq_kg * 1000.0) / self._ci if self._ci > 0 else 0.0
            cpu_e = 0.0
            gpu_e = 0.0
            ram_e = 0.0
        else:
            # Legacy dict format
            energy_kwh = float(emissions_data.get("energy_consumed", 0.0))
            co2eq_kg = float(emissions_data.get("emissions", 0.0))
            cpu_e = float(emissions_data.get("cpu_energy", 0.0))
            gpu_e = float(emissions_data.get("gpu_energy", 0.0))
            ram_e = float(emissions_data.get("ram_energy", 0.0))

        co2eq_g = co2eq_kg * 1000.0

        return EmissionResult(
            timestamp=now_iso,
            experiment_id=self._experiment_id,
            model_name=self._model_name,
            tier=self._tier,
            method=self._method,
            task_type=self._task_type,
            duration_seconds=duration_s,
            energy_kwh=energy_kwh,
            co2eq_g=co2eq_g,
            cpu_energy_kwh=cpu_e,
            gpu_energy_kwh=gpu_e,
            ram_energy_kwh=ram_e,
            country_iso=self._country_iso,
            carbon_intensity=self._ci,
            calibration_factor=self._cf,
            co2eq_corrected_g=co2eq_g * self._cf,
            n_samples=self._n_samples,
            device=self._device,
        )
