"""
Standardised logging configuration for all experiment modules.

Provides consistent timestamped logging to both stdout and rotating log files,
so every experiment run produces an auditable record of emissions measurements.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    experiment_id: str,
    log_dir: Optional[Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Create a logger that writes to both stdout and a per-experiment log file.

    Args:
        experiment_id: Unique experiment identifier (e.g., 'E01_baseline_linear_regression').
        log_dir: Directory for log files. Defaults to project results directory.
        level: Logging level as string ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logging("E01_linear_regression")
        >>> logger.info("Starting baseline training measurement...")
    """
    if log_dir is None:
        from carbon_benchmark.config import LOG_DIR
        log_dir = LOG_DIR

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_id}_{timestamp}.log"

    logger = logging.getLogger(experiment_id)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()  # Prevent duplicate handlers on re-invocation

    # File handler — detailed, timestamped
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler — concise
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "[%(levelname)-5s] %(name)s: %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging to {log_file}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a child logger for a specific module within an experiment.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Logger instance inheriting from the experiment logger.
    """
    return logging.getLogger(name)
