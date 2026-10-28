"""Unit tests for the standardised logging configuration."""

import logging
from pathlib import Path

import pytest

import carbon_benchmark.utils.logging_setup as ls


class TestSetupLogging:
    def test_creates_log_file_and_handlers(self, tmp_path: Path) -> None:
        logger = ls.setup_logging("E_TEST_LOGGING", log_dir=tmp_path)
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        # File handler (DEBUG) + console handler (INFO)
        assert len(logger.handlers) == 2
        log_files = list(tmp_path.glob("E_TEST_LOGGING_*.log"))
        assert len(log_files) == 1
        logger.info("hello from test")
        assert "hello from test" in log_files[0].read_text(encoding="utf-8")

    def test_level_string_applied(self, tmp_path: Path) -> None:
        logger = ls.setup_logging("E_TEST_LEVEL", log_dir=tmp_path, level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_lowercase_level_accepted(self, tmp_path: Path) -> None:
        logger = ls.setup_logging("E_TEST_LOWER", log_dir=tmp_path, level="warning")
        assert logger.level == logging.WARNING

    def test_invalid_level_falls_back_to_info(self, tmp_path: Path) -> None:
        logger = ls.setup_logging("E_TEST_BOGUS", log_dir=tmp_path, level="NOT_A_LEVEL")
        assert logger.level == logging.INFO

    def test_default_log_dir_branch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """log_dir=None uses config.LOG_DIR (monkeypatched to tmp)."""
        monkeypatch.setattr("carbon_benchmark.config.LOG_DIR", tmp_path)
        logger = ls.setup_logging("E_TEST_DEFAULT_DIR")
        assert logger is not None
        assert len(list(tmp_path.glob("E_TEST_DEFAULT_DIR_*.log"))) == 1

    def test_reinvocation_clears_duplicate_handlers(self, tmp_path: Path) -> None:
        ls.setup_logging("E_TEST_DUP", log_dir=tmp_path)
        logger = ls.setup_logging("E_TEST_DUP", log_dir=tmp_path)
        assert len(logger.handlers) == 2

    def test_file_handler_writes_debug(self, tmp_path: Path) -> None:
        logger = ls.setup_logging("E_TEST_DEBUG_WRITE", log_dir=tmp_path, level="DEBUG")
        logger.debug("debug message written")
        log_file = list(tmp_path.glob("E_TEST_DEBUG_WRITE_*.log"))[0]
        assert "debug message written" in log_file.read_text(encoding="utf-8")


class TestGetLogger:
    def test_returns_logger_for_name(self) -> None:
        logger = ls.get_logger("carbon_benchmark.tests.fake")
        assert logger.name == "carbon_benchmark.tests.fake"

    def test_same_name_same_logger(self) -> None:
        assert ls.get_logger("x") is ls.get_logger("x")
