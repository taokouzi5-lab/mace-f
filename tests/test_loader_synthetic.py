"""Smoke tests for dataset loaders — synthetic fallback paths fully covered.

Every test runs against an isolated, empty cache directory under tmp_path.
Download functions are monkeypatched so NO real network access is ever
attempted; real-data code paths are exercised with small locally generated
files instead of the genuine datasets under data/cache.
"""

import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from carbon_benchmark.data import loader
from carbon_benchmark.data.loader import (
    DatasetTuple,
    _download_cmapss,
    _download_cnc_tool_wear,
    _download_neu_steel,
    _download_secom,
    _find_neu_steel_dir,
    _load_image_folder,
    _synthetic_images,
    _synthetic_tabular,
    _train_test_split,
    get_dataset_info,
    load_cnc_tool_wear,
    load_nasa_cmapss,
    load_neu_steel_defects,
    load_secom,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point loader.DATA_DIR at an empty tmp cache so real datasets are never read."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(loader, "DATA_DIR", cache)
    return cache


@pytest.fixture(autouse=True)
def seeded_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the module-level RNG so synthetic data is reproducible per test."""
    monkeypatch.setattr(loader, "RNG", np.random.default_rng(42))


def _block_network(monkeypatch: pytest.MonkeyPatch, fn, exc: Exception) -> None:
    """Replace a loader download helper with one that raises immediately."""
    monkeypatch.setattr(loader, fn.__name__, lambda *a, **k: (_ for _ in ()).throw(exc))


def _assert_split_shapes(data: DatasetTuple, n_samples: int, n_features: int) -> None:
    X_train, X_test, y_train, y_test = data
    assert X_train.shape == (int(n_samples * 0.8), n_features)
    assert X_test.shape == (n_samples - int(n_samples * 0.8), n_features)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


# ---------------------------------------------------------------------------
# get_dataset_info
# ---------------------------------------------------------------------------

class TestGetDatasetInfo:
    def test_known_dataset_returns_copy(self) -> None:
        info = get_dataset_info("secom")
        assert info["name"] == "SECOM Semiconductor Manufacturing"
        info["name"] = "mutated"
        assert get_dataset_info("secom")["name"] == "SECOM Semiconductor Manufacturing"

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown dataset 'nope'"):
            get_dataset_info("nope")


# ---------------------------------------------------------------------------
# load_secom
# ---------------------------------------------------------------------------

class TestLoadSecom:
    def _write_real_secom(self, cache: Path) -> None:
        cache.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(7)
        X = rng.normal(size=(50, 8))
        X[0, 0] = np.nan
        X[3, 5] = np.nan
        np.savetxt(cache / "secom.data", X, fmt="%.4f")
        y = rng.choice([-1, 1], size=50)
        np.savetxt(cache / "secom_labels.data", y, fmt="%d")

    def test_real_file_path(self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._write_real_secom(isolated_cache / "secom")
        _block_network(monkeypatch, _download_secom, AssertionError("must not download"))
        X_train, X_test, y_train, y_test = load_secom()
        assert X_train.shape == (40, 8)
        assert X_test.shape == (10, 8)
        assert X_train.dtype == np.float32
        assert not np.isnan(X_train).any()
        assert np.all(np.isin(np.concatenate([y_train, y_test]), [0, 1]))

    def test_download_failure_falls_back_to_synthetic(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _block_network(monkeypatch, _download_secom, ConnectionError("offline"))
        data = load_secom()
        _assert_split_shapes(data, 1500, 100)
        assert np.all(np.isin(np.concatenate([data[2], data[3]]), [0, 1]))
        assert "using synthetic data" in caplog.text


# ---------------------------------------------------------------------------
# load_nasa_cmapss
# ---------------------------------------------------------------------------

class TestLoadCmapss:
    def _write_real_cmapss(self, cache: Path) -> None:
        cache.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(3)
        rows = []
        for unit in range(1, 4):
            for cycle in range(1, 41):
                sensors = rng.normal(size=23)
                ops = [10.0, 0.5, 100.0]
                rows.append([unit, cycle, *sensors, *ops])
        np.savetxt(cache / "train_FD001.txt", np.array(rows), fmt="%.4f")

    def test_real_file_path(self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._write_real_cmapss(isolated_cache / "nasa_cmapss")
        _block_network(monkeypatch, _download_cmapss, AssertionError("must not download"))
        X_train, X_test, y_train, y_test = load_nasa_cmapss()
        assert X_train.shape == (96, 26)
        assert X_test.shape == (24, 26)
        # RUL = max_cycle - cycle; three units × cycle 1 → three rows with RUL 39
        all_y = np.concatenate([y_train, y_test])
        assert all_y.min() == pytest.approx(0.0)
        assert all_y.max() == pytest.approx(39.0)
        assert np.sum(all_y == 39.0) == 3
        assert y_train.dtype == np.float32

    def test_download_failure_falls_back_to_synthetic(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _block_network(monkeypatch, _download_cmapss, ConnectionError("offline"))
        data = load_nasa_cmapss()
        _assert_split_shapes(data, 5000, 26)
        assert data[2].dtype == np.float32  # regression targets
        assert "using synthetic data" in caplog.text


# ---------------------------------------------------------------------------
# load_neu_steel_defects
# ---------------------------------------------------------------------------

def _write_neu_images(cache: Path, n_per_class: int = 2) -> None:
    """Create a NEU-DET-style image tree with tiny real JPEG files."""
    pytest.importorskip("PIL")
    from PIL import Image

    images_dir = cache / "neu_steel" / "NEU-DET" / "images"
    for cls in ("class_a", "class_b"):
        (images_dir / cls).mkdir(parents=True)
        for i in range(n_per_class):
            Image.new("RGB", (12, 12), (i * 40, 0, 255 - i * 40)).save(
                images_dir / cls / f"{i}.jpg"
            )


class TestLoadNeuSteel:
    def test_real_images_present(self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # 10 images (5 per class) so the stratified split yields ≥2 test samples
        _write_neu_images(isolated_cache, n_per_class=5)
        monkeypatch.setattr(loader, "_download_neu_steel", lambda *a: pytest.fail("no download"))
        X_train, X_test, y_train, y_test = load_neu_steel_defects()
        total = len(X_train) + len(X_test)
        assert total == 10
        assert X_train.shape[1:] == (224, 224, 3)
        assert X_train.dtype == np.float32
        assert X_train.min() >= 0.0 and X_train.max() <= 1.0
        assert np.all(np.isin(np.concatenate([y_train, y_test]), [0, 1]))

    def test_download_fails_falls_back_to_synthetic(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(loader, "_download_neu_steel", lambda *a: False)
        calls: list = []

        def fake_synthetic(*args: Any, **kwargs: Any) -> DatasetTuple:
            calls.append((args, kwargs))
            X = np.zeros((8, 4, 4, 3), dtype=np.float32)
            y = np.zeros(8, dtype=np.int64)
            return X[:6], X[6:], y[:6], y[6:]

        monkeypatch.setattr(loader, "_synthetic_images", fake_synthetic)
        data = load_neu_steel_defects()
        assert calls == [((), {"n_samples": 500, "n_classes": 6, "size": (224, 224)})]
        assert len(data[0]) == 6
        assert "not found and auto-download failed" in caplog.text

    def test_download_success_then_load(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_download(cache_dir: Path) -> bool:
            _write_neu_images(isolated_cache, n_per_class=5)
            return True

        monkeypatch.setattr(loader, "_download_neu_steel", fake_download)
        X_train, X_test, y_train, y_test = load_neu_steel_defects()
        assert len(X_train) + len(X_test) == 10
        assert X_train.shape[1:] == (224, 224, 3)

    def test_first_load_fails_then_fallback(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_neu_images(isolated_cache)
        monkeypatch.setattr(loader, "_load_image_folder", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad folder")))
        monkeypatch.setattr(loader, "_download_neu_steel", lambda *a: False)
        monkeypatch.setattr(loader, "_synthetic_images",
                            lambda *a, **k: (np.zeros((1, 2, 2, 3), np.float32),) * 2 + (np.zeros(1, np.int64),) * 2)
        data = load_neu_steel_defects()
        assert len(data[0]) == 1
        assert "NEU Steel load failed" in caplog.text

    def test_load_fails_after_download_then_final_fallback(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(loader, "_download_neu_steel",
                            lambda cache_dir: (_write_neu_images(isolated_cache), True)[1])
        monkeypatch.setattr(loader, "_load_image_folder", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad folder")))
        monkeypatch.setattr(loader, "_synthetic_images",
                            lambda *a, **k: (np.zeros((1, 2, 2, 3), np.float32),) * 2 + (np.zeros(1, np.int64),) * 2)
        data = load_neu_steel_defects()
        assert len(data[0]) == 1
        assert "not loadable after download" in caplog.text


# ---------------------------------------------------------------------------
# load_cnc_tool_wear
# ---------------------------------------------------------------------------

def _write_cnc_experiment(cache: Path, name: str = "experiment_01.csv", n: int = 200) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    df = pd.DataFrame({
        "S1_OutputPower": rng.normal(500, 50, n),
        "SensorA": rng.normal(0, 1, n),
        "SensorB": rng.normal(1, 0.2, n),
        "Machining_Process": ["OP-A"] * n,
    })
    df.to_csv(cache / name, index=False)


class TestLoadCncToolWear:
    def test_real_sensor_csv_with_target_column(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_cnc_experiment(isolated_cache / "cnc_tool_wear")
        monkeypatch.setattr(loader, "_download_cnc_tool_wear", lambda *a: pytest.fail("no download"))
        X_train, X_test, y_train, y_test = load_cnc_tool_wear()
        assert X_train.shape == (160, 2)  # SensorA/B only — Machining_Process dropped
        assert X_test.shape == (40, 2)
        assert y_train.dtype == np.float32

    def test_target_falls_back_to_last_numeric(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = isolated_cache / "cnc_tool_wear"
        cache.mkdir(parents=True)
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "ColA": rng.normal(size=150),
            "ColB": rng.normal(size=150),
            "ColC": rng.normal(size=150),
        })
        df.to_csv(cache / "experiment_01.csv", index=False)
        monkeypatch.setattr(loader, "_download_cnc_tool_wear", lambda *a: pytest.fail("no download"))
        X_train, _, y_train, _ = load_cnc_tool_wear()
        assert X_train.shape == (120, 2)
        assert y_train.shape == (120,)

    def test_download_fails_falls_back_to_synthetic(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(loader, "_download_cnc_tool_wear", lambda *a: False)
        data = load_cnc_tool_wear()
        _assert_split_shapes(data, 2000, 10)
        assert "not found and auto-download failed" in caplog.text

    def test_metadata_only_csv_falls_back_to_synthetic(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(loader, "_download_cnc_tool_wear", lambda *a: pytest.fail("no download"))
        cache = isolated_cache / "cnc_tool_wear"
        cache.mkdir(parents=True)
        df = pd.DataFrame({"Machining_Process": ["A", "B", "C"], "Label": ["x", "y", "z"]})
        df.to_csv(cache / "train.csv", index=False)
        data = load_cnc_tool_wear()
        _assert_split_shapes(data, 2000, 10)
        assert "no valid sensor CSV found" in caplog.text

    def test_too_few_rows_falls_back_to_synthetic(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(loader, "_download_cnc_tool_wear", lambda *a: pytest.fail("no download"))
        _write_cnc_experiment(isolated_cache / "cnc_tool_wear", n=20)
        data = load_cnc_tool_wear()
        _assert_split_shapes(data, 2000, 10)
        assert "no valid sensor CSV found" in caplog.text

    def test_bad_csv_skipped_then_valid_loaded(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = isolated_cache / "cnc_tool_wear"
        cache.mkdir(parents=True)
        (cache / "experiment_bad.csv").write_text("", encoding="utf-8")
        _write_cnc_experiment(cache, name="experiment_good.csv", n=120)
        monkeypatch.setattr(loader, "_download_cnc_tool_wear", lambda *a: pytest.fail("no download"))
        X_train, _, _, _ = load_cnc_tool_wear()
        assert X_train.shape == (96, 2)

    def test_download_success_then_load(
        self, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_download(cache_dir: Path) -> bool:
            _write_cnc_experiment(cache_dir, n=120)
            return True

        monkeypatch.setattr(loader, "_download_cnc_tool_wear", fake_download)
        X_train, _, _, _ = load_cnc_tool_wear()
        assert X_train.shape == (96, 2)


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------

class TestSyntheticGenerators:
    def test_synthetic_tabular_classification(self) -> None:
        data = _synthetic_tabular(n_samples=100, n_features=5, classification=True)
        _assert_split_shapes(data, 100, 5)
        assert data[2].dtype == np.int64
        assert np.all(np.isin(np.concatenate([data[2], data[3]]), [0, 1]))

    def test_synthetic_tabular_regression(self) -> None:
        data = _synthetic_tabular(n_samples=100, n_features=5, classification=False)
        _assert_split_shapes(data, 100, 5)
        assert data[2].dtype == np.float32
        # Regression target is derived from the first feature — correlated
        corr = np.corrcoef(data[0][:, 0], data[2])[0, 1]
        assert corr > 0.5

    def test_synthetic_images(self) -> None:
        data = _synthetic_images(n_samples=16, n_classes=4, size=(8, 8))
        X_train, X_test, y_train, y_test = data
        assert X_train.shape == (12, 8, 8, 3)
        assert X_test.shape == (4, 8, 8, 3)
        assert len(y_train) == 12 and len(y_test) == 4
        assert X_train.dtype == np.float32
        assert X_train.min() >= 0.0 and X_train.max() <= 1.0
        assert np.all(np.isin(np.concatenate([y_train, y_test]), range(4)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestFindNeuSteelDir:
    def test_finds_neu_det_images(self, isolated_cache: Path) -> None:
        p = isolated_cache / "NEU-DET" / "images"
        p.mkdir(parents=True)
        assert _find_neu_steel_dir(isolated_cache) == p

    def test_finds_neu_det_itself(self, isolated_cache: Path) -> None:
        p = isolated_cache / "NEU-DET"
        p.mkdir(parents=True)
        assert _find_neu_steel_dir(isolated_cache) == p

    def test_deep_search_images(self, isolated_cache: Path) -> None:
        p = isolated_cache / "a" / "b" / "images"
        p.mkdir(parents=True)
        assert _find_neu_steel_dir(isolated_cache) == p

    def test_dir_with_jpg_files(self, isolated_cache: Path) -> None:
        p = isolated_cache / "photos"
        p.mkdir(parents=True)
        (p / "x.jpg").touch()
        assert _find_neu_steel_dir(isolated_cache) == p

    def test_nothing_found(self, isolated_cache: Path) -> None:
        (isolated_cache / "empty").mkdir(parents=True)
        assert _find_neu_steel_dir(isolated_cache) is None


class TestTrainTestSplit:
    def test_stratified_classification(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        y = rng.integers(0, 2, 100)
        X_train, X_test, y_train, y_test = _train_test_split(X, y)
        assert len(X_train) == 80 and len(X_test) == 20
        assert set(y_train) == {0, 1}
        assert set(y_test) == {0, 1}

    def test_unstratified_regression(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        y = rng.normal(size=50)
        X_train, X_test, y_train, y_test = _train_test_split(X, y)
        assert len(X_train) == 40 and len(X_test) == 10

    def test_non_float_labels_disable_stratification(self) -> None:
        """Class-counting guard falls back to unstratified split on bad labels."""
        X = np.random.default_rng(1).normal(size=(30, 2))
        y = np.array(["a"] * 15 + ["b"] * 15, dtype=object)
        X_train, X_test, y_train, y_test = _train_test_split(X, y)
        assert len(X_train) == 24 and len(X_test) == 6


class TestLoadImageFolder:
    def test_loads_class_organised_images(self, tmp_path: Path) -> None:
        pytest.importorskip("PIL")
        from PIL import Image

        folder = tmp_path / "images"
        for cls, n in (("cr", 3), ("pa", 2)):
            (folder / cls).mkdir(parents=True)
            for i in range(n):
                Image.new("RGB", (10, 10), (i, 0, 0)).save(folder / cls / f"{i}.png")
        X, y = _load_image_folder(folder, target_size=(32, 32))
        assert X.shape == (5, 32, 32, 3)
        assert X.dtype == np.float32
        assert sorted(set(y.tolist())) == [0, 1]
        assert sorted(y.tolist()) == [0, 0, 0, 1, 1]

    def test_skips_unreadable_images(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        pytest.importorskip("PIL")
        import logging

        caplog.set_level(logging.DEBUG)
        from PIL import Image

        folder = tmp_path / "images"
        (folder / "cls").mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(folder / "cls" / "ok.png")
        (folder / "cls" / "broken.jpg").write_bytes(b"not a real jpeg")
        X, y = _load_image_folder(folder, target_size=(8, 8))
        assert X.shape == (1, 8, 8, 3)
        assert list(y) == [0]
        assert "Skipping unreadable image" in caplog.text


# ---------------------------------------------------------------------------
# Download helpers (mocked transport, no real network)
# ---------------------------------------------------------------------------

class TestDownloadSecom:
    def test_downloads_both_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_retrieve(url: str, dest: Path) -> None:
            dest.write_text(f"content:{url}", encoding="utf-8")

        monkeypatch.setattr(loader, "urlretrieve", fake_retrieve)
        _download_secom(tmp_path)
        assert (tmp_path / "secom.data").exists()
        assert (tmp_path / "secom_labels.data").exists()


class TestDownloadCmapss:
    def test_zip_mirror_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_retrieve(url: str, dest: Path) -> None:
            with zipfile.ZipFile(dest, "w") as zf:
                zf.writestr("train_FD001.txt", "1 1 0.1 0.2\n")

        monkeypatch.setattr(loader, "urlretrieve", fake_retrieve)
        _download_cmapss(tmp_path)
        assert (tmp_path / "train_FD001.txt").exists()
        assert not (tmp_path / "_temp_cmapss.zip").exists()

    def test_falls_through_to_txt_mirror(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_retrieve(url: str, dest: Path) -> None:
            if url.endswith(".zip"):
                raise ConnectionError("mirror down")
            dest.write_text("1 1 0.1 0.2\n", encoding="utf-8")

        monkeypatch.setattr(loader, "urlretrieve", fake_retrieve)
        _download_cmapss(tmp_path)
        assert (tmp_path / "train_FD001.txt").exists()

    def test_all_mirrors_fail_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(loader, "urlretrieve",
                            lambda url, dest: (_ for _ in ()).throw(ConnectionError("down")))
        with pytest.raises(FileNotFoundError, match="Could not download CMAPSS"):
            _download_cmapss(tmp_path)


class TestDownloadNeuSteel:
    def test_gdown_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_download(id: str, output: str, quiet: bool) -> None:
            with zipfile.ZipFile(output, "w") as zf:
                zf.writestr("big_payload.bin", b"\0" * 200_000)

        fake_gdown = SimpleNamespace(download=fake_download)
        monkeypatch.setitem(sys.modules, "gdown", fake_gdown)
        assert _download_neu_steel(tmp_path) is True
        assert not (tmp_path / "NEU-DET.zip").exists()
        assert (tmp_path / "big_payload.bin").exists()

    def test_gdown_small_zip_falls_through(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_download(id: str, output: str, quiet: bool) -> None:
            with zipfile.ZipFile(output, "w") as zf:
                zf.writestr("tiny.txt", b"x")

        monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=fake_download))
        monkeypatch.setitem(sys.modules, "kagglehub", None)
        assert _download_neu_steel(tmp_path) is False

    def test_kagglehub_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = tmp_path / "kaggle_src"
        source.mkdir()
        (source / "data_dir").mkdir()
        (source / "data_dir" / "img.jpg").touch()
        (source / "meta.txt").touch()

        def fake_dataset_download(handle: str) -> str:
            return str(source)

        monkeypatch.setitem(sys.modules, "gdown", None)
        monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(dataset_download=fake_dataset_download))
        assert _download_neu_steel(tmp_path) is True
        assert (tmp_path / "data_dir" / "img.jpg").exists()
        assert (tmp_path / "meta.txt").exists()

    def test_both_sources_unavailable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "gdown", None)
        monkeypatch.setitem(sys.modules, "kagglehub", None)
        assert _download_neu_steel(tmp_path) is False


class TestDownloadCncToolWear:
    def test_kagglehub_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = tmp_path / "cnc_src"
        source.mkdir()
        (source / "experiment_01.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (source / "docs").mkdir()

        def fake_dataset_download(handle: str) -> str:
            return str(source)

        monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(dataset_download=fake_dataset_download))
        assert _download_cnc_tool_wear(tmp_path) is True
        assert (tmp_path / "experiment_01.csv").exists()
        assert (tmp_path / "docs").is_dir()

    def test_direct_url_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        def fake_retrieve(url: str, dest: Path) -> None:
            with zipfile.ZipFile(dest, "w") as zf:
                zf.writestr("experiment_01.csv", "a,b\n1,2\n3,4\n")
                # Incompressible padding so the archive exceeds the 1000-byte guard
                zf.writestr("noise.bin", os.urandom(5000))

        monkeypatch.setitem(sys.modules, "kagglehub", None)
        monkeypatch.setattr(loader, "urlretrieve", fake_retrieve)
        assert _download_cnc_tool_wear(tmp_path) is True
        assert (tmp_path / "experiment_01.csv").exists()
        assert not (tmp_path / "cnc_temp.zip").exists()

    def test_direct_url_too_small_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_retrieve(url: str, dest: Path) -> None:
            dest.write_bytes(b"tiny")

        monkeypatch.setitem(sys.modules, "kagglehub", None)
        monkeypatch.setattr(loader, "urlretrieve", fake_retrieve)
        assert _download_cnc_tool_wear(tmp_path) is False

    def test_all_sources_fail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "kagglehub", None)
        monkeypatch.setattr(loader, "urlretrieve",
                            lambda url, dest: (_ for _ in ()).throw(ConnectionError("down")))
        assert _download_cnc_tool_wear(tmp_path) is False


# ---------------------------------------------------------------------------
# Module-level type export (types module check for downstream tooling)
# ---------------------------------------------------------------------------

class TestTypes:
    def test_module_exports_expected_names(self) -> None:
        for name in (
            "load_secom", "load_nasa_cmapss", "load_neu_steel_defects",
            "load_cnc_tool_wear", "get_dataset_info",
        ):
            assert callable(getattr(loader, name))


# ---------------------------------------------------------------------------
# Zip-slip defense
# ---------------------------------------------------------------------------

class TestSafeExtractall:
    def test_rejects_path_traversal_member(self, tmp_path: Path) -> None:
        import io

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../evil.txt", "payload")
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            with pytest.raises(ValueError, match="Unsafe zip member path"):
                loader._safe_extractall(zf, tmp_path)
        assert not (tmp_path.parent / "evil.txt").exists()

    def test_extracts_safe_members(self, tmp_path: Path) -> None:
        import io

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data.txt", "content")
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            loader._safe_extractall(zf, tmp_path)
        assert (tmp_path / "data.txt").read_text() == "content"
