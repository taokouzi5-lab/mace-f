"""
Dataset loaders for manufacturing AI benchmarks.

Each loader tries to load real data first, then falls back to synthetic
data so the pipeline always runs without manual downloads.

Datasets:
  - SECOM: Semiconductor manufacturing quality data (UCI)
  - NASA CMAPSS: Turbofan engine degradation (NASA Prognostics Center)
  - NEU Steel Defects: Steel surface defects (Northeastern Univ.)
  - CNC Tool Wear: Multi-sensor tool condition monitoring
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from carbon_benchmark.config import DATA_DIR, DATASETS, RANDOM_SEED

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, pd.DataFrame]
DatasetTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
RNG = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset_info(name: str) -> Dict[str, str]:
    """Return metadata dictionary for a named dataset."""
    info = DATASETS.get(name)
    if info is None:
        valid = ", ".join(DATASETS.keys())
        raise KeyError(f"Unknown dataset '{name}'. Valid keys: {valid}")
    return dict(info)


def load_secom() -> DatasetTuple:
    """SECOM semiconductor manufacturing (binary classification, 1567×591).
    Falls back to synthetic data if download fails."""
    cache_dir = DATA_DIR / "secom"
    data_file = cache_dir / "secom.data"
    labels_file = cache_dir / "secom_labels.data"

    try:
        if not data_file.exists():
            logger.info("Downloading SECOM...")
            _download_secom(cache_dir)

        X = pd.read_csv(data_file, sep=" ", header=None)
        y = pd.read_csv(labels_file, sep=" ", header=None)
        y = y.iloc[:, 0].map({-1: 0, 1: 1}).values
        X = X.replace(np.nan, 0).values.astype(np.float32)
        logger.info(f"SECOM loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return _train_test_split(X, y)

    except Exception as exc:
        logger.warning(f"SECOM unavailable ({exc}), using synthetic data")
        return _synthetic_tabular(n_samples=1500, n_features=100, classification=True)


def load_nasa_cmapss() -> DatasetTuple:
    """NASA CMAPSS turbofan RUL (regression, 26 sensors).
    Falls back to synthetic data if download fails."""
    cache_dir = DATA_DIR / "nasa_cmapss"
    train_file = cache_dir / "train_FD001.txt"

    try:
        if not train_file.exists():
            _download_cmapss(cache_dir)

        df = pd.read_csv(train_file, sep=r"\s+", header=None)
        df.columns = (
            ["unit", "cycle"]
            + [f"sensor_{i}" for i in range(1, 24)]
            + ["op1", "op2", "op3"]
        )
        max_cycle = df.groupby("unit")["cycle"].transform("max")
        df["rul"] = max_cycle - df["cycle"]
        feature_cols = [c for c in df.columns if c not in ("unit", "cycle", "rul")]
        X = df[feature_cols].values.astype(np.float32)
        y = df["rul"].values.astype(np.float32)
        logger.info(f"CMAPSS loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return _train_test_split(X, y)

    except Exception as exc:
        logger.warning(f"CMAPSS unavailable ({exc}), using synthetic data")
        return _synthetic_tabular(n_samples=5000, n_features=26, classification=False)


def load_neu_steel_defects() -> DatasetTuple:
    """NEU steel surface defects (6-class, 200×200 images).
    Auto-downloads via gdown (Google Drive) or kagglehub, falls back to synthetic data."""
    cache_dir = DATA_DIR / "neu_steel"

    # Try real data first
    image_dir = _find_neu_steel_dir(cache_dir)
    if image_dir is not None:
        try:
            X, y = _load_image_folder(image_dir, target_size=(224, 224))
            logger.info(f"NEU Steel loaded: {X.shape[0]} images, {len(np.unique(y))} classes")
            return _train_test_split(X, y)
        except Exception as exc:
            logger.warning(f"NEU Steel load failed ({exc}), using synthetic data")

    # Auto-download attempt: try multiple sources
    if not _download_neu_steel(cache_dir):
        logger.warning(
            "NEU Steel not found and auto-download failed. "
            "Manual download: https://www.kaggle.com/datasets/kaustubhdhikshit/neu-surface-defect-database"
        )
        return _synthetic_images(n_samples=500, n_classes=6, size=(224, 224))

    # Retry loading after download
    image_dir = _find_neu_steel_dir(cache_dir)
    if image_dir is not None:
        try:
            X, y = _load_image_folder(image_dir, target_size=(224, 224))
            logger.info(f"NEU Steel loaded: {X.shape[0]} images, {len(np.unique(y))} classes")
            return _train_test_split(X, y)
        except Exception as exc:
            logger.warning(f"NEU Steel load failed after download ({exc}), using synthetic data")

    logger.warning("NEU Steel not loadable after download, using synthetic images")
    return _synthetic_images(n_samples=500, n_classes=6, size=(224, 224))


def load_cnc_tool_wear() -> DatasetTuple:
    """CNC tool wear prediction (regression).

    Auto-downloads via kagglehub, falls back to synthetic data if no valid
    numeric sensor data is found.  The Kaggle CNC dataset contains experiment_*.csv
    files with 44 sensor channels.  We use S1_OutputPower (spindle power) as the
    regression target — spindle power increases with tool wear.

    Files like train.csv (metadata with categorical labels) and non-numeric
    columns (Machining_Process) are automatically filtered out.
    """
    cache_dir = DATA_DIR / "cnc_tool_wear"

    # Auto-download if needed
    if not any(cache_dir.glob("**/*.csv")):
        if not _download_cnc_tool_wear(cache_dir):
            logger.warning(
                "CNC Tool Wear not found and auto-download failed. "
                "Manual download: https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill"
            )
            return _synthetic_tabular(n_samples=2000, n_features=10, classification=False)

    # Find a valid experiment CSV — skip train.csv (metadata), look for sensor data
    csv_files = sorted(cache_dir.glob("**/*.csv"))
    # Filter out clearly non-sensor files
    sensor_files = [f for f in csv_files
                    if "experiment" in f.name.lower() or f.stat().st_size > 50_000]

    if not sensor_files:
        sensor_files = csv_files  # fallback: try any CSV

    for csv_path in sensor_files:
        try:
            df = pd.read_csv(csv_path)

            # Drop purely non-numeric columns (e.g. Machining_Process)
            numeric_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns.tolist()
            if len(numeric_cols) < 3:
                logger.debug(f"Skipping {csv_path.name}: only {len(numeric_cols)} numeric columns")
                continue

            # Target: spindle output power (S1_OutputPower) — correlates with tool wear
            # Fallback: last numeric column
            target_col = "S1_OutputPower" if "S1_OutputPower" in numeric_cols else numeric_cols[-1]
            feature_cols = [c for c in numeric_cols if c != target_col]

            df_numeric = df[numeric_cols].dropna()
            X = df_numeric[feature_cols].values.astype(np.float32)
            y = df_numeric[target_col].values.astype(np.float32)

            if len(X) < 100:
                logger.debug(f"Skipping {csv_path.name}: only {len(X)} rows after cleaning")
                continue

            logger.info(
                f"CNC loaded from {csv_path.name}: {X.shape[0]} samples, "
                f"{X.shape[1]} features, target={target_col}"
            )
            return _train_test_split(X, y)

        except Exception as exc:
            logger.debug(f"CNC file {csv_path.name} failed: {exc}")
            continue

    # Nothing worked — fall back to synthetic
    logger.warning("CNC Tool Wear: no valid sensor CSV found, using synthetic data")
    return _synthetic_tabular(n_samples=2000, n_features=10, classification=False)


# ---------------------------------------------------------------------------
# Synthetic data generators (fallbacks)
# ---------------------------------------------------------------------------

def _synthetic_tabular(
    n_samples: int,
    n_features: int,
    classification: bool = True,
) -> DatasetTuple:
    """Generate synthetic tabular data for pipeline testing."""
    X = RNG.normal(0, 1, (n_samples, n_features)).astype(np.float32)

    if classification:
        y = RNG.integers(0, 2, n_samples).astype(np.int64)
    else:
        y = (X[:, 0] * 2 + RNG.normal(0, 0.5, n_samples)).astype(np.float32)

    logger.info(f"Synthetic tabular: {n_samples}×{n_features} ({'cls' if classification else 'reg'})")
    return _train_test_split(X, y)


def _synthetic_images(
    n_samples: int,
    n_classes: int,
    size: Tuple[int, int],
) -> DatasetTuple:
    """Generate synthetic RGB images for pipeline testing."""
    h, w = size
    X = RNG.uniform(0, 1, (n_samples, h, w, 3)).astype(np.float32)
    y = RNG.integers(0, n_classes, n_samples).astype(np.int64)
    logger.info(f"Synthetic images: {n_samples}×{h}×{w}×3, {n_classes} classes")
    return _train_test_split(X, y)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_neu_steel_dir(cache_dir: Path) -> Optional[Path]:
    """Locate NEU Steel images directory, if present."""
    candidates = [
        cache_dir / "NEU-DET" / "images",
        cache_dir / "NEU-DET",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # Deep search
    for subdir in sorted(cache_dir.glob("**/images")):
        if subdir.is_dir():
            return subdir
    # Check if any subdir has image files
    for subdir in sorted(cache_dir.glob("**/*")):
        if subdir.is_dir() and list(subdir.glob("*.jpg")):
            return subdir
    return None


def _safe_extractall(zf: "zipfile.ZipFile", target_dir: Path) -> None:
    """Extract zip members, rejecting path-traversal entries (zip-slip defense)."""
    target_root = target_dir.resolve()
    for member in zf.namelist():
        member_path = (target_dir / member).resolve()
        if not member_path.is_relative_to(target_root):
            raise ValueError(f"Unsafe zip member path: {member!r}")
    zf.extractall(target_dir)


def _download_secom(cache_dir: Path) -> None:
    """Download SECOM from UCI repository."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"
    for filename in ["secom.data", "secom_labels.data"]:
        urlretrieve(f"{base_url}/{filename}", cache_dir / filename)  # nosec B310
        logger.info(f"  Downloaded {filename}")


def _download_cmapss(cache_dir: Path) -> None:
    """Download CMAPSS FD001 — tries multiple mirrors."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
        "https://raw.githubusercontent.com/juanrodriguez-net/cmapss-data/main/train_FD001.txt",
        "https://raw.githubusercontent.com/akshaykapoor347/CMAPSS-Dataset/main/train_FD001.txt",
    ]
    for url in urls:
        try:
            if url.endswith(".zip"):
                import zipfile
                zip_path = cache_dir / "_temp_cmapss.zip"
                urlretrieve(url, zip_path)  # nosec B310
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extract("train_FD001.txt", cache_dir)
                zip_path.unlink()
                logger.info(f"Downloaded and extracted CMAPSS from {url}")
            else:
                urlretrieve(url, cache_dir / "train_FD001.txt")  # nosec B310
                logger.info(f"Downloaded CMAPSS from {url}")
            return
        except Exception:
            continue
    raise FileNotFoundError(f"Could not download CMAPSS. Place train_FD001.txt in {cache_dir}")


def _download_neu_steel(cache_dir: Path) -> bool:
    """Download NEU Steel Defects dataset. Returns True on success.

    Tries multiple sources in order:
      1. gdown (Google Drive mirror from official NEU page)
      2. kagglehub (requires network, no Kaggle account needed for read)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Strategy 1: gdown from Google Drive (official NEU mirror) ---
    try:
        import gdown
        import zipfile

        gdrive_id = "1qrDZ1aD1272eA79b0uCwwqPrm2Q_WI3k"
        zip_path = cache_dir / "NEU-DET.zip"
        logger.info("Downloading NEU Steel Defects from Google Drive...")
        gdown.download(id=gdrive_id, output=str(zip_path), quiet=False)

        if zip_path.exists() and zip_path.stat().st_size > 100_000:
            logger.info("Extracting NEU Steel Defects...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                _safe_extractall(zf, cache_dir)
            zip_path.unlink()  # clean up zip
            logger.info("NEU Steel Defects downloaded and extracted via gdown")
            return True
        else:
            logger.debug("gdown download too small, trying next source")
    except Exception as exc:
        logger.debug(f"gdown download failed: {exc}")

    # --- Strategy 2: kagglehub ---
    try:
        import kagglehub
        import shutil

        logger.info("Downloading NEU Steel Defects via kagglehub...")
        neu_path = kagglehub.dataset_download(
            "kaustubhdhikshit/neu-surface-defect-database"
        )
        for item in Path(neu_path).iterdir():
            dst = cache_dir / item.name
            if item.is_dir() and not dst.exists():
                shutil.copytree(item, dst)
            elif item.is_file():
                shutil.copy2(item, dst)
        logger.info("NEU Steel Defects downloaded via kagglehub")
        return True
    except Exception as exc:
        logger.debug(f"kagglehub download failed: {exc}")

    return False


def _download_cnc_tool_wear(cache_dir: Path) -> bool:
    """Download CNC Tool Wear dataset. Returns True on success.

    Tries:
      1. kagglehub
      2. Direct URL mirrors (Kaggle raw)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Strategy 1: kagglehub ---
    try:
        import kagglehub
        import shutil

        logger.info("Downloading CNC Tool Wear via kagglehub...")
        cnc_path = kagglehub.dataset_download(
            "shasun/tool-wear-detection-in-cnc-mill"
        )
        for item in Path(cnc_path).iterdir():
            dst = cache_dir / item.name
            if item.is_dir() and not dst.exists():
                shutil.copytree(item, dst)
            elif item.is_file():
                shutil.copy2(item, dst)
        logger.info("CNC Tool Wear downloaded via kagglehub")
        return True
    except Exception as exc:
        logger.debug(f"kagglehub download failed: {exc}")

    # --- Strategy 2: Direct download attempts ---
    urls = [
        "https://storage.googleapis.com/kaggle-data-sets/543098/999999/compressed/tool-wear-detection-in-cnc-mill.zip",
    ]
    for url in urls:
        try:
            import zipfile

            zip_path = cache_dir / "cnc_temp.zip"
            logger.info(f"Trying direct download: {url}")
            urlretrieve(url, zip_path)  # nosec B310
            if zip_path.stat().st_size > 1000:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    _safe_extractall(zf, cache_dir)
                zip_path.unlink()
                logger.info("CNC Tool Wear downloaded via direct URL")
                return True
        except Exception as exc:
            logger.debug(f"Direct download failed: {exc}")

    return False


def _train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> DatasetTuple:
    """Deterministic train/test split with stratification for classification."""
    from sklearn.model_selection import train_test_split

    # Stratify when y is discrete (classification); skip for regression
    try:
        n_classes = len(set(float(v) for v in y[:100]))  # sample first 100
        stratify = y if n_classes <= 20 else None
    except Exception:
        stratify = None

    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=stratify,
    )


def _load_image_folder(
    folder: Path,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """Load images from class-organised subdirectories into numpy arrays."""
    from PIL import Image

    images: list = []
    labels: list = []
    class_names = sorted(
        d.name for d in folder.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    for label_idx, class_name in enumerate(class_names):
        class_dir = folder / class_name
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(target_size)
                    images.append(np.array(img, dtype=np.float32) / 255.0)
                    labels.append(label_idx)
                except Exception:
                    logger.debug(f"Skipping unreadable image: {img_path}")

    return np.array(images), np.array(labels, dtype=np.int64)
