"""
Standardised preprocessing pipelines for manufacturing datasets.

Each pipeline returns data in a format ready for model training/inference
with consistent scaling, imputation, and reshaping.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_tabular(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    impute_strategy: str = "mean",
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess tabular data for traditional ML models (T1-T2).

    Steps:
      1. Impute missing values.
      2. Standard-scale features (optional, recommended for regression).
      3. Return numpy arrays.

    Args:
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.
        impute_strategy: 'mean', 'median', 'most_frequent', or 'constant'.
        scale: Whether to apply StandardScaler.

    Returns:
        (X_train_processed, X_test_processed, y_train, y_test).
    """
    # Flatten if needed (e.g., for image data passed to tabular models)
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if X_test.ndim > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)

    # Impute missing values
    imputer = SimpleImputer(strategy=impute_strategy)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return (
        X_train.astype(np.float32),
        X_test.astype(np.float32),
        y_train,
        y_test,
    )


def preprocess_images(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_classes: Optional[int] = None,
    add_channel_dim: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess image data for deep learning models (T2-T4).

    Steps:
      1. Add channel dimension if grayscale (H, W) → (1, H, W).
      2. Label-encode string targets if needed.
      3. Return float32 arrays in (N, C, H, W) or (N, H, W, C) format.

    Args:
        X_train, X_test: Image arrays. Expect (N, H, W) or (N, H, W, C).
        y_train, y_test: Labels — can be strings or integers.
        num_classes: Number of classes (auto-detected if None).
        add_channel_dim: Add a channel axis for grayscale images.

    Returns:
        (X_train_processed, X_test_processed, y_train_encoded, y_test_encoded).
    """
    # Convert to PyTorch-compatible (N, C, H, W) format
    if X_train.ndim == 3:
        # Grayscale: (N, H, W) → (N, 1, H, W)
        X_train = X_train[:, np.newaxis, :, :]
    elif X_train.ndim == 4 and X_train.shape[-1] == 3:
        # RGB loaded as (N, H, W, 3) → (N, 3, H, W) for PyTorch
        X_train = np.transpose(X_train, (0, 3, 1, 2))

    if X_test.ndim == 3:
        X_test = X_test[:, np.newaxis, :, :]
    elif X_test.ndim == 4 and X_test.shape[-1] == 3:
        X_test = np.transpose(X_test, (0, 3, 1, 2))

    # Label-encode if string targets
    if y_train.dtype.kind in ("U", "S", "O"):
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

    return (
        X_train.astype(np.float32),
        X_test.astype(np.float32),
        y_train.astype(np.int64),
        y_test.astype(np.int64),
    )


def preprocess_time_series(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    sequence_length: int = 30,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess time-series data for LSTM models (T3).

    Steps:
      1. Create sliding-window sequences of fixed length.
      2. Standard-scale features.
      3. Return (N, seq_len, n_features) format for LSTM input.

    Args:
        X_train, X_test: Raw time-series feature arrays (N_samples, n_features).
        y_train, y_test: Target arrays.
        sequence_length: Number of time steps per sequence window.
        scale: Whether to apply StandardScaler.

    Returns:
        (X_train_seq, X_test_seq, y_train_seq, y_test_seq) in LSTM-compatible shape.
    """
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_train_seq, y_train_seq = _create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = _create_sequences(X_test, y_test, sequence_length)

    return (
        X_train_seq.astype(np.float32),
        X_test_seq.astype(np.float32),
        y_train_seq.astype(np.float32),
        y_test_seq.astype(np.float32),
    )


def _create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences from time-series data.

    Args:
        X: Feature array (N, features).
        y: Target array (N,).
        seq_len: Window length.

    Returns:
        (X_seq (N-seq_len, seq_len, features), y_seq (N-seq_len,)).
    """
    if len(X) <= seq_len:
        raise ValueError(
            f"Data length ({len(X)}) is <= sequence length ({seq_len}). "
            f"Reduce seq_len or provide more data. "
            f"A single truncated sample would produce meaningless training results."
        )

    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(xs), np.array(ys)
