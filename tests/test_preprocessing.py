"""Unit tests for the data preprocessing pipelines."""

import numpy as np
import pytest

from carbon_benchmark.data.preprocessing import (
    _create_sequences,
    preprocess_images,
    preprocess_tabular,
    preprocess_time_series,
)


class TestPreprocessTabular:
    def test_imputes_and_scales(self) -> None:
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(50, 4))
        X_test = rng.normal(size=(10, 4))
        X_train[0, 1] = np.nan
        y_train = rng.integers(0, 2, 50)
        y_test = rng.integers(0, 2, 10)
        out = preprocess_tabular(X_train, X_test, y_train, y_test)
        Xt_train, Xt_test, yt_train, yt_test = out
        assert Xt_train.shape == (50, 4)
        assert Xt_test.shape == (10, 4)
        assert not np.isnan(Xt_train).any()
        # Scaled: mean ≈ 0, std ≈ 1 on train
        assert abs(Xt_train.mean()) < 1e-4
        assert np.allclose(Xt_train.std(axis=0), 1.0, atol=1e-3)
        assert yt_train is y_train and yt_test is y_test

    def test_no_scaling(self) -> None:
        rng = np.random.default_rng(1)
        X_train = rng.normal(size=(20, 3))
        X_test = rng.normal(size=(5, 3))
        out = preprocess_tabular(
            X_train, X_test, np.zeros(20), np.zeros(5), scale=False,
        )
        assert np.allclose(out[0], X_train)

    def test_flattens_image_input(self) -> None:
        X_train = np.random.default_rng(2).random((10, 8, 8, 3))
        X_test = np.random.default_rng(3).random((4, 8, 8, 3))
        out = preprocess_tabular(X_train, X_test, np.zeros(10), np.zeros(4))
        assert out[0].shape == (10, 8 * 8 * 3)
        assert out[1].shape == (4, 8 * 8 * 3)

    def test_median_imputation(self) -> None:
        X_train = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 9.0]])
        out = preprocess_tabular(
            X_train, X_train.copy(), np.zeros(3), np.zeros(3),
            impute_strategy="median", scale=False,
        )
        # Median of observed column values [3.0, 9.0] is 6.0
        assert out[0][0, 1] == 6.0


class TestPreprocessImages:
    def test_grayscale_adds_channel_dim(self) -> None:
        X_train = np.random.default_rng(0).random((10, 28, 28))
        X_test = np.random.default_rng(1).random((4, 28, 28))
        out = preprocess_images(X_train, X_test, np.zeros(10), np.zeros(4))
        assert out[0].shape == (10, 1, 28, 28)
        assert out[1].shape == (4, 1, 28, 28)

    def test_rgb_transposed_to_channels_first(self) -> None:
        X_train = np.random.default_rng(2).random((6, 32, 32, 3))
        X_test = np.random.default_rng(3).random((2, 32, 32, 3))
        out = preprocess_images(X_train, X_test, np.zeros(6), np.zeros(2))
        assert out[0].shape == (6, 3, 32, 32)
        assert out[1].shape == (2, 3, 32, 32)

    def test_channels_first_passthrough(self) -> None:
        X_train = np.random.default_rng(4).random((5, 3, 32, 32))
        X_test = np.random.default_rng(5).random((1, 3, 32, 32))
        out = preprocess_images(X_train, X_test, np.zeros(5), np.zeros(1))
        assert out[0].shape == (5, 3, 32, 32)

    def test_string_labels_encoded(self) -> None:
        X_train = np.random.default_rng(6).random((6, 3, 16, 16))
        X_test = np.random.default_rng(7).random((2, 3, 16, 16))
        y_train = np.array(["crazing", "patches", "crazing", "patches", "scratches", "patches"])
        y_test = np.array(["scratches", "patches"])
        out = preprocess_images(X_train, X_test, y_train, y_test)
        assert out[2].dtype == np.int64
        assert set(out[2].tolist()) == {0, 1, 2}
        # y_test encoded with the same mapping
        assert out[3].tolist() == [2, 1]


class TestPreprocessTimeSeries:
    def test_windowing_and_scaling(self) -> None:
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(100, 5))
        X_test = rng.normal(size=(50, 5))
        y_train = rng.normal(size=100)
        y_test = rng.normal(size=50)
        out = preprocess_time_series(X_train, X_test, y_train, y_test, sequence_length=10)
        Xt_train, Xt_test, yt_train, yt_test = out
        assert Xt_train.shape == (90, 10, 5)
        assert Xt_test.shape == (40, 10, 5)
        assert yt_train.shape == (90,)
        assert yt_test.shape == (40,)
        assert Xt_train.dtype == np.float32

    def test_no_scaling_option(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 3))
        y = rng.normal(size=20)
        out = preprocess_time_series(X, X, y, y, sequence_length=5, scale=False)
        assert out[0].shape == (15, 5, 3)

    def test_create_sequences_windows(self) -> None:
        X = np.arange(10, dtype=float).reshape(10, 1)
        y = np.arange(10, dtype=float)
        X_seq, y_seq = _create_sequences(X, y, 3)
        assert X_seq.shape == (7, 3, 1)
        assert X_seq[0, :, 0].tolist() == [0.0, 1.0, 2.0]
        assert y_seq[0] == 2.0
        # Last window covers indices 6..8 → target is the window's last value
        assert y_seq[-1] == 8.0

    def test_create_sequences_too_short_raises(self) -> None:
        X = np.zeros((5, 2))
        y = np.zeros(5)
        with pytest.raises(ValueError, match="Reduce seq_len"):
            _create_sequences(X, y, 5)

    def test_create_sequences_exactly_length_raises(self) -> None:
        X = np.zeros((4, 2))
        y = np.zeros(4)
        with pytest.raises(ValueError):
            _create_sequences(X, y, 4)
