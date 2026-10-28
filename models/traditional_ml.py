"""
Traditional ML model factories for manufacturing AI benchmarks (T1–T2).

Each factory returns a (model, train_fn, inference_fn) tuple so that
CarbonTracker can wrap training and inference uniformly.

Models:
  T1: Linear Regression — tool wear prediction (regression)
  T1: Decision Tree (max_depth≤10) — quality pass/fail (classification)
  T2: Random Forest (100 trees) — quality classification
  T2: XGBoost — remaining useful life prediction (regression)

References:
  Milutinović & Lukić (2025) — CodeCarbon benchmarking of these model types
  Geißler et al. (2024) — hyperparameter impact on energy consumption
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type aliases for the factory return signature
ModelFactory = Tuple[Any, Callable[..., Any], Callable[..., np.ndarray]]


# ---------------------------------------------------------------------------
# T1: Linear Regression
# ---------------------------------------------------------------------------

def create_linear_regression(
    random_seed: int = 42,
) -> ModelFactory:
    """Create a linear regression model for manufacturing regression tasks.

    Use case: CNC tool wear prediction from cutting force and temperature data.

    Returns:
        (model, train_fn, inference_fn) tuple.
    """
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()

    def train_fn(X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """Train linear regression. Fast — ~seconds for manufacturing datasets."""
        logger.info(f"Training LinearRegression on {X.shape[0]:,} samples "
                     f"with {X.shape[1]} features")
        return model.fit(X, y)

    def inference_fn(X: np.ndarray) -> np.ndarray:
        """Predict wear values from sensor data."""
        return model.predict(X)

    logger.info("Created LinearRegression (T1)")
    return model, train_fn, inference_fn


# ---------------------------------------------------------------------------
# T1: Decision Tree
# ---------------------------------------------------------------------------

def create_decision_tree(
    max_depth: int = 10,
    random_seed: int = 42,
) -> ModelFactory:
    """Create a shallow decision tree for manufacturing quality classification.

    Use case: Pass/fail classification on semiconductor fabrication data (SECOM).
    max_depth≤10 ensures T1 lightweight classification with interpretability.

    Args:
        max_depth: Maximum tree depth (≤10 for T1 classification).
        random_seed: Random state for reproducibility.

    Returns:
        (model, train_fn, inference_fn) tuple.
    """
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_seed,
        class_weight="balanced",
    )

    def train_fn(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        """Train decision tree. Fast — <1 second typical."""
        logger.info(f"Training DecisionTree (max_depth={max_depth}) "
                     f"on {X.shape[0]:,} samples")
        return model.fit(X, y)

    def inference_fn(X: np.ndarray) -> np.ndarray:
        """Predict binary quality label (0=pass, 1=fail)."""
        return model.predict(X)

    logger.info(f"Created DecisionTreeClassifier (T1, max_depth={max_depth})")
    return model, train_fn, inference_fn


# ---------------------------------------------------------------------------
# T2: Random Forest
# ---------------------------------------------------------------------------

def create_random_forest(
    n_estimators: int = 100,
    max_depth: int = 15,
    random_seed: int = 42,
) -> ModelFactory:
    """Create a random forest classifier for manufacturing quality tasks.

    Use case: Same quality classification as decision tree (T1) — enables
    direct T1→T2 carbon efficiency comparison on identical task.

    Args:
        n_estimators: Number of trees (100 is standard per Milutinović, 2025).
        max_depth: Per-tree depth limit (None for unlimited).
        random_seed: Random state for reproducibility.

    Returns:
        (model, train_fn, inference_fn) tuple.
    """
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        n_jobs=-1,  # Use all CPU cores — increases power but reduces wall time
        class_weight="balanced",
    )

    def train_fn(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Train random forest. Moderate compute — ~seconds to minutes."""
        logger.info(f"Training RandomForest (n={n_estimators}) "
                     f"on {X.shape[0]:,} samples")
        return model.fit(X, y)

    def inference_fn(X: np.ndarray) -> np.ndarray:
        """Predict quality label."""
        return model.predict(X)

    logger.info(f"Created RandomForestClassifier (T2, n_estimators={n_estimators})")
    return model, train_fn, inference_fn


# ---------------------------------------------------------------------------
# T2: XGBoost
# ---------------------------------------------------------------------------

def create_xgboost(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_seed: int = 42,
) -> ModelFactory:
    """Create an XGBoost regressor for remaining useful life (RUL) prediction.

    Use case: NASA CMAPSS turbofan engine RUL prediction.
    Provides T2 regression baseline for comparison with T3 LSTM on same task.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Tree depth per round.
        learning_rate: Step size shrinkage.
        random_seed: Random state for reproducibility.

    Returns:
        (model, train_fn, inference_fn) tuple.
    """
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_seed,
        n_jobs=-1,
        verbosity=0,
    )

    def train_fn(X: np.ndarray, y: np.ndarray) -> XGBRegressor:
        """Train XGBoost. Efficient gradient boosting — moderate compute."""
        logger.info(f"Training XGBoost (n={n_estimators}, lr={learning_rate}) "
                     f"on {X.shape[0]:,} samples")
        return model.fit(X, y)

    def inference_fn(X: np.ndarray) -> np.ndarray:
        """Predict RUL in cycles."""
        return model.predict(X)

    logger.info(f"Created XGBRegressor (T2, n_estimators={n_estimators})")
    return model, train_fn, inference_fn


# ---------------------------------------------------------------------------
# Utility: train/test accuracy metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "classification",
) -> Dict[str, float]:
    """Compute standard evaluation metrics for model benchmarking.

    Args:
        y_true: Ground-truth labels.
        y_pred: Model predictions.
        task_type: 'classification' or 'regression'.

    Returns:
        Dictionary of metric_name → value.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        r2_score,
    )

    metrics: Dict[str, float] = {}

    if task_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        # Use weighted F1 for multi-class, binary F1 for binary
        n_classes = len(np.unique(y_true))
        average = "weighted" if n_classes > 2 else "binary"
        try:
            metrics["f1"] = float(f1_score(y_true, y_pred, average=average))
        except ValueError:
            metrics["f1"] = float(f1_score(y_true, y_pred, average="micro"))
    else:
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    return metrics
