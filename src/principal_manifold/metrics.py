from __future__ import annotations

import numpy as np

Array = np.ndarray


def _as_matching_1d_arrays(y_true: Array, y_pred: Array) -> tuple[Array, Array]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    return y_true, y_pred


def prediction_rmse(y_true: Array, y_pred: Array) -> float:
    y_true, y_pred = _as_matching_1d_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def prediction_sad(y_true: Array, y_pred: Array) -> float:
    """
    Sum of absolute deviations for scalar prediction errors:

        SAD = sum_i |y_i - yhat_i|
    """
    y_true, y_pred = _as_matching_1d_arrays(y_true, y_pred)
    return float(np.sum(np.abs(y_true - y_pred)))


def prediction_mae(y_true: Array, y_pred: Array) -> float:
    """
    Mean absolute error:

        MAE = mean_i |y_i - yhat_i|
    """
    y_true, y_pred = _as_matching_1d_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def geometric_sad(X: Array, X_projected: Array) -> float:
    """
    Geometric sum of absolute deviations from data points to their
    projections on the learned principal manifold:

        SAD = sum_i ||x_i - proj_M(x_i)||_2
    """
    X = np.asarray(X, dtype=float)
    X_projected = np.asarray(X_projected, dtype=float)

    if X.shape != X_projected.shape:
        raise ValueError("X and X_projected must have the same shape.")

    distances = np.linalg.norm(X - X_projected, axis=1)
    return float(np.sum(distances))


def geometric_mad(X: Array, X_projected: Array) -> float:
    """
    Mean absolute geometric deviation:

        MAD = mean_i ||x_i - proj_M(x_i)||_2
    """
    X = np.asarray(X, dtype=float)
    X_projected = np.asarray(X_projected, dtype=float)

    if X.shape != X_projected.shape:
        raise ValueError("X and X_projected must have the same shape.")

    distances = np.linalg.norm(X - X_projected, axis=1)
    return float(np.mean(distances))


__all__ = [
    "prediction_rmse",
    "prediction_sad",
    "prediction_mae",
    "geometric_sad",
    "geometric_mad",
]