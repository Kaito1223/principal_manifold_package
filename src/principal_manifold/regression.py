from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class RegressionPredictionResult:
    """
    Detailed output from PrincipalManifoldRegressor.predict(..., return_details=True).

    Attributes
    ----------
    y_pred:
        Predicted target values in the original target scale.

    projected_points:
        Points on the learned joint-space manifold in original coordinates.
        Shape is (n_samples, n_features + 1), where the last column is y.

    x_squared_distance:
        Squared distance between each query x and the X-part of its projected
        manifold point.

    nearest_edge_index:
        Index of the selected edge for each query point.
        If the manifold has no edges, this is -1.

    edge_t:
        Interpolation coordinate on the selected edge.
        0 means the first endpoint, 1 means the second endpoint.
    """

    y_pred: Array
    projected_points: Array
    x_squared_distance: Array
    nearest_edge_index: Array
    edge_t: Array


def _as_2d_float_array(X: Array) -> Array:
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("Expected X to be a 1D or 2D numeric array.")

    return X


def _as_1d_float_array(y: Array) -> Array:
    y = np.asarray(y, dtype=float).reshape(-1)

    if y.ndim != 1:
        raise ValueError("Expected y to be one-dimensional.")

    return y


def _safe_scale(std: Array) -> Array:
    std = np.asarray(std, dtype=float)
    return np.where(std <= 1e-15, 1.0, std)


def _chain_edges(n_vertices: int) -> Array:
    if n_vertices < 2:
        return np.empty((0, 2), dtype=int)

    return np.column_stack(
        [
            np.arange(n_vertices - 1, dtype=int),
            np.arange(1, n_vertices, dtype=int),
        ]
    )


def _extract_vertices_edges(estimator: Any) -> Tuple[Array, Array]:
    """
    Extract vertices and edges from any principal_manifold estimator.

    Curves such as HS and KK usually provide vertices but not explicit edges,
    so we create chain edges.

    Graph methods provide both vertices and edges through their result object.
    """

    # Prefer result_, because it is the public fitted-result interface.
    result = None
    try:
        result = estimator.result_
    except Exception:
        result = None

    if result is not None and hasattr(result, "vertices"):
        vertices = np.asarray(result.vertices, dtype=float)

        if hasattr(result, "edges"):
            edges = np.asarray(result.edges, dtype=int)
        else:
            edges = _chain_edges(vertices.shape[0])

        return vertices, edges

    # Fallback for graph-like estimators.
    if hasattr(estimator, "graph_") and getattr(estimator, "graph_") is not None:
        graph = estimator.graph_
        vertices = np.asarray(graph.vertices, dtype=float)
        edges = np.asarray(graph.edges, dtype=int)
        return vertices, edges

    # Fallback for curve-like estimators.
    if hasattr(estimator, "vertices_") and getattr(estimator, "vertices_") is not None:
        vertices = np.asarray(estimator.vertices_, dtype=float)
        edges = _chain_edges(vertices.shape[0])
        return vertices, edges

    raise RuntimeError(
        "Could not extract vertices/edges. Make sure the base estimator has been fitted."
    )


def _partial_project_x_to_joint_manifold(
    X: Array,
    vertices: Array,
    edges: Array,
) -> Tuple[Array, Array, Array, Array]:
    """
    Project observed X onto the X-part of a joint-space manifold.

    The learned manifold lives in joint space:

        z = [x1, x2, ..., xd, y]

    But at prediction time, only x is observed. For each query x, this finds
    the point q on the learned manifold whose X-coordinates are closest to x,
    then returns q.

    Returns
    -------
    projected:
        Full projected joint-space points [X_projected, y_projected].

    x_squared_distance:
        Squared distance between x and X_projected.

    nearest_edge_index:
        Selected edge index for each sample. If no edges exist, this is -1.

    edge_t:
        Interpolation coordinate on the selected edge.
    """

    X = _as_2d_float_array(X)
    vertices = _as_2d_float_array(vertices)
    edges = np.asarray(edges, dtype=int)

    if vertices.shape[1] != X.shape[1] + 1:
        raise ValueError(
            "Dimension mismatch. The manifold vertices must have one more column "
            "than X, because vertices are [X, y]. "
            f"Got X.shape={X.shape}, vertices.shape={vertices.shape}."
        )

    VX = vertices[:, :-1]

    n_samples = X.shape[0]
    projected = np.empty((n_samples, vertices.shape[1]), dtype=float)
    x_squared_distance = np.empty(n_samples, dtype=float)
    nearest_edge_index = np.full(n_samples, -1, dtype=int)
    edge_t = np.zeros(n_samples, dtype=float)

    # Degenerate case: no edges, use nearest vertex.
    if edges.size == 0:
        d2 = np.sum((X[:, None, :] - VX[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(d2, axis=1)

        projected[:] = vertices[nearest]
        x_squared_distance[:] = d2[np.arange(n_samples), nearest]

        return projected, x_squared_distance, nearest_edge_index, edge_t

    for sample_idx, x in enumerate(X):
        best_dist = np.inf
        best_projected = None
        best_edge = -1
        best_t = 0.0

        for edge_idx, (i, j) in enumerate(edges):
            i = int(i)
            j = int(j)

            a = vertices[i]
            b = vertices[j]

            a_x = a[:-1]
            b_x = b[:-1]

            direction_x = b_x - a_x
            denom = float(np.dot(direction_x, direction_x))

            if denom <= 1e-15:
                t = 0.0
            else:
                t = float(np.dot(x - a_x, direction_x) / denom)
                t = min(1.0, max(0.0, t))

            q = a + t * (b - a)
            q_x = q[:-1]

            dist = float(np.sum((x - q_x) ** 2))

            if dist < best_dist:
                best_dist = dist
                best_projected = q
                best_edge = edge_idx
                best_t = t

        projected[sample_idx] = best_projected
        x_squared_distance[sample_idx] = best_dist
        nearest_edge_index[sample_idx] = best_edge
        edge_t[sample_idx] = best_t

    return projected, x_squared_distance, nearest_edge_index, edge_t


class PrincipalManifoldRegressor:
    """
    Supervised regression adapter for principal_manifold estimators.

    The base estimator is still unsupervised. This wrapper turns it into a
    regressor by fitting the manifold in joint space:

        Z_train = [X_train, y_train]

    Then prediction is performed by partial projection:

        given x_new,
        find the point [x_projected, y_projected] on the learned manifold
        whose X-part is closest to x_new,
        return y_projected.

    Parameters
    ----------
    base_estimator:
        Any fitted-style principal_manifold estimator with fit(...) and result_
        or vertices_/graph_ after fitting.

    scale:
        Whether to standardize X and y before fitting the joint-space manifold.
        This is recommended because otherwise the target scale can dominate the
        geometry.

    copy_estimator:
        If True, deep-copy the base estimator before fitting. If False, fit the
        estimator object passed in.
    """

    def __init__(
        self,
        base_estimator: Any,
        *,
        scale: bool = True,
        copy_estimator: bool = False,
    ) -> None:
        self.base_estimator = base_estimator
        self.scale = bool(scale)
        self.copy_estimator = bool(copy_estimator)

        self.estimator_: Optional[Any] = None

        self.x_mean_: Optional[Array] = None
        self.x_scale_: Optional[Array] = None
        self.y_mean_: Optional[float] = None
        self.y_scale_: Optional[float] = None

        self.vertices_: Optional[Array] = None
        self.edges_: Optional[Array] = None

        self.vertices_original_: Optional[Array] = None
        self.edges_original_: Optional[Array] = None

    def fit(self, X: Array, y: Array) -> "PrincipalManifoldRegressor":
        X = _as_2d_float_array(X)
        y = _as_1d_float_array(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows. "
                f"Got X.shape={X.shape}, y.shape={y.shape}."
            )

        if self.copy_estimator:
            import copy

            self.estimator_ = copy.deepcopy(self.base_estimator)
        else:
            self.estimator_ = self.base_estimator

        if self.scale:
            self.x_mean_ = X.mean(axis=0)
            self.x_scale_ = _safe_scale(X.std(axis=0))

            self.y_mean_ = float(y.mean())
            self.y_scale_ = float(max(y.std(), 1e-15))

            X_fit = (X - self.x_mean_[None, :]) / self.x_scale_[None, :]
            y_fit = (y - self.y_mean_) / self.y_scale_
        else:
            self.x_mean_ = np.zeros(X.shape[1], dtype=float)
            self.x_scale_ = np.ones(X.shape[1], dtype=float)
            self.y_mean_ = 0.0
            self.y_scale_ = 1.0

            X_fit = X
            y_fit = y

        Z_fit = np.column_stack([X_fit, y_fit])

        self.estimator_.fit(Z_fit)

        vertices, edges = _extract_vertices_edges(self.estimator_)

        self.vertices_ = vertices
        self.edges_ = edges

        self.vertices_original_ = self._inverse_transform_joint(vertices)
        self.edges_original_ = edges.copy()

        return self

    def _check_is_fitted(self) -> None:
        if self.estimator_ is None or self.vertices_ is None or self.edges_ is None:
            raise ValueError("Call fit before predict.")

    def _transform_X(self, X: Array) -> Array:
        X = _as_2d_float_array(X)

        if self.x_mean_ is None or self.x_scale_ is None:
            raise ValueError("Call fit before transforming X.")

        return (X - self.x_mean_[None, :]) / self.x_scale_[None, :]

    def _inverse_transform_joint(self, Z: Array) -> Array:
        Z = _as_2d_float_array(Z)

        if (
            self.x_mean_ is None
            or self.x_scale_ is None
            or self.y_mean_ is None
            or self.y_scale_ is None
        ):
            raise ValueError("Call fit before inverse-transforming joint points.")

        X_original = Z[:, :-1] * self.x_scale_[None, :] + self.x_mean_[None, :]
        y_original = Z[:, -1] * self.y_scale_ + self.y_mean_

        return np.column_stack([X_original, y_original])

    def predict(self, X: Array, *, return_details: bool = False):
        self._check_is_fitted()

        X_scaled = self._transform_X(X)

        projected_scaled, x_d2_scaled, edge_idx, edge_t = (
            _partial_project_x_to_joint_manifold(
                X_scaled,
                self.vertices_,
                self.edges_,
            )
        )

        projected_original = self._inverse_transform_joint(projected_scaled)
        y_pred = projected_original[:, -1]

        if not return_details:
            return y_pred

        return RegressionPredictionResult(
            y_pred=y_pred,
            projected_points=projected_original,
            x_squared_distance=x_d2_scaled,
            nearest_edge_index=edge_idx,
            edge_t=edge_t,
        )

    def fit_predict(self, X: Array, y: Array, *, return_details: bool = False):
        self.fit(X, y)
        return self.predict(X, return_details=return_details)

    def score(self, X: Array, y: Array) -> float:
        y = _as_1d_float_array(y)
        y_pred = self.predict(X)
        return -float(np.mean((y - y_pred) ** 2))