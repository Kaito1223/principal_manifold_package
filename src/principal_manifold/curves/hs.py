from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .._types import CurveSnapshot, PrincipalCurveResult, copy_curve_snapshot as _copy_snapshot
from ..geometry import (
    _as_2d_float_array,
    _choose_window_size,
    _dataset_radius,
    _mean_squared_distance_to_polyline,
    _pca_init_line,
    _polyline_lambda_from_segment_parameter,
    _polyline_length,
    _project_onto_polyline,
    _project_to_polyline_parameter,
    _sort_nodes_by_lambda,
)

Array = np.ndarray

NUMBA_OK = False

def njit(func=None, *args, **kwargs):
    if func is not None and callable(func):
        return func
    def wrap(f):
        return f
    return wrap

@dataclass
class HSConfig:
    """
    Configuration for the finite-sample HS principal-curve algorithm.

    Parameters
    ----------
    w
        Neighborhood fraction. The window size is computed as k=floor(w*N),
        then adjusted to be at least 3 and odd whenever possible.
    max_iter
        Maximum number of outer HS iterations.
    tol
        Relative MSD improvement stopping tolerance.
    verbose
        If True, print one diagnostics record per iteration.
    store_trace
        If True, store curve snapshots in ``trace_`` with the same structure as
        the KK framework so that the same visual.py can be reused unchanged.
    """

    w: float = 0.10
    max_iter: int = 10
    tol: float = 1e-4
    verbose: bool = False
    store_trace: bool = False


class HSPrincipalCurve:
    """
    Finite-sample HS principal-curve estimator rewritten in the same class-based
    style as the KK framework.

    This implementation preserves the algorithmic structure of the uploaded HS
    script:
      1. initialize all nodes on the PCA line,
      2. project points onto the current polyline,
      3. convert those projections into updated internal coordinates,
      4. build exact contiguous k-neighborhood windows in sorted lambda order,
      5. perform weighted local linear regression at each lambda,
      6. rebuild the polyline and repeat until convergence.

    The trace snapshots intentionally use the same field names as the KK
    framework, so the same visual.py can be used directly.
    """

    def __init__(self, config: Optional[HSConfig] = None) -> None:
        self.config = config or HSConfig()
        self.vertices_: Optional[Array] = None
        self.history_: List[Dict[str, float]] = []
        self.trace_: List[CurveSnapshot] = []
        self._X_fit_: Optional[Array] = None
        self._lambda_nodes_: Optional[Array] = None
        self._mean_: Optional[Array] = None
        self._radius_: Optional[float] = None
        self.numba_available_: bool = bool(NUMBA_OK)

    def fit(self, X: Array) -> "HSPrincipalCurve":
        X = _as_2d_float_array(X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("HS principal curve requires at least two samples.")
        if n_features < 1:
            raise ValueError("Input must have at least one feature.")

        self._X_fit_ = X.copy()
        self._mean_ = X.mean(axis=0)
        self._radius_ = _dataset_radius(X, self._mean_)
        self.history_ = []
        self.trace_ = []

        mu, direction, lam0 = _pca_init_line(X)
        initial_vertices = mu[None, :] + lam0[:, None] * direction[None, :]
        lam_nodes, vertices = _sort_nodes_by_lambda(lam0, initial_vertices)

        self._lambda_nodes_ = lam_nodes
        self.vertices_ = vertices

        init_msd = _mean_squared_distance_to_polyline(X, self.vertices_)
        if self.config.store_trace:
            self._append_trace(
                X,
                self.vertices_,
                phase="init",
                outer_iteration=0,
                sweep=None,
                mean_squared_distance=init_msd,
            )

        prev_msd = init_msd

        for outer_iteration in range(1, self.config.max_iter + 1):
            if self.vertices_ is None or self._lambda_nodes_ is None:
                raise RuntimeError("Internal state corrupted: fit initialization failed.")

            # E-step: project onto the current polyline.
            t, projected, d2 = _project_to_polyline_parameter(X, self.vertices_)
            projected_msd = float(np.mean(d2))
            if self.config.store_trace:
                self._append_trace(
                    X,
                    self.vertices_,
                    phase="projected",
                    outer_iteration=outer_iteration,
                    sweep=None,
                    mean_squared_distance=projected_msd,
                )

            lam_proj = _polyline_lambda_from_segment_parameter(t, self._lambda_nodes_)
            k_neighbors = _choose_window_size(self.config.w, n_samples)

            sort_idx = np.argsort(lam_proj)
            lam_sorted = lam_proj[sort_idx]
            X_sorted = X[sort_idx]

            Ls, Rs = hs_windows_sorted(lam_sorted, k_neighbors)
            new_vertices = hs_update_local_regression_sorted(lam_sorted, X_sorted, Ls, Rs)

            self._lambda_nodes_ = lam_sorted
            self.vertices_ = new_vertices

            updated_msd = _mean_squared_distance_to_polyline(X, self.vertices_)
            rel_improvement = (prev_msd - updated_msd) / max(abs(prev_msd), 1e-15)
            record = {
                "iteration": float(outer_iteration),
                "k_neighbors": float(k_neighbors),
                "mean_squared_distance": float(updated_msd),
                "root_mean_squared_distance": float(np.sqrt(max(updated_msd, 0.0))),
                "polyline_length": float(_polyline_length(self.vertices_)),
                "relative_improvement": float(rel_improvement),
            }
            self.history_.append(record)
            if self.config.verbose:
                print(record)

            if self.config.store_trace:
                self._append_trace(
                    X,
                    self.vertices_,
                    phase="updated",
                    outer_iteration=outer_iteration,
                    sweep=None,
                    mean_squared_distance=updated_msd,
                )

            if rel_improvement < self.config.tol:
                break
            prev_msd = updated_msd

        return self

    def fit_result(self, X: Array) -> PrincipalCurveResult:
        self.fit(X)
        return self.result_

    @property
    def result_(self) -> PrincipalCurveResult:
        if self.vertices_ is None or self._X_fit_ is None:
            raise AttributeError("Model has not been fitted yet.")
        projected, arc_coords, kinds, indices = self.project(self._X_fit_)
        return PrincipalCurveResult(
            vertices=self.vertices_.copy(),
            projected_points=projected,
            arc_length_coordinates=arc_coords,
            nearest_object_kind=kinds,
            nearest_object_index=indices,
            history=[dict(item) for item in self.history_],
            trace=[_copy_snapshot(snapshot) for snapshot in self.trace_],
        )

    def project(self, X: Array) -> Tuple[Array, Array, Array, Array]:
        if self.vertices_ is None:
            raise ValueError("Call fit before project.")
        X = _as_2d_float_array(X)
        return _project_onto_polyline(X, self.vertices_)

    def transform(self, X: Array) -> Array:
        _, arc_coords, _, _ = self.project(X)
        return arc_coords

    def score(self, X: Array) -> float:
        if self.vertices_ is None:
            raise ValueError("Call fit before score.")
        X = _as_2d_float_array(X)
        return -_mean_squared_distance_to_polyline(X, self.vertices_)

    def _append_trace(
        self,
        X: Array,
        vertices: Array,
        phase: str,
        outer_iteration: int,
        sweep: Optional[int],
        mean_squared_distance: Optional[float] = None,
    ) -> None:
        vertices = np.asarray(vertices, dtype=float)
        msd = (
            float(mean_squared_distance)
            if mean_squared_distance is not None
            else _mean_squared_distance_to_polyline(X, vertices)
        )
        self.trace_.append(
            CurveSnapshot(
                phase=phase,
                outer_iteration=int(outer_iteration),
                sweep=None if sweep is None else int(sweep),
                vertices=vertices.copy(),
                mean_squared_distance=float(msd),
                root_mean_squared_distance=float(np.sqrt(max(msd, 0.0))),
                lambda_p=0.0,
                segments=int(max(vertices.shape[0] - 1, 0)),
                polyline_length=float(_polyline_length(vertices)),
            )
        )


@njit

def hs_windows_sorted(lam_sorted: np.ndarray, k: int):
    """For each point i in sorted lambda order, choose the exact contiguous
    window [L, R] of size k minimizing the maximum left/right lambda radius."""
    n_samples = lam_sorted.shape[0]
    left_bounds = np.empty(n_samples, dtype=np.int64)
    right_bounds = np.empty(n_samples, dtype=np.int64)

    for i in range(n_samples):
        t_min = 0
        if i + k - n_samples > 0:
            t_min = i + k - n_samples
        t_max = k - 1
        if i < t_max:
            t_max = i

        lo = t_min
        hi = t_max
        while lo < hi:
            mid = (lo + hi) // 2
            left = i - mid
            right = left + k - 1
            left_dist = lam_sorted[i] - lam_sorted[left]
            right_dist = lam_sorted[right] - lam_sorted[i]
            if left_dist >= right_dist:
                hi = mid
            else:
                lo = mid + 1

        best_t = lo
        best_val = 1e300
        for cand in (lo, lo - 1):
            if cand < t_min or cand > t_max:
                continue
            left = i - cand
            right = left + k - 1
            left_dist = lam_sorted[i] - lam_sorted[left]
            right_dist = lam_sorted[right] - lam_sorted[i]
            value = left_dist if left_dist > right_dist else right_dist
            if value < best_val:
                best_val = value
                best_t = cand

        left = i - best_t
        right = left + k - 1
        left_bounds[i] = left
        right_bounds[i] = right

    return left_bounds, right_bounds


@njit

def hs_update_local_regression_sorted(
    lam_sorted: np.ndarray,
    X_sorted: np.ndarray,
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
):
    """Weighted local linear regression y(lambda)=a*lambda+b at every sorted
    lambda using the HS compact-support kernel."""
    n_samples, n_features = X_sorted.shape
    updated = np.empty((n_samples, n_features), dtype=np.float64)

    for i in range(n_samples):
        left = left_bounds[i]
        right = right_bounds[i]

        lam_i = lam_sorted[i]
        d_left = lam_i - lam_sorted[left]
        d_right = lam_sorted[right] - lam_i
        dmax = d_left if d_left > d_right else d_right

        if dmax < 1e-12:
            for feature_idx in range(n_features):
                total = 0.0
                count = 0.0
                for j in range(left, right + 1):
                    total += X_sorted[j, feature_idx]
                    count += 1.0
                updated[i, feature_idx] = total / count
            continue

        S0 = 0.0
        S1 = 0.0
        S2 = 0.0
        T0 = np.zeros(n_features, dtype=np.float64)
        T1 = np.zeros(n_features, dtype=np.float64)

        for j in range(left, right + 1):
            delta = lam_sorted[j] - lam_i
            abs_delta = delta if delta >= 0 else -delta
            ratio = abs_delta / dmax
            weight = (1.0 - ratio * ratio * ratio) ** (1.0 / 3.0)

            lam_j = lam_sorted[j]
            S0 += weight
            S1 += weight * lam_j
            S2 += weight * lam_j * lam_j
            for feature_idx in range(n_features):
                x_j = X_sorted[j, feature_idx]
                T0[feature_idx] += weight * x_j
                T1[feature_idx] += weight * lam_j * x_j

        denom = S0 * S2 - S1 * S1
        if denom < 1e-12:
            for feature_idx in range(n_features):
                updated[i, feature_idx] = T0[feature_idx] / (S0 + 1e-12)
        else:
            for feature_idx in range(n_features):
                a = (S0 * T1[feature_idx] - S1 * T0[feature_idx]) / denom
                b = (T0[feature_idx] - a * S1) / (S0 + 1e-12)
                updated[i, feature_idx] = a * lam_i + b

    return updated

