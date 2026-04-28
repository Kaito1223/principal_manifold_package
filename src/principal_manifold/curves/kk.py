from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .._types import CurveSnapshot, PrincipalCurveResult, copy_curve_snapshot as _copy_snapshot
from ..geometry import (
    _as_2d_float_array,
    _dataset_radius,
    _mean_squared_distance_to_polyline,
    _polyline_length,
    _project_onto_polyline,
    _segment_distance_squared,
)

Array = np.ndarray

@dataclass
class OptimizerConfig:
    max_inner_sweeps: int = 50
    max_vertex_iterations: int = 50
    relative_vertex_tolerance: float = 1e-6
    relative_curve_tolerance: float = 1e-5
    gradient_epsilon: float = 1e-5
    directional_hessian_epsilon: float = 1e-4
    line_search_shrink: float = 0.5
    min_step_size: float = 1e-8
    armijo_c: float = 1e-4


@dataclass
class KeglKrzyzakConfig:
    beta: float = 0.3
    lambda0_p: float = 0.13
    max_segments: Optional[int] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    verbose: bool = False
    store_trace: bool = False
    trace_inner_sweeps: bool = False


@dataclass
class _Partition:
    vertex_indices: List[Array]
    segment_indices: List[Array]
    point_labels: Array


class KeglKrzyzakPrincipalCurve:
    """
    Kégl-Krzyzak polygonal-line algorithm with the formulas written literally as in
    the chapter / draft used in this conversation.

    Kept from the original structure:
    - same public class name and public API
    - same config/result/trace dataclasses
    - same expectation -> optimization -> adaptation outer loop

    Matched literally at formula level:
    - U(X,Y) = MSD(X,Y) + lambda/(k+1) * sum_i CP(i)
    - CP endpoint penalties are squared segment lengths
    - interior CP(i) = r^2 * (1 + cos gamma(i))
    - r = max_x dist(x, MF(X))
    - stopping rule uses beta * N^(1/3) * r / MSD(X,Y)
    - default lambda uses lambda' * (k / N^(1/3)) * MSD(X,Y) / r
    - partition ties go to vertices

    The chapter states a gradient update y <- y - eta * grad(U_fixed) but does not
    specify a unique eta schedule, so the implementation uses backtracking gradient
    descent for that step while keeping the objective itself literal.
    """

    def __init__(self, config: Optional[KeglKrzyzakConfig] = None) -> None:
        self.config = config or KeglKrzyzakConfig()
        self.vertices_: Optional[Array] = None
        self.history_: List[Dict[str, float]] = []
        self.trace_: List[CurveSnapshot] = []
        self._mean_: Optional[Array] = None
        self._radius_: Optional[float] = None
        self._X_fit_: Optional[Array] = None

    def fit(self, X: Array) -> "KeglKrzyzakPrincipalCurve":
        X = _as_2d_float_array(X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("Kegl-Krzyzak principal curve requires at least two samples.")
        if n_features < 1:
            raise ValueError("Input must have at least one feature.")

        self._X_fit_ = X.copy()
        self._mean_ = X.mean(axis=0)
        self._radius_ = _dataset_radius(X, self._mean_)
        self.vertices_ = self._initialize_with_pca_segment(X)
        self.history_ = []
        self.trace_ = []

        if self.config.store_trace:
            self._append_trace(X, self.vertices_, phase="init", outer_iteration=0, sweep=None)

        outer_iteration = 0
        while True:
            outer_iteration += 1

            sweep_callback: Optional[Callable[[Array, int, float, float], None]] = None
            if self.config.store_trace and self.config.trace_inner_sweeps:
                sweep_callback = lambda vertices, sweep, objective_value, lambda_p: self._append_trace(
                    X,
                    vertices,
                    phase="inner_sweep",
                    outer_iteration=outer_iteration,
                    sweep=sweep,
                    mean_squared_distance=_mean_squared_distance_to_polyline(X, vertices),
                    lambda_p=lambda_p,
                )

            vertices, diagnostics = self._optimize_current_curve(
                X,
                self.vertices_,
                trace_callback=sweep_callback,
            )
            self.vertices_ = vertices

            current_delta = _mean_squared_distance_to_polyline(X, self.vertices_)
            current_rms = float(np.sqrt(max(current_delta, 0.0)))
            n_segments = self.vertices_.shape[0] - 1
            stop_threshold = self._segment_stop_threshold(n_samples, current_delta)
            lambda_p = self._curvature_penalty(n_segments, n_samples, current_delta)
            polyline_length = _polyline_length(self.vertices_)

            record = {
                "segments": float(n_segments),
                "mean_squared_distance": float(current_delta),
                "root_mean_squared_distance": float(current_rms),
                "lambda_p": float(lambda_p),
                "stop_threshold": float(stop_threshold),
                "polyline_length": float(polyline_length),
                "inner_sweeps": float(diagnostics["inner_sweeps"]),
                "max_vertex_move": float(diagnostics["max_vertex_move"]),
            }
            self.history_.append(record)
            if self.config.verbose:
                print(record)

            if self.config.store_trace:
                self._append_trace(
                    X,
                    self.vertices_,
                    phase="optimized",
                    outer_iteration=outer_iteration,
                    sweep=None,
                    mean_squared_distance=current_delta,
                    lambda_p=lambda_p,
                )

            if self.config.max_segments is not None and n_segments >= self.config.max_segments:
                break
            if current_delta <= 1e-15:
                break
            if n_segments > stop_threshold:
                break

            self.vertices_ = self._insert_new_vertex(X, self.vertices_)
            if self.config.store_trace:
                self._append_trace(
                    X,
                    self.vertices_,
                    phase="inserted",
                    outer_iteration=outer_iteration,
                    sweep=None,
                )

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
        lambda_p: Optional[float] = None,
    ) -> None:
        vertices = np.asarray(vertices, dtype=float)
        msd = (
            float(mean_squared_distance)
            if mean_squared_distance is not None
            else _mean_squared_distance_to_polyline(X, vertices)
        )
        segments = vertices.shape[0] - 1
        lam = (
            float(lambda_p)
            if lambda_p is not None
            else self._curvature_penalty(segments, X.shape[0], msd)
        )
        self.trace_.append(
            CurveSnapshot(
                phase=phase,
                outer_iteration=int(outer_iteration),
                sweep=None if sweep is None else int(sweep),
                vertices=vertices.copy(),
                mean_squared_distance=float(msd),
                root_mean_squared_distance=float(np.sqrt(max(msd, 0.0))),
                lambda_p=float(lam),
                segments=int(segments),
                polyline_length=float(_polyline_length(vertices)),
            )
        )

    def _initialize_with_pca_segment(self, X: Array) -> Array:
        mu = X.mean(axis=0)
        centered = X - mu
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = vt[0]
        pc1 = pc1 / max(np.linalg.norm(pc1), 1e-15)
        t = centered @ pc1
        y1 = mu + t.min() * pc1
        y2 = mu + t.max() * pc1
        if np.allclose(y1, y2):
            y2 = y1 + 1e-12 * np.ones_like(y1)
        return np.vstack([y1, y2])

    def _optimize_current_curve(
        self,
        X: Array,
        vertices: Array,
        trace_callback: Optional[Callable[[Array, int, float, float], None]] = None,
    ) -> Tuple[Array, Dict[str, float]]:
        vertices = np.asarray(vertices, dtype=float).copy()
        opt = self.config.optimizer
        n_samples = X.shape[0]
        n_segments = vertices.shape[0] - 1

        # The chapter's optimisation step starts from the partition obtained in the
        # preceding projection step and keeps it fixed while taking gradients.
        partition = _partition_points(X, vertices)
        initial_delta = _mean_squared_distance_to_polyline(X, vertices)
        lambda_p = self._curvature_penalty(n_segments, n_samples, initial_delta)

        prev_objective = _fixed_partition_objective(
            X=X,
            vertices=vertices,
            partition=partition,
            lambda_p=lambda_p,
            radius=float(self._radius_ if self._radius_ is not None else 1.0),
        )
        max_vertex_move_seen = 0.0
        completed_sweeps = 0

        for sweep in range(opt.max_inner_sweeps):
            max_vertex_move = 0.0
            for i in range(vertices.shape[0]):
                old_vertex = vertices[i].copy()
                new_vertex = self._optimize_single_vertex(
                    X=X,
                    vertices=vertices,
                    partition=partition,
                    vertex_index=i,
                    lambda_p=lambda_p,
                )
                vertices[i] = new_vertex
                max_vertex_move = max(max_vertex_move, float(np.linalg.norm(new_vertex - old_vertex)))

            current_objective = _fixed_partition_objective(
                X=X,
                vertices=vertices,
                partition=partition,
                lambda_p=lambda_p,
                radius=float(self._radius_ if self._radius_ is not None else 1.0),
            )
            completed_sweeps = sweep + 1
            max_vertex_move_seen = max(max_vertex_move_seen, max_vertex_move)

            if trace_callback is not None:
                trace_callback(vertices.copy(), completed_sweeps, current_objective, lambda_p)

            rel_improvement = (prev_objective - current_objective) / max(abs(prev_objective), 1e-15)
            if rel_improvement < opt.relative_curve_tolerance and max_vertex_move < opt.relative_curve_tolerance:
                break
            prev_objective = current_objective

        return vertices, {
            "inner_sweeps": float(completed_sweeps),
            "max_vertex_move": float(max_vertex_move_seen),
        }

    def _optimize_single_vertex(
        self,
        X: Array,
        vertices: Array,
        partition: _Partition,
        vertex_index: int,
        lambda_p: float,
    ) -> Array:
        opt = self.config.optimizer
        y = vertices[vertex_index].copy()
        radius = float(self._radius_ if self._radius_ is not None else 1.0)

        def objective(candidate: Array) -> float:
            return _local_vertex_objective_exact(
                X=X,
                vertices=vertices,
                partition=partition,
                vertex_index=vertex_index,
                y=candidate,
                lambda_p=lambda_p,
                radius=radius,
            )

        f_prev = objective(y)

        for _ in range(opt.max_vertex_iterations):
            grad = _finite_difference_gradient(objective, y, opt.gradient_epsilon)
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm < 1e-12:
                break

            direction = -grad
            step = 1.0
            improved = False
            while step >= opt.min_step_size:
                candidate = y + step * direction
                f_candidate = objective(candidate)
                if f_candidate <= f_prev - opt.armijo_c * step * (grad_norm ** 2):
                    rel = (f_prev - f_candidate) / max(abs(f_prev), 1e-15)
                    y = candidate
                    f_prev = f_candidate
                    improved = True
                    if rel < opt.relative_vertex_tolerance:
                        return y
                    break
                step *= opt.line_search_shrink

            if not improved:
                break

        return y

    def _insert_new_vertex(self, X: Array, vertices: Array) -> Array:
        partition = _partition_points(X, vertices)
        seg_counts = np.array([len(idx) for idx in partition.segment_indices], dtype=int)
        if seg_counts.size == 0:
            return vertices

        max_count = seg_counts.max()
        candidates = np.flatnonzero(seg_counts == max_count)
        seg_lengths = np.array(
            [np.linalg.norm(vertices[i + 1] - vertices[i]) for i in range(vertices.shape[0] - 1)],
            dtype=float,
        )
        best_seg = int(candidates[np.argmax(seg_lengths[candidates])])
        midpoint = 0.5 * (vertices[best_seg] + vertices[best_seg + 1])
        return np.vstack([vertices[: best_seg + 1], midpoint[None, :], vertices[best_seg + 1 :]])

    def _curvature_penalty(self, n_segments: int, n_samples: int, delta_n: float) -> float:
        radius = float(self._radius_ if self._radius_ is not None else 1.0)
        if radius <= 1e-15:
            radius = 1.0
        return self.config.lambda0_p * (n_segments / max(n_samples ** (1.0 / 3.0), 1e-15)) * (delta_n / radius)

    def _segment_stop_threshold(self, n_samples: int, delta_n: float) -> float:
        radius = float(self._radius_ if self._radius_ is not None else 1.0)
        if delta_n <= 1e-15:
            return np.inf
        return self.config.beta * (n_samples ** (1.0 / 3.0)) * radius / delta_n


def _partition_points(X: Array, vertices: Array) -> _Partition:
    n_vertices = vertices.shape[0]
    n_segments = n_vertices - 1

    vertex_d2 = np.stack([np.sum((X - v[None, :]) ** 2, axis=1) for v in vertices], axis=1)
    segment_d2_cols: List[Array] = []
    for i in range(n_segments):
        d2, _, _ = _segment_distance_squared(X, vertices[i], vertices[i + 1])
        segment_d2_cols.append(d2)

    if segment_d2_cols:
        segment_d2 = np.stack(segment_d2_cols, axis=1)
        # Vertices are concatenated before segments, so exact ties go to vertices.
        all_d2 = np.concatenate([vertex_d2, segment_d2], axis=1)
    else:
        all_d2 = vertex_d2

    labels = np.argmin(all_d2, axis=1)
    vertex_indices = [np.flatnonzero(labels == i) for i in range(n_vertices)]
    segment_indices = [np.flatnonzero(labels == (n_vertices + i)) for i in range(n_segments)]
    return _Partition(vertex_indices=vertex_indices, segment_indices=segment_indices, point_labels=labels)


def _cos_gamma(vertices: Array, idx: int) -> float:
    if idx <= 0 or idx >= vertices.shape[0] - 1:
        return 1.0

    left = vertices[idx - 1] - vertices[idx]
    right = vertices[idx + 1] - vertices[idx]
    nl = float(np.linalg.norm(left))
    nr = float(np.linalg.norm(right))
    if nl <= 1e-15 or nr <= 1e-15:
        return 1.0

    value = float(np.dot(left, right) / (nl * nr))
    return float(np.clip(value, -1.0, 1.0))


def _cp_term(vertices: Array, cp_index: int, radius: float) -> float:
    last = vertices.shape[0] - 1
    if cp_index == 0:
        return float(np.sum((vertices[0] - vertices[1]) ** 2))
    if cp_index == last:
        return float(np.sum((vertices[last - 1] - vertices[last]) ** 2))
    return float((radius ** 2) * (1.0 + _cos_gamma(vertices, cp_index)))


def _all_cp_terms(vertices: Array, radius: float) -> Array:
    return np.array([_cp_term(vertices, i, radius) for i in range(vertices.shape[0])], dtype=float)


def _penalty_sum(vertices: Array, radius: float) -> float:
    return float(np.sum(_all_cp_terms(vertices, radius)))


def _fixed_partition_data_sum(X: Array, vertices: Array, partition: _Partition) -> float:
    total = 0.0
    for j, idx in enumerate(partition.vertex_indices):
        if idx.size:
            total += float(np.sum((X[idx] - vertices[j][None, :]) ** 2))
    for j, idx in enumerate(partition.segment_indices):
        if idx.size:
            d2, _, _ = _segment_distance_squared(X[idx], vertices[j], vertices[j + 1])
            total += float(np.sum(d2))
    return total


def _fixed_partition_objective(
    X: Array,
    vertices: Array,
    partition: _Partition,
    lambda_p: float,
    radius: float,
) -> float:
    n = X.shape[0]
    k = vertices.shape[0] - 1
    data_term = _fixed_partition_data_sum(X, vertices, partition) / max(n, 1)
    penalty_term = (lambda_p / max(k + 1, 1)) * _penalty_sum(vertices, radius)
    return float(data_term + penalty_term)


def _affected_cp_indices(vertex_index: int, n_vertices: int) -> List[int]:
    affected = [vertex_index - 1, vertex_index, vertex_index + 1]
    return [i for i in affected if 0 <= i < n_vertices]


def _local_data_sum_exact(
    X: Array,
    vertices: Array,
    partition: _Partition,
    vertex_index: int,
    y: Array,
) -> float:
    vv = np.asarray(vertices, dtype=float).copy()
    vv[vertex_index] = np.asarray(y, dtype=float)
    k = vv.shape[0] - 1
    total = 0.0

    v_idx = partition.vertex_indices[vertex_index]
    if v_idx.size:
        total += float(np.sum((X[v_idx] - vv[vertex_index][None, :]) ** 2))

    if vertex_index > 0:
        s_idx = partition.segment_indices[vertex_index - 1]
        if s_idx.size:
            d2, _, _ = _segment_distance_squared(X[s_idx], vv[vertex_index - 1], vv[vertex_index])
            total += float(np.sum(d2))

    if vertex_index < k:
        s_idx = partition.segment_indices[vertex_index]
        if s_idx.size:
            d2, _, _ = _segment_distance_squared(X[s_idx], vv[vertex_index], vv[vertex_index + 1])
            total += float(np.sum(d2))

    return total


def _local_penalty_sum_exact(vertices: Array, vertex_index: int, y: Array, radius: float) -> float:
    vv = np.asarray(vertices, dtype=float).copy()
    vv[vertex_index] = np.asarray(y, dtype=float)
    affected = _affected_cp_indices(vertex_index, vv.shape[0])
    return float(sum(_cp_term(vv, idx, radius) for idx in affected))


def _local_vertex_objective_exact(
    X: Array,
    vertices: Array,
    partition: _Partition,
    vertex_index: int,
    y: Array,
    lambda_p: float,
    radius: float,
) -> float:
    n = X.shape[0]
    k = vertices.shape[0] - 1
    data_term = _local_data_sum_exact(X, vertices, partition, vertex_index, y) / max(n, 1)
    penalty_term = (lambda_p / max(k + 1, 1)) * _local_penalty_sum_exact(vertices, vertex_index, y, radius)
    return float(data_term + penalty_term)


def _finite_difference_gradient(func, y: Array, eps: float) -> Array:
    grad = np.zeros_like(y)
    scale = eps * (1.0 + np.linalg.norm(y))
    for j in range(y.shape[0]):
        step = np.zeros_like(y)
        step[j] = scale
        f_plus = func(y + step)
        f_minus = func(y - step)
        grad[j] = (f_plus - f_minus) / (2.0 * scale)
    return grad


def _directional_curvature(func, y: Array, direction: Array, eps: float) -> float:
    h = eps * (1.0 + np.linalg.norm(y))
    f0 = func(y)
