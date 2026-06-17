from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._types import (
    CurveSnapshot,
    GraphSnapshot,
    PrincipalCurveResult,
    PrincipalGraphResult,
    copy_curve_snapshot as _copy_snapshot,
    copy_graph_snapshot as _copy_graph_snapshot,
)
from ..geometry import (
    _as_2d_float_array,
    _choose_window_size,
    _dataset_radius,
    _initialize_intrinsic_coordinates_on_first_k_pcs,
    _mean_squared_distance_to_graph,
    _mean_squared_distance_to_polyline,
    _mean_squared_distance_to_surface,
    _pca_init_line,
    _polyline_lambda_from_segment_parameter,
    _polyline_length,
    _project_onto_complex,
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
    Configuration for finite-sample HS principal-curve algorithm.

    Parameters
    ----------
    w
        Neighborhood fraction. Window size computed as k=floor(w*N),
        then adjusted to be at least 3 and odd whenever possible.
    max_iter
        Maximum number of outer HS iterations.
    tol
        Relative MSD improvement stopping tolerance.
    verbose
        If True, print one diagnostics record per iteration.
    store_trace
        If True, store curve snapshots in ``trace_`` with same structure as
        KK framework so same visual.py can be reused unchanged.
    """

    w: float = 0.10
    max_iter: int = 10
    tol: float = 1e-4
    verbose: bool = False
    store_trace: bool = False


@dataclass
class IntrinsicHSConfig:
    intrinsic_dim: int = 2
    grid_shape: Optional[Tuple[int, ...]] = None
    w: float = 0.10
    max_iter: int = 10
    tol: float = 1e-4
    verbose: bool = False
    store_trace: bool = False
    ridge: float = 1e-10


class HSPrincipalCurve:
    """
    Finite-sample HS principal-curve estimator rewritten in same class-based
    style as KK framework.

    This implementation preserves algorithmic structure of uploaded HS
    script:
      1. initialize all nodes on PCA line,
      2. project points onto current polyline,
      3. convert projections into updated internal coordinates,
      4. build exact contiguous k-neighborhood windows in sorted lambda order,
      5. perform weighted local linear regression at each lambda,
      6. rebuild polyline and repeat until convergence.

    Trace snapshots intentionally use same field names as KK framework, so same
    visual.py can be used directly.
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

    def fit(self, X: Array) -> 'HSPrincipalCurve':
        X = _as_2d_float_array(X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError('HS principal curve requires at least two samples.')
        if n_features < 1:
            raise ValueError('Input must have at least one feature.')

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
                phase='init',
                outer_iteration=0,
                sweep=None,
                mean_squared_distance=init_msd,
            )

        prev_msd = init_msd

        for outer_iteration in range(1, self.config.max_iter + 1):
            if self.vertices_ is None or self._lambda_nodes_ is None:
                raise RuntimeError('Internal state corrupted: fit initialization failed.')

            t, projected, d2 = _project_to_polyline_parameter(X, self.vertices_)
            projected_msd = float(np.mean(d2))
            if self.config.store_trace:
                self._append_trace(
                    X,
                    self.vertices_,
                    phase='projected',
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
                'iteration': float(outer_iteration),
                'k_neighbors': float(k_neighbors),
                'mean_squared_distance': float(updated_msd),
                'root_mean_squared_distance': float(np.sqrt(max(updated_msd, 0.0))),
                'polyline_length': float(_polyline_length(self.vertices_)),
                'relative_improvement': float(rel_improvement),
            }
            self.history_.append(record)
            if self.config.verbose:
                print(record)

            if self.config.store_trace:
                self._append_trace(
                    X,
                    self.vertices_,
                    phase='updated',
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
            raise AttributeError('Model has not been fitted yet.')
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
            raise ValueError('Call fit before project.')
        X = _as_2d_float_array(X)
        return _project_onto_polyline(X, self.vertices_)

    def transform(self, X: Array) -> Array:
        _, arc_coords, _, _ = self.project(X)
        return arc_coords

    def score(self, X: Array) -> float:
        if self.vertices_ is None:
            raise ValueError('Call fit before score.')
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


class IntrinsicHSPrincipalManifold:
    """Fixed-k HS-style intrinsic manifold estimator.

    Keeps legacy HS curve implementation untouched. Uses multivariate latent
    coordinates, regular intrinsic grid, intrinsic-space kNN neighborhoods, and
    centered weighted local affine regression with singular fallback.
    """

    def __init__(self, config: Optional[IntrinsicHSConfig] = None) -> None:
        self.config = config or IntrinsicHSConfig()
        _validate_intrinsic_hs_config(self.config)
        self.vertices_: Optional[Array] = None
        self.intrinsic_vertices_: Optional[Array] = None
        self.cells_by_dim_: Optional[Dict[int, Array]] = None
        self.history_: List[Dict[str, float]] = []
        self.trace_: List[GraphSnapshot] = []
        self._X_fit_: Optional[Array] = None
        self._faces: Array = np.empty((0, 3), dtype=int)

    def fit(self, X: Array) -> 'IntrinsicHSPrincipalManifold':
        X = _as_2d_float_array(X)
        _require_all_finite('X', X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError('Intrinsic HS principal manifold requires at least two samples.')
        if self.config.intrinsic_dim > n_features:
            raise ValueError(f'intrinsic_dim must be <= n_features ({n_features}).')

        self._X_fit_ = X.copy()
        self.history_ = []
        self.trace_ = []

        U_samples = _initialize_intrinsic_coordinates_on_first_k_pcs(X, self.config.intrinsic_dim)
        _require_all_finite('intrinsic_samples', U_samples)
        grid_shape = _resolved_grid_shape(self.config)
        intrinsic_vertices = _build_intrinsic_grid(_intrinsic_bounds(U_samples), grid_shape)
        cells_by_dim = _build_cells_by_dim(grid_shape)
        self.intrinsic_vertices_ = intrinsic_vertices.copy()
        self.cells_by_dim_ = _copy_cells_by_dim(cells_by_dim)
        self._faces = np.asarray(cells_by_dim.get(2, np.empty((0, 3), dtype=int)), dtype=int)

        init_neighbors = _intrinsic_knn_indices(intrinsic_vertices, U_samples, _intrinsic_window_size(self.config.w, n_samples, self.config.intrinsic_dim))
        self.vertices_ = _hs_update_local_affine_intrinsic(
            centers=intrinsic_vertices,
            samples_u=U_samples,
            samples_x=X,
            neighbor_indices=init_neighbors,
            ridge=float(self.config.ridge),
        )
        _require_all_finite('vertices_', self.vertices_)

        init_msd = self._current_msd(X)
        if self.config.store_trace:
            self._append_trace(phase='init', outer_iteration=0, sweep=None, mean_squared_distance=init_msd)

        prev_msd = init_msd
        for outer_iteration in range(1, self.config.max_iter + 1):
            if self.intrinsic_vertices_ is None or self.vertices_ is None:
                raise RuntimeError('Internal state corrupted: fit initialization failed.')

            projected = self.project(X)
            projected_msd = self._current_msd(X)
            if self.config.store_trace:
                self._append_trace(
                    phase='projected',
                    outer_iteration=outer_iteration,
                    sweep=None,
                    mean_squared_distance=projected_msd,
                )

            projected_u = _project_samples_to_intrinsic_vertices(projected, self.vertices_, self.intrinsic_vertices_)
            neighbors = _intrinsic_knn_indices(
                self.intrinsic_vertices_,
                projected_u,
                _intrinsic_window_size(self.config.w, n_samples, self.config.intrinsic_dim),
            )
            new_vertices = _hs_update_local_affine_intrinsic(
                centers=self.intrinsic_vertices_,
                samples_u=projected_u,
                samples_x=X,
                neighbor_indices=neighbors,
                ridge=float(self.config.ridge),
            )
            _require_all_finite('new_vertices', new_vertices)
            self.vertices_ = new_vertices

            updated_msd = self._current_msd(X)
            rel_improvement = (prev_msd - updated_msd) / max(abs(prev_msd), 1e-15)
            record = {
                'iteration': float(outer_iteration),
                'intrinsic_dim': float(self.config.intrinsic_dim),
                'nodes': float(self.vertices_.shape[0]),
                'k_neighbors': float(neighbors.shape[1]),
                'mean_squared_distance': float(updated_msd),
                'root_mean_squared_distance': float(np.sqrt(max(updated_msd, 0.0))),
                'polyline_length': float(_polyline_length(self.vertices_)),
                'relative_improvement': float(rel_improvement),
            }
            self.history_.append(record)
            if self.config.verbose:
                print(record)

            if self.config.store_trace:
                self._append_trace(
                    phase='updated',
                    outer_iteration=outer_iteration,
                    sweep=None,
                    mean_squared_distance=updated_msd,
                )

            if rel_improvement < self.config.tol:
                break
            prev_msd = updated_msd

        return self

    def fit_result(self, X: Array) -> PrincipalGraphResult:
        self.fit(X)
        return self.result_

    @property
    def result_(self) -> PrincipalGraphResult:
        if self.vertices_ is None or self._X_fit_ is None or self.cells_by_dim_ is None:
            raise AttributeError('Model has not been fitted yet.')
        edges = np.asarray(self.cells_by_dim_.get(1, np.empty((0, 2), dtype=int)), dtype=int)
        return PrincipalGraphResult(
            vertices=self.vertices_.copy(),
            edges=edges.copy(),
            projected_points=self.project(self._X_fit_),
            history=[dict(item) for item in self.history_],
            trace=[_copy_graph_snapshot(item) for item in self.trace_],
            faces=None if self._faces.size == 0 else self._faces.copy(),
            cells_by_dim=_copy_cells_by_dim(self.cells_by_dim_),
        )

    def project(self, X: Array) -> Array:
        if self.vertices_ is None or self.cells_by_dim_ is None:
            raise ValueError('Call fit before project.')
        X = _as_2d_float_array(X)
        projected = _project_onto_complex(
            X,
            self.vertices_,
            edge_index_pairs=_edge_pairs_from_cells(self.cells_by_dim_),
            faces=None if self._faces.size == 0 else self._faces,
            prefer_dim=2 if self._faces.size > 0 else 1,
        )
        _require_all_finite('projected_points', projected)
        return projected

    def transform(self, X: Array) -> Array:
        if self.vertices_ is None or self.intrinsic_vertices_ is None:
            raise ValueError('Call fit before transform.')
        X = _as_2d_float_array(X)
        projected = self.project(X)
        return _project_samples_to_intrinsic_vertices(projected, self.vertices_, self.intrinsic_vertices_)

    def score(self, X: Array) -> float:
        if self.vertices_ is None or self.cells_by_dim_ is None:
            raise ValueError('Call fit before score.')
        X = _as_2d_float_array(X)
        return -self._current_msd(X)

    def _current_msd(self, X: Array) -> float:
        if self.vertices_ is None or self.cells_by_dim_ is None:
            raise ValueError('Model has not been initialized.')
        faces = np.asarray(self.cells_by_dim_.get(2, np.empty((0, 3), dtype=int)), dtype=int)
        edges = _edge_pairs_from_cells(self.cells_by_dim_)
        if faces.size > 0:
            return _mean_squared_distance_to_surface(X, self.vertices_, faces)
        return _mean_squared_distance_to_graph(X, self.vertices_, edges)

    def _append_trace(
        self,
        phase: str,
        outer_iteration: int,
        sweep: Optional[int],
        mean_squared_distance: Optional[float] = None,
    ) -> None:
        if self.vertices_ is None or self.cells_by_dim_ is None:
            raise ValueError('No manifold state for trace.')
        msd = self._current_msd(self._X_fit_) if mean_squared_distance is None else float(mean_squared_distance)
        edges = np.asarray(self.cells_by_dim_.get(1, np.empty((0, 2), dtype=int)), dtype=int)
        self.trace_.append(
            GraphSnapshot(
                phase=phase,
                outer_iteration=int(outer_iteration),
                sweep=None if sweep is None else int(sweep),
                vertices=self.vertices_.copy(),
                edges=edges.copy(),
                mean_squared_distance=float(msd),
                root_mean_squared_distance=float(np.sqrt(max(msd, 0.0))),
                lambda_p=0.0,
                segments=int(edges.shape[0]),
                polyline_length=float(_polyline_length(self.vertices_)),
                elastic_energy=0.0,
                operation='hs_intrinsic',
                construction_complexity=int(outer_iteration),
                structural_complexity=float(self.vertices_.shape[0]),
                faces=None if self._faces.size == 0 else self._faces.copy(),
                cells_by_dim=_copy_cells_by_dim(self.cells_by_dim_),
            )
        )


@njit

def hs_windows_sorted(lam_sorted: np.ndarray, k: int):
    """For each point i in sorted lambda order, choose exact contiguous
    window [L, R] of size k minimizing maximum left/right lambda radius."""
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
    lambda using HS compact-support kernel."""
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


def _validate_intrinsic_hs_config(config: IntrinsicHSConfig) -> None:
    if int(config.intrinsic_dim) < 1:
        raise ValueError('intrinsic_dim must be at least 1.')
    if config.grid_shape is not None:
        if len(config.grid_shape) != int(config.intrinsic_dim):
            raise ValueError('grid_shape length must match intrinsic_dim.')
        if any(int(count) < 2 for count in config.grid_shape):
            raise ValueError('grid_shape entries must be at least 2.')
    if float(config.w) <= 0.0:
        raise ValueError('w must be positive.')
    if int(config.max_iter) < 1:
        raise ValueError('max_iter must be at least 1.')
    if float(config.ridge) < 0.0:
        raise ValueError('ridge must be nonnegative.')


def _resolved_grid_shape(config: IntrinsicHSConfig) -> Tuple[int, ...]:
    if config.grid_shape is not None:
        return tuple(int(count) for count in config.grid_shape)
    return tuple(3 for _ in range(int(config.intrinsic_dim)))


def _intrinsic_bounds(U: Array) -> List[Tuple[float, float]]:
    U = np.asarray(U, dtype=float)
    bounds: List[Tuple[float, float]] = []
    for dim in range(U.shape[1]):
        bounds.append((float(np.min(U[:, dim])), float(np.max(U[:, dim]))))
    return bounds


def _build_intrinsic_grid(bounds: Sequence[Tuple[float, float]], grid_shape: Tuple[int, ...]) -> Array:
    axes = [np.linspace(lo, hi, int(count)) for (lo, hi), count in zip(bounds, grid_shape)]
    mesh = np.meshgrid(*axes, indexing='ij')
    return np.stack([axis.ravel() for axis in mesh], axis=1)


def _build_cells_by_dim(grid_shape: Tuple[int, ...]) -> Dict[int, Array]:
    shape = tuple(int(count) for count in grid_shape)
    total = int(np.prod(shape))
    if total < 1:
        raise ValueError('grid_shape must define at least one node.')
    index_grid = np.arange(total, dtype=int).reshape(shape)
    edges: List[Tuple[int, int]] = []
    for dim in range(len(shape)):
        slicer_a = [slice(None)] * len(shape)
        slicer_b = [slice(None)] * len(shape)
        slicer_a[dim] = slice(0, shape[dim] - 1)
        slicer_b[dim] = slice(1, shape[dim])
        a = index_grid[tuple(slicer_a)].ravel()
        b = index_grid[tuple(slicer_b)].ravel()
        edges.extend((int(i), int(j)) for i, j in zip(a, b))

    cells_by_dim: Dict[int, Array] = {1: np.asarray(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)}
    if len(shape) == 2:
        faces: List[Tuple[int, int, int]] = []
        for i in range(shape[0] - 1):
            for j in range(shape[1] - 1):
                v00 = int(index_grid[i, j])
                v10 = int(index_grid[i + 1, j])
                v01 = int(index_grid[i, j + 1])
                v11 = int(index_grid[i + 1, j + 1])
                faces.append((v00, v10, v11))
                faces.append((v00, v11, v01))
        cells_by_dim[2] = np.asarray(faces, dtype=int) if faces else np.empty((0, 3), dtype=int)
    return cells_by_dim


def _copy_cells_by_dim(cells_by_dim: Optional[Dict[int, Array]]) -> Optional[Dict[int, Array]]:
    if cells_by_dim is None:
        return None
    return {int(dim): np.asarray(cells, dtype=int).copy() for dim, cells in cells_by_dim.items()}


def _edge_pairs_from_cells(cells_by_dim: Dict[int, Array]) -> List[Tuple[int, int]]:
    edges = np.asarray(cells_by_dim.get(1, np.empty((0, 2), dtype=int)), dtype=int)
    return [(int(i), int(j)) for i, j in edges]


def _intrinsic_window_size(w: float, n_samples: int, intrinsic_dim: int) -> int:
    base = _choose_window_size(w, n_samples)
    return int(max(base, intrinsic_dim + 2))


def _intrinsic_knn_indices(centers: Array, samples_u: Array, k_neighbors: int) -> Array:
    centers = np.asarray(centers, dtype=float)
    samples_u = np.asarray(samples_u, dtype=float)
    k = int(min(max(1, k_neighbors), samples_u.shape[0]))
    d2 = np.sum((centers[:, None, :] - samples_u[None, :, :]) ** 2, axis=2)
    order = np.argsort(d2, axis=1)
    return order[:, :k]


def _hs_kernel_weight(dist: float, radius: float) -> float:
    if radius <= 1e-15:
        return 1.0
    ratio = min(max(dist / radius, 0.0), 1.0)
    return float(max(0.0, 1.0 - ratio ** 3) ** (1.0 / 3.0))


def _weighted_local_mean(X_local: Array, weights: Array) -> Array:
    denom = float(np.sum(weights))
    if denom <= 1e-15:
        return np.mean(X_local, axis=0)
    return (weights[:, None] * X_local).sum(axis=0) / denom


def _hs_update_local_affine_intrinsic(
    centers: Array,
    samples_u: Array,
    samples_x: Array,
    neighbor_indices: Array,
    ridge: float,
) -> Array:
    centers = np.asarray(centers, dtype=float)
    samples_u = np.asarray(samples_u, dtype=float)
    samples_x = np.asarray(samples_x, dtype=float)
    neighbor_indices = np.asarray(neighbor_indices, dtype=int)
    n_centers, intrinsic_dim = centers.shape
    ambient_dim = samples_x.shape[1]
    updated = np.empty((n_centers, ambient_dim), dtype=float)

    for idx in range(n_centers):
        local_idx = neighbor_indices[idx]
        U_local = samples_u[local_idx]
        X_local = samples_x[local_idx]
        center = centers[idx]
        du = U_local - center[None, :]
        distances = np.linalg.norm(du, axis=1)
        radius = float(np.max(distances))

        if radius <= 1e-12:
            updated[idx] = np.mean(X_local, axis=0)
            continue

        weights = np.asarray([_hs_kernel_weight(float(dist), radius) for dist in distances], dtype=float)
        positive = weights > 1e-14
        if int(np.sum(positive)) < intrinsic_dim + 1:
            updated[idx] = _weighted_local_mean(X_local, weights)
            continue

        Z = np.concatenate([np.ones((U_local.shape[0], 1), dtype=float), du], axis=1)
        sqrt_w = np.sqrt(weights)[:, None]
        Z_w = Z * sqrt_w
        X_w = X_local * sqrt_w
        gram = Z_w.T @ Z_w
        gram[1:, 1:] += float(ridge) * np.eye(intrinsic_dim)

        if np.linalg.matrix_rank(gram) < intrinsic_dim + 1:
            updated[idx] = _weighted_local_mean(X_local, weights)
            continue

        try:
            coef = np.linalg.solve(gram, Z_w.T @ X_w)
        except np.linalg.LinAlgError:
            updated[idx] = _weighted_local_mean(X_local, weights)
            continue

        intercept = coef[0]
        if not np.all(np.isfinite(intercept)):
            updated[idx] = _weighted_local_mean(X_local, weights)
            continue
        updated[idx] = intercept

    return updated


def _project_samples_to_intrinsic_vertices(projected: Array, ambient_vertices: Array, intrinsic_vertices: Array) -> Array:
    projected = np.asarray(projected, dtype=float)
    ambient_vertices = np.asarray(ambient_vertices, dtype=float)
    intrinsic_vertices = np.asarray(intrinsic_vertices, dtype=float)
    d2 = np.sum((projected[:, None, :] - ambient_vertices[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(d2, axis=1)
    return intrinsic_vertices[nearest].copy()


def _require_all_finite(name: str, value: Array) -> None:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f'Non-finite {name} encountered.')
