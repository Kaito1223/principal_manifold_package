from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

Array = np.ndarray

# Intrinsic-manifold symbol contract.
# N: number of observed samples (N >= 1)
# M: number of manifold vertices/nodes (M >= 1)
# d: ambient-space dimension (d >= 1)
# k: intrinsic manifold dimension (1 <= k <= d)
# Y: manifold vertices, shape (M, d), finite float values expected
# U: intrinsic coordinates for observed samples, shape (N, k), finite float values expected
# topology_cells[q]: q-dimensional intrinsic cells encoded by integer vertex indices.
#   Contracted shape: topology_cells[q].shape == (n_q, q + 1), dtype integer,
#   with each entry in [0, M). In particular:
#   - k = 1 (curve/graph chain): topology_cells[1] is edge/segment list.
#   - k = 2 (surface-like): topology_cells[2] is face list (typically triangles).
IntrinsicData = Array
IntrinsicCoordinates = Array
ManifoldVertices = Array
TopologyCellList = List[Array]


@dataclass(frozen=True)
class IntrinsicManifoldContract:
    """Named dimensions/symbols for intrinsic-manifold outputs.

    This type is a documentation-and-typing contract only. It does not enforce
    runtime validation and does not change existing estimator behavior.

    Invariants (assumed by downstream code):
    - N, M, d are positive integers
    - k is a positive integer with k <= d
    - Y has shape (M, d) and finite float values
    - U has shape (N, k) and finite float values
    - topology_cells indexes vertices in [0, M)
    """

    N: int
    M: int
    d: int
    k: int

@dataclass
class CurveSnapshot:
    # Kept intentionally identical to the KK snapshot structure so that the
    # same visual.py works without any changes.
    phase: str
    outer_iteration: int
    sweep: Optional[int]
    vertices: Array
    mean_squared_distance: float
    root_mean_squared_distance: float
    lambda_p: float
    segments: int
    polyline_length: float


@dataclass
class PrincipalCurveResult:
    # Kept intentionally identical to the KK result structure.
    vertices: Array
    projected_points: Array
    # 1D compatibility field. For intrinsic contract this corresponds to U[:, 0]
    # when k = 1.
    arc_length_coordinates: Array
    nearest_object_kind: Array
    nearest_object_index: Array
    history: List[Dict[str, float]]
    trace: List[CurveSnapshot]


@dataclass
class GraphSnapshot:
    phase: str
    outer_iteration: int
    sweep: Optional[int]
    vertices: Array
    edges: Array
    mean_squared_distance: float
    root_mean_squared_distance: float
    lambda_p: float
    segments: int
    polyline_length: float
    elastic_energy: float
    operation: Optional[str]
    construction_complexity: int
    structural_complexity: float
    faces: Optional[Array] = None
    # Optional intrinsic-k topology cells keyed by intrinsic dimension.
    # Each value stores integer simplices over vertex indices.
    cells_by_dim: Optional[Dict[int, Array]] = None


@dataclass
class PrincipalGraphResult:
    vertices: Array
    edges: Array
    projected_points: Array
    history: List[Dict[str, float]]
    trace: List[GraphSnapshot]
    faces: Optional[Array] = None
    # Optional intrinsic-k topology cells keyed by intrinsic dimension.
    cells_by_dim: Optional[Dict[int, Array]] = None


def _copy_topology_cells(
    cells_by_dim: Optional[Dict[int, Array]],
) -> Optional[Dict[int, Array]]:
    if cells_by_dim is None:
        return None
    return {
        int(dim): np.asarray(cells, dtype=int).copy()
        for dim, cells in cells_by_dim.items()
    }



def copy_curve_snapshot(snapshot: CurveSnapshot) -> CurveSnapshot:
    return CurveSnapshot(
        phase=snapshot.phase,
        outer_iteration=int(snapshot.outer_iteration),
        sweep=None if snapshot.sweep is None else int(snapshot.sweep),
        vertices=np.asarray(snapshot.vertices, dtype=float).copy(),
        mean_squared_distance=float(snapshot.mean_squared_distance),
        root_mean_squared_distance=float(snapshot.root_mean_squared_distance),
        lambda_p=float(snapshot.lambda_p),
        segments=int(snapshot.segments),
        polyline_length=float(snapshot.polyline_length),
    )


def copy_graph_snapshot(snapshot: GraphSnapshot) -> GraphSnapshot:
    return GraphSnapshot(
        phase=str(snapshot.phase),
        outer_iteration=int(snapshot.outer_iteration),
        sweep=None if snapshot.sweep is None else int(snapshot.sweep),
        vertices=np.asarray(snapshot.vertices, dtype=float).copy(),
        edges=np.asarray(snapshot.edges, dtype=int).copy(),
        mean_squared_distance=float(snapshot.mean_squared_distance),
        root_mean_squared_distance=float(snapshot.root_mean_squared_distance),
        lambda_p=float(snapshot.lambda_p),
        segments=int(snapshot.segments),
        polyline_length=float(snapshot.polyline_length),
        elastic_energy=float(snapshot.elastic_energy),
        operation=None if snapshot.operation is None else str(snapshot.operation),
        construction_complexity=int(snapshot.construction_complexity),
        structural_complexity=float(snapshot.structural_complexity),
        faces=None if snapshot.faces is None else np.asarray(snapshot.faces, dtype=int).copy(),
        cells_by_dim=_copy_topology_cells(snapshot.cells_by_dim),
    )
