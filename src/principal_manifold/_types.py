from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

Array = np.ndarray

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


@dataclass
class PrincipalGraphResult:
    vertices: Array
    edges: Array
    projected_points: Array
    history: List[Dict[str, float]]
    trace: List[GraphSnapshot]



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
    )
