from __future__ import annotations

import numpy as np
import pytest

from principal_manifold._types import CurveSnapshot, GraphSnapshot
from principal_manifold.visual import plot_snapshot
from principal_manifold.visual._utils import _infer_render_mode, _normalize_render_mode


def test_legacy_routing_compatibility() -> None:
    curve_snapshot = CurveSnapshot(
        phase="init",
        outer_iteration=0,
        sweep=None,
        vertices=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        mean_squared_distance=0.0,
        root_mean_squared_distance=0.0,
        lambda_p=0.0,
        segments=1,
        polyline_length=1.0,
    )
    assert _infer_render_mode(method_name="HS", snapshot=curve_snapshot) == "CURVE"
    assert _infer_render_mode(method_name="KK", snapshot=curve_snapshot) == "CURVE"
    graph_snapshot = GraphSnapshot(
        phase="init",
        outer_iteration=0,
        sweep=None,
        vertices=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        edges=np.array([[0, 1]], dtype=int),
        mean_squared_distance=0.0,
        root_mean_squared_distance=0.0,
        lambda_p=0.0,
        segments=1,
        polyline_length=1.0,
        elastic_energy=0.0,
        operation=None,
        construction_complexity=0,
        structural_complexity=2.0,
    )
    assert _infer_render_mode(method_name="ELASTIC_MAP", snapshot=graph_snapshot) == "GRAPH"
    assert _infer_render_mode(method_name="PRINCIPAL_ELASTIC_GRAPH", snapshot=graph_snapshot) == "GRAPH"


def test_intrinsic_mode_selection() -> None:
    snapshot = GraphSnapshot(
        phase="init",
        outer_iteration=0,
        sweep=None,
        vertices=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        edges=np.array([[0, 1], [1, 2]], dtype=int),
        mean_squared_distance=0.0,
        root_mean_squared_distance=0.0,
        lambda_p=0.0,
        segments=2,
        polyline_length=2.0,
        elastic_energy=0.0,
        operation=None,
        construction_complexity=0,
        structural_complexity=3.0,
        faces=np.array([[0, 1, 2]], dtype=int),
        cells_by_dim={2: np.array([[0, 1, 2]], dtype=int)},
    )
    assert _infer_render_mode(snapshot=snapshot) == "INTRINSIC"
    ax = plot_snapshot(
        X=np.array([[0.2, 0.2], [0.8, 0.1]], dtype=float),
        snapshot=snapshot,
        render_mode="intrinsic",
        target_dim=2,
    )
    assert ax is not None


def test_invalid_render_mode() -> None:
    with pytest.raises(ValueError, match="render_mode must be one of"):
        _normalize_render_mode("bad-mode")
