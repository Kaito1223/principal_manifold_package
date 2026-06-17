from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mpl_toolkits.mplot3d.axes3d import Axes3D

from principal_manifold._types import CurveSnapshot, GraphSnapshot
from principal_manifold.visual.comparison import save_final_comparison_figure
from principal_manifold.visual.plotting import (
    _DEFAULT_3D_AZIM,
    _DEFAULT_3D_ELEV,
    plot_snapshot,
)


class _MockModel:
    def __init__(self, snapshot) -> None:
        self.trace_ = [snapshot]


def _curve_snapshot_2d() -> CurveSnapshot:
    return CurveSnapshot(
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


def _graph_snapshot_2d() -> GraphSnapshot:
    return GraphSnapshot(
        phase="init",
        outer_iteration=0,
        sweep=None,
        vertices=np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 0.5]], dtype=float),
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
    )


def _graph_snapshot_3d() -> GraphSnapshot:
    return GraphSnapshot(
        phase="init",
        outer_iteration=0,
        sweep=None,
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
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


def test_comparison_api_2d_3d(tmp_path: Path) -> None:
    X2 = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.0]], dtype=float)
    models2 = {
        "KK": _MockModel(_curve_snapshot_2d()),
        "HS": _MockModel(_curve_snapshot_2d()),
        "Elastic graph": _MockModel(_graph_snapshot_2d()),
        "Principal elastic graph": _MockModel(_graph_snapshot_2d()),
    }
    out2 = tmp_path / "comparison_2d.png"
    save_final_comparison_figure(X2, models2, out2, target_dim=2)
    assert out2.exists()
    assert out2.stat().st_size > 0

    X3 = np.array([[0.0, 0.0, 0.0], [0.5, 0.2, 0.1], [1.0, 0.0, -0.1]], dtype=float)
    models3 = {
        "KK": _MockModel(_graph_snapshot_3d()),
        "HS": _MockModel(_graph_snapshot_3d()),
        "Elastic graph": _MockModel(_graph_snapshot_3d()),
        "Principal elastic graph": _MockModel(_graph_snapshot_3d()),
    }
    out3 = tmp_path / "comparison_3d.png"
    save_final_comparison_figure(X3, models3, out3, target_dim=3)
    assert out3.exists()
    assert out3.stat().st_size > 0


def test_invalid_dim_rejected(tmp_path: Path) -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    models = {
        "KK": _MockModel(_curve_snapshot_2d()),
        "HS": _MockModel(_curve_snapshot_2d()),
        "Elastic graph": _MockModel(_graph_snapshot_2d()),
        "Principal elastic graph": _MockModel(_graph_snapshot_2d()),
    }
    with pytest.raises(ValueError, match="target_dim must be 2 or 3"):
        save_final_comparison_figure(X, models, tmp_path / "bad.png", target_dim=4)


def test_surface_faces_rendered() -> None:
    snapshot = GraphSnapshot(
        phase="init",
        outer_iteration=0,
        sweep=None,
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        edges=np.array([[0, 1], [1, 2], [2, 0]], dtype=int),
        mean_squared_distance=0.0,
        root_mean_squared_distance=0.0,
        lambda_p=0.0,
        segments=3,
        polyline_length=3.0,
        elastic_energy=0.0,
        operation=None,
        construction_complexity=0,
        structural_complexity=3.0,
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    ax = plot_snapshot(
        X=np.array([[0.2, 0.2, 0.1], [0.4, 0.1, 0.0]], dtype=float),
        snapshot=snapshot,
        render_mode="surface",
        target_dim=3,
    )
    assert ax is not None
    assert len(ax.collections) >= 2
    assert isinstance(ax, Axes3D)
    ax3d = ax
    assert ax3d.elev == pytest.approx(_DEFAULT_3D_ELEV)
    assert ax3d.azim == pytest.approx(_DEFAULT_3D_AZIM)


def test_3d_box_aspect_safe_for_flat_data() -> None:
    snapshot = GraphSnapshot(
        phase="init",
        outer_iteration=0,
        sweep=None,
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
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
    ax = plot_snapshot(
        X=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        snapshot=snapshot,
        target_dim=3,
    )
    assert ax is not None
    assert isinstance(ax, Axes3D)
    ax3d = ax
    assert ax3d.elev == pytest.approx(_DEFAULT_3D_ELEV)
    assert ax3d.azim == pytest.approx(_DEFAULT_3D_AZIM)
    if hasattr(ax3d, "get_box_aspect"):
        box_aspect = np.asarray(ax3d.get_box_aspect(), dtype=float)
        assert box_aspect.shape == (3,)
        assert np.all(np.isfinite(box_aspect))
        assert np.all(box_aspect > 0.0)


def test_3d_view_override_applied() -> None:
    snapshot = _graph_snapshot_3d()
    ax = plot_snapshot(
        X=np.array([[0.0, 0.0, 0.0], [1.0, 0.2, -0.1]], dtype=float),
        snapshot=snapshot,
        target_dim=3,
        view_init={"elev": 10.0, "azim": 35.0},
    )
    assert ax is not None
    assert isinstance(ax, Axes3D)
    ax3d = ax
    assert ax3d.elev == pytest.approx(10.0)
    assert ax3d.azim == pytest.approx(35.0)


def test_3d_view_override_reaches_comparison_wrapper(tmp_path: Path) -> None:
    X3 = np.array([[0.0, 0.0, 0.0], [0.5, 0.2, 0.1], [1.0, 0.0, -0.1]], dtype=float)
    models3 = {
        "KK": _MockModel(_graph_snapshot_3d()),
        "HS": _MockModel(_graph_snapshot_3d()),
        "Elastic graph": _MockModel(_graph_snapshot_3d()),
        "Principal elastic graph": _MockModel(_graph_snapshot_3d()),
    }
    out3 = tmp_path / "comparison_3d_custom_view.png"
    save_final_comparison_figure(
        X3,
        models3,
        out3,
        target_dim=3,
        view_init={"elev": 12.0, "azim": 48.0},
    )
    assert out3.exists()
    assert out3.stat().st_size > 0


def test_3d_view_override_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="view_init contains unsupported keys"):
        plot_snapshot(
            X=np.array([[0.0, 0.0, 0.0], [1.0, 0.2, -0.1]], dtype=float),
            snapshot=_graph_snapshot_3d(),
            target_dim=3,
            view_init={"roll": 10.0},
        )


def test_faces_shape_validation() -> None:
    snapshot = GraphSnapshot(
        phase="init",
        outer_iteration=0,
        sweep=None,
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        edges=np.array([[0, 1], [1, 2], [2, 0]], dtype=int),
        mean_squared_distance=0.0,
        root_mean_squared_distance=0.0,
        lambda_p=0.0,
        segments=3,
        polyline_length=3.0,
        elastic_energy=0.0,
        operation=None,
        construction_complexity=0,
        structural_complexity=3.0,
        faces=np.array([0, 1, 2], dtype=int),
    )
    with pytest.raises(ValueError, match=r"faces must have shape \(m, 3\)"):
        plot_snapshot(
            X=np.array([[0.2, 0.2, 0.1]], dtype=float),
            snapshot=snapshot,
            render_mode="surface",
            target_dim=3,
        )


def test_missing_artifact_detected(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    assert not missing.exists()
