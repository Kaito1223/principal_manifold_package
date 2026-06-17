from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D

from ._utils import (
    _as_float_matrix,
    _build_snapshot_title,
    _get_cached_projection,
    _infer_render_mode,
    _pad_to_dim,
    _polyline_edges,
    _snapshot_edges,
    _supported_plot_dimension,
    _visual_options_for_render_mode,
    _visual_options_for_method,
)

Array = np.ndarray

_DEFAULT_3D_ELEV = 22.0
_DEFAULT_3D_AZIM = -58.0
_VIEW_INIT_KEYS = frozenset({"elev", "azim"})


def _resolve_3d_view_init(view_init: Mapping[str, float] | None) -> tuple[float, float]:
    if view_init is None:
        return _DEFAULT_3D_ELEV, _DEFAULT_3D_AZIM

    unknown_keys = set(view_init) - _VIEW_INIT_KEYS
    if unknown_keys:
        keys = ", ".join(sorted(repr(key) for key in unknown_keys))
        raise ValueError(f"view_init contains unsupported keys: {keys}.")

    elev = float(view_init.get("elev", _DEFAULT_3D_ELEV))
    azim = float(view_init.get("azim", _DEFAULT_3D_AZIM))
    return elev, azim


def _configure_3d_axes(
    ax: Axes3D,
    mins: Array,
    maxs: Array,
    view_init: Mapping[str, float] | None = None,
) -> None:
    span = np.asarray(maxs, dtype=float) - np.asarray(mins, dtype=float)
    safe_span = np.where(span > 1e-12, span, 1.0)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(safe_span)
    elev, azim = _resolve_3d_view_init(view_init)
    ax.view_init(elev=elev, azim=azim)


def _require_axes3d(ax: Axes) -> Axes3D:
    if not isinstance(ax, Axes3D):
        raise TypeError("Expected 3D axes.")
    return ax


def _padded_3d_limits(mins: Array, maxs: Array) -> tuple[Array, Array]:
    mins_arr = np.asarray(mins, dtype=float)
    maxs_arr = np.asarray(maxs, dtype=float)
    span = maxs_arr - mins_arr
    pad = np.where(span > 1e-12, 0.0, 0.5)
    return mins_arr - pad, maxs_arr + pad

def plot_principal_object(
    X,
    vertices,
    edges=None,
    faces=None,
    projected=None,
    ax=None,
    target_dim: Optional[int] = None,
    show_projections: bool = False,
    show_projected_points: bool = False,
    show_structure_segments: bool = True,
    show_structure_vertices: bool = True,
    render_mode: Optional[str] = None,
    title: Optional[str] = None,
    view_init: Mapping[str, float] | None = None,
):
    X = _as_float_matrix("X", X)
    vertices = _as_float_matrix("vertices", vertices)
    if edges is None:
        edges = _polyline_edges(vertices.shape[0])
    else:
        edges = np.asarray(edges, dtype=int)
    faces_arr = None if faces is None else np.asarray(faces, dtype=int)
    if faces_arr is not None and faces_arr.size > 0:
        if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
            raise ValueError("faces must have shape (m, 3).")

    resolved_dim = _supported_plot_dimension(X, vertices, target_dim)
    if resolved_dim is None:
        return None

    Xp = _pad_to_dim(X, resolved_dim)
    Vp = _pad_to_dim(vertices, resolved_dim)
    Pp = None if projected is None else _pad_to_dim(_as_float_matrix("projected", projected), resolved_dim)

    if ax is None:
        if resolved_dim == 3:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
        else:
            _, ax = plt.subplots(figsize=(10, 7))

    if resolved_dim == 2:
        ax2d = ax
        ax2d.scatter(Xp[:, 0], Xp[:, 1], s=24, alpha=0.65, label="data")

        if show_structure_segments and len(edges) > 0:
            segments = np.stack([Vp[edges[:, 0]], Vp[edges[:, 1]]], axis=1)
            line_collection = LineCollection(segments.tolist(), linewidths=2, alpha=0.95, colors="black",)
            ax2d.add_collection(line_collection)

        if show_structure_vertices and len(Vp) > 0:
            ax2d.scatter(
                Vp[:, 0],
                Vp[:, 1],
                s=28,
                alpha=0.9,
                label="structure vertices" if not show_structure_segments else None,
            )

        if Pp is not None:
            if show_projections:
                projection_segments = np.stack([Xp[:, :2], Pp[:, :2]], axis=1)
                line_collection = LineCollection(projection_segments.tolist(), linewidths=0.4, alpha=0.25, color='crimson')
                ax2d.add_collection(line_collection)
            if show_projected_points:
                ax2d.scatter(Pp[:, 0], Pp[:, 1], s=16, alpha=0.75, marker="x", label="projected points")

        ax2d.autoscale()
        ax2d.set_aspect("equal", adjustable="box")
        ax2d.set_xlabel("x1", fontsize = 15)
        ax2d.set_ylabel("x2", fontsize = 15)
    else:
        ax3d = _require_axes3d(ax)
        ax3d.plot(Xp[:, 0], Xp[:, 1], Xp[:, 2], linestyle="", marker="o", markersize=4.5, alpha=0.55, label="data")

        if render_mode in {"SURFACE", "INTRINSIC"} and faces_arr is not None and faces_arr.size > 0:
            ax3d.plot_trisurf(
                Vp[:, 0],
                Vp[:, 1],
                Vp[:, 2],
                triangles=faces_arr,
                color="cornflowerblue",
                alpha=0.35,
                edgecolor="black",
                linewidth=0.5,
            )

        if show_structure_segments and len(edges) > 0:
            segments = np.stack([Vp[edges[:, 0]], Vp[edges[:, 1]]], axis=1)
            line_collection = Line3DCollection(segments.tolist(), linewidths=2, alpha=0.95, colors="black",)
            ax3d.add_collection(line_collection)

        if show_structure_vertices and len(Vp) > 0:
            ax3d.plot(Vp[:, 0], Vp[:, 1], Vp[:, 2], linestyle="", marker="o", markersize=5.0, alpha=0.9, label="structure vertices")

        if Pp is not None and show_projections:
            segs = np.stack([Xp, Pp], axis=1)
            ax3d.add_collection(Line3DCollection(segs.tolist(), linewidths=0.35, alpha=0.22))

        mins = np.min(np.vstack([Xp, Vp]), axis=0)
        maxs = np.max(np.vstack([Xp, Vp]), axis=0)
        mins_plot, maxs_plot = _padded_3d_limits(mins, maxs)
        ax3d.set_xlim(mins_plot[0], maxs_plot[0])
        ax3d.set_ylim(mins_plot[1], maxs_plot[1])
        ax3d.set_zlim(mins_plot[2], maxs_plot[2])
        _configure_3d_axes(ax3d, mins, maxs, view_init=view_init)
        ax3d.set_xlabel("x1", fontsize = 15)
        ax3d.set_ylabel("x2", fontsize = 15)
        ax3d.set_zlabel("x3", fontsize = 15)

    if title is not None:
        ax.set_title(title, fontsize = 17)

    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
    if filtered and resolved_dim == 2:
        ax.legend(*zip(*filtered))

    return ax


def plot_curve_snapshot(
    X,
    snapshot,
    method_name: Optional[str] = None,
    ax=None,
    target_dim: Optional[int] = None,
    show_projections: bool = False,
    render_mode: Optional[str] = None,
    title: Optional[str] = None,
    view_init: Mapping[str, float] | None = None,
):
    X = _as_float_matrix("X", X)
    vertices = _as_float_matrix("snapshot.vertices", snapshot.vertices)
    edges = _snapshot_edges(snapshot)
    faces = getattr(snapshot, "faces", None)
    resolved_render_mode = _infer_render_mode(
        render_mode=render_mode,
        method_name=method_name,
        snapshot=snapshot,
        edges=edges,
        faces=faces,
        cells_by_dim=getattr(snapshot, "cells_by_dim", None),
    )
    if method_name is None:
        options = _visual_options_for_render_mode(resolved_render_mode, show_projections)
    else:
        options = _visual_options_for_render_mode(
            resolved_render_mode,
            show_projections,
            method_name=method_name,
        )
    projected = _get_cached_projection(X, snapshot, vertices) if options["need_projection"] else None

    if title is None:
        title = _build_snapshot_title(snapshot)

    return plot_principal_object(
        X,
        vertices,
        edges=edges,
        faces=faces,
        projected=projected,
        ax=ax,
        target_dim=target_dim,
        show_projections=options["show_projections"],
        show_projected_points=options["show_projected_points"],
        show_structure_segments=options["show_structure_segments"],
        show_structure_vertices=options["show_structure_vertices"],
        render_mode=resolved_render_mode,
        title=title,
        view_init=view_init,
    )


def plot_snapshot(
    X,
    snapshot,
    render_mode: Optional[str] = None,
    method_name: Optional[str] = None,
    ax=None,
    target_dim: Optional[int] = None,
    show_projections: bool = False,
    title: Optional[str] = None,
    view_init: Mapping[str, float] | None = None,
):
    return plot_curve_snapshot(
        X=X,
        snapshot=snapshot,
        method_name=method_name,
        ax=ax,
        target_dim=target_dim,
        show_projections=show_projections,
        render_mode=render_mode,
        title=title,
        view_init=view_init,
    )


def plot_curve_trace_grid(
    X,
    snapshots: Sequence,
    method_name: Optional[str] = None,
    cols: int = 3,
    max_panels: Optional[int] = None,
    figsize=None,
    target_dim: Optional[int] = None,
    show_projections: bool = False,
    render_mode: Optional[str] = None,
    view_init: Mapping[str, float] | None = None,
):
    if len(snapshots) == 0:
        raise ValueError("snapshots must contain at least one snapshot.")

    X = _as_float_matrix("X", X)
    sample_vertices = _as_float_matrix("snapshot.vertices", snapshots[0].vertices)
    resolved_dim = _supported_plot_dimension(X, sample_vertices, target_dim)
    if resolved_dim is None:
        return None, None

    selected = list(snapshots)
    if max_panels is not None and len(selected) > max_panels:
        idx = np.linspace(0, len(selected) - 1, max_panels, dtype=int)
        selected = [selected[i] for i in idx]

    cols = max(int(cols), 1)
    rows = int(math.ceil(len(selected) / cols))

    if figsize is None:
        figsize = (5 * cols, 4 * rows)

    if resolved_dim == 3:
        fig = plt.figure(figsize=figsize)
        axes_flat = [fig.add_subplot(rows, cols, i + 1, projection="3d") for i in range(rows * cols)]
    else:
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes_flat = axes.ravel().tolist()

    for ax, snapshot in zip(axes_flat, selected):
        plot_snapshot(
            X,
            snapshot,
            render_mode=render_mode,
            method_name=method_name,
            ax=ax,
            target_dim=resolved_dim,
            show_projections=show_projections,
            view_init=view_init,
        )

    for ax in axes_flat[len(selected):]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes_flat[: len(selected)]

def plot_curve_snapshot_2d(X, snapshot, method_name: str, ax=None, show_projections: bool = False, title: Optional[str] = None):
    return plot_curve_snapshot(
        X=X,
        snapshot=snapshot,
        method_name=method_name,
        ax=ax,
        target_dim=2,
        show_projections=show_projections,
        title=title,
    )


def plot_trace_grid(
    X,
    snapshots: Sequence,
    render_mode: Optional[str] = None,
    method_name: Optional[str] = None,
    cols: int = 3,
    max_panels: Optional[int] = None,
    figsize=None,
    target_dim: Optional[int] = None,
    show_projections: bool = False,
    view_init: Mapping[str, float] | None = None,
):
    return plot_curve_trace_grid(
        X=X,
        snapshots=snapshots,
        method_name=method_name,
        cols=cols,
        max_panels=max_panels,
        figsize=figsize,
        target_dim=target_dim,
        show_projections=show_projections,
        render_mode=render_mode,
        view_init=view_init,
    )


def plot_curve_trace_grid_2d(X, snapshots: Sequence, method_name: str, cols: int = 3, max_panels: Optional[int] = None, figsize=None, show_projections: bool = False):
    return plot_curve_trace_grid(
        X=X,
        snapshots=snapshots,
        method_name=method_name,
        cols=cols,
        max_panels=max_panels,
        figsize=figsize,
        target_dim=2,
        show_projections=show_projections,
    )
