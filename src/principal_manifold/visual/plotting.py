from __future__ import annotations

import math
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D

from ._utils import (
    _as_float_matrix,
    _build_snapshot_title,
    _get_cached_projection,
    _pad_to_dim,
    _polyline_edges,
    _snapshot_edges,
    _supported_plot_dimension,
    _visual_options_for_method,
)

Array = np.ndarray

def plot_principal_object(
    X,
    vertices,
    edges=None,
    projected=None,
    ax=None,
    target_dim: Optional[int] = None,
    show_projections: bool = False,
    show_projected_points: bool = False,
    show_structure_segments: bool = True,
    show_structure_vertices: bool = True,
    title: Optional[str] = None,
):
    X = _as_float_matrix("X", X)
    vertices = _as_float_matrix("vertices", vertices)
    if edges is None:
        edges = _polyline_edges(vertices.shape[0])
    else:
        edges = np.asarray(edges, dtype=int)

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
        ax.scatter(Xp[:, 0], Xp[:, 1], s=24, alpha=0.65, label="data")

        if show_structure_segments and len(edges) > 0:
            segments = np.stack([Vp[edges[:, 0]], Vp[edges[:, 1]]], axis=1)
            line_collection = LineCollection(segments, linewidths=2, alpha=0.95, colors="black",)
            ax.add_collection(line_collection)

        if show_structure_vertices and len(Vp) > 0:
            ax.scatter(
                Vp[:, 0],
                Vp[:, 1],
                s=28,
                alpha=0.9,
                label="structure vertices" if not show_structure_segments else None,
            )

        if Pp is not None:
            if show_projections:
                projection_segments = np.stack([Xp[:, :2], Pp[:, :2]], axis=1)
                line_collection = LineCollection(projection_segments, linewidths=0.4, alpha=0.25, color='crimson')
                ax.add_collection(line_collection)
            if show_projected_points:
                ax.scatter(Pp[:, 0], Pp[:, 1], s=16, alpha=0.75, marker="x", label="projected points")

        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x1", fontsize = 15)
        ax.set_ylabel("x2", fontsize = 15)
    else:
        ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], s=20, alpha=0.55, label="data")

        if show_structure_segments and len(edges) > 0:
            segments = np.stack([Vp[edges[:, 0]], Vp[edges[:, 1]]], axis=1)
            line_collection = Line3DCollection(segments, linewidths=2, alpha=0.95, colors="black",)
            ax.add_collection(line_collection)

        if show_structure_vertices and len(Vp) > 0:
            ax.scatter(Vp[:, 0], Vp[:, 1], Vp[:, 2], s=26, alpha=0.9, label="structure vertices")

        if Pp is not None and show_projections:
            segs = np.stack([Xp, Pp], axis=1)
            ax.add_collection(Line3DCollection(segs, linewidths=0.35, alpha=0.22))

        mins = np.min(np.vstack([Xp, Vp]), axis=0)
        maxs = np.max(np.vstack([Xp, Vp]), axis=0)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel("x1", fontsize = 15)
        ax.set_ylabel("x2", fontsize = 15)
        ax.set_zlabel("x3", fontsize = 15)

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
    method_name: str,
    ax=None,
    target_dim: Optional[int] = None,
    show_projections: bool = False,
    title: Optional[str] = None,
):
    X = _as_float_matrix("X", X)
    vertices = _as_float_matrix("snapshot.vertices", snapshot.vertices)
    edges = _snapshot_edges(snapshot)
    options = _visual_options_for_method(method_name, show_projections)
    projected = _get_cached_projection(X, snapshot, vertices) if options["need_projection"] else None

    if title is None:
        title = _build_snapshot_title(snapshot)

    return plot_principal_object(
        X,
        vertices,
        edges=edges,
        projected=projected,
        ax=ax,
        target_dim=target_dim,
        show_projections=options["show_projections"],
        show_projected_points=options["show_projected_points"],
        show_structure_segments=options["show_structure_segments"],
        show_structure_vertices=options["show_structure_vertices"],
        title=title,
    )


def plot_curve_trace_grid(
    X,
    snapshots: Sequence,
    method_name: str,
    cols: int = 3,
    max_panels: Optional[int] = None,
    figsize=None,
    target_dim: Optional[int] = None,
    show_projections: bool = False,
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
        plot_curve_snapshot(
            X,
            snapshot,
            method_name=method_name,
            ax=ax,
            target_dim=resolved_dim,
            show_projections=show_projections,
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

