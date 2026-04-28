from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from PIL import Image

from ._utils import (
    _as_float_matrix,
    _build_snapshot_title,
    _get_cached_projection,
    _pad_to_dim,
    _snapshot_edges,
    _supported_plot_dimension,
    _visual_options_for_method,
)
from .plotting import plot_curve_snapshot

Array = np.ndarray

def animate_curve_trace(
    X,
    snapshots: Sequence,
    method_name: str,
    interval: int = 700,
    repeat: bool = False,
    target_dim: Optional[int] = None,
    show_projections: bool = False,
):
    if len(snapshots) == 0:
        raise ValueError("snapshots must contain at least one snapshot.")

    X = _as_float_matrix("X", X)
    sample_vertices = _as_float_matrix("snapshot.vertices", snapshots[0].vertices)
    resolved_dim = _supported_plot_dimension(X, sample_vertices, target_dim)
    if resolved_dim is None:
        return None
    if resolved_dim != 2:
        raise ValueError("Animation is implemented for 2D traces only.")

    options = _visual_options_for_method(method_name, show_projections)
    fig, ax = plt.subplots(figsize=(7, 5))

    X2 = _pad_to_dim(X, 2)
    x_margin = 0.05 * max(np.ptp(X2[:, 0]), 1.0)
    y_margin = 0.05 * max(np.ptp(X2[:, 1]), 1.0)
    ax.set_xlim(X2[:, 0].min() - x_margin, X2[:, 0].max() + x_margin)
    ax.set_ylim(X2[:, 1].min() - y_margin, X2[:, 1].max() + y_margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    data_artist = ax.scatter(X2[:, 0], X2[:, 1], s=24, alpha=0.65, label="data")
    structure_collection = LineCollection([], linewidths=2, alpha=0.95)
    ax.add_collection(structure_collection)
    vertex_artist = ax.scatter([], [], s=28, alpha=0.9, label="structure vertices")
    projected_points_artist = ax.scatter([], [], s=16, alpha=0.75, marker="x", label="projected points")
    projection_lines = LineCollection([], linewidths=0.4, alpha=0.25)
    ax.add_collection(projection_lines)

    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
    if filtered:
        ax.legend(*zip(*filtered))

    empty_offsets = np.empty((0, 2), dtype=float)
    empty_segments = np.empty((0, 2, 2), dtype=float)

    def update(frame_index: int):
        snapshot = snapshots[frame_index]
        vertices = _as_float_matrix("snapshot.vertices", snapshot.vertices)
        vertices2 = _pad_to_dim(vertices, 2)
        edges = _snapshot_edges(snapshot)

        if options["show_structure_segments"] and len(edges) > 0:
            structure_collection.set_segments(np.stack([vertices2[edges[:, 0]], vertices2[edges[:, 1]]], axis=1))
        else:
            structure_collection.set_segments(empty_segments)

        if options["show_structure_vertices"]:
            vertex_artist.set_offsets(vertices2)
        else:
            vertex_artist.set_offsets(empty_offsets)

        projected = _get_cached_projection(X, snapshot, vertices) if options["need_projection"] else None
        projected2 = None if projected is None else _pad_to_dim(projected, 2)

        if options["show_projected_points"] and projected2 is not None:
            projected_points_artist.set_offsets(projected2)
        else:
            projected_points_artist.set_offsets(empty_offsets)

        if options["show_projections"] and projected2 is not None:
            projection_lines.set_segments(np.stack([X2, projected2], axis=1))
        else:
            projection_lines.set_segments(empty_segments)

        ax.set_title(_build_snapshot_title(snapshot))
        return data_artist, structure_collection, vertex_artist, projected_points_artist, projection_lines

    return FuncAnimation(fig, update, frames=len(snapshots), interval=interval, repeat=repeat, blit=False)


def save_curve_trace_frames(
    X,
    snapshots,
    method_name: str,
    output_dir="trace_frames",
    target_dim: Optional[int] = None,
    show_projections: bool = False,
    dpi: int = 150,
):
    if len(snapshots) == 0:
        raise ValueError("snapshots must contain at least one snapshot.")

    X = _as_float_matrix("X", X)
    sample_vertices = _as_float_matrix("snapshot.vertices", snapshots[0].vertices)
    resolved_dim = _supported_plot_dimension(X, sample_vertices, target_dim)
    if resolved_dim is None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, snapshot in enumerate(snapshots):
        if resolved_dim == 3:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(7, 5))

        plot_curve_snapshot(
            X,
            snapshot,
            method_name=method_name,
            ax=ax,
            target_dim=resolved_dim,
            show_projections=show_projections,
        )

        filename = f"{idx:03d}_outer_{snapshot.outer_iteration:03d}_{snapshot.phase}"
        if getattr(snapshot, "sweep", None) is not None:
            filename += f"_sweep_{snapshot.sweep:03d}"
        filename += ".png"

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def frames_to_gif(frames_dir, output_path="trace_steps/evolution.gif", pattern="*.png", duration=400, loop=0):
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)

    frame_paths = sorted(frames_dir.glob(pattern))
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir} matching {pattern!r}")

    images = [Image.open(path).convert("RGBA") for path in frame_paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, rest = images[0], images[1:]
    first.save(output_path, save_all=True, append_images=rest, duration=duration, loop=loop, optimize=False)


def save_trace_gif(X: Array, snapshots, method_name: str, output_dir: Path, gif_name: str) -> Path:
    save_curve_trace_frames(
        X=X,
        snapshots=snapshots,
        method_name=method_name,
        output_dir=str(output_dir),
        target_dim=3,
        show_projections=False,
        dpi=140,
    )
    gif_path = output_dir / gif_name
    frames_to_gif(frames_dir=output_dir, output_path=gif_path, duration=450)
    return gif_path

def animate_curve_trace_2d(X, snapshots: Sequence, method_name: str, interval: int = 700, repeat: bool = False, show_projections: bool = False):
    return animate_curve_trace(
        X=X,
        snapshots=snapshots,
        method_name=method_name,
        interval=interval,
        repeat=repeat,
        target_dim=2,
        show_projections=show_projections,
    )


def save_curve_trace_frames_2d(X, snapshots, method_name: str, output_dir="trace_steps", show_projections: bool = False, dpi: int = 150):
    return save_curve_trace_frames(
        X=X,
        snapshots=snapshots,
        method_name=method_name,
        output_dir=output_dir,
        target_dim=2,
        show_projections=show_projections,
        dpi=dpi,
    )
