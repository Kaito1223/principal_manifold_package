from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from principal_manifold.elastic import ElasticSurfaceConfig, ElasticSurfacePrincipalManifold
from principal_manifold.visual.plotting import _configure_3d_axes
from principal_manifold.visual import save_trace_gif

Array = np.ndarray


def _padded_3d_limits(mins: Array, maxs: Array) -> tuple[Array, Array]:
    mins_arr = np.asarray(mins, dtype=float)
    maxs_arr = np.asarray(maxs, dtype=float)
    span = maxs_arr - mins_arr
    pad = np.where(span > 1e-12, 0.0, 0.5)
    return mins_arr - pad, maxs_arr + pad


def make_plane_patch_3d(
    n_u: int = 14,
    n_v: int = 12,
    noise: float = 0.04,
    seed: int = 7,
) -> Array:
    """Small synthetic 2D manifold in R^3."""
    rng = np.random.default_rng(seed)
    u = np.linspace(-1.0, 1.0, n_u)
    v = np.linspace(-0.9, 0.9, n_v)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    zz = 0.45 * uu - 0.25 * vv + 0.12 * np.sin(np.pi * uu) * np.cos(np.pi * vv)
    X = np.column_stack([uu.ravel(), vv.ravel(), zz.ravel()])
    X += noise * rng.normal(size=X.shape)
    return X


def print_result_summary(result, runtime_seconds: float | None = None) -> dict:
    summary = {
        "label": "Elastic surface",
        "vertices_shape": tuple(int(v) for v in result.vertices.shape),
        "edges": int(result.edges.shape[0]),
        "faces": int(result.faces.shape[0]) if result.faces is not None else 0,
    }
    if result.history:
        last = result.history[-1]
        for key in (
            "mean_squared_distance",
            "root_mean_squared_distance",
            "polyline_length",
            "elastic_energy",
        ):
            if key in last:
                summary[key] = float(last[key])
    if runtime_seconds is not None:
        summary["runtime_seconds"] = float(runtime_seconds)

    print("\n=== Elastic surface ===")
    for key, value in summary.items():
        if key != "label":
            print(f"{key}: {value}")
    return summary


def save_surface_figure(
    X: Array,
    model: ElasticSurfacePrincipalManifold,
    output_path: Path,
    view_init: dict[str, float] | None = None,
) -> None:
    result = model.result_
    vertices = np.asarray(result.vertices, dtype=float)
    faces = np.asarray(result.faces, dtype=int)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if not isinstance(ax, Axes3D):
        raise TypeError("Expected 3D axes.")
    ax3d = ax
    ax3d.plot(X[:, 0], X[:, 1], X[:, 2], linestyle="", marker="o", markersize=4.2, alpha=0.5, label="data")
    ax3d.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color="cornflowerblue",
        alpha=0.45,
        edgecolor="black",
        linewidth=0.6,
    )
    ax3d.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], linestyle="", marker="o", markersize=4.8, alpha=0.85, color="black", label="surface vertices")
    mins = np.min(np.vstack([X, vertices]), axis=0)
    maxs = np.max(np.vstack([X, vertices]), axis=0)
    mins_plot, maxs_plot = _padded_3d_limits(mins, maxs)
    ax3d.set_xlim(mins_plot[0], maxs_plot[0])
    ax3d.set_ylim(mins_plot[1], maxs_plot[1])
    ax3d.set_zlim(mins_plot[2], maxs_plot[2])
    _configure_3d_axes(ax3d, mins, maxs, view_init=view_init)
    ax3d.set_title("Fixed-topology 2D elastic principal manifold", fontsize=16)
    ax3d.set_xlabel("x1", fontsize=13)
    ax3d.set_ylabel("x2", fontsize=13)
    ax3d.set_zlabel("x3", fontsize=13)
    ax3d.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "demo_surface_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic 2D manifold in R^3.
    X = make_plane_patch_3d()

    # Fit fixed-topology elastic surface on a small rectangular grid.
    t0 = perf_counter()
    model = ElasticSurfacePrincipalManifold(
        ElasticSurfaceConfig(
            lam=0.02,
            mu=0.05,
            n_u=7,
            n_v=6,
            max_iter=60,
            tol=1e-6,
            softening=(1e1, 1.0),
            verbose=False,
            store_trace=True,
        )
    ).fit(X)
    runtime = perf_counter() - t0

    summary = print_result_summary(model.result_, runtime)

    np.save(out_dir / "dataset_surface.npy", X)
    save_surface_figure(X, model, out_dir / "surface_comparison_3d.png")
    gif_path = save_trace_gif(
        X,
        model.trace_,
        "ELASTIC_MAP",
        out_dir / "elastic_surface_steps_3d",
        "evolution.gif",
    )

    report = {
        "run_command": "uv run python tests/surface_evolution_demo.py",
        "dataset_shape": tuple(int(v) for v in X.shape),
        "summary": summary,
        "artifacts": {
            "comparison_figure": str(out_dir / "surface_comparison_3d.png"),
            "gif": str(gif_path),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved outputs to:", out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
