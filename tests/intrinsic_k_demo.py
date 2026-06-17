from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np

from principal_manifold import (
    IntrinsicElasticMapConfig,
    IntrinsicElasticMapPrincipalManifold,
    IntrinsicHSConfig,
    IntrinsicHSPrincipalManifold,
)
from principal_manifold.visual import plot_principal_object, plot_snapshot, save_trace_gif

Array = np.ndarray


class _TraceModel(Protocol):
    trace_: list[Any]

    @property
    def result_(self) -> Any:
        ...


def make_intrinsic_sheet_3d(seed: int = 7) -> Array:
    rng = np.random.default_rng(seed)
    u = np.linspace(-1.0, 1.0, 10)
    v = np.linspace(-0.8, 0.8, 9)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    zz = 0.45 * uu - 0.25 * vv + 0.12 * np.sin(np.pi * uu) * np.cos(np.pi * vv)
    X = np.column_stack([uu.ravel(), vv.ravel(), zz.ravel()])
    X += 0.03 * rng.normal(size=X.shape)
    return X


def print_result_summary(result, label: str, runtime_seconds: float | None = None) -> dict:
    summary = {
        "label": label,
        "vertices_shape": tuple(int(v) for v in result.vertices.shape),
        "edges": int(result.edges.shape[0]) if hasattr(result, "edges") else 0,
        "faces": int(result.faces.shape[0]) if getattr(result, "faces", None) is not None else 0,
    }
    if result.history:
        last = result.history[-1]
        for key in ("mean_squared_distance", "root_mean_squared_distance", "polyline_length", "elastic_energy"):
            if key in last:
                summary[key] = float(last[key])
    if runtime_seconds is not None:
        summary["runtime_seconds"] = float(runtime_seconds)

    print(f"\n=== {label} ===")
    for key, value in summary.items():
        if key != "label":
            print(f"{key}: {value}")
    return summary


def save_intrinsic_comparison_figure(
    X: Array,
    models: dict[str, _TraceModel],
    output_path: Path,
    target_dim: int,
) -> None:
    ordered = list(models.items())
    cols = 2
    rows = int(np.ceil(len(ordered) / cols))
    fig = plt.figure(figsize=(7 * cols, 5 * rows))

    for idx, (label, model) in enumerate(ordered, start=1):
        if target_dim == 3:
            ax = fig.add_subplot(rows, cols, idx, projection="3d")
            plot_snapshot(
                X=X,
                snapshot=model.trace_[-1],
                render_mode="intrinsic",
                ax=ax,
                target_dim=3,
                show_projections=False,
                title=f"{label} final",
                view_init={"elev": 28, "azim": 135},
            )
        else:
            ax = fig.add_subplot(rows, cols, idx)
            snapshot = model.trace_[-1]
            plot_principal_object(
                X=X[:, :2],
                vertices=np.asarray(snapshot.vertices, dtype=float)[:, :2],
                edges=np.asarray(snapshot.edges, dtype=int),
                ax=ax,
                target_dim=2,
                show_projections=False,
                title=f"{label} final",
            )

    for idx in range(len(ordered) + 1, rows * cols + 1):
        ax = fig.add_subplot(rows, cols, idx)
        ax.set_visible(False)

    fig.suptitle(f"Intrinsic-k manifold demo ({target_dim}D view)", fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_demo(output_dir: Path, intrinsic_dim: int = 2) -> dict:
    X = make_intrinsic_sheet_3d()
    if intrinsic_dim > X.shape[1]:
        raise ValueError(f"intrinsic_dim must be <= n_features ({X.shape[1]}).")

    output_dir.mkdir(parents=True, exist_ok=True)

    runtimes: dict[str, float] = {}
    models: dict[str, _TraceModel] = {}

    t0 = perf_counter()
    models["Intrinsic elastic map"] = IntrinsicElasticMapPrincipalManifold(
        IntrinsicElasticMapConfig(
            intrinsic_dim=intrinsic_dim,
            grid_shape=(3, 4) if intrinsic_dim == 2 else (6,),
            optimizer_max_iter=20,
            softening=(10.0, 1.0),
            topology_epochs=1,
            store_trace=True,
        )
    ).fit(X)
    runtimes["Intrinsic elastic map"] = perf_counter() - t0

    t0 = perf_counter()
    models["Intrinsic HS"] = IntrinsicHSPrincipalManifold(
        IntrinsicHSConfig(
            intrinsic_dim=intrinsic_dim,
            grid_shape=(3, 4) if intrinsic_dim == 2 else (6,),
            w=0.20,
            max_iter=3,
            tol=1e-6,
            store_trace=True,
            ridge=1e-10,
        )
    ).fit(X)
    runtimes["Intrinsic HS"] = perf_counter() - t0

    summaries = {
        label: print_result_summary(model.result_, label, runtimes[label])
        for label, model in models.items()
    }

    np.save(output_dir / "dataset_intrinsic_k.npy", X)
    save_intrinsic_comparison_figure(X, models, output_dir / "comparison_2d.png", target_dim=2)
    save_intrinsic_comparison_figure(X, models, output_dir / "comparison_3d.png", target_dim=3)

    gif_paths = {
        label: str(save_trace_gif(X, model.trace_, "ELASTIC_MAP", output_dir / f"{label.lower().replace(' ', '_')}_steps", "evolution.gif", view_init={"elev": 28, "azim": 135}))
        for label, model in models.items()
    }

    report = {
        "run_command": "uv run python tests/intrinsic_k_demo.py",
        "dataset_shape": tuple(int(v) for v in X.shape),
        "summaries": summaries,
        "artifacts": {
            "comparison_2d": str(output_dir / "comparison_2d.png"),
            "comparison_3d": str(output_dir / "comparison_3d.png"),
            "gifs": gif_paths,
        },
        "notes": {
            "regression_adapter": "PrincipalManifoldRegressor remains edge-oriented; intrinsic demo is unsupervised.",
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved outputs to:", output_dir)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "demo_intrinsic_k_outputs"
    run_demo(out_dir, intrinsic_dim=2)


if __name__ == "__main__":
    main()
