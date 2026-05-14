from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from principal_manifold import (
    KeglKrzyzakConfig,
    KeglKrzyzakPrincipalCurve,
    HSPrincipalCurve,
    HSConfig,
    OptimizerConfig,
    ElasticGraphConfig,
    ElasticGraphPrincipalCurve,
    ElasticPrincipalGraphConfig,
    ElasticPrincipalGraph,
)
from principal_manifold.visual import plot_curve_snapshot, save_trace_gif


Array = np.ndarray


# =============================================================================
# Data generators
# =============================================================================

def make_half_circle(n: int = 250, noise: float = 0.08, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, np.pi, size=n)
    x = np.cos(t)
    y = np.sin(t)
    return np.column_stack([x, y]) + noise * rng.normal(size=(n, 2))


def make_sin(n: int = 450, seed: int = 0):
    rng = np.random.default_rng(seed)
    t0 = np.linspace(0, 2 * np.pi, n)
    X = np.column_stack(
        [
            t0 + 0.35 * rng.normal(size=n),
            np.sin(t0) + 0.25 * rng.normal(size=n),
        ]
    )
    return X


def make_sine_2d(
    n=180,
    x_min=-1.2,
    x_max=1.2,
    noise=0.15,
    seed=7,
):
    rng = np.random.default_rng(seed)
    x = np.linspace(x_min, x_max, n)
    y = np.sin(2.5 * np.pi * x) + 0.20 * x
    X = np.column_stack([x, y])
    X += noise * rng.normal(size=X.shape)
    return X


def make_sin_3d(n: int = 450, seed: int = 0):
    rng = np.random.default_rng(seed)
    t0 = np.linspace(0, 2 * np.pi, n)
    X = np.column_stack(
        [
            t0 + 0.30 * rng.normal(size=n),
            np.sin(t0) + 0.20 * rng.normal(size=n),
            np.cos(t0) + 0.20 * rng.normal(size=n),
        ]
    )
    return X


# =============================================================================
# Prediction helpers
# =============================================================================

def train_test_split_points(
    X: Array,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)

    n = X.shape[0]
    idx = rng.permutation(n)

    n_test = max(1, int(round(test_fraction * n)))

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx]


def rmse(y_true: Array, y_pred: Array) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def chain_edges(n_vertices: int) -> Array:
    if n_vertices < 2:
        return np.empty((0, 2), dtype=int)

    return np.column_stack(
        [
            np.arange(n_vertices - 1, dtype=int),
            np.arange(1, n_vertices, dtype=int),
        ]
    )


def extract_vertices_edges(model) -> tuple[Array, Array]:
    """
    Extract learned vertices and edges from any principal-manifold model.

    KK / HS usually behave like ordered curves, so if explicit edges are not
    present, we create chain edges:

        0 -- 1 -- 2 -- ... -- n

    Elastic graph / principal elastic graph usually have explicit graph edges.
    """
    result = model.result_

    vertices = np.asarray(result.vertices, dtype=float)

    if hasattr(result, "edges"):
        edges = np.asarray(result.edges, dtype=int)

        if edges.size == 0:
            edges = chain_edges(vertices.shape[0])

        return vertices, edges

    return vertices, chain_edges(vertices.shape[0])


def predict_y_from_2d_manifold(
    x_query: Array,
    vertices: Array,
    edges: Array,
) -> tuple[Array, Array, Array, Array]:
    """
    Predict y from a learned 2D principal curve/graph.

    The learned manifold is in joint space:

        z = [x, y]

    For each new x, this finds the point [x_proj, y_proj] on the learned
    curve/graph whose x-coordinate is closest to x.

    If a curve segment crosses exactly through x, this is ordinary linear
    interpolation along that segment.
    """
    x_query = np.asarray(x_query, dtype=float).reshape(-1)
    vertices = np.asarray(vertices, dtype=float)
    edges = np.asarray(edges, dtype=int)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError(
            "This helper expects 2D vertices with shape (n_vertices, 2), "
            "where columns are [x, y]."
        )

    vx = vertices[:, 0]
    vy = vertices[:, 1]

    y_pred = np.empty(x_query.shape[0], dtype=float)
    x_squared_distance = np.empty(x_query.shape[0], dtype=float)
    nearest_edge_index = np.full(x_query.shape[0], -1, dtype=int)
    edge_t = np.zeros(x_query.shape[0], dtype=float)

    if edges.size == 0:
        for row_idx, x in enumerate(x_query):
            d2 = (vx - x) ** 2
            best_idx = int(np.argmin(d2))

            y_pred[row_idx] = vy[best_idx]
            x_squared_distance[row_idx] = float(d2[best_idx])
            nearest_edge_index[row_idx] = -1
            edge_t[row_idx] = 0.0

        return y_pred, x_squared_distance, nearest_edge_index, edge_t

    for row_idx, x in enumerate(x_query):
        best_dist = np.inf
        best_y = np.nan
        best_edge = -1
        best_t = 0.0

        for edge_idx, (i, j) in enumerate(edges):
            i = int(i)
            j = int(j)

            ax = vx[i]
            ay = vy[i]

            bx = vx[j]
            by = vy[j]

            dx = bx - ax

            if abs(dx) <= 1e-15:
                # Vertical segment in x. Choose the closer endpoint in x.
                t = 0.0
            else:
                # Interpolation coordinate in x-space.
                t = (x - ax) / dx
                t = min(1.0, max(0.0, float(t)))

            x_proj = ax + t * (bx - ax)
            y_proj = ay + t * (by - ay)

            dist = float((x - x_proj) ** 2)

            if dist < best_dist:
                best_dist = dist
                best_y = y_proj
                best_edge = edge_idx
                best_t = t

        y_pred[row_idx] = best_y
        x_squared_distance[row_idx] = best_dist
        nearest_edge_index[row_idx] = best_edge
        edge_t[row_idx] = best_t

    return y_pred, x_squared_distance, nearest_edge_index, edge_t


def predict_with_model(model, X_test: Array) -> dict:
    """
    Predict y for test points using the learned principal curve/graph.

    X_test is a 2D array whose first column is observed x and second column
    is true y. The model only receives x_test for prediction.
    """
    vertices, edges = extract_vertices_edges(model)

    x_test = X_test[:, 0]
    y_true = X_test[:, 1]

    y_pred, x_d2, edge_idx, edge_t = predict_y_from_2d_manifold(
        x_query=x_test,
        vertices=vertices,
        edges=edges,
    )

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "x_test": x_test,
        "rmse": rmse(y_true, y_pred),
        "x_squared_distance": x_d2,
        "nearest_edge_index": edge_idx,
        "edge_t": edge_t,
    }


# =============================================================================
# Summary / plotting
# =============================================================================

def print_result_summary(result, label: str, prediction: dict | None = None) -> dict:
    summary = {
        "label": label,
        "vertices_shape": tuple(int(v) for v in result.vertices.shape),
        "segments": (
            int(result.edges.shape[0])
            if hasattr(result, "edges")
            else int(max(result.vertices.shape[0] - 1, 0))
        ),
    }

    if prediction is not None:
        summary["prediction_rmse"] = float(prediction["rmse"])
        summary["mean_x_projection_error"] = float(
            np.sqrt(np.mean(prediction["x_squared_distance"]))
        )

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

    print(f"\n=== {label} ===")

    for key, value in summary.items():
        if key != "label":
            print(f"{key}: {value}")

    return summary


def last_snapshot(model):
    trace = getattr(model, "trace_", None)

    if trace is None or len(trace) == 0:
        return None

    return trace[-1]


def save_final_comparison_figure(
    X_train: Array,
    models: dict[str, object],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 12))

    ordered = [
        ("KK", "KK"),
        ("HS", "HS"),
        ("Elastic graph", "ELASTIC_MAP"),
        ("Principal elastic graph", "PRINCIPAL_ELASTIC_GRAPH"),
    ]

    for idx, (label, method_name) in enumerate(ordered, start=1):
        ax = fig.add_subplot(2, 2, idx)

        snapshot = last_snapshot(models[label])

        if snapshot is not None:
            plot_curve_snapshot(
                X=X_train,
                snapshot=snapshot,
                method_name=method_name,
                ax=ax,
                target_dim=2,
                show_projections=False,
                title=f"{label} final",
            )
        else:
            ax.scatter(X_train[:, 0], X_train[:, 1], s=15, alpha=0.6)
            ax.set_title(f"{label} final")

    fig.suptitle("Final learned principal manifolds", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_prediction_comparison_figure(
    X_train: Array,
    X_test: Array,
    models: dict[str, object],
    predictions: dict[str, dict],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 12))

    ordered = [
        ("KK", "KK"),
        ("HS", "HS"),
        ("Elastic graph", "ELASTIC_MAP"),
        ("Principal elastic graph", "PRINCIPAL_ELASTIC_GRAPH"),
    ]

    for idx, (label, method_name) in enumerate(ordered, start=1):
        ax = fig.add_subplot(2, 2, idx)

        snapshot = last_snapshot(models[label])

        if snapshot is not None:
            plot_curve_snapshot(
                X=X_train,
                snapshot=snapshot,
                method_name=method_name,
                ax=ax,
                target_dim=2,
                show_projections=False,
                title=f"{label}: RMSE={predictions[label]['rmse']:.4f}",
            )
        else:
            ax.scatter(X_train[:, 0], X_train[:, 1], s=15, alpha=0.6, label="train")
            ax.set_title(f"{label}: RMSE={predictions[label]['rmse']:.4f}")

        x_test = predictions[label]["x_test"]
        y_true = predictions[label]["y_true"]
        y_pred = predictions[label]["y_pred"]

        ax.scatter(
            x_test,
            y_true,
            s=28,
            marker="x",
            alpha=0.75,
            label="test true",
        )

        order = np.argsort(x_test)

        ax.plot(
            x_test[order],
            y_pred[order],
            linewidth=2.2,
            alpha=0.9,
            label="predicted",
        )

        ax.scatter(
            x_test,
            y_pred,
            s=18,
            alpha=0.75,
            label="predicted points",
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(fontsize=12)

    fig.suptitle("Principal manifold regression prediction", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def safe_npz_key(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "demo_2d_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    X = make_sine_2d()
    # X = make_half_circle()
    # X = make_sin()

    X_train, X_test = train_test_split_points(
        X,
        test_fraction=0.25,
        seed=42,
    )

    model_KK = KeglKrzyzakPrincipalCurve(
        KeglKrzyzakConfig(
            max_segments=10,
            store_trace=True,
            trace_inner_sweeps=False,
            optimizer=OptimizerConfig(
                backend="torch_armijo",
                max_gradient_steps=250,
                torch_dtype="float64",
                torch_device="cpu",
            ),
        )
    ).fit(X_train)

    model_HS = HSPrincipalCurve(
        HSConfig(
            w=0.10,
            max_iter=8,
            tol=1e-4,
            store_trace=True,
        )
    ).fit(X_train)

    model_elastic_graph = ElasticGraphPrincipalCurve(
        ElasticGraphConfig(
            lam=0.02,
            mu=0.05,
            n_nodes=45,
            max_iter=80,
            tol=1e-6,
            softening=(1e1, 1.0),
            verbose=False,
            store_trace=True,
        )
    ).fit(X_train)

    model_principal_elastic_graph = ElasticPrincipalGraph(
        ElasticPrincipalGraphConfig(
            lam=0.02,
            mu=0.05,
            grammar_sequence=(
                ("add_node", "bisect_edge"),
                ("add_node", "bisect_edge"),
                ("remove_leaf", "remove_edge"),
            ),
            sc_measure="n_vertices",
            sc_max=20,
            cc_max=25,
            softening=(1e1, 1.0),
            verbose=False,
            store_trace=True,
        )
    ).fit(X_train)

    models = {
        "KK": model_KK,
        "HS": model_HS,
        "Elastic graph": model_elastic_graph,
        "Principal elastic graph": model_principal_elastic_graph,
    }

    predictions = {
        label: predict_with_model(model, X_test)
        for label, model in models.items()
    }

    summaries = {
        label: print_result_summary(
            model.result_,
            label,
            prediction=predictions[label],
        )
        for label, model in models.items()
    }

    np.save(out_dir / "dataset_2d.npy", X)
    np.save(out_dir / "train_2d.npy", X_train)
    np.save(out_dir / "test_2d.npy", X_test)

    np.savez(
        out_dir / "predictions_2d.npz",
        x_test=X_test[:, 0],
        y_test=X_test[:, 1],
        **{
            f"{safe_npz_key(label)}_y_pred": pred["y_pred"]
            for label, pred in predictions.items()
        },
    )

    save_final_comparison_figure(
        X_train,
        models,
        out_dir / "comparison_2d.png",
    )

    save_prediction_comparison_figure(
        X_train,
        X_test,
        models,
        predictions,
        out_dir / "prediction_comparison_2d.png",
    )

    gif_paths = {}

    gif_paths["KK"] = save_trace_gif(
        X_train,
        model_KK.trace_,
        "KK",
        out_dir / "kk_steps_2d",
        "evolution.gif",
    )

    gif_paths["HS"] = save_trace_gif(
        X_train,
        model_HS.trace_,
        "HS",
        out_dir / "hs_steps_2d",
        "evolution.gif",
    )

    gif_paths["Elastic graph"] = save_trace_gif(
        X_train,
        model_elastic_graph.trace_,
        "ELASTIC_MAP",
        out_dir / "elastic_graph_steps_2d",
        "evolution.gif",
    )

    peg_snapshots = [
        s
        for s in model_principal_elastic_graph.trace_
        if s.phase in ("init", "accepted")
    ]

    gif_paths["Principal elastic graph"] = save_trace_gif(
        X_train,
        peg_snapshots,
        "PRINCIPAL_ELASTIC_GRAPH",
        out_dir / "principal_elastic_graph_steps_2d",
        "evolution.gif",
    )

    report = {
        "dataset_shape": tuple(int(v) for v in X.shape),
        "train_shape": tuple(int(v) for v in X_train.shape),
        "test_shape": tuple(int(v) for v in X_test.shape),
        "summaries": summaries,
        "prediction_rmse": {
            label: float(pred["rmse"])
            for label, pred in predictions.items()
        },
        "artifacts": {
            "comparison_figure": str(out_dir / "comparison_2d.png"),
            "prediction_comparison_figure": str(out_dir / "prediction_comparison_2d.png"),
            "predictions_npz": str(out_dir / "predictions_2d.npz"),
            "gifs": {
                key: str(value)
                for key, value in gif_paths.items()
            },
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved outputs to:", out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()