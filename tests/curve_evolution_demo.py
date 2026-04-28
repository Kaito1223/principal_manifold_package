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
    ElasticGraphConfig,
    ElasticGraphPrincipalCurve,
    ElasticPrincipalGraphConfig,
    ElasticPrincipalGraph,
)
from principal_manifold.visual import plot_curve_snapshot, save_trace_gif

Array = np.ndarray

def make_half_circle(n: int = 250, noise: float = 0.08, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, np.pi, size=n)
    x = np.cos(t)
    y = np.sin(t)
    return np.column_stack([x, y]) + noise * rng.normal(size=(n, 2))


def make_sin(n: int = 450, seed: int = 0):
    rng = np.random.default_rng(seed)
    t0 = np.linspace(0, 2 * np.pi, n)
    X = np.column_stack([
        t0 + 0.35 * rng.normal(size=n),
        np.sin(t0) + 0.25 * rng.normal(size=n),
    ])
    return X


def make_sin_3d(n: int = 450, seed: int = 0):
    rng = np.random.default_rng(seed)
    t0 = np.linspace(0, 2 * np.pi, n)
    X = np.column_stack([
        t0 + 0.30 * rng.normal(size=n),
        np.sin(t0) + 0.20 * rng.normal(size=n),
        np.cos(t0) + 0.20 * rng.normal(size=n),
    ])
    return X

def print_result_summary(result, label: str) -> dict:
    summary = {
        'label': label,
        'vertices_shape': tuple(int(v) for v in result.vertices.shape),
        'segments': int(result.edges.shape[0]) if hasattr(result, 'edges') else int(max(result.vertices.shape[0] - 1, 0)),
    }
    if result.history:
        last = result.history[-1]
        for key in ('mean_squared_distance', 'root_mean_squared_distance', 'polyline_length', 'elastic_energy'):
            if key in last:
                summary[key] = float(last[key])
    print(f'\n=== {label} ===')
    for key, value in summary.items():
        if key != 'label':
            print(f'{key}: {value}')
    return summary



def save_final_comparison_figure(X: Array, models: dict[str, object], output_path: Path) -> None:
    fig = plt.figure(figsize=(16, 12))
    ordered = [
        ('KK', 'KK'),
        ('HS', 'HS'),
        ('Elastic graph', 'ELASTIC_MAP'),
        ('Principal elastic graph', 'PRINCIPAL_ELASTIC_GRAPH'),
    ]

    for idx, (label, method_name) in enumerate(ordered, start=1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        snapshot = models[label].trace_[-1]
        plot_curve_snapshot(
            X=X,
            snapshot=snapshot,
            method_name=method_name,
            ax=ax,
            target_dim=2,
            show_projections=False,
            title=f'{label} final',
        )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / 'demo_2d_outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    X = make_sin()
    # X = make_half_circle()

    model_KK = KeglKrzyzakPrincipalCurve(
        KeglKrzyzakConfig(
            max_segments=10,
            store_trace=True,   
            trace_inner_sweeps=False,
        )
    ).fit(X)

    model_HS = HSPrincipalCurve(
        HSConfig(
            w=0.10,
            max_iter=8,
            tol=1e-4,
            store_trace=True,
        )
    ).fit(X)

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
    ).fit(X)

    model_principal_elastic_graph = ElasticPrincipalGraph(
        ElasticPrincipalGraphConfig(
            lam=0.02,
            mu=0.05,
            grammar_sequence=(
                ('add_node', 'bisect_edge'),
                ('add_node', 'bisect_edge'),
                ('remove_leaf', 'remove_edge'),
            ),
            sc_measure='n_vertices',
            sc_max=20,
            cc_max=25,
            softening=(1e1, 1.0),
            verbose=False,
            store_trace=True,
        )
    ).fit(X)

    # print_trace_summary(model_KK, "KK")
    # print_trace_summary(model_HS, "HS")
    # print_trace_summary(model_elastic_graph, "Elastic graph")
    # print_trace_summary(model_principal_elastic_graph, "Principal elastic graph")

    models = {
        'KK': model_KK,
        'HS': model_HS,
        'Elastic graph': model_elastic_graph,
        'Principal elastic graph': model_principal_elastic_graph,
    }

    summaries = {
        label: print_result_summary(model.result_, label)
        for label, model in models.items()
    }

    np.save(out_dir / 'dataset_2d.npy', X)

    save_final_comparison_figure(
        X,
        models,
        out_dir / 'comparison_2d.png',
    )

    gif_paths = {}
    gif_paths['KK'] = save_trace_gif(X, model_KK.trace_, 'KK', out_dir / 'kk_steps_2d', 'evolution.gif')
    gif_paths['HS'] = save_trace_gif(X, model_HS.trace_, 'HS', out_dir / 'hs_steps_2d', 'evolution.gif')
    gif_paths['Elastic graph'] = save_trace_gif(
        X,
        model_elastic_graph.trace_,
        'ELASTIC_MAP',
        out_dir / 'elastic_graph_steps_2d',
        'evolution.gif',
    )
    peg_snapshots = [s for s in model_principal_elastic_graph.trace_ if s.phase in ('init', 'accepted')]
    gif_paths['Principal elastic graph'] = save_trace_gif(
        X,
        peg_snapshots,
        'PRINCIPAL_ELASTIC_GRAPH',
        out_dir / 'principal_elastic_graph_steps_2d',
        'evolution.gif',
    )

    report = {
        'dataset_shape': tuple(int(v) for v in X.shape),
        'summaries': summaries,
        'artifacts': {
            'comparison_figure': str(out_dir / 'comparison_2d.png'),
            'gifs': {key: str(value) for key, value in gif_paths.items()},
        },
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('\nSaved outputs to:', out_dir)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()