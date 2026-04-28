from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

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
from principal_manifold.visual import (
    save_curve_trace_frames,
    frames_to_gif,
    plot_curve_snapshot,
    save_trace_gif,
    save_final_comparison_figure,
)

Array = np.ndarray

def make_helix_3d(
    n: int = 180,
    turns: float = 2.75,
    radius: float = 1.0,
    pitch: float = 0.45,
    noise: float = 0.10,
    seed: int = 7,
) -> Array:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, turns * 2.0 * np.pi, n)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * t
    X = np.column_stack([x, y, z])
    X += noise * rng.normal(size=X.shape)
    return X


def make_sin_3d(n: int = 180, seed: int = 7) -> Array:
    rng = np.random.default_rng(seed)
    t0 = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([
        t0 + 0.30 * rng.normal(size=n),
        np.sin(t0) + 0.20 * rng.normal(size=n),
        np.cos(t0) + 0.20 * rng.normal(size=n),
    ])


def print_result_summary(result, label: str, runtime_seconds: float | None = None) -> dict:
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

    if runtime_seconds is not None:
        summary['runtime_seconds'] = float(runtime_seconds)

    print(f'\n=== {label} ===')
    for key, value in summary.items():
        if key != 'label':
            print(f'{key}: {value}')
    return summary

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / 'demo_3d_outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    X = make_helix_3d()

    t0 = perf_counter()
    model_KK = KeglKrzyzakPrincipalCurve(
        KeglKrzyzakConfig(
            max_segments=30,
            store_trace=True,
            trace_inner_sweeps=False,
        )
    ).fit(X)
    runtime_KK = perf_counter() - t0

    t0 = perf_counter()
    model_HS = HSPrincipalCurve(
        HSConfig(
            w=0.10,
            max_iter=8,
            tol=1e-4,
            store_trace=True,
        )
    ).fit(X)
    runtime_HS = perf_counter() - t0

    t0 = perf_counter()
    model_elastic_graph = ElasticGraphPrincipalCurve(
        ElasticGraphConfig(
            lam=0.02,
            mu=0.05,
            n_nodes=28,
            max_iter=80,
            tol=1e-6,
            softening=(1e1, 1.0),
            verbose=False,
            store_trace=True,
        )
    ).fit(X)
    runtime_elastic_graph = perf_counter() - t0

    t0 = perf_counter()
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
            sc_max=64,
            cc_max=72,
            softening=(1e1, 1.0),
            verbose=False,
            store_trace=True,
        )
    ).fit(X)
    runtime_principal_elastic_graph = perf_counter() - t0

    runtimes = {
        'KK': runtime_KK,
        'HS': runtime_HS,
        'Elastic graph': runtime_elastic_graph,
        'Principal elastic graph': runtime_principal_elastic_graph,
    }
    models = {
        'KK': model_KK,
        'HS': model_HS,
        'Elastic graph': model_elastic_graph,
        'Principal elastic graph': model_principal_elastic_graph,
    }

    summaries = {
        label: print_result_summary(model.result_, label, runtimes[label])
        for label, model in models.items()
    }

    np.save(out_dir / 'dataset_3d.npy', X)

    save_final_comparison_figure(
        X,
        models,
        out_dir / 'comparison_3d.png',
    )

    gif_paths = {}
    gif_paths['KK'] = save_trace_gif(X, model_KK.trace_, 'KK', out_dir / 'kk_steps_3d', 'evolution.gif')
    gif_paths['HS'] = save_trace_gif(X, model_HS.trace_, 'HS', out_dir / 'hs_steps_3d', 'evolution.gif')
    gif_paths['Elastic graph'] = save_trace_gif(
        X,
        model_elastic_graph.trace_,
        'ELASTIC_MAP',
        out_dir / 'elastic_graph_steps_3d',
        'evolution.gif',
    )
    peg_snapshots = [s for s in model_principal_elastic_graph.trace_ if s.phase in ('init', 'accepted')]
    gif_paths['Principal elastic graph'] = save_trace_gif(
        X,
        peg_snapshots,
        'PRINCIPAL_ELASTIC_GRAPH',
        out_dir / 'principal_elastic_graph_steps_3d',
        'evolution.gif',
    )

    report = {
        'dataset_shape': tuple(int(v) for v in X.shape),
        'summaries': summaries,
        'artifacts': {
            'comparison_figure': str(out_dir / 'comparison_3d.png'),
            'gifs': {key: str(value) for key, value in gif_paths.items()},
        },
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('\nSaved outputs to:', out_dir)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
