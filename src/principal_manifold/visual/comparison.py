from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

import matplotlib.pyplot as plt
import numpy as np

from .plotting import plot_curve_snapshot

Array = np.ndarray


class _TraceModel(Protocol):
    trace_: list[Any]


def save_final_comparison_figure(
    X: Array,
    models: Mapping[str, _TraceModel],
    output_path: Path,
    target_dim: int = 3,
    view_init: Mapping[str, float] | None = None,
) -> None:
    if target_dim not in (2, 3):
        raise ValueError("target_dim must be 2 or 3.")

    fig = plt.figure(figsize=(16, 12))
    ordered = [
        ('KK', 'KK'),
        ('HS', 'HS'),
        ('Elastic graph', 'ELASTIC_MAP'),
        ('Principal elastic graph', 'PRINCIPAL_ELASTIC_GRAPH'),
    ]

    for idx, (label, method_name) in enumerate(ordered, start=1):
        if target_dim == 3:
            ax = fig.add_subplot(2, 2, idx, projection='3d')
        else:
            ax = fig.add_subplot(2, 2, idx)
        snapshot = models[label].trace_[-1]
        plot_curve_snapshot(
            X=X,
            snapshot=snapshot,
            method_name=method_name,
            ax=ax,
            target_dim=target_dim,
            show_projections=False,
            title=f'{label} final',
            view_init=view_init,
        )

    fig.suptitle(f'{target_dim}D principal-curve / graph demo', fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
