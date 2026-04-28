from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plotting import plot_curve_snapshot

Array = np.ndarray

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
            target_dim=3,
            show_projections=False,
            title=f'{label} final',
        )

    fig.suptitle('3D principal-curve / graph demo', fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
