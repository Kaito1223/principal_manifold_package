from __future__ import annotations

from pathlib import Path

import pytest

from tests.intrinsic_k_demo import run_demo


def test_demo_invalid_k_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="intrinsic_dim must be <= n_features"):
        run_demo(tmp_path / "demo_intrinsic_k_outputs", intrinsic_dim=4)
