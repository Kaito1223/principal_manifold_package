import numpy as np
import pytest

from principal_manifold.elastic.optimizer import (
    Edge,
    _elastic_energy_terms_from_node_assignments_weighted,
    _solve_paper_linear_system_weighted,
)


def test_term_decomposition_happy_path() -> None:
    X = np.array([[0.0], [2.0]], dtype=float)
    vertices = np.array([[0.0], [2.0]], dtype=float)
    assignment = np.array([0, 1], dtype=int)
    edges = [Edge(0, 1, 0.5)]
    sample_weight = np.ones(X.shape[0], dtype=float)

    terms = _elastic_energy_terms_from_node_assignments_weighted(
        X=X,
        vertices=vertices,
        assignment=assignment,
        edges=edges,
        stars=[],
        sample_weight=sample_weight,
    )

    assert np.isfinite(terms.data_term)
    assert np.isfinite(terms.elastic_term)
    assert np.isfinite(terms.total)
    assert terms.data_term == pytest.approx(0.0)
    assert terms.elastic_term == pytest.approx(2.0)
    assert terms.total == pytest.approx(terms.data_term + terms.elastic_term)


def test_singular_system_guard() -> None:
    X = np.array([[0.0], [1.0]], dtype=float)
    vertices = np.array([[0.0], [0.0]], dtype=float)
    assignment = np.array([0, 1], dtype=int)
    sample_weight = np.ones(X.shape[0], dtype=float)

    updated = _solve_paper_linear_system_weighted(
        X=X,
        vertices=vertices,
        assignment=assignment,
        edges=[],
        stars=[],
        sample_weight=sample_weight,
    )

    assert updated.shape == vertices.shape
    assert np.all(np.isfinite(updated))
    assert updated[0, 0] == pytest.approx(0.0, abs=1e-8)
    assert updated[1, 0] == pytest.approx(1.0, abs=1e-8)
