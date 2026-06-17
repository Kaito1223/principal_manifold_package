from __future__ import annotations

import pytest

from principal_manifold.elastic.intrinsic_topology import IntrinsicTopologyAdapter
from principal_manifold.elastic.principal_graph import ElasticPrincipalGraph, ElasticPrincipalGraphConfig


def test_split_prune_happy_path() -> None:
    adapter = IntrinsicTopologyAdapter(ops=("split", "prune"), max_ops_per_epoch=2)
    assert adapter.select_epoch_ops(0) == ("split", "prune")
    assert adapter.select_epoch_ops(1) == ("prune", "split")
    assert adapter.expand_to_legacy_grammar("split") == ("add_node", "bisect_edge")
    assert adapter.expand_to_legacy_grammar("prune") == ("remove_leaf", "remove_edge")


def test_disallowed_op_rejected() -> None:
    with pytest.raises(ValueError, match="disallows operation: reconnect"):
        ElasticPrincipalGraph(
            ElasticPrincipalGraphConfig(
                intrinsic_topology_enabled=True,
                intrinsic_topology_ops=("split", "reconnect"),
            )
        )


def test_epoch_cap_enforced_deterministically() -> None:
    model = ElasticPrincipalGraph(
        ElasticPrincipalGraphConfig(
            intrinsic_topology_enabled=True,
            intrinsic_topology_ops=("split", "prune"),
            intrinsic_topology_max_ops_per_epoch=1,
        )
    )
    assert model._resolved_epoch_grammar_sequence(1) == (("add_node", "bisect_edge"),)
    assert model._resolved_epoch_grammar_sequence(2) == (("remove_leaf", "remove_edge"),)
    assert model._resolved_epoch_grammar_sequence(3) == (("add_node", "bisect_edge"),)
