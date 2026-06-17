from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SUPPORTED_INTRINSIC_TOPOLOGY_OPS: Tuple[str, ...] = ("split", "prune")
DISALLOWED_INTRINSIC_TOPOLOGY_OPS: Tuple[str, ...] = ("reconnect", "refine", "merge")


def _validate_intrinsic_topology_ops(ops: Tuple[str, ...]) -> Tuple[str, ...]:
    normalized = tuple(str(op) for op in ops)
    if len(normalized) == 0:
        raise ValueError("intrinsic topology ops must be nonempty.")

    for op in normalized:
        if op in DISALLOWED_INTRINSIC_TOPOLOGY_OPS:
            raise ValueError(
                f"intrinsic topology MVP disallows operation: {op}. "
                "Allowed ops are split and prune only."
            )
        if op not in SUPPORTED_INTRINSIC_TOPOLOGY_OPS:
            raise ValueError(
                f"unsupported intrinsic topology operation: {op}. "
                "Allowed ops are split and prune only."
            )
    return normalized


@dataclass(frozen=True)
class IntrinsicTopologyAdapter:
    ops: Tuple[str, ...] = ("split", "prune")
    max_ops_per_epoch: int = 1

    def __post_init__(self) -> None:
        validated_ops = _validate_intrinsic_topology_ops(tuple(self.ops))
        if int(self.max_ops_per_epoch) < 1:
            raise ValueError("intrinsic topology max_ops_per_epoch must be at least 1.")
        object.__setattr__(self, "ops", validated_ops)
        object.__setattr__(self, "max_ops_per_epoch", int(self.max_ops_per_epoch))

    def select_epoch_ops(self, epoch_index: int) -> Tuple[str, ...]:
        if len(self.ops) == 0:
            return ()
        start = int(epoch_index) % len(self.ops)
        rotated = self.ops[start:] + self.ops[:start]
        return tuple(rotated[: self.max_ops_per_epoch])

    def expand_to_legacy_grammar(self, op: str) -> Tuple[str, ...]:
        op = str(op)
        if op == "split":
            return ("add_node", "bisect_edge")
        if op == "prune":
            return ("remove_leaf", "remove_edge")
        raise ValueError(
            f"unsupported intrinsic topology operation: {op}. "
            "Allowed ops are split and prune only."
        )
