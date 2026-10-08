"""Optional fixed-policy reference compilation for SIM-only offline audits.

Never reuse a producer factory. Compile the original deterministic clipped
decoder once, and deeply own its untouched initial state for every audit.
Default auditors still use the original per-episode constructor.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.canonical_json_snapshot as snapshot_module
from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot

from rosclaw_soccer.rsi.recurrent_clipped_motor import CompiledRecurrentClippedMotor
from rosclaw_soccer.sim.contracts import hash_bytes


def _numeric_graph_hash(root: Any) -> str:
    """Bind trusted reference objects, including aliases and private state.

    No pickle, callbacks, arbitrary repr, or caller-provided object classes.
    Only built-in containers/numerics and project-owned reference objects are
    traversed. This is private prototype integrity, not physical provenance.
    """
    seen: dict[int, int] = {}
    retained: list[Any] = []
    tokens: list[Any] = []
    byte_count = 0

    def visit(value: Any, depth: int) -> None:
        nonlocal byte_count
        if depth > 128 or len(tokens) > 65536:
            raise ValueError("bounded reference object graph required")
        if value is None or type(value) in (str, bool, int, float):
            tokens.append([type(value).__name__, value])
            return
        if isinstance(value, np.generic):
            visit(value.item(), depth + 1)
            return
        if id(value) in seen:
            tokens.append(["alias", seen[id(value)]])
            return
        seen[id(value)] = len(seen)
        # Retain temporary public views too: recycled Python ids must never
        # masquerade as aliases later in the same graph traversal.
        retained.append(value)
        if type(value) is bytes:
            byte_count += len(value)
            tokens.append(["bytes", len(value), hashlib.sha256(value).hexdigest()])
        elif type(value) is np.ndarray:
            if value.dtype.kind not in "fiub" or not np.isfinite(value).all():
                raise ValueError("finite reference numeric arrays required")
            byte_count += value.nbytes
            tokens.append(
                [
                    "array",
                    value.shape,
                    value.dtype.str,
                    hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                ]
            )
        elif type(value) in (list, tuple):
            tokens.append([type(value).__name__, len(value)])
            for item in value:
                visit(item, depth + 1)
        elif type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("ordinary reference object keys required")
            # Serialized model/receipt subtrees are complete ordinary JSON.
            # Bind their EXACT typed content compactly instead of expanding
            # millions of scalar tokens. Object/array aliases remain bound;
            # aliases internal to JSON are not semantic model content.
            if type(value.get("schema")) is str:
                try:
                    encoded = json.dumps(
                        value, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                except TypeError:
                    pass  # Numeric reference objects still use explicit traversal.
                else:
                    byte_count += len(encoded)
                    if byte_count > 256 * 1024**2:
                        raise ValueError("bounded reference numeric/byte storage required")
                    tokens.append(["whole-json", len(encoded), hashlib.sha256(encoded).hexdigest()])
                    return
            tokens.append(["dict", sorted(value)])
            for key in sorted(value):
                visit(value[key], depth + 1)
        elif type(value).__module__ == "scipy.spatial._ckdtree":
            from scipy.spatial import cKDTree  # type: ignore[import-untyped]
            from scipy.spatial._ckdtree import cKDTreeNode  # type: ignore[import-untyped]

            names: tuple[str, ...]
            if type(value) is cKDTree:
                # Complete topology in a bounded dense representation, not
                # thousands of repeatedly expanded Python node dictionaries.
                nodes = [value.tree]
                rows = []
                for node in nodes:
                    if len(nodes) > 65536:
                        raise ValueError("bounded complete reference spatial tree required")
                    if not np.array_equal(
                        node.indices, value.indices[node.start_idx : node.end_idx]
                    ):
                        raise ValueError("complete reference node permutation differs")
                    child_ids = []
                    for child in (node.lesser, node.greater):
                        child_ids.append(-1 if child is None else len(nodes))
                        if child is not None:
                            nodes.append(child)
                    rows.append(
                        [
                            node.children,
                            node.start_idx,
                            node.end_idx,
                            node.level,
                            node.split,
                            node.split_dim,
                            *child_ids,
                        ]
                    )
                if len(nodes) != value.size:
                    raise ValueError("whole reference spatial topology required")
                names = (
                    "data",
                    "indices",
                    "mins",
                    "maxes",
                    "boxsize",
                    "leafsize",
                    "m",
                    "n",
                    "size",
                )
                fields = {name: getattr(value, name) for name in names}
                fields["complete_node_topology"] = np.asarray(rows, dtype=np.float64)
                tokens.append(["spatial-index", type(value).__qualname__])
                visit(fields, depth + 1)
                return
            elif type(value) is cKDTreeNode:
                names = (
                    "children",
                    "end_idx",
                    "greater",
                    "indices",
                    "lesser",
                    "level",
                    "split",
                    "split_dim",
                    "start_idx",
                )
            else:
                raise ValueError("exact reference spatial index type required")
            tokens.append(["spatial-index", type(value).__qualname__])
            visit({name: getattr(value, name) for name in names}, depth + 1)
        elif type(value) is CanonicalJSONSnapshot:
            value.verify()
            tokens.append(["canonical-snapshot"])
            visit({"data": value._data, "content_hash": value.content_hash}, depth + 1)
        elif type(value).__module__.startswith(("rosclaw.", "rosclaw_soccer.")):
            tokens.append(["object", type(value).__module__, type(value).__qualname__])
            visit(vars(value), depth + 1)
        else:
            raise ValueError("unsupported reference object graph")
        if byte_count > 256 * 1024**2:
            raise ValueError("bounded reference numeric/byte storage required")

    visit(root, 0)
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(tokens, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


class FixedRecurrentReferenceAuditCompiler:
    """One exact full deterministic policy; no seed rebinding or execution.

    Source authentication and full physics/Foundation auditing remain external.
    The caller must qualify actual archive parity before selecting this option
    in a new experiment. It is never automatically used by live experiments.
    """

    def __init__(self, policy: dict[str, Any]) -> None:
        if type(policy) is not dict or "recurrent_clipped_motor_proof" not in policy:
            raise ValueError("fixed original deterministic clipped policy required")
        self._policy = CanonicalJSONSnapshot(policy)
        paths = list(Path(__file__).parent.glob("*.py"))
        paths += list(Path(snapshot_module.__file__).parent.glob("*.py"))
        paths += [Path(__file__).parents[1] / "sim/contracts.py"]
        self._pins = {str(path): hash_bytes(path.read_bytes()) for path in paths}
        # The original complete constructor validates actor/critic, ancestry,
        # source, original preview, frozen parent and all action boundaries.
        self._prototype = CompiledRecurrentClippedMotor(self._policy.restore())
        self._prototype_hash = _numeric_graph_hash(self._prototype)
        spatial_module = sys.modules.get("scipy.spatial._ckdtree")
        if spatial_module is not None:
            spatial_path = Path(str(spatial_module.__file__))
            self._pins[str(spatial_path)] = hash_bytes(spatial_path.read_bytes())
        self._policy_hash = str(policy["policy_hash"])
        self._stable()

    def _stable(self) -> None:
        if (
            any(
                hash_bytes(Path(path).read_bytes()) != expected
                for path, expected in self._pins.items()
            )
            or self._policy.verify() != self._policy.content_hash
            or self._prototype._policy_hash != self._policy_hash
            or _numeric_graph_hash(self._prototype) != self._prototype_hash
        ):
            raise ValueError("fixed reference compiler source, policy or prototype changed")

    def verify_policy(self, policy: dict[str, Any]) -> None:
        self._stable()
        if CanonicalJSONSnapshot(policy).content_hash != self._policy.content_hash:
            raise ValueError("complete fixed reference policy differs")

    def new_episode(self) -> CompiledRecurrentClippedMotor:
        self._stable()
        episode = copy.deepcopy(self._prototype)
        if _numeric_graph_hash(episode) != self._prototype_hash:
            raise ValueError("whole original fresh reference episode changed")
        self._stable()
        return episode

    def contract(self) -> dict[str, Any]:
        self._stable()
        return {
            "schema": "soccer.rsi.fixed_original_reference_audit_compiler.v1",
            "policy_hash": self._policy_hash,
            "complete_policy_numeric_hash": self._policy.content_hash,
            "initial_reference_graph_hash": self._prototype_hash,
            "source_pins": dict(self._pins),
            "original_complete_reference_constructor_used_at_allocation": True,
            "producer_factory_reused": False,
            "all_initial_reference_objects_deeply_owned_per_episode": True,
            "physics_parity_requires_external_evidence": True,
            "actual_archive_parity_qualified_here": False,
            "physical_action_bounds_changed": False,
            "execution_ceiling": "OFFLINE_SIM_AUDIT_ONLY",
            "runtime_execution_authorized": False,
            "promotion_authorized": False,
            "hardware_authorized": False,
        }
