"""Explicit SIM parent-conditioned velocity corrections, not absolute cloning.

The verified geometry/clearance parent stays in the loop. A separate zero head
retains it exactly. Old absolute command-head weights are never evaluated.
Uses Core's dimension-independent numeric primitives, not G1 torque semantics.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_role_command import NativeRoleCommandPolicy

ROLE_RESIDUAL_SCHEMA = "soccer.native_role_parent_residual.v1"
ROLE_RESIDUAL_SEMANTICS = "PARENT_PLUS_BOUNDED_BODY_VELOCITY_RESIDUAL_BEFORE_FINAL_GUARD"
ROLE_COMMAND_LIMITS = np.array([1.0, 0.5, 1.0], np.float64)
ROLE_COMMAND_LIMITS.flags.writeable = False
_HASH = re.compile(r"sha256:[0-9a-f]{64}")


def qualified_role_parent_hash() -> str:
    """Function content + physical caller clock, independent of checkout paths."""
    from rosclaw_soccer.rsi import native_role_teacher, native_team_intent
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.skills import planar_interception
    from rosclaw_soccer.world import player_clearance

    sources = {}
    for module in (native_role_teacher, native_team_intent, planar_interception, player_clearance):
        file = module.__file__
        if not isinstance(file, str):
            raise ValueError("ordinary qualified parent implementation source required")
        sources[module.__name__] = "sha256:" + hashlib.sha256(Path(file).read_bytes()).hexdigest()
    return str(
        hash_json(
            dict(schema="soccer.qualified_coached_role_parent.v1", control_hz=50, sources=sources)
        )
    )


def serialize_role_residual(
    *,
    encoder_hash: str,
    binding_hash: str,
    parent_hash: str,
    dataset_hash: str,
    update_index: int = 0,
    maximum_ratio: float = 0.05,
    head_weight: np.ndarray | None = None,
    head_bias: np.ndarray | None = None,
) -> dict[str, Any]:
    value = dict(
        schema=ROLE_RESIDUAL_SCHEMA,
        action_semantics=ROLE_RESIDUAL_SEMANTICS,
        encoder_hash=encoder_hash,
        binding_hash=binding_hash,
        parent_hash=parent_hash,
        dataset_hash=dataset_hash,
        update_index=update_index,
        maximum_ratio=maximum_ratio,
        observation_dim=121,
        feature_dim=131,
        head_weight=np.zeros((3, 131)).tolist()
        if head_weight is None
        else np.asarray(head_weight).tolist(),
        head_bias=np.zeros(3).tolist() if head_bias is None else np.asarray(head_bias).tolist(),
        hardware_authorized=False,
        activation_ceiling="SIM_ONLY",
    )
    _decode(value)
    return value


def _decode(value: dict[str, Any]) -> dict[str, np.ndarray]:
    if set(value) != {
        "schema",
        "action_semantics",
        "encoder_hash",
        "binding_hash",
        "parent_hash",
        "dataset_hash",
        "update_index",
        "maximum_ratio",
        "observation_dim",
        "feature_dim",
        "head_weight",
        "head_bias",
        "hardware_authorized",
        "activation_ceiling",
    } or (
        value["schema"] != ROLE_RESIDUAL_SCHEMA
        or value["action_semantics"] != ROLE_RESIDUAL_SEMANTICS
        or value["observation_dim"] != 121
        or value["feature_dim"] != 131
        or value["hardware_authorized"] is not False
        or value["activation_ceiling"] != "SIM_ONLY"
        or type(value["update_index"]) is not int
        or not 0 <= value["update_index"] <= 10**9
        or type(value["maximum_ratio"]) not in (float, int)
        or not math.isfinite(value["maximum_ratio"])
        or not 0 < value["maximum_ratio"] <= 0.1
        or any(
            not isinstance(value[k], str) or not _HASH.fullmatch(value[k])
            for k in ("encoder_hash", "binding_hash", "parent_hash", "dataset_hash")
        )
    ):
        raise ValueError("strict bound SIM-only role residual contract required")
    tensors: dict[str, np.ndarray] = {}
    for key, shape in (("head_weight", (3, 131)), ("head_bias", (3,))):
        array = np.array(value[key], np.float32, copy=True)
        if array.shape != shape or not np.isfinite(array).all() or (abs(array) > 1e4).any():
            raise ValueError("finite owned exact-shaped residual tensors required")
        array.flags.writeable = False
        tensors[key] = array
    if value["update_index"] == 0 and any(np.any(v) for v in tensors.values()):
        raise ValueError("untrained initialization must be an exact zero head")
    return tensors


@dataclass(frozen=True)
class RoleResidualProposal:
    context121: np.ndarray
    features131: np.ndarray
    normalized_action: np.ndarray
    delta: np.ndarray
    raw_command: np.ndarray
    parent_command: np.ndarray
    exact_parent: bool
    maximum_ratio: float


class NativeRoleResidualPolicy:
    def __init__(
        self,
        path: Path,
        *,
        expected_hash: str,
        encoder: NativeRoleCommandPolicy,
        expected_encoder_hash: str,
        expected_binding: str,
        expected_parent_hash: str,
    ) -> None:
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size < 262144:
            raise ValueError("bounded ordinary residual model required")
        payload = path.read_bytes()
        if "sha256:" + hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError("exact residual artifact required")
        value = json.loads(payload)
        self.tensors = _decode(value)
        if (
            value["encoder_hash"] != expected_encoder_hash
            or encoder.artifact_hash != expected_encoder_hash
            or value["binding_hash"] != expected_binding
            or encoder.metadata["binding_hash"] != expected_binding
            or value["parent_hash"] != expected_parent_hash
        ):
            raise ValueError("exact encoder, body/gait and qualified parent required")
        self.metadata = {k: v for k, v in value.items() if k not in self.tensors}
        self.encoder = encoder

    def propose(
        self, observation: np.ndarray, parent: np.ndarray, *, latent_noise: np.ndarray | None = None
    ) -> RoleResidualProposal:
        from rosclaw.control.parent_residual import compose_parent_residual, parent_context

        parent = np.asarray(parent)
        if (
            parent.shape != (3,)
            or parent.dtype != np.float32
            or not np.isfinite(parent).all()
            or not -0.5 <= parent[0] <= 1
            or abs(parent[1]) > 0.5
            or abs(parent[2]) > 1
        ):
            raise ValueError("actual finite float32 qualified parent body command required")
        context = parent_context(np.asarray(observation), parent, ROLE_COMMAND_LIMITS)
        features = parent_context(self.encoder.encode(observation), parent, ROLE_COMMAND_LIMITS)
        latent = self.tensors["head_weight"] @ features + self.tensors["head_bias"]
        noise = np.zeros(3) if latent_noise is None else np.asarray(latent_noise, dtype=np.float64)
        if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(abs(noise) > 10):
            raise ValueError("finite bounded explicit latent exploration required")
        action = np.tanh(latent.astype(np.float64) + noise)
        delta = action * ROLE_COMMAND_LIMITS * self.metadata["maximum_ratio"]
        raw = compose_parent_residual(
            parent, delta, ROLE_COMMAND_LIMITS, maximum_ratio=self.metadata["maximum_ratio"]
        )
        return RoleResidualProposal(
            context,
            features,
            action,
            delta,
            raw,
            parent.copy(),
            not np.any(delta),
            float(self.metadata["maximum_ratio"]),
        )


def guard_role_residual(
    proposal: RoleResidualProposal,
    yaw: float,
    neighbor_offsets: np.ndarray,
    *,
    maximum_speed_mps: float,
) -> tuple[np.ndarray, bool, bool]:
    from rosclaw_soccer.world.player_clearance import propose_clearance_velocity

    neighbors = np.asarray(neighbor_offsets)
    if (
        type(proposal) is not RoleResidualProposal
        or not math.isfinite(yaw)
        or not 0.1 <= maximum_speed_mps <= 1.0
        or neighbors.ndim != 2
        or neighbors.shape[1] != 2
        or len(neighbors) > 16
        or not np.isfinite(neighbors).all()
        or any(
            np.asarray(v).shape != (3,) or not np.isfinite(v).all()
            for v in (proposal.delta, proposal.raw_command, proposal.parent_command)
        )
        or proposal.parent_command.dtype != np.float32
        or not -0.5 <= proposal.parent_command[0] <= 1
        or abs(proposal.parent_command[1]) > 0.5
        or abs(proposal.parent_command[2]) > 1
        or np.linalg.norm(proposal.parent_command[:2]) > maximum_speed_mps + 1e-6
        or proposal.exact_parent != (not np.any(proposal.delta))
        or not math.isfinite(proposal.maximum_ratio)
        or not 0 < proposal.maximum_ratio <= 0.1
        or np.any(abs(proposal.delta) > ROLE_COMMAND_LIMITS * proposal.maximum_ratio)
        or not np.array_equal(
            proposal.raw_command, proposal.parent_command.astype(np.float64) + proposal.delta
        )
    ):
        raise ValueError("finite linked role proposal and actual caller guard context required")
    if proposal.exact_parent:
        # The parent is already qualified and projected. Never project it twice.
        return proposal.parent_command.copy(), True, False
    command = proposal.raw_command.copy()
    command[2] = np.clip(command[2], -1, 1)
    c, s = math.cos(yaw), math.sin(yaw)
    forward, lateral = np.array([c, s]), np.array([-s, c])
    parent, budget = proposal.parent_command, ROLE_COMMAND_LIMITS * proposal.maximum_ratio
    # Projection must retain the parent's bounded deviation as well as body
    # domain/clearance, rather than undoing the residual's declared authority.
    planes = np.array(
        [
            [*forward, -0.5],
            [*(-forward), -1.0],
            [*lateral, -0.5],
            [*(-lateral), -0.5],
            [*forward, float(parent[0]) - budget[0]],
            [*(-forward), -float(parent[0]) - budget[0]],
            [*lateral, float(parent[1]) - budget[1]],
            [*(-lateral), -float(parent[1]) - budget[1]],
        ]
    )
    projected = propose_clearance_velocity(
        forward * command[0] + lateral * command[1],
        neighbors,
        maximum_speed_mps=maximum_speed_mps,
        clearance_m=0.6,
        influence_m=1.2,
        approach_gain=0.6,
        additional_halfplanes=planes,
    )
    if not projected.feasible:
        return parent.copy(), False, True
    world = np.asarray(projected.velocity_mps)
    final = np.array([world @ forward, world @ lateral, command[2]], np.float32)
    return final, True, bool(not np.array_equal(final.astype(np.float64), proposal.raw_command))
