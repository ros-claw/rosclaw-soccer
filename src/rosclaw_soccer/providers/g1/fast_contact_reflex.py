"""Private, stateless numerical 500 Hz G1 contact-residual proposals.

No simulator, optimizer, transport, activation or execution authority. The caller
owns the physical clock, observation provenance, control arbitration and guards.
This complements a frozen whole-body policy; it is not an end-to-end controller.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.g1_fast_contact_reflex.81x128x128x29.float64.500hz.v1"
PHYSICS_PERIOD_SEC = 0.002


def fast_contact_features(
    *,
    policy_frame: int,
    contact_latched: bool,
    joint_position: NDArray[Any],
    joint_velocity: NDArray[Any],
    ball_minus_foot_world: NDArray[Any],
    ball_velocity_world: NDArray[Any],
    root_velocity: NDArray[Any],
    foot_spatial_velocity: NDArray[Any],
    goal_minus_ball_world: NDArray[Any],
) -> NDArray[np.float64]:
    """Current pre-physics state, not post-step labels or predicted ball states.

    Foot origin is the right ankle body origin. Spatial foot velocity is MuJoCo's
    six-component cvel at that body's subtree COM reference, not foot-point linear
    velocity. Root velocity follows the world's six free-joint tangent components.
    World axes and the committed goal must match training; no automatic mirroring.
    """
    if (
        type(policy_frame) is not int
        or not 0 <= policy_frame <= 100000
        or type(contact_latched) is not bool
    ):
        raise ValueError("explicit bounded policy frame and causal contact latch required")
    pieces = [np.asarray([(policy_frame - 256) / 10, float(contact_latched)])]
    for value, size, divisor in (
        (joint_position, 29, 1),
        (joint_velocity, 29, 10),
        (ball_minus_foot_world, 3, 1),
        (ball_velocity_world, 3, 10),
        (root_velocity, 6, 5),
        (foot_spatial_velocity, 6, 10),
        (goal_minus_ball_world, 3, 10),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (size,)
            or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(value).all()
            or (np.abs(value) > 1e4).any()
        ):
            raise ValueError("finite bounded current-state arrays required")
        pieces.append(value.astype(np.float64) / divisor)
    return np.concatenate(pieces)


def fast_contact_weight_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "0.weight": (128, 81),
        "0.bias": (128,),
        "2.weight": (128, 128),
        "2.bias": (128,),
        "4.weight": (29, 128),
        "4.bias": (29,),
        "center": (81,),
        "scale": (81,),
        "output_mask": (29,),
    }


def _copy_weights(weights: Mapping[str, NDArray[Any]]) -> dict[str, NDArray[np.float64]]:
    shapes = fast_contact_weight_shapes()
    if not isinstance(weights, Mapping) or set(weights) != set(shapes):
        raise ValueError("exact fast-contact numeric weight members required")
    result = {}
    for key, shape in shapes.items():
        value = weights[key]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != np.float64
            or not np.isfinite(value).all()
            or (np.abs(value) > 1e4).any()
        ):
            raise ValueError("finite bounded float64 fast-contact weights required")
        result[key] = value.copy()
        result[key].setflags(write=False)
    if (result["scale"] < 0.1).any() or not np.isin(result["output_mask"], [0, 1]).all():
        raise ValueError("qualified normalization floor and binary output mask required")
    return result


def fast_contact_numeric_hash(weights: Mapping[str, NDArray[Any]]) -> str:
    """Canonical numeric commitment, deliberately distinct from an NPZ file hash."""
    copied = _copy_weights(weights)
    return str(
        hash_json(
            {
                "schema": SCHEMA,
                "weights": {
                    key: dict(shape=list(value.shape), sha256=hash_bytes(value.tobytes()))
                    for key, value in copied.items()
                },
            }
        )
    )


@dataclass(frozen=True)
class FastContactProposal:
    agent_id: str
    residual_torque_nm: tuple[float, ...]
    contract_hash: str
    activation_ceiling: str = "SIM_ONLY"


class G1FastContactReflex:
    """One immutable model per player, with explicit 246–266 phase support.

    Accepts already decoded numeric arrays, never untrusted pickle/model files.
    The phase gate is a declared specialist support bound, not an OOD certificate.
    Stateless proposals prove neither timing nor body identity nor safe execution.
    """

    def __init__(
        self,
        weights: Mapping[str, NDArray[Any]],
        *,
        agent_id: str,
        expected_numeric_hash: str,
        body_hash: str,
        observation_contract_hash: str,
    ) -> None:
        if (
            not isinstance(agent_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", agent_id) is None
            or any(
                not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
                for value in (expected_numeric_hash, body_hash, observation_contract_hash)
            )
        ):
            raise ValueError(
                "explicit player, numeric content, body and observation hashes required"
            )
        self._weights = _copy_weights(weights)
        digest = fast_contact_numeric_hash(self._weights)
        if digest != expected_numeric_hash:
            raise ValueError("fast-contact numeric hash mismatch")
        self._agent_id = agent_id
        self._contract_hash = str(
            hash_json(
                dict(
                    schema=SCHEMA,
                    agent_id=agent_id,
                    numeric_hash=digest,
                    body_hash=body_hash,
                    observation_contract_hash=observation_contract_hash,
                    physics_period_sec=PHYSICS_PERIOD_SEC,
                    activation_ceiling="SIM_ONLY",
                )
            )
        )

    @property
    def contract_hash(self) -> str:
        return self._contract_hash

    def propose(self, *, agent_id: str, observation: NDArray[Any]) -> FastContactProposal:
        if (
            agent_id != self._agent_id
            or not isinstance(observation, np.ndarray)
            or observation.shape != (81,)
            or observation.dtype != np.float64
            or not np.isfinite(observation).all()
            or (np.abs(observation) > 1e4).any()
            or observation[1] not in (0, 1)
        ):
            raise ValueError("matching player and finite float64 fast-contact observation required")
        phase = observation[0] * 10 + 256
        if not 0 <= phase <= 100000 or abs(phase - round(phase)) > 1e-9:
            raise ValueError("integer policy phase required; physical substep is not policy phase")
        result: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        if 246 <= phase <= 266:
            value = (observation - self._weights["center"]) / self._weights["scale"]
            for layer in (0, 2, 4):
                value = value @ self._weights[f"{layer}.weight"].T + self._weights[f"{layer}.bias"]
                if layer != 4:
                    value = np.tanh(value)
            if not np.isfinite(value).all():
                raise ValueError("nonfinite contact inference; caller must reject proposal")
            result = np.clip(value * 80, -120, 120) * self._weights["output_mask"]
        return FastContactProposal(
            self._agent_id, tuple(float(value) for value in result), self._contract_hash
        )
