"""Numeric SIM reference decoder using the existing Core causal GRU.

No Runtime, simulator, fitting, dataset admission, driver or executor. Outputs
are proposals BEFORE the unchanged parent motor, never torque commands. A
caller must authenticate training and re-evaluate the parent on its own body.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.causal_residual_memory as memory_module
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi import pre_parent_reference_features as feature_module
from rosclaw_soccer.rsi.contact_motor_primitive import JOINT_NAMES
from rosclaw_soccer.rsi.taskspace_swing_probe import MAX_JOINT_DELTA_RAD
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _array(value: Any, shape: tuple[int, ...], bound: float) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if (
        raw.shape != shape
        or raw.dtype != np.float64
        or not np.isfinite(raw).all()
        or np.max(np.abs(raw)) > bound
    ):
        raise ValueError("aligned finite float64 numeric reference boundary required")
    return raw


class ReferenceStudentDecoder:
    """Per-episode owned numeric state, not an activated or qualified policy.

    Raw numeric weights carry no physical provenance. This adapter makes no
    claim that weights were trained at all. All permissions stay false, even
    when the numeric constructor accepts a nonzero head.
    """

    def __init__(
        self,
        parameters: dict[str, Any],
        *,
        mean: NDArray[np.float64],
        scale: NDArray[np.float64],
        world_hash: str,
        parent_model_hash: str,
    ) -> None:
        if any(
            type(h) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", h) is None
            for h in (world_hash, parent_model_hash)
        ):
            raise ValueError("explicit world and unchanged parent identity required")
        self._memory = memory_module.CausalResidualMemory(parameters)
        if (
            self._memory.input_dimension,
            self._memory.hidden_dimension,
            self._memory.output_dimension,
        ) != (135, 64, 12) or G1_DDS_JOINT_NAMES[:12] != JOINT_NAMES:
            raise ValueError("new 135-clock-feature / 64-GRU / 12-canonical-leg interface required")
        self._mean = _array(mean, (135,), 1e6).copy()
        self._scale = _array(scale, (135,), 1e6).copy()
        if np.any(self._scale < 0.05) or self._mean[-1] != 0 or self._scale[-1] != 1:
            raise ValueError(
                "training-only normalization with unchanged remaining-clock unit required"
            )
        self._mean.setflags(write=False)
        self._scale.setflags(write=False)
        self._previous = np.zeros(12, dtype=np.float64)
        self._world_hash, self._parent_hash = world_hash, parent_model_hash

    @property
    def next_frame(self) -> int:
        return int(self._memory.next_index)

    def contract(self) -> dict[str, Any]:
        """Numeric identity, NOT training provenance or an execution permit."""
        value = dict(
            schema="soccer.rsi.pre_parent_reference_gru_decoder.v1",
            architecture=[135, 64, 12],
            feature_names=list(feature_module.FEATURE_NAMES),
            joint_names=list(JOINT_NAMES),
            world_hash=self._world_hash,
            parent_model_hash=self._parent_hash,
            parameters_hash=hash_json(self._memory.parameters()),
            normalization_hash=hash_json(
                dict(mean=self._mean.tolist(), scale=self._scale.tolist())
            ),
            source_hashes=dict(
                decoder=hash_bytes(Path(__file__).read_bytes()),
                features=hash_bytes(Path(feature_module.__file__).read_bytes()),
                core_memory=hash_bytes(Path(memory_module.__file__).read_bytes()),
            ),
            control_dt_s=0.02,
            official_horizon_frames=300,
            decision_start_frame=30,
            reference_cap_rad=MAX_JOINT_DELTA_RAD,
            action_kind="pre-parent leg reference increment; not torque or post-parent residual",
            baseline="navigation-only; original contact teacher is not added",
            zero_output_claim="bitwise navigation target only; not full teacher trajectory parity",
            projection="no enlargement of existing navigation joint-limit violation",
            parent_must_recompute_after_reference=True,
            weights_owned_and_read_only=True,
            previous_reference_is_own_actually_projected_action=True,
            state_reset="construct a new decoder per episode; no implicit reset",
            normalization_clip=8.0,
            training_provenance_authenticated=False,
            activation_ceiling="SIM_ONLY",
            physics_qualified=False,
            teacher_qualified=False,
            runtime_execution_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
            fresh_holdout_open_authorized=False,
        )
        return dict(value, contract_hash=hash_json(value))

    def advance(
        self,
        measured_history: dict[str, Any],
        *,
        frame: int,
        navigation_target: NDArray[np.float64],
        joint_limits: NDArray[np.float64],
    ) -> dict[str, NDArray[Any]]:
        if type(frame) is not int or frame != self.next_frame or not 0 <= frame < 300:
            raise ValueError("one original ordered task frame at a time; no implicit episode reset")
        base = _array(navigation_target, (29,), 100)
        bounds = _array(joint_limits, (29, 2), 100)
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("original canonical ordered joint limits required")
        feature = feature_module.actor_features(
            measured_history,
            frame=frame,
            navigation_target=base,
            previous_applied_reference=self._previous,
        )
        normalized = np.clip((feature - self._mean) / self._scale, -8, 8)
        desired = self._memory.step(normalized, index=frame) * MAX_JOINT_DELTA_RAD
        # Do not force a previously out-of-box Foundation target into range:
        # the proposal may only maintain or reduce that existing violation.
        low = np.maximum(np.minimum(0.0, bounds[:12, 0] - base[:12]), -MAX_JOINT_DELTA_RAD)
        high = np.minimum(np.maximum(0.0, bounds[:12, 1] - base[:12]), MAX_JOINT_DELTA_RAD)
        applied = np.zeros(12) if frame < 30 else np.clip(desired, low, high)
        target = base.copy()
        # The numerical zero path must preserve signed zero and every bit of
        # the original 29 targets; do not add a synthetic all-zero vector.
        if np.any(applied != 0):
            target[:12] += applied
        self._previous = applied.copy()
        return dict(
            input_features=feature.copy(),
            normalized_input_features=normalized,
            requested_reference_increment_rad=desired,
            applied_reference_increment_rad=applied,
            pre_parent_joint_target_rad=target,
            recurrent_hidden_state=self._memory.state,
        )
