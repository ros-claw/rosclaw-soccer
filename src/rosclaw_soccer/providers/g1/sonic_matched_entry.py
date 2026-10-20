"""Constrained, matched-time reference handoff; never moves an actual body.

Only23 non-wrist JOINT channels match finite-difference velocity. Root pose and
wrist interpolation stay identical to the matched-time position control.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


class InfeasibleMatchedBridge(ValueError):
    """Finite proposal not available; caller must not retry a faulted actor."""

    def __init__(self, attempts: tuple[dict[str, Any], ...]) -> None:
        super().__init__("NO_FEASIBLE_MATCHED_BRIDGE")
        self.attempts = attempts


@dataclass(frozen=True)
class MatchedEntryPair:
    position: SonicPoseReference
    velocity: SonicPoseReference
    entry_intervals: int
    source_preparation_hash: str
    measured_velocity_hash: str
    position_initial_velocity_gap_rad_s: float
    velocity_initial_velocity_gap_rad_s: float
    maximum_joint_correction_rad: float
    native_joint_limits_hash: str
    rejected_intervals: tuple[dict[str, Any], ...]

    def report(self, mode: str) -> dict[str, Any]:
        if mode not in ("matched_position", "matched_velocity"):
            raise ValueError("explicit matched entry mode required")
        reference = self.position if mode == "matched_position" else self.velocity
        return dict(
            schema="soccer.sonic_matched_joint_entry.v1",
            mode=mode,
            reference_hash=reference.contract_hash,
            entry_intervals=self.entry_intervals,
            source_preparation_hash=self.source_preparation_hash,
            measured_velocity_hash=self.measured_velocity_hash,
            position_initial_velocity_gap_rad_s=self.position_initial_velocity_gap_rad_s,
            velocity_initial_velocity_gap_rad_s=self.velocity_initial_velocity_gap_rad_s,
            maximum_joint_correction_rad=self.maximum_joint_correction_rad,
            native_joint_limits_hash=self.native_joint_limits_hash,
            rejected_intervals=[dict(x) for x in self.rejected_intervals],
            matched_joint_count=23,
            root_or_wrist_velocity_matched=False,
            physical_body_or_history_modified=False,
            dynamics_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )


def _joint_bridge(
    start: np.ndarray, end: np.ndarray, start_v: np.ndarray, end_v: np.ndarray, intervals: int
) -> np.ndarray:
    u = 1 / intervals

    def powers(x: float) -> list[float]:
        return [x**i for i in range(6)]

    matrix = np.array(
        [powers(0), powers(u), powers(1 - u), powers(1), [0, 0, 2, 0, 0, 0], [0, 0, 2, 6, 12, 20]],
        dtype=np.float64,
    )
    values = np.stack(
        (
            start,
            start + 0.02 * start_v,
            end - 0.02 * end_v,
            end,
            np.zeros_like(start),
            np.zeros_like(start),
        )
    )
    coefficients = np.linalg.solve(matrix, values)
    curve = np.vander(np.arange(intervals + 1) / intervals, N=6, increasing=True) @ coefficients
    curve[0], curve[-1] = start, end
    return curve


def prepare_matched_entry_pair(
    source_poses: np.ndarray,
    measured_pose: np.ndarray,
    measured_qvel: np.ndarray,
    native_joint_limits: np.ndarray,
    *,
    source_evidence_hash: str,
) -> MatchedEntryPair:
    if (
        type(measured_qvel) is not np.ndarray
        or measured_qvel.shape != (41,)
        or measured_qvel.dtype != np.float64
        or not np.isfinite(measured_qvel).all()
        or type(native_joint_limits) is not np.ndarray
        or native_joint_limits.shape != (29, 2)
        or native_joint_limits.dtype != np.float64
        or not np.isfinite(native_joint_limits).all()
        or np.any(native_joint_limits[:, 0] >= native_joint_limits[:, 1])
    ):
        raise ValueError("finite canonical41 velocity and native29 joint limits required")
    # Existing preparation performs strict pose/source validation, not repair.
    initial = prepare_sonic_reference(
        source_poses,
        measured_pose,
        source_frequency_hz=120,
        entry_intervals=50,
        source_evidence_hash=source_evidence_hash,
    )
    ids = np.array([i for i, n in enumerate(G1_DDS_JOINT_NAMES) if "wrist" not in n])
    if len(ids) != 23:
        raise ValueError("exact23 non-wrist G1 reference channels required")
    speed = measured_qvel[6:35][ids]
    velocity_hash = hash_bytes(measured_qvel.tobytes())
    source = np.asarray(initial.reference.poses)[50:]
    end_v = (source[1, 7 + ids] - source[0, 7 + ids]) / 0.02
    attempts: list[dict[str, Any]] = []
    for intervals in range(10, 51):
        try:
            position = prepare_sonic_reference(
                source_poses,
                measured_pose,
                source_frequency_hz=120,
                entry_intervals=intervals,
                source_evidence_hash=source_evidence_hash,
            )
        except ValueError:
            # Fixed inputs already passed the50-interval preparation. Only
            # candidate continuity can fail when varying this interval count.
            attempts.append(dict(entry_intervals=intervals, reason="POSITION_CONTINUITY_REJECTED"))
            continue
        poses = np.asarray(position.reference.poses).copy()
        changed = poses.copy()
        changed[: intervals + 1, 7 + ids] = _joint_bridge(
            poses[0, 7 + ids], poses[intervals, 7 + ids], speed, end_v, intervals
        )
        if not np.isfinite(changed).all():
            raise ValueError("nonfinite matched-reference derivation")
        correction = float(
            np.max(abs(changed[: intervals + 1, 7 + ids] - poses[: intervals + 1, 7 + ids]))
        )
        violations = sorted(
            {
                G1_DDS_JOINT_NAMES[j]
                for p in (poses, changed)
                for j in np.flatnonzero(
                    np.any(
                        (p[: intervals + 1, 7:36] < native_joint_limits[:, 0])
                        | (p[: intervals + 1, 7:36] > native_joint_limits[:, 1]),
                        axis=0,
                    )
                )
            }
        )
        if correction > 0.12 or violations:
            attempts.append(
                dict(
                    entry_intervals=intervals,
                    reason="REFERENCE_TRUST_REGION_OR_NATIVE_LIMIT",
                    maximum_joint_correction_rad=correction,
                    outside_limit_joints=violations,
                )
            )
            continue
        evidence = hash_json(
            dict(
                operation="matched23_joint_finite_difference_bridge_v1",
                source_preparation=position.contract_hash,
                measured_velocity=velocity_hash,
                intervals=intervals,
            )
        )
        try:
            velocity = SonicPoseReference(
                tuple(tuple(float(x) for x in row) for row in changed), evidence
            )
        except ValueError:
            # Only candidate continuity checks may be infeasible; inputs were
            # fully validated by the original preparation above.
            attempts.append(dict(entry_intervals=intervals, reason="VELOCITY_CONTINUITY_REJECTED"))
            continue
        velocity_gap = float(
            np.max(abs((changed[1, 7 + ids] - changed[0, 7 + ids]) / 0.02 - speed))
        )
        end_gap = float(
            np.max(
                abs((changed[intervals, 7 + ids] - changed[intervals - 1, 7 + ids]) / 0.02 - end_v)
            )
        )
        if max(velocity_gap, end_gap) > 1e-8:
            raise ValueError("matched reference finite-difference constraint not met")
        untouched = [i for i in range(36) if i not in 7 + ids]
        if (
            not np.array_equal(changed[:, untouched], poses[:, untouched])
            or not np.array_equal(changed[intervals:], poses[intervals:])
            or not np.array_equal(changed[0], measured_pose)
        ):
            raise ValueError("matched entry changed root/wrist/source or measured entry")
        position_gap = float(np.max(abs((poses[1, 7 + ids] - poses[0, 7 + ids]) / 0.02 - speed)))
        return MatchedEntryPair(
            position.reference,
            velocity,
            intervals,
            position.contract_hash,
            velocity_hash,
            position_gap,
            velocity_gap,
            correction,
            hash_bytes(native_joint_limits.tobytes()),
            tuple(attempts),
        )
    raise InfeasibleMatchedBridge(tuple(attempts))
