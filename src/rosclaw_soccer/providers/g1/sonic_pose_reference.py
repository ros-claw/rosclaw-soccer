"""Immutable 50 Hz pose reference for a SIM_ONLY frozen-tracker probe.

This replaces navigation planning, not measured proprioception or physics.
Provenance and dynamic feasibility require external evidence; a hash alone
does not authenticate a demonstration or certify a collision-free motion.
"""

import re
from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class SonicPoseReference:
    poses: tuple[tuple[float, ...], ...]
    source_evidence_hash: str

    def __post_init__(self) -> None:
        if (
            type(self.poses) is not tuple
            or not 10 <= len(self.poses) <= 1001
            or type(self.source_evidence_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.source_evidence_hash) is None
            or any(
                type(row) is not tuple
                or len(row) != 36
                or any(type(x) not in (int, float) or not np.isfinite(x) for x in row)
                for row in self.poses
            )
        ):
            raise ValueError(
                "bounded immutable 50 Hz reference poses and evidence binding required"
            )
        poses = np.asarray(self.poses)
        if (
            np.any(abs(poses[:, :2]) > 50)
            or np.any((poses[:, 2] < 0.4) | (poses[:, 2] > 1.2))
            or not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1, rtol=0, atol=1e-5)
            or np.any(abs(poses[:, 7:]) > 3.2)
            or np.any(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) > 0.1)
            or np.any(abs(np.diff(poses[:, 7:], axis=0)) > 0.25)
            or np.any(abs(np.sum(poses[1:, 3:7] * poses[:-1, 3:7], axis=1)) < 0.98)
        ):
            raise ValueError(
                "finite continuous upright reference required; not a safety certificate"
            )

    @property
    def contract_hash(self) -> str:
        return str(
            hash_json(
                dict(
                    schema="soccer.sonic_pose_reference.v1",
                    poses=self.poses,
                    source_evidence_hash=self.source_evidence_hash,
                    frequency_hz=50,
                    joint_order="G1_DDS_JOINT_NAMES",
                    padding="explicit_terminal_hold",
                    navigation_commands_control_this_reference=False,
                    activation_ceiling="SIM_ONLY",
                )
            )
        )

    def initialize(self, measured_qpos: np.ndarray, *, frames: int) -> np.ndarray:
        """Require exact measured entry; return a private terminal-padded copy."""
        q = np.asarray(measured_qpos)
        if (
            q.shape != (36,)
            or q.dtype.kind not in "fiu"
            or not np.isfinite(q).all()
            or not np.allclose(q, np.asarray(self.poses[0]), rtol=0, atol=1e-7)
            or type(frames) is not int
            or not len(self.poses) <= frames <= 1200
        ):
            raise ValueError("reference must bind the current measured pose and bounded horizon")
        poses = np.asarray(self.poses, dtype=np.float64)
        return np.concatenate((poses, np.repeat(poses[-1:], frames - len(poses), axis=0)))
