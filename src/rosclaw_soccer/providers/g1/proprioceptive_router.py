"""Proprioceptive approach proposals for qualified G1 simulation courses.

The network selects a motion reference, not torques or task success. A caller
must retain contact attribution, body safety, continuous-handoff validation
and promotion gates. Course coordinates are explicit, not arbitrary world axes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.approach_router import (
    ApproachReferenceProposal,
    load_bounded_reference_parameters,
)
from rosclaw_soccer.sim.contracts import hash_json


def g1_approach_proprioception(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """74 features from 29-joint G1 + free ball, both in the declared course.

    Root velocity retains MuJoCo's global-linear/local-angular convention.
    This only reads observations; it never resets or modifies a physics world.
    """
    q, v = np.asarray(qpos), np.asarray(qvel)
    if (
        q.shape != (43,)
        or v.shape != (41,)
        or q.dtype.kind not in "fi"
        or v.dtype.kind not in "fi"
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("finite numeric G1 course qpos[43]/qvel[41] required")
    if abs(float(np.linalg.norm(q[3:7])) - 1.0) > 1e-3:
        raise ValueError("normalized root quaternion required")
    w, x, y, z = q[3:7]
    gravity = [2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1]
    features: np.ndarray = np.concatenate(
        (q[7:36], v[6:35] * 0.1, gravity, v[:3], v[3:6] * 0.2, q[36:39] - q[:3], v[35:38], [q[2]])
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("proprioceptive feature conversion overflow")
    return features


class G1ProprioceptiveApproachRouter:
    """74 → 128 → 64 → K frozen utility network with body/reference binding."""

    def __init__(
        self,
        manifest: Path,
        *,
        expected_body_actor_hash: str,
        expected_reference_library_hash: str,
    ) -> None:
        for value in (expected_body_actor_hash, expected_reference_library_hash):
            if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ValueError("explicit body and reference content bindings required")
        if manifest.stat().st_size > 32768:
            raise ValueError("proprioceptive manifest exceeds size bound")
        metadata = json.loads(manifest.read_text())
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema") != "rosclaw_soccer.g1_proprioceptive_router.v1"
            or metadata.get("activation_ceiling") != "SIM_ONLY"
            or metadata.get("body_actor_hash") != expected_body_actor_hash
            or metadata.get("reference_library_hash") != expected_reference_library_hash
            or metadata.get("feature_contract") != "g1_course_proprioception_74.v1"
        ):
            raise ValueError("proprioceptive router bindings or feature contract differ")
        choices = metadata.get("reference_indices")
        if (
            not isinstance(choices, list)
            or not 2 <= len(choices) <= 32
            or any(type(i) is not int or not 0 <= i < 4096 for i in choices)
            or len(set(choices)) != len(choices)
        ):
            raise ValueError("unique bounded reference indices required")
        self.reference_indices = tuple(choices)
        self._mean = self._vector(metadata.get("feature_mean"))
        self._scale = self._vector(metadata.get("feature_scale"))
        if np.any(self._scale < 0.05):
            raise ValueError("proprioceptive scales must be at least 0.05")
        shapes = {
            "0.weight": (128, 74),
            "0.bias": (128,),
            "2.weight": (64, 128),
            "2.bias": (64,),
            "4.weight": (len(choices), 64),
            "4.bias": (len(choices),),
        }
        self._parameters, self.policy_hash = load_bounded_reference_parameters(
            manifest.with_suffix(".npz"), metadata.get("weights_hash"), shapes
        )
        self.contract_hash = str(hash_json(metadata))

    @staticmethod
    def _vector(value: object) -> np.ndarray:
        if (
            not isinstance(value, list)
            or len(value) != 74
            or any(type(x) not in (float, int) for x in value)
        ):
            raise ValueError("74 finite numeric normalization values required")
        result = np.asarray(value, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("finite proprioceptive normalization required")
        result.setflags(write=False)
        return result

    def propose(
        self, *, course_qpos: np.ndarray, course_qvel: np.ndarray
    ) -> ApproachReferenceProposal:
        features = g1_approach_proprioception(course_qpos, course_qvel)
        x = (features - self._mean) / self._scale
        if np.any(np.abs(x) > 8.0):
            raise ValueError("proprioception exceeds admitted normalization domain")
        p = self._parameters
        x = np.tanh(p["0.weight"] @ x + p["0.bias"])
        x = np.tanh(p["2.weight"] @ x + p["2.bias"])
        scores = p["4.weight"] @ x + p["4.bias"]
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite proprioceptive utility")
        index = int(np.argmax(scores))
        return ApproachReferenceProposal(
            reference_index=self.reference_indices[index],
            predicted_utility=float(scores[index]),
            policy_hash=self.policy_hash,
            observation_hash=str(hash_json(features.tolist())),
        )
