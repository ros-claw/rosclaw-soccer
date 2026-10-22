"""Declared SIM reference-conditioning teacher, NOT learned motor output.

Starts from the receiving body's measured joint pose at the actual handoff.
Only the reference input to the frozen NN changes; never a joint target,
physical body state or saved-action interpolation. Native tests must qualify
every nonzero bridge horizon. No prior standing/kicking proof is inherited.
"""

import numpy as np


class CausalReferenceBridge:
    def __init__(self, initial_joint_pos_policy: np.ndarray, *, controls: int) -> None:
        if (
            type(controls) is not int
            or not 1 <= controls <= 50
            or initial_joint_pos_policy.shape != (29,)
            or not np.isfinite(initial_joint_pos_policy).all()
        ):
            raise ValueError(
                "measured finite policy-order29 pose and explicit1..50 controls required"
            )
        self.initial = initial_joint_pos_policy.astype(np.float32, copy=True)
        self.controls = controls

    def reference(
        self, position: np.ndarray, velocity: np.ndarray, angular: np.ndarray, *, frame: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            type(frame) is not int
            or frame < 0
            or any(
                a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all()
                for a, shape in zip(
                    (position, velocity, angular), ((29,), (29,), (3,)), strict=True
                )
            )
        ):
            raise ValueError("own monotonic task frame and exact finite reference ABI required")
        if frame >= self.controls:
            return position.copy(), velocity.copy(), angular.copy()
        u = frame / self.controls
        weight = 3 * u * u - 2 * u * u * u
        derivative = (6 * u - 6 * u * u) / (0.02 * self.controls)
        # Initial pose is stationary as a declared reference, not an assertion
        # that the actual body is stationary. Actual joint velocity remains in
        # the NN's separate measured-proprioception observation throughout.
        q = (1 - weight) * self.initial + weight * position
        v = weight * velocity + derivative * (position - self.initial)
        w = weight * angular
        return q.astype(np.float32), v.astype(np.float32), w.astype(np.float32)
