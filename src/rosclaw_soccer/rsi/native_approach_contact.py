"""New SIM approach/contact protocol; no learning reward or old qualification.

Reuse original native PD, measured observation, anatomical tracking and world
freeze mechanics, but not the old24 deadline. Hand off using observed10N foot
contact+14 following controls, or the actual source end, never a lead label.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.rsi.contact_option_clock import enter_measured_recovery
from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse
from rosclaw_soccer.rsi.native_recovery_tail import NativeRecoveryTail


class NativeApproachContactCourse(NativeContactCourse):
    protocol = "native-approach-measured-contact-v1"

    def __init__(self, **kwargs: Any) -> None:
        # Preserve the old constructor's24-sample bank admission. Do not mutate
        # the qualified class or pretend its qualification covers this task.
        super().__init__(**kwargs)
        for frame, source in zip(self.bank["reference_frame"], self.bank["source_id"], strict=True):
            remaining = len(self.plans[int(source)]["joint"]) - int(frame)
            if not 1 <= remaining <= 300:
                raise ValueError("entire finite source remainder must be within300 controls")
        self.window = 300

    def reset(self) -> NDArray[np.float32]:
        obs = super().reset()
        self.reference_horizon = np.array(
            [
                len(self.plans[int(self.bank["source_id"][i])]["joint"])
                - int(self.bank["reference_frame"][i])
                for i in self.index
            ],
            dtype=np.int64,
        )
        self.first_contact_control = np.full(self.worlds, -1, dtype=np.int64)
        self.head_failed = np.zeros(self.worlds, dtype=bool)
        return obs

    def step(self, action: NDArray[np.float32]) -> dict[str, NDArray[Any]]:
        info = super().step(action)
        try:
            active = info["valid"]
            contact = (info["forces"][:, :, :2] > 10).any(axis=(1, 2)) & active
            first = contact & (self.first_contact_control < 0)
            self.first_contact_control[first] = self.elapsed[first] - 1
            q, rq = info["qpos"], self.layout.rq
            height = q[:, :, rq + 2].min(axis=1)
            upright = (1 - 2 * (q[:, :, rq + 4] ** 2 + q[:, :, rq + 5] ** 2)).min(axis=1)
            body_failed = active & ((height < 0.65) | (upright < 0.9))
            terminated = info["terminated"] | body_failed
            handoff = np.zeros(self.worlds, dtype=bool)
            for i in np.flatnonzero(active & ~terminated):
                first_control = int(self.first_contact_control[i])
                handoff[i] = enter_measured_recovery(
                    completed_controls=int(self.elapsed[i]),
                    first_contact_control=None if first_control < 0 else first_control,
                    reference_horizon=int(self.reference_horizon[i]),
                )
            if (info["truncated"] & ~terminated & ~handoff).any():
                raise ValueError("legacy deadline ended before actual measured approach boundary")
            self.head_failed |= terminated
            self.ended |= terminated | handoff
            info["terminated"] = terminated
            info["truncated"] = handoff
            info["ended"] = self.ended.copy()
            info["body_failed"] = body_failed
            info["minimum_height"] = height
            info["minimum_upright"] = upright
            info["first_contact_control"] = self.first_contact_control.copy()
            # A positive per-control short-course reward could reward waiting
            # in a longer task. It is diagnostics ONLY; a learner cannot consume
            # this as the old reward interface without explicitly changing it.
            info["legacy24_reward_diagnostic"] = info.pop("reward")
            return info
        except Exception:
            self.faulted = True
            raise


class NativeApproachRecoveryTail(NativeRecoveryTail):
    protocol = "native-approach-recovery60-probe-v1"

    def __init__(self, course: NativeApproachContactCourse) -> None:
        if (
            not isinstance(course, NativeApproachContactCourse)
            or course.protocol != "native-approach-measured-contact-v1"
            or course.batch is None
            or course._closed
            or course.faulted
            or not course.ended.all()
        ):
            raise ValueError("completed unfaulted new approach course required")
        for i in np.flatnonzero(~course.head_failed):
            first = int(course.first_contact_control[i])
            if not enter_measured_recovery(
                completed_controls=int(course.elapsed[i]),
                first_contact_control=None if first < 0 else first,
                reference_horizon=int(course.reference_horizon[i]),
            ):
                raise ValueError("cannot bypass measured contact/reference end handoff")
        self.course, self.batch = course, course.batch
        self.model, self.layout, self.params = course.model, course.layout, course.params
        self.worlds = course.worlds
        self.phase = course.phase.copy()
        self.last_action = course.last_action.copy()
        self.ended = course.head_failed.copy()
        self.failed = course.head_failed.copy()
        self.elapsed = np.zeros(self.worlds, dtype=np.int64)
        self._observation = course.observation.copy()
        self._next_phase = self.phase.copy()
        self.faulted = False
        self._prepare(np.flatnonzero(~self.ended))
