"""SIM-only continuous zero-command recovery after the qualified contact24.

Owns no reset or reward. Reuses the SAME native data objects and original PD.
The caller must use the immutable foundation on active observations, not a
trained contact adapter. No reference end-frame extension or synthetic state.
"""

from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.velocity_policy import observation98, target29


class NativeRecoveryTail:
    protocol = "native-contact24-recovery60-probe-v1"
    window = 60

    def __init__(self, course: Any, *, contact_failed: NDArray[np.bool_]):
        if (
            course.protocol != "native-contact24-v1"
            or course.batch is None
            or course._closed
            or course.faulted
            or not course.ended.all()
            or contact_failed.shape != (course.worlds,)
            or contact_failed.dtype != np.bool_
            or np.any(~contact_failed & (course.elapsed != 24))
        ):
            raise ValueError("completed contact24 and explicit accumulated failures required")
        self.course, self.batch = course, course.batch
        self.model, self.layout, self.params = course.model, course.layout, course.params
        self.worlds = course.worlds
        self.phase = course.phase.copy()
        self.last_action = course.last_action.copy()
        self.ended = contact_failed.copy()
        self.failed = contact_failed.copy()
        self.elapsed = np.zeros(self.worlds, dtype=np.int64)
        self._observation = course.observation.copy()
        self._next_phase = self.phase.copy()
        self.faulted = False
        self._prepare(np.flatnonzero(~self.ended))

    @property
    def active_ids(self) -> NDArray[np.int64]:
        return np.flatnonzero(~self.ended).astype(np.int64)

    @property
    def observation(self) -> NDArray[np.float32]:
        if self.faulted or self.course._closed:
            raise ValueError("live recovery tail required")
        return np.asarray(self._observation, dtype=np.float32).copy()

    def _prepare(self, ids: NDArray[np.int64]) -> None:
        for i in ids:
            d = self.batch.data[i]
            mujoco.mj_forward(self.model, d)
            q, v = self.layout.canonical(d)
            obs, phase = observation98(
                q,
                v,
                self.layout.angular_velocity_body(d),
                np.zeros(3, dtype=np.float32),
                self.last_action[i],
                self.phase[i],
                self.params,
            )
            self._observation[i] = np.r_[obs, np.zeros(79, dtype=np.float32)]
            self._next_phase[i] = phase

    def step(self, action: NDArray[np.float32]) -> dict[str, NDArray[Any]]:
        if self.faulted or self.course._closed or self.ended.all():
            raise ValueError("active recovery tail required")
        if (
            action.shape != (self.worlds, 29)
            or action.dtype != np.float32
            or not np.isfinite(action).all()
        ):
            raise ValueError("finite float32 foundation actions required")
        ids, active = self.active_ids, ~self.ended.copy()
        before = self.observation
        phase, last = self.phase.copy(), self.last_action.copy()
        target = np.stack([target29(x, self.params) for x in action])
        ctrl = np.zeros((self.worlds, self.model.nu))
        ctrl[:, self.layout.ai] = target
        qs, vs, fs = [], [], []
        try:
            for _ in range(10):
                self.batch.advance(ctrl, substeps=1, world_ids=ids)
                qs.append(np.stack([d.qpos.copy() for d in self.batch.data]))
                vs.append(np.stack([d.qvel.copy() for d in self.batch.data]))
                forces = np.zeros((self.worlds, 3))
                for i in ids:
                    d, ai = self.batch.data[i], self.layout.ai
                    forces[i] = self.course._forces(d)
                    expected = np.clip(
                        self.model.actuator_gainprm[ai, 0] * target[i]
                        + self.model.actuator_biasprm[ai, 0]
                        + self.model.actuator_biasprm[ai, 1] * d.actuator_length[ai]
                        + self.model.actuator_biasprm[ai, 2] * d.actuator_velocity[ai],
                        self.model.actuator_forcerange[ai, 0],
                        self.model.actuator_forcerange[ai, 1],
                    )
                    if not np.allclose(d.actuator_force[ai], expected, atol=1e-8, rtol=0):
                        raise ValueError("original PD law changed during recovery")
                fs.append(forces)
            q, v, forces = (np.stack(x, axis=1) for x in (qs, vs, fs))
            rq = self.layout.rq
            height = q[:, :, rq + 2].min(axis=1)
            upright = (1 - 2 * (q[:, :, rq + 4] ** 2 + q[:, :, rq + 5] ** 2)).min(axis=1)
            failed = active & ((height < 0.65) | (upright < 0.9))
            self.elapsed[active] += 1
            timeout = active & ~failed & (self.elapsed >= self.window)
            self.phase[active] = self._next_phase[active]
            self.last_action[active] = action[active]
            self._prepare(ids)
            self.failed |= failed
            self.ended |= failed | timeout
            return dict(
                observation_before=before,
                observation=self.observation,
                phase_before=phase,
                previous_action=last,
                action=action.copy(),
                target=target,
                qpos=q,
                qvel=v,
                forces=forces,
                valid=active,
                terminated=failed,
                completed=timeout,
                ended=self.ended.copy(),
                minimum_height=height,
                minimum_upright=upright,
            )
        except Exception:
            self.faulted = True
            raise
