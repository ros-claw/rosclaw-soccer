"""Diagnostic neural feedback from GPU q/v; CPU forward is NOT integration.

Original177 foundation only, no learned adapter, optimizer, recorded actions or
future outcomes. Native forward/Jacobian observation reads are counted. This
hybrid observation implementation is not an all-GPU training environment.
"""

import mujoco
import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import observation98, target29
from rosclaw_soccer.rsi.native_reference_observation import native_reference_observation177


class FoundationFeedback:
    def __init__(self, *, model, layout, params, bank, indices, plans, actor):
        self.model, self.layout, self.params = model, layout, params
        self.bank, self.indices, self.plans = bank, indices, plans
        self.actor = actor.eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.data = [mujoco.MjData(model) for _ in indices]
        self.phase = bank["phase_before"][indices].astype(np.float32).copy()
        self.last = bank["previous_action"][indices].copy()
        self.torso = model.body("robot/torso_link").id
        self.control = self.neural_rows = self.native_forward_calls = 0

    def next(self, qpos, qvel, control):
        if control != self.control or not 0 <= control < 84:
            raise ValueError("one continuous actual24+60 control clock required")
        if (
            qpos.shape != (len(self.indices), 43)
            or qvel.shape != (len(self.indices), 41)
            or not np.isfinite(qpos).all()
            or not np.isfinite(qvel).all()
        ):
            raise ValueError("actual finite GPU state required")
        observations, phases = [], []
        for w, (data, index) in enumerate(zip(self.data, self.indices, strict=True)):
            data.qpos[:] = qpos[w]
            data.qvel[:] = qvel[w]
            mujoco.mj_forward(self.model, data)
            self.native_forward_calls += 1
            if control < 24:
                obs, phase = native_reference_observation177(
                    data=data,
                    layout=self.layout,
                    params=self.params,
                    plan=self.plans[int(self.bank["source_id"][index])],
                    frame=int(self.bank["reference_frame"][index]) + control,
                    origin=self.bank["origin"][index],
                    goal=self.bank["goal_world"][index],
                    last_action=self.last[w],
                    phase=self.phase[w],
                    torso=self.torso,
                )
            else:
                q, v = self.layout.canonical(data)
                obs, phase = observation98(
                    q,
                    v,
                    self.layout.angular_velocity_body(data),
                    np.zeros(3, dtype=np.float32),
                    self.last[w],
                    self.phase[w],
                    self.params,
                )
                obs = np.r_[obs, np.zeros(79, dtype=np.float32)]
            observations.append(obs)
            phases.append(phase)
        observations = np.stack(observations)
        with torch.no_grad():
            action = self.actor(torch.from_numpy(observations)).numpy()
        self.neural_rows += len(self.indices)
        targets = np.stack([target29(x, self.params) for x in action])
        self.phase[:] = phases
        self.last[:] = action
        self.control += 1
        return dict(observation=observations, action=action, target=targets)
