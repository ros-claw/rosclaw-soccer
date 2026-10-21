"""Strict SIM phase replay with actual ball/contact-body diagnostics.

Restores a complete recorded integration state, never invents missing initial
forces or warmstart. Replays recorded torques, not new neural inference.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_continuity_audit import identical, require_integration_fields
from rosclaw_soccer.rsi.paid_skill_handoff import snapshot


class PaidNativePhaseReplay:
    def __init__(self, auditor: Any) -> None:
        import mujoco

        self.auditor = auditor
        self.model = auditor.layout.model
        if mujoco.get_mjcb_control() is not None or self.model.opt.timestep != 0.002:
            raise ValueError("callback-free original 2ms SIM model required")
        self.feet = [self.model.body(s + "_ankle_roll_link").id for s in ("left", "right")]
        self.robot = set()
        for body in range(1, self.model.nbody):
            parent = body
            while parent not in (0, auditor.layout.pelvis):
                parent = int(self.model.body_parentid[parent])
            if parent == auditor.layout.pelvis:
                self.robot.add(body)
        self.native_steps = 0
        self.phases = 0

    def contacts(self, data: Any) -> tuple[np.ndarray, list[tuple[str, float]]]:
        import mujoco

        forces = np.zeros(3)
        nonfoot = []
        for ci in range(data.ncon):
            contact = data.contact[ci]
            x, y = (int(self.model.geom_bodyid[g]) for g in contact.geom)
            ball = self.auditor.layout.ball
            other = y if x == ball else x if y == ball else -1
            if other not in self.robot:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, ci, wrench)
            magnitude = float(np.linalg.norm(wrench[:3]))
            role = self.feet.index(other) if other in self.feet else 2
            forces[role] = max(forces[role], magnitude)
            if role == 2 and magnitude > 1:
                nonfoot.append((self.model.body(other).name, magnitude))
        return forces, nonfoot

    def replay(
        self,
        raw: dict[str, np.ndarray],
        *,
        entry: dict[str, np.ndarray],
        terminal: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        import mujoco

        require_integration_fields(entry)
        require_integration_fields(terminal)
        n = len(raw["target"])
        if not 1 <= n <= 300 or any(not np.isfinite(x).all() for x in raw.values()):
            raise ValueError("bounded finite recorded SIM phase required")
        m, layout = self.model, self.auditor.layout
        data = mujoco.MjData(m)
        for key, value in entry.items():
            if key == "time":
                data.time = float(value)
            else:
                getattr(data, key)[:] = value
        # Build unrecorded derived fields without consuming an integration step.
        mujoco.mj_forward(m, data)
        data.qacc_warmstart[:] = entry["qacc_warmstart"]
        warnings = tuple(int(w.number) for w in data.warning)
        events: dict[str, dict[str, Any]] = {}
        first_foot = None
        for frame in range(n):
            mujoco.mj_forward(m, data)
            if (
                not np.array_equal(data.qpos, raw["native_qpos_before"][frame])
                or not np.array_equal(data.qvel, raw["native_qvel_before"][frame])
                or float(data.time) != float(raw["time_before"][frame])
            ):
                raise ValueError("strict recorded control entry differs")
            for sub in range(10):
                data.ctrl[layout.ai] = raw["torque"][frame, sub]
                mujoco.mj_step(m, data)
                self.native_steps += 1
                q, v = layout.canonical(data)
                force, pairs = self.contacts(data)
                if (
                    not np.array_equal(q, raw["qpos"][frame, sub])
                    or not np.array_equal(v, raw["qvel"][frame, sub])
                    or not np.array_equal(force, raw["forces"][frame, sub])
                    or not np.array_equal(data.actuator_force[layout.ai], raw["torque"][frame, sub])
                    or tuple(int(w.number) for w in data.warning) != warnings
                ):
                    raise ValueError(
                        f"strict 2ms q/v/contact/effort replay differs at {frame}:{sub}"
                    )
                step = frame * 10 + sub
                if first_foot is None and (force[:2] > 10).any():
                    first_foot = step
                for body, magnitude in pairs:
                    event = events.setdefault(
                        body,
                        dict(
                            body=body,
                            first_substep=step,
                            last_substep=step,
                            peak_force_N=0.0,
                            contact_rows=0,
                            impulse_sum_Ns=0.0,
                            first_before_any_foot_contact=first_foot is None,
                        ),
                    )
                    event["last_substep"] = step
                    event["peak_force_N"] = max(event["peak_force_N"], magnitude)
                    event["contact_rows"] += 1
                    event["impulse_sum_Ns"] += magnitude * 0.002
        identical(snapshot(data), terminal)
        self.phases += 1
        return dict(
            native_steps=n * 10,
            exact_all_q_v_force_and_terminal=True,
            first_foot_contact_substep=first_foot,
            nonfoot_contacts=list(events.values()),
            optimizer_updates=0,
            own_neural_rows=0,
            hardware_authorized=False,
        )
