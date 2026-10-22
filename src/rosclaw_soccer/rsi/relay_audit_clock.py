"""SIM audit-only native clock, independent of the producer advance method.

Scene/layout validation is shared. Integration, torque assembly and contact
extraction below are recomputed, never copied from a producer trajectory.
Neural codecs may be shared by an audit but must run on these own states.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_skill_handoff import snapshot


class RelayAuditClock:
    def __init__(self, initialized: Any) -> None:
        if initialized.control_steps or initialized.native_steps or initialized.failed:
            raise ValueError("unused initialized world required; no mid-trial restoration")
        self.model, self.data = initialized.model, initialized.data
        self.layouts = initialized.layouts
        self.parameters = {p: x.owned_copy() for p, x in initialized.parameters.items()}
        self.control_steps = self.native_steps = 0
        self.failed = False
        self.last_ball_contact_bodies: dict[str, dict[str, float]] = {p: {} for p in self.layouts}
        self.last_targets: dict[str, np.ndarray] = {}
        self.boundary = snapshot(self.data)
        self.owners = {}
        for index, (p, layout) in enumerate(self.layouts.items()):
            for body in range(1, self.model.nbody):
                ancestor = body
                while ancestor not in (0, layout.pelvis):
                    ancestor = int(self.model.body_parentid[ancestor])
                if ancestor == layout.pelvis:
                    name = self.model.body(body).name
                    part = (
                        0
                        if name == p + "left_ankle_roll_link"
                        else 1
                        if name == p + "right_ankle_roll_link"
                        else 2
                    )
                    self.owners[body] = (index, part)

    def check_boundary(self) -> None:
        current = snapshot(self.data)
        if any(not np.array_equal(v, current[k]) for k, v in self.boundary.items()):
            raise ValueError("audit physical boundary changed outside native integration")

    def last_applied_target(self, prefix: str) -> np.ndarray:
        self.check_boundary()
        return self.last_targets[prefix].copy()

    def switch_pd_profile(self, prefix: str, parameters: Any) -> dict[str, Any]:
        self.check_boundary()
        old, new = self.parameters[prefix], parameters.owned_copy()
        if (
            self.failed
            or not np.array_equal(old.effort, new.effort)
            or any((x <= 0).any() or (x > 2000).any() for x in (new.stiffness, new.damping))
        ):
            raise ValueError("healthy same-cap bounded software profile required")
        before = snapshot(self.data)
        self.parameters[prefix] = new
        after = snapshot(self.data)
        if any(not np.array_equal(v, after[k]) for k, v in before.items()):
            raise ValueError("audit profile switch changed physics")
        return dict(before=before, after=after)

    def advance(self, targets: dict[str, np.ndarray]) -> dict[str, Any]:
        import mujoco

        m, d = self.model, self.data
        if (
            self.failed
            or set(targets) != set(self.layouts)
            or mujoco.get_mjcb_control() is not None
            or np.any(d.qfrc_applied)
            or np.any(d.xfrc_applied)
            or any(x.shape != (29,) or not np.isfinite(x).all() for x in targets.values())
        ):
            raise ValueError("all own finite targets and live force-free world required")
        start = float(d.time)
        rows = []
        prefixes = list(self.layouts)
        contacts: dict[str, dict[str, float]] = {p: {} for p in prefixes}
        ball = next(iter(self.layouts.values())).ball
        for _ in range(10):
            torque = np.zeros(m.nu)
            for p, layout in self.layouts.items():
                law = self.parameters[p]
                torque[layout.ai] = np.clip(
                    law.stiffness * (targets[p] - d.qpos[layout.qi])
                    - law.damping * d.qvel[layout.vi],
                    -law.effort,
                    law.effort,
                )
            d.ctrl[:] = torque
            mujoco.mj_step(m, d)
            self.native_steps += 1
            if (
                any(int(w.number) for w in d.warning)
                or not np.isfinite(d.qpos).all()
                or not np.isfinite(d.qvel).all()
                or not np.array_equal(d.actuator_force, torque)
            ):
                raise ValueError("audit native warning, nonfinite state or changed torque law")
            forces = np.zeros((len(prefixes), 3))
            for ci in range(d.ncon):
                b1, b2 = [int(m.geom_bodyid[g]) for g in d.contact[ci].geom]
                other = b2 if b1 == ball else b1 if b2 == ball else -1
                if other not in self.owners:
                    continue
                player, part = self.owners[other]
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, wrench)
                force = float(np.linalg.norm(wrench[:3]))
                forces[player, part] = max(forces[player, part], force)
                if force > 1:
                    name = m.body(other).name or prefixes[player] + f"unknown_body_{other}"
                    seen = contacts[prefixes[player]]
                    seen[name] = max(seen.get(name, 0.0), force)
            rows.append(
                dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), torque=torque, ball_forces=forces)
            )
        raw = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        failed = {}
        for p, layout in self.layouts.items():
            root = raw["qpos"][:, layout.rq : layout.rq + 7]
            failed[p] = bool(
                ((root[:, 2] < 0.65) | (1 - 2 * (root[:, 4] ** 2 + root[:, 5] ** 2) < 0.9)).any()
            )
        self.failed = any(failed.values())
        self.control_steps += 1
        self.last_targets = {p: target.copy() for p, target in targets.items()}
        self.last_ball_contact_bodies = contacts
        self.boundary = snapshot(d)
        return dict(**raw, time_before=start, body_failed=failed)
