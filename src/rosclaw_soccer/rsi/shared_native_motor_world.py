"""SIM-only joint-target motors sharing one native world and one clock.

This adapter does not choose skills, reset bodies, move a ball, or fabricate
policy observations. Every declared player must submit a target before ANY
control is written. All PD torques are refreshed before one shared 2ms step.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout


@dataclass(frozen=True)
class NativePDParameters:
    stiffness: np.ndarray
    damping: np.ndarray
    effort: np.ndarray

    def owned_copy(self) -> "NativePDParameters":
        arrays = [np.asarray(x) for x in (self.stiffness, self.damping, self.effort)]
        if any(x.shape != (29,) or not np.isfinite(x).all() for x in arrays) or (
            (arrays[0] < 0).any() or (arrays[1] < 0).any() or (arrays[2] <= 0).any()
        ):
            raise ValueError("finite nonnegative HG29 gains and positive effort required")
        copies = [x.astype(np.float64, copy=True) for x in arrays]
        for x in copies:
            x.flags.writeable = False
        return NativePDParameters(*copies)


class SharedNativeMotorWorld:
    """Own a bounded group of 1..8 G1s and exactly one free shared ball.

    The constructor does not forward or integrate. Initialization belongs to
    the caller's declared protocol and must finish BEFORE this clock is made.
    No per-player step API exists: one player cannot run ahead of teammates.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        players: Mapping[str, NativePDParameters],
        ball_body_name: str,
    ) -> None:
        import mujoco

        if (
            not 1 <= len(players) <= 8
            or any(not isinstance(p, str) for p in players)
            or mujoco.get_mjcb_control() is not None
            or float(model.opt.timestep) != 0.002
            or not np.isfinite(data.time)
            or (data.qpos.shape, data.qvel.shape, data.ctrl.shape)
            != ((model.nq,), (model.nv,), (model.nu,))
        ):
            raise ValueError("callback-free native2ms world and bounded explicit players required")
        self.model, self.data = model, data
        self.layouts = {
            p: NativeVelocityLayout(
                model, robot_prefix=p, ball_body_name=ball_body_name, player_count=len(players)
            )
            for p in players
        }
        self.parameters = {p: v.owned_copy() for p, v in players.items()}
        first = next(iter(self.layouts.values()))
        self.ball = first.ball
        self.contact_owners: dict[int, tuple[int, int]] = {}
        for index, (prefix, layout) in enumerate(self.layouts.items()):
            feet = [
                int(model.body(prefix + side + "_ankle_roll_link").id)
                if mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, prefix + side + "_ankle_roll_link"
                )
                >= 0
                else -1
                for side in ("left", "right")
            ]
            for body in range(1, model.nbody):
                parent = body
                while parent not in (0, layout.pelvis):
                    parent = int(model.body_parentid[parent])
                if parent == layout.pelvis:
                    self.contact_owners[body] = (index, feet.index(body) if body in feet else 2)
        ball_geoms = np.flatnonzero(model.geom_bodyid == first.ball)
        if (
            len(ball_geoms) != 1
            or model.geom_type[ball_geoms[0]] != int(mujoco.mjtGeom.mjGEOM_SPHERE)
            or not np.isclose(model.geom_size[ball_geoms[0], 0], 0.11, rtol=0, atol=1e-12)
            or not np.isclose(model.body_mass[first.ball], 0.43, rtol=0, atol=1e-12)
        ):
            raise ValueError("explicit standard .11m radius/.43kg single sphere ball required")
        qids = list(range(first.bq, first.bq + 7))
        vids = list(range(first.bv, first.bv + 6))
        aids: list[int] = []
        for layout in self.layouts.values():
            qids.extend([*range(layout.rq, layout.rq + 7), *layout.qi])
            vids.extend([*range(layout.rv, layout.rv + 6), *layout.vi])
            aids.extend(layout.ai)
        if (
            sorted(qids) != list(range(model.nq))
            or sorted(vids) != list(range(model.nv))
            or sorted(aids) != list(range(model.nu))
            or model.nmocap > 1
        ):
            raise ValueError("complete disjoint player ownership and exactly one ball required")
        # Exact torque transmission, no position servos, activation dynamics,
        # actuator clipping, external forces or pre-existing native warnings.
        gear = np.zeros((model.nu, 6))
        gear[:, 0] = 1
        gain = np.zeros((model.nu, 10))
        gain[:, 0] = 1
        if (
            not np.array_equal(model.actuator_gear, gear)
            or not np.array_equal(model.actuator_gainprm, gain)
            or np.any(model.actuator_biastype)
            or np.any(model.actuator_dyntype)
            or np.any(model.actuator_gaintype)
            or np.any(model.actuator_ctrllimited)
            or np.any(model.actuator_forcelimited)
            or np.any(data.qfrc_applied)
            or np.any(data.xfrc_applied)
            or any(int(w.number) for w in data.warning)
        ):
            raise ValueError("plain unit-gear torque motors without hidden forces required")
        for p, layout in self.layouts.items():
            layout.canonical(data)
            cap = model.jnt_actfrcrange[layout.joints]
            limited = model.jnt_actfrclimited[layout.joints]
            if np.any(
                limited
                & (
                    (cap[:, 0] > -self.parameters[p].effort)
                    | (cap[:, 1] < self.parameters[p].effort)
                )
            ):
                raise ValueError("declared effort cannot exceed actual joint force caps")
        self.native_steps = 0
        self.control_steps = 0
        self.expected_time = float(data.time)
        self.failed = False
        self._last_targets: dict[str, np.ndarray] = {}
        self.last_ball_contact_bodies: dict[str, dict[str, float]] = {p: {} for p in players}
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

        self._boundary_snapshot = snapshot(data)

    def _require_current_boundary(self, *, allow_failed: bool = False) -> None:
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

        actual = snapshot(self.data)
        if (self.failed and not allow_failed) or not all(
            np.array_equal(value, actual[key]) for key, value in self._boundary_snapshot.items()
        ):
            raise ValueError("unchanged healthy actual shared control boundary required")

    def last_applied_target(self, prefix: str) -> np.ndarray:
        self._require_current_boundary(allow_failed=True)
        if prefix not in self._last_targets:
            raise ValueError("actual prior player control required")
        return self._last_targets[prefix].copy()

    def switch_pd_profile(self, prefix: str, parameters: NativePDParameters) -> dict[str, Any]:
        """Explicit SIM controller handoff: unchanged body/model, same effort caps.

        Call after the preceding control is fully committed, before advancing
        or rewriting any physical field. This changes software gains ONLY.
        It grants no policy promotion, robot authority or future stability.
        """
        import mujoco

        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

        self._require_current_boundary()
        if prefix not in self.parameters or mujoco.get_mjcb_control() is not None:
            raise ValueError("own player and callback-free simulation required")
        new = parameters.owned_copy()
        old = self.parameters[prefix]
        if (
            not np.array_equal(new.effort, old.effort)
            or (new.stiffness <= 0).any()
            or (new.damping <= 0).any()
            or (new.stiffness > 2000).any()
            or (new.damping > 2000).any()
        ):
            raise ValueError(
                "same original effort caps and positive bounded software gains required"
            )
        before = snapshot(self.data)
        self.parameters[prefix] = new
        after = snapshot(self.data)
        if any(not np.array_equal(before[k], after[k]) for k in before):
            raise ValueError("PD profile handoff changed physics")
        return dict(
            prefix=prefix,
            control_step=self.control_steps,
            time=float(self.data.time),
            previous={k: getattr(old, k).tolist() for k in ("stiffness", "damping", "effort")},
            current={k: getattr(new, k).tolist() for k in ("stiffness", "damping", "effort")},
            physics_unchanged=True,
            before=before,
            after=after,
        )

    def advance(self, targets: Mapping[str, np.ndarray]) -> dict[str, Any]:
        """Integrate exactly ten shared substeps, not ten per player.

        Input HG29 targets are joint positions, not normalized policy actions.
        The caller retains its private observation/action/RNN provenance.
        """
        import mujoco

        m, d = self.model, self.data
        if (
            self.failed
            or mujoco.get_mjcb_control() is not None
            or not np.isclose(float(d.time), self.expected_time, rtol=0, atol=1e-12)
            or set(targets) != set(self.layouts)
            or np.any(d.qfrc_applied)
            or np.any(d.xfrc_applied)
        ):
            raise ValueError("live unmodified world and all player targets required")
        owned = {}
        for p, target in targets.items():
            x = np.asarray(target)
            if x.shape != (29,) or x.dtype.kind != "f" or not np.isfinite(x).all():
                raise ValueError("all finite HG29 joint targets required before any physics")
            owned[p] = x.copy()
        samples = []
        witnesses: dict[str, dict[str, float]] = {p: {} for p in self.layouts}
        prefixes = list(self.layouts)
        before = float(d.time)
        for _ in range(10):
            all_torques = np.zeros(m.nu)
            for p, layout in self.layouts.items():
                param = self.parameters[p]
                torque = np.clip(
                    param.stiffness * (owned[p] - d.qpos[layout.qi])
                    - param.damping * d.qvel[layout.vi],
                    -param.effort,
                    param.effort,
                )
                if not np.isfinite(torque).all():
                    self.failed = True
                    raise ValueError("nonfinite PD torque; no state repair")
                all_torques[layout.ai] = torque
            d.ctrl[:] = all_torques
            mujoco.mj_step(m, d)
            self.native_steps += 1
            if (
                any(int(w.number) for w in d.warning)
                or not np.array_equal(d.actuator_force, all_torques)
                or not np.isfinite(d.qpos).all()
                or not np.isfinite(d.qvel).all()
            ):
                self.failed = True
                raise ValueError("native warning, nonfinite state or changed motor law")
            forces = np.zeros((len(self.layouts), 3))
            for ci in range(d.ncon):
                contact = d.contact[ci]
                b1, b2 = (int(m.geom_bodyid[g]) for g in contact.geom)
                other = b2 if b1 == self.ball else b1 if b2 == self.ball else -1
                if other in self.contact_owners:
                    player, part = self.contact_owners[other]
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, wrench)
                    force = float(np.linalg.norm(wrench[:3]))
                    forces[player, part] = max(forces[player, part], force)
                    if force > 1:
                        name = m.body(other).name or prefixes[player] + f"unknown_body_{other}"
                        seen = witnesses[prefixes[player]]
                        seen[name] = max(seen.get(name, 0.0), force)
            samples.append(
                dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), torque=all_torques, ball_forces=forces)
            )
        self.control_steps += 1
        self.expected_time = float(d.time)
        raw = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
        body_failed = {}
        for p, layout in self.layouts.items():
            q = raw["qpos"][:, layout.rq : layout.rq + 7]
            body_failed[p] = bool(
                ((q[:, 2] < 0.65) | (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2) < 0.9)).any()
            )
        self.failed = any(body_failed.values())
        self._last_targets = {p: x.copy() for p, x in owned.items()}
        self.last_ball_contact_bodies = witnesses
        from rosclaw_soccer.rsi.paid_skill_handoff import snapshot

        self._boundary_snapshot = snapshot(d)
        return dict(**raw, time_before=before, body_failed=body_failed)
