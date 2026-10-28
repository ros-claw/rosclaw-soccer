"""Experimental learned receive residual on the shared SIM native clock."""

from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld
from scripts.rsi_diagnose_pass_contact_timing import digest


class SharedNativeReceiverTorqueActor:
    """Own state and previous actual contacts only; never a hardware executor."""

    def __init__(
        self,
        world: SharedNativeMotorWorld,
        prefix: str,
        *,
        checkpoint: Path,
        checkpoint_hash: str,
        dataset_hash: str,
        horizon_controls: int,
    ):
        world._require_current_boundary()
        if (
            type(world) is not SharedNativeMotorWorld
            or len(world.layouts) != 2
            or prefix not in world.layouts
            or prefix in world._torque_layer_owners
            or type(horizon_controls) is not int
            or not 1 <= horizon_controls <= 600
            or checkpoint.is_symlink()
            or digest(checkpoint) != checkpoint_hash
        ):
            raise ValueError("own live SIM boundary and exact research checkpoint required")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            state.get("architecture") != "tanh-256-256-tanh20"
            or state.get("dataset_hash") != dataset_hash
            or any(
                state.get(k) is not False
                for k in ("teacher_qualified", "promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("unpromoted exact research actor lineage required")
        self.model = nn.Sequential(
            nn.Linear(140, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 29)
        )
        weights = {k.removeprefix("net."): v for k, v in state["actor"].items()}
        if len(weights) != len(state["actor"]) or any(
            not k.startswith("net.") for k in state["actor"]
        ):
            raise ValueError("exact network keys required")
        self.model.load_state_dict(weights, strict=True)
        self.model.eval().requires_grad_(False)
        self.mean, self.std = state["observation_mean"], state["observation_std"]
        if (
            self.mean.shape != (140,)
            or self.std.shape != (140,)
            or not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or (self.std < 0.01).any()
            or any(not torch.isfinite(x).all() for x in self.model.parameters())
            or digest(checkpoint) != checkpoint_hash
        ):
            raise ValueError("finite exact normalization and network required")
        self.world, self.prefix, self.policy = world, prefix, self
        self.layout = world.layouts[prefix]
        self.teammate = world.layouts[next(p for p in world.layouts if p != prefix)]
        self.previous_torque = world.data.actuator_force[self.layout.ai].copy()
        self.previous_contact = np.zeros(3, dtype=bool)
        self.last_foot_contact_step: int | None = None
        self.next_native_step = world.native_steps
        self.horizon_native_steps = self.next_native_step + horizon_controls * 10
        self.pending = self.failed = False
        self.commands = self.applied = self.active_commands = 0
        self.last_observation: np.ndarray | None = None
        world._torque_layer_owners[prefix] = self

    def observation(self, parent_torque: np.ndarray) -> np.ndarray:
        w, r, t = self.world, self.layout, self.teammate
        q, v = w.data.qpos, w.data.qvel
        age = (
            1.0
            if self.last_foot_contact_step is None
            else min((self.next_native_step - self.last_foot_contact_step) / 150.0, 1.0)
        )
        seen = float(self.last_foot_contact_step is not None)
        parts = (
            q[r.qi],
            v[r.vi],
            q[r.rq + 3 : r.rq + 7],
            q[r.rq + 2 : r.rq + 3],
            v[r.rv : r.rv + 6],
            q[r.bq : r.bq + 3] - q[r.rq : r.rq + 3],
            v[r.bv : r.bv + 3],
            q[t.rq : t.rq + 2] - q[r.rq : r.rq + 2],
            self.previous_torque,
            parent_torque,
            self.previous_contact.astype(np.float64),
            np.array([seen, age]),
        )
        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape != (140,) or not np.isfinite(obs).all():
            raise ValueError("finite actual causal 140D observation required")
        return obs

    def command(self, parent_torque: np.ndarray) -> np.ndarray:
        w = self.world
        parent = np.asarray(parent_torque)
        if (
            self.failed
            or self.pending
            or w.failed
            or w.native_steps != self.next_native_step
            or w._torque_layer_owners.get(self.prefix) is not self
            or self.next_native_step >= self.horizon_native_steps
            or parent.shape != (29,)
            or parent.dtype != np.float64
            or not np.isfinite(parent).all()
            or np.any(np.abs(parent) > w.parameters[self.prefix].effort)
        ):
            self.failed = True
            raise ValueError("actual consecutive parent torque required")
        try:
            obs = self.observation(parent)
            with torch.no_grad():
                delta = (
                    (20.0 * torch.tanh(self.model((torch.from_numpy(obs) - self.mean) / self.std)))
                    .numpy()
                    .astype(np.float64)
                )
            if (
                delta.shape != (29,)
                or not np.isfinite(delta).all()
                or np.abs(delta).max() > 20.00001
            ):
                raise ValueError("bounded finite actor proposal required")
            applied = np.clip(
                parent + delta, -w.parameters[self.prefix].effort, w.parameters[self.prefix].effort
            )
            self.last_observation = obs
            self.active_commands += int(np.any(applied != parent))
        except BaseException:
            self.failed = True
            raise
        self.pending = True
        self.commands += 1
        return applied

    def note_applied(self, actual: np.ndarray) -> None:
        w, r = self.world, self.layout
        if (
            self.failed
            or not self.pending
            or w.native_steps != self.next_native_step + 1
            or actual.shape != (29,)
            or actual.dtype != np.float64
            or not np.isfinite(actual).all()
            or not np.array_equal(actual, w.data.actuator_force[r.ai])
        ):
            self.failed = True
            raise ValueError("actual applied native torque required")
        groups = np.zeros(3)
        receiver_index = list(w.layouts).index(self.prefix)
        for i, contact in enumerate(w.data.contact):
            b1, b2 = (int(w.model.geom_bodyid[g]) for g in contact.geom)
            other = b2 if b1 == w.ball else b1 if b2 == w.ball else -1
            if other in w.contact_owners:
                player, part = w.contact_owners[other]
                if player == receiver_index:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(w.model, w.data, i, force)
                    groups[part] = max(groups[part], float(np.linalg.norm(force[:3])))
        self.previous_contact = groups > 1.0
        if self.previous_contact[:2].any():
            self.last_foot_contact_step = self.next_native_step
        self.previous_torque = actual.copy()
        self.pending = False
        self.next_native_step += 1
        self.applied += 1
