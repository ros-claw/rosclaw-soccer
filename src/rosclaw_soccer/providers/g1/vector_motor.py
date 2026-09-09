"""Role-neutral G1 motor learning sandbox on MuJoCo Warp.

No goalkeeper curriculum, locomotion policy, ball launcher, implicit reset or
reward lives here. A trainer supplies bounded torques or a PD teacher; physics
and the shared joint guard own execution. CUDA results require CPU agreement.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json

if TYPE_CHECKING:
    from rosclaw_soccer.world.field import G1TrainingGoalSpec


@dataclass(frozen=True)
class G1VectorMotorConfig:
    environment_count: int = 32
    device: str = "cuda:0"
    physics_substeps: int = 10
    guard_margin_rad: float = 0.08
    activation_ceiling: str = "SIM_ONLY"
    torque_limit_scale: float = 1.0

    def __post_init__(self) -> None:
        if (
            type(self.environment_count) is not int
            or not 1 <= self.environment_count <= 4096
            or not isinstance(self.device, str)
            or re.fullmatch(r"cuda:[0-9]+", self.device) is None
            or type(self.physics_substeps) is not int
            or not 1 <= self.physics_substeps <= 10
            or not math.isfinite(self.guard_margin_rad)
            or not 0.04 <= self.guard_margin_rad <= 0.10
            or self.activation_ceiling != "SIM_ONLY"
            or type(self.torque_limit_scale) not in (int, float)
            or not math.isfinite(self.torque_limit_scale)
            or not 0.5 <= self.torque_limit_scale <= 1.0
        ):
            raise ValueError("bounded simulation-only vector motor configuration required")

    @property
    def config_hash(self) -> str:
        value = asdict(self)
        if self.torque_limit_scale == 1.0:
            value.pop("torque_limit_scale")
        return str(hash_json(value))

    @property
    def torque_limits_nm(self) -> tuple[float, ...]:
        """Only lower the hard envelope; bind the training motor authority."""
        return tuple(float(v * self.torque_limit_scale) for v in G1_HARD_TORQUE_LIMITS)


class G1VectorMotorBatch:
    """One process/device; explicit episode reset and 2 ms guarded motor steps."""

    def __init__(
        self,
        asset_root: Path,
        config: G1VectorMotorConfig | None = None,
        *,
        goal_spec: G1TrainingGoalSpec | None = None,
    ) -> None:
        import mujoco
        import mujoco_warp as mjw
        import torch
        import warp as wp

        from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
        from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

        self.config = config or G1VectorMotorConfig()
        self.qualification = qualify_g1_assets(asset_root)
        self.qualification.require_eligible()
        if goal_spec is not None and not isinstance(goal_spec, G1TrainingGoalSpec):
            raise ValueError("explicit typed simulation world specification required")
        goal = goal_spec or G1TrainingGoalSpec()
        self.world_contract = {
            "goal": asdict(goal),
            "body_hash": self.qualification.body_hash,
            "physics_dt_sec": 0.002,
            "activation_ceiling": "SIM_ONLY",
        }
        if self.config.torque_limit_scale != 1.0:
            self.world_contract["torque_limit_scale"] = self.config.torque_limit_scale
        self.world_hash = hash_json(self.world_contract)
        self._torch, self._mjw = torch, mjw
        self.device = torch.device(self.config.device)
        wp.init()
        wp.set_device(self.config.device)
        if str(wp.get_device()) != self.config.device:
            raise RuntimeError("Warp physics device binding differs")
        self.cpu_model = build_g1_stadium_model(asset_root, spec=goal)
        self.cpu_model.opt.timestep = 0.002
        from rosclaw_soccer.sim.mjwarp_contract import qualify_mjwarp_damping

        self.damping_qualification = qualify_mjwarp_damping(
            self.cpu_model, device=self.config.device
        )
        self.world_contract["passive_damping_contract"] = self.damping_qualification[
            "contract_hash"
        ]
        self.world_hash = hash_json(self.world_contract)
        self.joint_ids = np.asarray([self.cpu_model.joint(n).id for n in G1_DDS_JOINT_NAMES])
        self.actuator_ids = np.asarray([self.cpu_model.actuator(n).id for n in G1_DDS_JOINT_NAMES])
        self.joint_qpos = self.cpu_model.jnt_qposadr[self.joint_ids].copy()
        self.joint_qvel = self.cpu_model.jnt_dofadr[self.joint_ids].copy()
        if self.cpu_model.nu != 29 or not np.array_equal(
            self.cpu_model.actuator_trnid[self.actuator_ids, 0], self.joint_ids
        ):
            raise ValueError("G1 motor/joint transmission identity differs")
        initial = mujoco.MjData(self.cpu_model)
        mujoco.mj_forward(self.cpu_model, initial)
        self._model = mjw.put_model(self.cpu_model)
        self._data = mjw.put_data(
            self.cpu_model, initial, nworld=self.config.environment_count, nconmax=256, njmax=1024
        )
        self._qpos, self._qvel = wp.to_torch(self._data.qpos), wp.to_torch(self._data.qvel)
        self._ctrl = wp.to_torch(self._data.ctrl)
        self._overflow = wp.to_torch(self._data.overflow)
        self._ranges = torch.as_tensor(
            self.cpu_model.jnt_range[self.joint_ids], device=self.device, dtype=torch.float32
        )
        self._limited = torch.as_tensor(
            self.cpu_model.jnt_limited[self.joint_ids].astype(bool), device=self.device
        )
        self._limits = torch.as_tensor(
            self.config.torque_limits_nm, device=self.device, dtype=torch.float32
        )
        self._qids = torch.as_tensor(self.joint_qpos, device=self.device, dtype=torch.long)
        self._vids = torch.as_tensor(self.joint_qvel, device=self.device, dtype=torch.long)
        self._aids = torch.as_tensor(self.actuator_ids, device=self.device, dtype=torch.long)
        self.episode_resets = 0
        self.physics_steps = 0
        self._ready = False
        self._physics_graph: Any = None
        self.warmup_physics_steps = 0

    def prepare_physics_graph(self) -> None:
        """Optional launch optimization, before the first declared episode.

        Warmup state is never evidence. The caller must still initialize the
        episode explicitly. Torch motor feedback remains outside the graph.
        """
        if self._ready or self.episode_resets or self._physics_graph is not None:
            raise RuntimeError("physics graph preparation must precede the first episode")
        import warp as wp

        stream = wp.Stream(self.config.device)
        with wp.ScopedStream(stream):
            self._mjw.step(self._model, self._data)
            self.warmup_physics_steps += 1
            wp.synchronize_stream(stream)
            with wp.ScopedCapture(stream=stream) as capture:
                self._mjw.step(self._model, self._data)
        self._physics_graph = capture.graph

    @property
    def physics_graph_enabled(self) -> bool:
        return self._physics_graph is not None

    def reset(self, qpos: Any, qvel: Any) -> None:
        """Explicit episode initialization; never called by a motor step."""
        torch = self._torch
        position = torch.as_tensor(qpos, device=self.device, dtype=torch.float32)
        velocity = torch.as_tensor(qvel, device=self.device, dtype=torch.float32)
        n = self.config.environment_count
        if position.shape != (n, self.cpu_model.nq) or velocity.shape != (n, self.cpu_model.nv):
            raise ValueError("vector initial state shape differs")
        if not bool(torch.isfinite(position).all() and torch.isfinite(velocity).all()):
            raise ValueError("vector initial state must be finite")
        for joint in np.flatnonzero(self.cpu_model.jnt_type == 0):
            address = int(self.cpu_model.jnt_qposadr[joint]) + 3
            norm = torch.linalg.vector_norm(position[:, address : address + 4], dim=1)
            if not bool(torch.all(torch.abs(norm - 1) < 1e-4)):
                raise ValueError("vector free-joint quaternion must be normalized")
        self._qpos.copy_(position)
        self._qvel.copy_(velocity)
        self._ctrl.zero_()
        # Clear warm-start acceleration at a declared episode boundary.
        import warp as wp

        wp.to_torch(self._data.qacc_warmstart).zero_()
        wp.to_torch(self._data.time).zero_()
        self._overflow.zero_()
        self._mjw.forward(self._model, self._data)
        self.episode_resets += 1
        self._ready = True

    def state(self) -> dict[str, Any]:
        return {"qpos": self._qpos.clone(), "qvel": self._qvel.clone(), "ctrl": self._ctrl.clone()}

    def _action(self, value: Any, *, name: str) -> Any:
        torch = self._torch
        value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if value.shape != (self.config.environment_count, 29) or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"finite batched 29-joint {name} required")
        return value

    def step_torque(self, normalized_torque: Any) -> dict[str, Any]:
        """Direct 29-motor student action; only the guard/clips intervene."""
        torque = self._action(normalized_torque, name="normalized torque")
        if bool((torque.abs() > 1).any()):
            raise ValueError("normalized torque exceeds hard motor envelope")
        return self._step(torque * self._limits, None)

    def step_pd(self, target: Any, kp: Any, kd: Any) -> dict[str, Any]:
        """Teacher-only target tracking, with feedback recomputed every 2 ms."""
        target = self._action(target, name="PD target")
        kp, kd = self._action(kp, name="PD kp"), self._action(kd, name="PD kd")
        if bool((kp < 0).any() or (kp > 300).any() or (kd < 0).any() or (kd > 30).any()):
            raise ValueError("PD teacher gain envelope exceeded")
        return self._step(None, (target, kp, kd))

    def _step(self, torque: Any, pd: tuple[Any, Any, Any] | None) -> dict[str, Any]:
        from rosclaw_soccer.training.joint_guard import project_joint_safe_torque_torch

        if not self._ready:
            raise RuntimeError("explicit finite episode reset required before motor execution")
        torch = self._torch
        for _ in range(self.config.physics_substeps):
            if pd is not None:
                target, kp, kd = pd
                torque = kp * (target - self._qpos[:, self._qids]) - kd * self._qvel[:, self._vids]
            projected, _ = project_joint_safe_torque_torch(
                joint_position=self._qpos[:, self._qids],
                joint_velocity=self._qvel[:, self._vids],
                commanded_torque=torque,
                joint_ranges=self._ranges,
                limited=self._limited,
                margin_rad=self.config.guard_margin_rad,
            )
            self._ctrl[:, self._aids] = torch.clamp(projected, -self._limits, self._limits)
            import warp as wp

            # Shared buffers must follow the caller's active Torch stream.
            stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
            with wp.ScopedStream(stream):
                if self._physics_graph is None:
                    self._mjw.step(self._model, self._data)
                else:
                    wp.capture_launch(self._physics_graph, stream=stream)
            self.physics_steps += 1
            if not bool(torch.isfinite(self._qpos).all() and torch.isfinite(self._qvel).all()):
                self._ready = False
                self._ctrl.zero_()
                raise FloatingPointError(
                    "nonfinite vector physics; failed state retained, no implicit reset"
                )
            if bool((self._overflow != 0).any()):
                self._ready = False
                self._ctrl.zero_()
                raise RuntimeError("vector contact/solver capacity overflow; failed state retained")
        return self.state()
