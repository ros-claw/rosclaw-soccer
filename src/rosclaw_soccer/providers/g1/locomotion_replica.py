"""Private frozen locomotion replica for simulation forecast/replay research.

This is an inference adapter, not a motor option, runtime executor, full-world
checkpoint or learned teacher. External deployment code must be trusted and
content-bound. No live controller/model handles are accepted or returned.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.providers.g1.locomotion_memory import (
    LocomotionMemory,
    capture_locomotion_memory,
    restore_private_locomotion_memory,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import load_robonaldo
from rosclaw_soccer.providers.g1.recurrent_inference_state import detach_frozen_recurrent_state
from rosclaw_soccer.sim.contracts import hash_bytes


def _vector(value: Any, size: int, bound: float) -> NDArray[Any]:
    array = np.asarray(value)
    if (
        array.shape != (size,)
        or array.dtype.kind not in "fiu"
        or not np.isfinite(array).all()
        or np.any(array > bound)
        or np.any(array < -bound)
    ):
        raise ValueError("bounded finite locomotion vector required")
    return array.copy()


class FrozenLocomotionReplica:
    """Own one independent CPU foundation and reconstruct its qualified inputs.

    Artifact hashes bind the loaded deployment policy, configuration and adapter
    source; the enclosing experiment must also pin this library implementation.
    Restored memory is post-inference: ``step`` computes the *next* inference.
    """

    activation_ceiling = "SIM_ONLY"

    def __init__(
        self,
        asset_root: Path,
        *,
        policy_hash: str,
        configuration_hash: str,
        adapter_hash: str,
        synchronize_action_frame: bool,
        correct_mirrored_yaw: bool,
    ) -> None:
        if any(
            type(v) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", v) is None
            for v in (policy_hash, configuration_hash, adapter_hash)
        ) or any(type(v) is not bool for v in (synchronize_action_frame, correct_mirrored_yaw)):
            raise ValueError("explicit artifact bindings and reflection configuration required")
        root = Path(asset_root).resolve()
        paths = {
            root / "policy/loco_mode/model/policy_29dof.pt": policy_hash,
            root / "policy/loco_mode/config/LocoMode.yaml": configuration_hash,
            root / "policy/loco_mode/LocoMode.py": adapter_hash,
        }
        self._verify_files(paths)
        qualify_g1_assets(root).require_eligible()
        with contextlib.redirect_stdout(io.StringIO()):
            state_type, output_type, _, _ = load_robonaldo(root)
            module = importlib.import_module("policy.loco_mode.LocoMode")
            module_file = getattr(module, "__file__", None)
            if (
                not module_file
                or Path(module_file).resolve() != root / "policy/loco_mode/LocoMode.py"
            ):
                raise ValueError("locomotion adapter imported outside the bound asset root")
            state, output = state_type(29), output_type(29)
            policy = module.LocoMode(state, output)
            policy.enter()
        if Path(policy.policy_path).resolve() != root / "policy/loco_mode/model/policy_29dof.pt":
            raise ValueError("locomotion adapter loaded a different policy artifact")
        self._verify_files(paths)
        self.policy_hash = policy_hash
        self.configuration_hash = configuration_hash
        self.adapter_hash = adapter_hash
        self._controller: Any = SimpleNamespace(
            state=state,
            output=output,
            policy=policy,
            keeper_reach=None,
            locomotion_reflection_frame=None,
            locomotion_frame_switch_count=0,
        )
        self._synchronize = synchronize_action_frame
        self._correct_yaw = correct_mirrored_yaw
        self._restored = False

    @staticmethod
    def _verify_files(paths: dict[Path, str]) -> None:
        for path, expected in paths.items():
            if not path.is_file() or not 0 < path.stat().st_size <= 1024**3:
                raise ValueError("bounded external locomotion artifact required")
            if hash_bytes(path.read_bytes()) != expected:
                raise ValueError("external locomotion artifact differs from binding")

    def restore(
        self, memory: LocomotionMemory, previous_raw_action: Any, *, reflected: bool
    ) -> None:
        action = _vector(previous_raw_action, 29, 100.0)
        if type(reflected) is not bool:
            raise ValueError("explicit previous reflection frame required")
        restore_private_locomotion_memory(
            self._controller.policy.policy,
            memory,
            policy_hash=self.policy_hash,
            private_replay=True,
        )
        self._controller.policy.action = action
        self._controller.locomotion_reflection_frame = reflected
        self._controller.locomotion_frame_switch_count = 0
        self._restored = True

    def step(self, qpos: Any, qvel: Any, world_command: Any) -> NDArray[Any]:
        """Advance only this private policy; return an owned joint-target copy."""
        from rosclaw_soccer.skills.team.independent_team_world import (
            _gravity_orientation,
            _normalized_locomotion_command,
            _pelvis_yaw,
            _rotate_z,
            _run_locomotion,
        )

        if not self._restored:
            raise ValueError("restore measured locomotion memory before forecasting")
        q, v, command = (
            _vector(qpos, 43, 1e4),
            _vector(qvel, 41, 1e4),
            _vector(world_command, 3, 100.0),
        )
        if any(abs(float(np.linalg.norm(q[a:b])) - 1) > 1e-4 for a, b in ((3, 7), (39, 43))):
            raise ValueError("unit body and ball quaternion required")
        controller = self._controller
        controller.state.q = q[7:36].copy()
        controller.state.dq = v[6:35].copy()
        controller.state.gravity_ori = _gravity_orientation(q[3:7])
        controller.state.ang_vel = v[3:6].copy()
        local = _rotate_z(command, -_pelvis_yaw(q[3:7]))
        controller.state.vel_cmd = _normalized_locomotion_command(controller.policy, local)
        self._restored = False  # A partial failed inference requires an explicit fresh restore.
        _run_locomotion(
            controller,
            mirror=bool(local[1] < -1e-6),
            correct_mirrored_yaw=self._correct_yaw,
            synchronize_action_frame=self._synchronize,
        )
        detach_frozen_recurrent_state(controller.policy.policy, frozen_inference=True)
        target = _vector(controller.output.actions, 29, 100.0)
        self._restored = True
        return target

    @property
    def observation(self) -> NDArray[Any]:
        return np.asarray(self._controller.policy.obs).copy()

    @property
    def raw_action(self) -> NDArray[Any]:
        return np.asarray(self._controller.policy.action).copy()

    @property
    def memory(self) -> LocomotionMemory:
        return capture_locomotion_memory(
            self._controller.policy.policy, policy_hash=self.policy_hash
        )
