"""Owned short SIM forecasts of the complete low-latency SONIC navigation.

Shares ONLY the three qualified CPU ONNX sessions and immutable MuJoCo model.
All controller/history/reference and native data are copied. No cold body is
constructed and its clock patched. This is a privileged training teacher tool,
not a learned policy, hardware interface or official M0 qualification.
"""

import copy
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, _StreamingBackend
from rosclaw_soccer.rsi.full_task_branch import integration_hash
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation

_SESSIONS = {"_planner", "_encoder", "_decoder"}
_NAV_KEYS = {
    "agent_id",
    "config",
    "backend",
    "contract_hash",
    "_next_frame",
    "_faulted",
    "_retired",
    "_origin_frame",
    "_ready_from_handoff",
    "_ready_from_observation",
    "_boundary_observation_hash",
    "_batch_pending",
}
_BACKEND_KEYS = {
    "navigation",
    "command",
    "facing",
    "planner_calls",
    "last_planned_frame",
    "last_planned_command",
    "events",
    "latent_records",
    "config",
    "_variant",
    "qualification",
    "_planner",
    "_encoder",
    "_decoder",
    "_history",
    "reference",
    "reference_digest",
    "action",
    "target",
    "_history_handoff_binding",
    "_kp",
    "_kd",
    "_action_scale",
}
_BODY_KEYS = {
    "record_all_ball_contacts",
    "model",
    "data",
    "qi",
    "vi",
    "ai",
    "ball",
    "bg",
    "pelvis",
    "bq",
    "bv",
    "kp",
    "kd",
    "caps",
    "contacts",
    "warnings",
    "next_frame",
    "attempted_steps",
    "integrated_steps",
    "failed",
}


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("finite numeric controller buffers required")
        return dict(
            dtype=value.dtype.str, shape=list(value.shape), bytes_hash=hash_bytes(value.tobytes())
        )
    if isinstance(value, np.generic):
        return _encode(value.item())
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError("finite controller scalars required")
        return value
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("named controller state fields required")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, deque)):
        return dict(kind=type(value).__name__, values=[_encode(v) for v in value])
    if is_dataclass(value) and not isinstance(value, type):
        return dict(
            kind=type(value).__qualname__,
            fields={f.name: _encode(getattr(value, f.name)) for f in fields(value)},
        )
    raise ValueError("unqualified opaque mutable controller state: " + type(value).__qualname__)


def navigation_state_hash(nav: G1SonicNavigation) -> str:
    """Hash every numeric/configuration/history field, not internal ORT caches."""
    if type(nav) is not G1SonicNavigation or type(nav.backend) is not _StreamingBackend:
        raise ValueError("exact qualified streaming navigation types required")
    if (
        set(nav.__dict__) != _NAV_KEYS
        or set(nav.backend.__dict__) - {"default_angles"} != _BACKEND_KEYS
    ):
        raise ValueError("navigation state schema changed; requalify full snapshot ownership")
    backend = {k: v for k, v in nav.backend.__dict__.items() if k not in _SESSIONS}
    backend["default_angles"] = nav.backend.default_angles
    return hash_json(
        dict(
            navigation=_encode({k: v for k, v in nav.__dict__.items() if k != "backend"}),
            backend=_encode(backend),
            shared_sessions="qualified_stateless_CPU_only",
        )
    )


def _arrays(value: Any) -> list[np.ndarray]:
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, dict):
        return [a for v in value.values() for a in _arrays(v)]
    if isinstance(value, (list, tuple, deque)):
        return [a for v in value for a in _arrays(v)]
    if is_dataclass(value) and not isinstance(value, type):
        return [a for f in fields(value) for a in _arrays(getattr(value, f.name))]
    return []


def clone_navigation(nav: G1SonicNavigation) -> G1SonicNavigation:
    """Copy the whole qualified state graph; reject unknown fields, not skip them."""
    import onnxruntime

    before = navigation_state_hash(nav)
    if (
        nav._faulted
        or nav._retired
        or nav._batch_pending
        or nav.config.model_variant != "low_latency"
        or nav.config.pose_reference is not None
        or nav.config.latent_schedule is not None
        or nav.backend.navigation is not nav.config
    ):
        raise ValueError("healthy locomotion-only navigation without pending inference required")
    memo = {}
    for key in _SESSIONS:
        session = getattr(nav.backend, key)
        if not isinstance(session, onnxruntime.InferenceSession) or session.get_providers() != [
            "CPUExecutionProvider"
        ]:
            raise ValueError("only qualified CPU ONNX sessions may be shared")
        memo[id(session)] = session
    nav.backend.qualification.require_eligible()
    result = copy.deepcopy(nav, memo)
    # This mutable class-level array must become a private instance buffer.
    result.backend.default_angles = nav.backend.default_angles.copy()
    if result.backend.navigation is not result.config:
        raise ValueError("navigation/config alias graph lost during cloning")
    source_arrays = _arrays({k: v for k, v in nav.__dict__.items() if k != "backend"})
    source_arrays += _arrays({k: v for k, v in nav.backend.__dict__.items() if k not in _SESSIONS})
    source_arrays.append(nav.backend.default_angles)
    copied_arrays = _arrays({k: v for k, v in result.__dict__.items() if k != "backend"})
    copied_arrays += _arrays(
        {k: v for k, v in result.backend.__dict__.items() if k not in _SESSIONS}
    )
    if any(np.shares_memory(a, b) for a in source_arrays for b in copied_arrays):
        raise ValueError("forecast controller aliases mutable donor buffers")
    if navigation_state_hash(result) != before or navigation_state_hash(nav) != before:
        raise ValueError("cloned whole controller graph differs or donor was changed")
    return result


def _clone_body(body: G1TargetEpisode) -> G1TargetEpisode:
    if type(body) is not G1TargetEpisode or set(body.__dict__) != _BODY_KEYS or body.failed:
        raise ValueError("complete healthy exact target-body state required")
    before = integration_hash(body.model, body.data)
    data = mujoco.MjData(body.model)
    mujoco.mj_copyData(data, body.model, body.data)
    result = copy.deepcopy(body, {id(body.model): body.model, id(body.data): data})
    if integration_hash(result.model, result.data) != before or result.data is body.data:
        raise ValueError("native warm-start/integration state was not preserved")
    return result


@dataclass(frozen=True)
class SonicNavigationForecast:
    arrays: dict[str, np.ndarray]
    commands: np.ndarray
    reference_poses: np.ndarray
    neural_actions: np.ndarray
    donor_integration_hash: str
    donor_navigation_hash: str
    new_attempted_native_steps: int
    new_integrated_native_steps: int
    new_planner_calls: int
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False


class SonicNavigationForecaster:
    """Finite40-frame own-feedback previews; future truth is TEACHER_ONLY."""

    def __init__(self, *, world_hash: str, max_native_steps: int) -> None:
        if (
            type(max_native_steps) is not int
            or not 400 <= max_native_steps <= 600000
            or max_native_steps % 10
        ):
            raise ValueError("bounded explicit native forecast budget required")
        self.world_hash = world_hash
        self.max_native_steps = max_native_steps
        self.attempted_native_steps = self.integrated_native_steps = 0
        self.failed = False

    def forecast(
        self,
        nav: G1SonicNavigation,
        body: G1TargetEpisode,
        command: Callable[[np.ndarray, np.ndarray, int], tuple[float, float, float]],
    ) -> SonicNavigationForecast:
        if (
            self.failed
            or body.failed
            or body.next_frame != nav._next_frame
            or body.record_all_ball_contacts is not True
            or not 1 <= body.next_frame <= 260
            or self.attempted_native_steps + 400 > self.max_native_steps
        ):
            raise ValueError(
                "aligned current body/navigation and remaining40-frame budget required"
            )
        # Structural/model/gain admission only, never a body reset or step.
        G1PDTorqueForecaster(
            body.model,
            expected_world_hash=self.world_hash,
            kp=body.kp,
            kd=body.kd,
            max_forecast_steps=400,
        )
        donor_body = integration_hash(body.model, body.data)
        donor_nav = navigation_state_hash(nav)
        donor_fields = hash_json(
            _encode({k: v for k, v in body.__dict__.items() if k not in ("model", "data")})
        )
        branch = None
        origin_attempted, origin_integrated = body.attempted_steps, body.integrated_steps
        try:
            controller = clone_navigation(nav)
            branch = _clone_body(body)
            traces, commands, references, actions = [], [], [], []
            for frame in range(body.next_frame, body.next_frame + 40):
                q, v = branch.canonical()
                wanted = command(q.copy(), v.copy(), frame)
                observation = TeamMotorObservation(
                    agent_id=controller.agent_id,
                    frame=frame,
                    time_sec=float(branch.data.time),
                    intent="shoot",
                    prospective_owner=True,
                    qpos=tuple(float(x) for x in q),
                    qvel=tuple(float(x) for x in v),
                    target_position_m=(float(q[36]), float(q[37]), float(q[38])),
                    navigation_command=wanted,
                )
                proposal = controller.propose(observation)
                if not (
                    np.array_equal(proposal.kp, branch.kp)
                    and np.array_equal(proposal.kd, branch.kd)
                ):
                    raise ValueError("forecast changed source PD gains")
                local = frame - controller._origin_frame
                references.append(controller.backend.reference[local : local + 10].copy())
                actions.append(controller.backend.action.copy())
                commands.append(wanted)
                traces.append(branch.step(frame, np.asarray(proposal.target_rad, dtype=np.float64)))
            arrays = {k: np.concatenate([row[k] for row in traces]) for k in traces[0]}
            result = SonicNavigationForecast(
                arrays,
                np.asarray(commands),
                np.asarray(references),
                np.asarray(actions),
                donor_body,
                donor_nav,
                branch.attempted_steps - origin_attempted,
                branch.integrated_steps - origin_integrated,
                controller.backend.planner_calls - nav.backend.planner_calls,
            )
        except Exception:
            self.failed = True
            raise
        finally:
            if branch is not None:
                self.attempted_native_steps += branch.attempted_steps - origin_attempted
                self.integrated_native_steps += branch.integrated_steps - origin_integrated
            if (
                integration_hash(body.model, body.data) != donor_body
                or navigation_state_hash(nav) != donor_nav
                or hash_json(
                    _encode({k: v for k, v in body.__dict__.items() if k not in ("model", "data")})
                )
                != donor_fields
            ):
                self.failed = True
                raise ValueError("forecast changed donor body or whole navigation state")
        return result
