"""SIM-only frozen velocity-conditioned G1 policy, not a robot transport.

The 98D observation/action codec follows the pinned deployment parameter file.
No simulator, driver, SDK, ROS or hardware handle enters this provider.
"""

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

OBSERVATIONS = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("gait_phase", 2),
    ("joint_pos_rel", 29),
    ("joint_vel_rel", 29),
    ("last_action", 29),
)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return "sha256:" + h.hexdigest()


def _vector(value: Any, n: int, *, positive: bool = False) -> tuple[float, ...]:
    if (
        type(value) is not list
        or len(value) != n
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
    ):
        raise ValueError("finite ordinary numeric parameter vector required")
    a = np.asarray(value, dtype=np.float32)
    if not np.isfinite(a).all() or np.max(abs(a)) > 2000 or (positive and np.any(a <= 0)):
        raise ValueError("bounded deployment float32 parameters required")
    return tuple(float(x) for x in a)


@dataclass(frozen=True)
class VelocityPolicyParameters:
    config_hash: str
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    defaults: tuple[float, ...]
    action_scale: tuple[float, ...]
    action_offset: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            type(self.config_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.config_hash) is None
        ):
            raise ValueError("explicit deployment configuration identity required")
        for name in ("stiffness", "damping", "defaults", "action_scale", "action_offset"):
            value = getattr(self, name)
            if (
                type(value) is not tuple
                or len(value) != 29
                or any(type(x) is not float or not math.isfinite(x) or abs(x) > 2000 for x in value)
                or any(float(np.float32(x)) != x for x in value)
            ):
                raise ValueError("owned canonical deployment float32 parameter tuples required")
        if (
            any(x <= 0 for x in (*self.stiffness, *self.damping, *self.action_scale))
            or max(abs(x) for x in self.defaults) > 3.2
            or self.defaults != self.action_offset
        ):
            raise ValueError("positive gains/scales and bounded matching default pose required")

    @classmethod
    def load(cls, path: Path, *, expected_hash: str) -> "VelocityPolicyParameters":
        import yaml

        if path.stat().st_size > 65536 or file_hash(path) != expected_hash:
            raise ValueError("exact bounded deployment configuration required")
        c = yaml.safe_load(path.read_text())
        if (
            type(c) is not dict
            or set(c)
            != {
                "joint_ids_map",
                "step_dt",
                "stiffness",
                "damping",
                "default_joint_pos",
                "commands",
                "actions",
                "observations",
            }
            or c["joint_ids_map"] != list(range(29))
            or any(type(v) is not int for v in c["joint_ids_map"])
            or type(c["step_dt"]) is not float
            or c["step_dt"] != 0.02
        ):
            raise ValueError("explicit canonical29 identity mapping and 50Hz deployment required")
        if c["commands"] != {
            "base_velocity": {
                "ranges": {
                    "lin_vel_x": [-0.5, 1.0],
                    "lin_vel_y": [-0.5, 0.5],
                    "ang_vel_z": [-1.0, 1.0],
                    "heading": None,
                }
            }
        }:
            raise ValueError("exact qualified body-frame command ranges required")
        if type(c["actions"]) is not dict or set(c["actions"]) != {"JointPositionAction"}:
            raise ValueError("only declared joint position output codec allowed")
        act = c["actions"]["JointPositionAction"]
        if (
            set(act) != {"clip", "joint_names", "scale", "offset", "joint_ids"}
            or act["clip"] is not None
            or act["joint_ids"] is not None
            or act["joint_names"] != [".*"]
        ):
            raise ValueError("unclipped full29 action codec required")
        if tuple(c["observations"]) != tuple(k for k, _ in OBSERVATIONS):
            raise ValueError("exact ordered98 observation terms required")
        for name, n in OBSERVATIONS:
            term = c["observations"][name]
            params = (
                {"period": 0.6}
                if name == "gait_phase"
                else {"command_name": "base_velocity"}
                if name == "velocity_commands"
                else {}
            )
            if (
                set(term) != {"params", "clip", "scale", "history_length"}
                or term["params"] != params
                or term["clip"] is not None
                or type(term["history_length"]) is not int
                or term["history_length"] != 1
                or _vector(term["scale"], n) != (1.0,) * n
            ):
                raise ValueError("exact unscaled unfiltered observation term required: " + name)
        result = cls(
            expected_hash,
            _vector(c["stiffness"], 29, positive=True),
            _vector(c["damping"], 29, positive=True),
            _vector(c["default_joint_pos"], 29),
            _vector(act["scale"], 29, positive=True),
            _vector(act["offset"], 29),
        )
        if result.defaults != result.action_offset or max(abs(x) for x in result.defaults) > 3.2:
            raise ValueError("bounded declared initial pose and matching output offset required")
        return result

    @property
    def contract_hash(self) -> str:
        return hash_json(
            dict(
                schema="soccer.g1.frozen_velocity_parameters.v1",
                **self.__dict__,
                observation_terms=OBSERVATIONS,
                control_hz=50,
                gait_period_sec=0.6,
                gait_reset_primes_observations_once=True,
                action_math="float32_multiply_then_add_then_float64_target",
                activation_ceiling="SIM_ONLY",
                hardware_authorized=False,
            )
        )


def phase_delta() -> np.float32:
    return np.float32(np.float32(0.02) * np.float32(np.float32(1.0) / np.float32(0.6)))


def observation98(
    qpos: np.ndarray,
    qvel: np.ndarray,
    angular_velocity_body: np.ndarray,
    command_body: np.ndarray,
    last_action: np.ndarray,
    phase: np.float32,
    params: VelocityPolicyParameters,
) -> tuple[np.ndarray, np.float32]:
    for x, shape, dtype in (
        (qpos, (43,), np.float64),
        (qvel, (41,), np.float64),
        (angular_velocity_body, (3,), np.float64),
        (command_body, (3,), np.float32),
        (last_action, (29,), np.float32),
    ):
        if (
            type(x) is not np.ndarray
            or x.shape != shape
            or x.dtype != dtype
            or not np.isfinite(x).all()
        ):
            raise ValueError("owned finite typed current velocity-policy inputs required")
    if (
        abs(np.linalg.norm(qpos[3:7]) - 1) > 1e-5
        or type(phase) is not np.float32
        or not np.isfinite(phase)
        or not 0 <= phase < 1
        or not -0.5 <= command_body[0] <= 1
        or not -0.5 <= command_body[1] <= 0.5
        or abs(command_body[2]) > 1
    ):
        raise ValueError("unit root, bounded body commands and float32 phase required")
    phase = np.float32(np.fmod(np.float32(phase + phase_delta()), np.float32(1.0)))
    w = qpos[3:7].astype(np.float32)
    # Eigen Quaternionf conjugate-vector product: v + w*2(qv x v) + qv x 2(qv x v).
    gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    vector = -w[1:]
    uv = np.cross(vector, gravity)
    uv = uv + uv
    gravity = gravity + w[0] * uv + np.cross(vector, uv)
    gait = np.array(
        [math.sin(float(phase) * 2 * math.pi), math.cos(float(phase) * 2 * math.pi)],
        dtype=np.float32,
    )
    norm = np.float32(
        np.sqrt(
            np.float32(
                np.float32(command_body[0] * command_body[0] + command_body[1] * command_body[1])
                + command_body[2] * command_body[2]
            )
        )
    )
    if norm < np.float32(0.1):
        gait[:] = 0.0
    obs = np.concatenate(
        (
            angular_velocity_body.astype(np.float32),
            gravity,
            command_body,
            gait,
            qpos[7:36].astype(np.float32) - np.asarray(params.defaults, dtype=np.float32),
            qvel[6:35].astype(np.float32),
            last_action,
        )
    ).astype(np.float32)
    if obs.shape != (98,) or not np.isfinite(obs).all():
        raise ValueError("finite exact98 model observation required")
    return obs, phase


def target29(action: np.ndarray, params: VelocityPolicyParameters) -> np.ndarray:
    if (
        type(action) is not np.ndarray
        or action.shape != (29,)
        or action.dtype != np.float32
        or not np.isfinite(action).all()
    ):
        raise ValueError("finite raw float32 HG29 action required")
    # Preserve the deployment float32 two-operation rounding, no new clipping.
    target = action * np.asarray(params.action_scale, dtype=np.float32)
    target = target + np.asarray(params.action_offset, dtype=np.float32)
    if not np.isfinite(target).all() or np.max(abs(target)) > 10:
        raise ValueError("invalid model target; no target clipping or retry")
    return target.astype(np.float64)


class G1VelocityPolicy:
    def __init__(
        self, model: Path, config: Path, *, expected_model_hash: str, expected_config_hash: str
    ) -> None:
        self.params = VelocityPolicyParameters.load(config, expected_hash=expected_config_hash)
        if file_hash(model) != expected_model_hash:
            raise ValueError("exact frozen velocity model required before session allocation")
        import onnx
        import onnxruntime as ort

        graph = onnx.load(str(model), load_external_data=False)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
            raise ValueError("unbound external ONNX tensors not supported")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        ins, outs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(ins) != 1
            or ins[0].name != "obs"
            or ins[0].shape != [1, 98]
            or ins[0].type != "tensor(float)"
            or len(outs) != 1
            or outs[0].name != "actions"
            or outs[0].shape != [1, 29]
            or outs[0].type != "tensor(float)"
            or self.session.get_providers() != ["CPUExecutionProvider"]
        ):
            raise ValueError("exact CPU float32 98->29 actor contract required")
        self.contract_hash = hash_json(
            dict(
                schema="soccer.g1.frozen_velocity_policy.v1",
                model_hash=expected_model_hash,
                parameter_hash=self.params.contract_hash,
                activation_ceiling="SIM_ONLY",
                hardware_authorized=False,
            )
        )
        self.phase = (
            phase_delta()
        )  # Upstream reset() primes observation terms once, without inference.
        self.last_action = np.zeros(29, dtype=np.float32)
        self.next_frame = 0
        self.attempted_neural_calls = 0
        self.neural_calls = 0
        self.faulted = False
        self.last_observation = np.zeros(98, dtype=np.float32)

    def propose(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        angular_velocity_body: np.ndarray,
        command_body: np.ndarray,
        *,
        frame: int,
    ) -> np.ndarray:
        if self.faulted:
            raise ValueError("velocity actor fault latched; no implicit episode restart")
        try:
            if type(frame) is not int or frame != self.next_frame or not 0 <= frame < 300:
                raise ValueError("consecutive H300 actual actor observations required")
            obs, phase = observation98(
                qpos,
                qvel,
                angular_velocity_body,
                command_body,
                self.last_action,
                self.phase,
                self.params,
            )
            self.attempted_neural_calls += 1
            output = self.session.run(["actions"], {"obs": obs[None]})[0]
            self.neural_calls += 1
            if (
                type(output) is not np.ndarray
                or output.shape != (1, 29)
                or output.dtype != np.float32
            ):
                raise ValueError("actual finite model action tensor contract differs")
            target = target29(output[0], self.params)
            self.last_observation = obs.copy()
            self.last_action = output[0].copy()
            self.phase = phase
            self.next_frame += 1
            return target
        except Exception:
            self.faulted = True
            raise
