"""Strict research-only PAiD ONNX LSTM reader, not a hardware policy backend.

Weights remain external. Original reference is CC-BY-NC4.0; this loader neither
relicenses weights nor authorizes commercial publicity or robot execution.
One object owns one episode's h/c. No silent padding, fallback, restart or crop.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES

INPUT_SHAPES = dict(obs=[1, 160], h_in=[2, 1, 128], c_in=[2, 1, 128], time_step=[1, 1])
OUTPUT_SHAPES = dict(
    actions=[1, 29],
    h_out=[2, 1, 128],
    c_out=[2, 1, 128],
    joint_pos=[1, 29],
    joint_vel=[1, 29],
    body_pos_w=[1, 14, 3],
    body_quat_w=[1, 14, 4],
    body_lin_vel_w=[1, 14, 3],
    body_ang_vel_w=[1, 14, 3],
    time_step_total=[1, 1],
)
OBSERVATION_NAMES = (
    "command",
    "projected_gravity",
    "motion_ref_ang_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "target_point_pos",
    "target_destination_pos_local",
)


def strict_parameters(metadata: dict[str, str]) -> tuple[list[str], dict[str, np.ndarray]]:
    names = [s.strip() for s in metadata.get("joint_names", "").split(",")]
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(G1_DDS_JOINT_NAMES):
        raise ValueError("exact bijective named29 motor order required")
    if (
        tuple(metadata.get("observation_names", "").split(",")) != OBSERVATION_NAMES
        or metadata.get("command_names") != "motion"
        or metadata.get("anchor_body_name") != "torso_link"
    ):
        raise ValueError("explicit160 observation and motion-anchor semantics required")
    result = {}
    for key in ("joint_stiffness", "joint_damping", "default_joint_pos", "action_scale"):
        values = np.array([float(s) for s in metadata.get(key, "").split(",")], dtype=np.float32)
        if values.shape != (29,) or not np.isfinite(values).all():
            raise ValueError("finite exact29 metadata without fallback required: " + key)
        if key != "default_joint_pos" and (values <= 0).any():
            raise ValueError("positive actual PD scales/gains required")
        result[key] = values
    return names, result


class PaidResearchPolicy:
    def __init__(self, path: Path, *, expected_hash: str, horizon: int) -> None:
        import onnx
        import onnxruntime as ort

        if (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
            or type(horizon) is not int
            or not 1 <= horizon <= 260
        ):
            raise ValueError("pinned teacher and actual bounded source horizon required")
        graph = onnx.load(path, load_external_data=False)
        tensors = [
            *graph.graph.initializer,
            *[
                a.t
                for n in graph.graph.node
                for a in n.attribute
                if a.type == onnx.AttributeProto.TENSOR
            ],
        ]
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in tensors):
            raise ValueError("self-contained known ONNX weights required")
        if any(n.domain not in ("", "ai.onnx") for n in graph.graph.node):
            raise ValueError("no custom ONNX operators in research reader")
        onnx.checker.check_model(graph, full_check=True)
        self.joint_names, self.parameters = strict_parameters(
            {x.key: x.value for x in graph.metadata_props}
        )
        self.hg_from_policy = np.array(
            [self.joint_names.index(name) for name in G1_DDS_JOINT_NAMES]
        )
        self.policy_from_hg = np.argsort(self.hg_from_policy)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        for infos, expected in (
            (self.session.get_inputs(), INPUT_SHAPES),
            (self.session.get_outputs(), OUTPUT_SHAPES),
        ):
            if {x.name: x.shape for x in infos} != expected or any(
                x.type != "tensor(float)" for x in infos
            ):
                raise ValueError("exact static float32 LSTM160/29 IO contract required")
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.h = np.zeros((2, 1, 128), dtype=np.float32)
        self.c = np.zeros_like(self.h)
        self.horizon = horizon
        self.frame = self.neural_calls = self.attempted_neural_calls = 0
        self.faulted = False

    def step(self, observation: np.ndarray, *, frame: int) -> dict[str, Any]:
        if self.faulted:
            raise ValueError("teacher inference fault latched; new episode object required")
        try:
            if (
                type(frame) is not int
                or frame != self.frame
                or not 0 <= frame < self.horizon
                or observation.shape != (160,)
                or observation.dtype != np.float32
                or not np.isfinite(observation).all()
            ):
                raise ValueError(
                    "consecutive actual finite160 input, no crop/pad/reference extension"
                )
            before = dict(
                h_before=self.h.copy(), c_before=self.c.copy(), observation=observation.copy()
            )
            self.attempted_neural_calls += 1
            outputs = self.session.run(
                None,
                dict(
                    obs=observation[None],
                    h_in=self.h,
                    c_in=self.c,
                    time_step=np.array([[frame]], dtype=np.float32),
                ),
            )
            self.neural_calls += 1
            values = dict(zip(self.output_names, outputs, strict=True))
            for key, shape in OUTPUT_SHAPES.items():
                if (
                    list(values[key].shape) != shape
                    or values[key].dtype != np.float32
                    or not np.isfinite(values[key]).all()
                ):
                    raise ValueError("finite exact teacher output required: " + key)
            if values["time_step_total"][0, 0] != 260:
                raise ValueError("known260 exported reference table identity required")
            self.h, self.c = values["h_out"].copy(), values["c_out"].copy()
            self.frame += 1
            return dict(
                **before,
                action=values["actions"][0].copy(),
                h_after=self.h.copy(),
                c_after=self.c.copy(),
            )
        except Exception:
            self.faulted = True
            raise
