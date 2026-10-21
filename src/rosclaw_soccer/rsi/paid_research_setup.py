"""Pinned SIM-only PAiD benchmark setup shared by continuing research tasks."""

import hashlib
import sys
from pathlib import Path
from typing import Any


def open_paid_bench(pins: dict[str, str], cases: list[dict[str, Any]]) -> tuple[Any, Path, str]:
    from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench
    from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
    from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH

    policies = [
        Path(p)
        for p, h in pins.items()
        if p.endswith("/ckp/policy_30000.onnx") and h == PUBLISHED_HASH
    ]
    xmls = [Path(p) for p in pins if p.endswith("/cached-standard/declared-scene.xml")]
    if len(policies) != 1 or not xmls or len({pins[str(p)] for p in xmls}) != 1:
        raise ValueError("one exact qualified base actor and compiled standardbody required")
    policy, xml = policies[0], xmls[0]
    for p in (policy, xml):
        if "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest() != pins[str(p)]:
            raise ValueError("pinned base policy or physical recipe changed")
    contract = PaidResearchPolicy(policy, expected_hash=PUBLISHED_HASH, horizon=1)
    sys.path.insert(0, str(policy.parent.parent))
    from exp.mujoco_soccer.constants import BODY_NAMES, JOINT_EFFORT_LIMIT_ARRAY
    from exp.mujoco_soccer.math_utils import quat_apply
    from exp.mujoco_soccer.motion import load_motion_clips
    from exp.mujoco_soccer.observations import build_observation
    from exp.mujoco_soccer.scene import MujocoSoccer

    names = list(dict.fromkeys(c["source"] for c in cases))
    clips = {
        n: load_motion_clips(policy.parent.parent / "motions/soccer-standard" / n)[0] for n in names
    }
    bench = PaidResearchBench(
        xml=xml,
        contract=contract,
        clips=clips,
        scene_factory=MujocoSoccer,
        build_observation=build_observation,
        quat_apply=quat_apply,
        effort=JOINT_EFFORT_LIMIT_ARRAY,
        torso_index=BODY_NAMES.index("torso_link"),
    )
    for case in cases:
        bench.make(case)
    return bench, policy, pins[str(xml)]


def open_paid_auditor(
    pins: dict[str, str], cases: list[dict[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    """Independent native layout/effort/reference reader, no simulator steps."""
    import ast

    import mujoco
    import numpy as np
    import onnx

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_primitive_audit import PaidPrimitiveAuditor
    from rosclaw_soccer.rsi.paid_teacher_policy import strict_parameters

    policy = next(Path(p) for p in pins if p.endswith("/ckp/policy_30000.onnx"))
    xmls = [Path(p) for p in pins if p.endswith("/cached-standard/declared-scene.xml")]
    if not xmls or len({pins[str(p)] for p in xmls}) != 1:
        raise ValueError("one compiled standardbody required")
    graph = onnx.load(policy, load_external_data=False)
    names, params = strict_parameters({x.key: x.value for x in graph.metadata_props})
    order = np.array([names.index(n) for n in G1_DDS_JOINT_NAMES])
    effort = np.array(
        [
            139
            if ("hip_roll" in n or "knee" in n)
            else 88
            if ("hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint")
            else 50
            if ("ankle" in n or n in ("waist_roll_joint", "waist_pitch_joint"))
            else 5
            if ("wrist_pitch" in n or "wrist_yaw" in n)
            else 25
            for n in G1_DDS_JOINT_NAMES
        ]
    )
    model = mujoco.MjModel.from_xml_path(str(xmls[0]))
    layout = NativeVelocityLayout(model, robot_prefix="", ball_body_name="soccer_ball")
    goal_id = int(model.body_mocapid[model.body("soccer_goal_marker").id])
    auditor = PaidPrimitiveAuditor(
        layout=layout, parameters=params, order=order, effort=effort, goal_mocap_id=goal_id
    )
    const = policy.parent.parent / "exp/mujoco_soccer/constants.py"
    if str(const) not in pins:
        raise ValueError("source body ordering must be pinned")
    body_names = next(
        ast.literal_eval(n.value)
        for n in ast.parse(const.read_text()).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BODY_NAMES" for t in n.targets)
    )
    torso = body_names.index("torso_link")
    motions = {}
    for name in dict.fromkeys(c["source"] for c in cases):
        path = policy.parent.parent / "motions/soccer-standard" / name
        if str(path) not in pins:
            raise ValueError("motion reference must be pinned")
        with np.load(path, allow_pickle=False) as z:
            motions[name] = dict(
                q=z["joint_pos"].copy(),
                v=z["joint_vel"].copy(),
                w=z["body_ang_vel_w"][:, torso].copy(),
            )
    return auditor, motions
