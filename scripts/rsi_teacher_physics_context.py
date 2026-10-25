"""Load an already pinned CPU SIM teacher context; no rollout or hardware allocation."""

import json
from pathlib import Path


def load_teacher_context(pins, backbone_model):
    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.experimental_velocity_evidence import admit_heading_evidence
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from scripts.rsi_native_context_recovery_common import digest

    def one(suffix):
        rows = [Path(p) for p in pins if p.endswith(suffix)]
        if len(rows) != 1:
            raise ValueError("unique pinned teacher asset required: " + suffix)
        return rows[0]

    configs = [
        Path(p)
        for p in pins
        if p.endswith(".json")
        and Path(p).stat().st_size < 2048
        and json.loads(Path(p).read_text()).get("schema")
        == "soccer.native_single_finisher_experimental_approach.v1"
    ]
    if len(configs) != 1:
        raise ValueError("one qualified approach codec required")
    fast, _ = admit_heading_evidence(configs[0], pins[str(configs[0])])
    vm, vc = one("/velocity/v0/exported/policy.onnx"), one("/velocity/v0/params/deploy.yaml")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    report = json.loads(backbone_model.read_text())
    artifact = backbone_model.parent / report["artifact"]["path"]
    if pins.get(str(backbone_model)) != digest(backbone_model) or pins.get(str(artifact)) != digest(
        artifact
    ):
        raise ValueError("exact bound frozen motion basis required")
    with np.load(artifact, allow_pickle=False) as raw:
        basis = raw["basis_hg"].copy()
    return dict(
        model=model,
        layout=NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball"),
        types=_load_robonaldo(one("/policy/robonaldo/FreeKick.py").parents[2]),
        assets=dict(
            velocity_model=vm,
            velocity_config=vc,
            velocity_model_hash=pins[str(vm)],
            velocity_config_hash=pins[str(vc)],
            **fast,
        ),
        basis=basis,
        hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
    )
