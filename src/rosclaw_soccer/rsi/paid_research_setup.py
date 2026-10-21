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
