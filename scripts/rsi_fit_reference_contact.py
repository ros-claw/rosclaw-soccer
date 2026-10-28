"""Retrospective one-frame reference geometry only; no physics steps."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.coupled_reference_fit import fit_reference_contact
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.training.omnicontact_reference import LAB_TO_MUJOCO


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--scene-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--max-evaluations", type=int, default=150)
    parser.add_argument("--foot", choices=("left", "right"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if digest(a.scene) != a.scene_sha256 or digest(a.reference) != a.reference_sha256:
        raise ValueError("exact pinned scene and reference required")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    data = mujoco.MjData(model)
    with np.load(a.reference, allow_pickle=False) as ref:
        if not 0 <= a.frame < len(ref["joint_pos"]):
            raise ValueError("valid declared reference frame required")
        frame = a.frame
        data.qpos[layout.rq : layout.rq + 3] = ref["body_pos_w"][frame, 0]
        data.qpos[layout.rq + 3 : layout.rq + 7] = ref["body_quat_w"][frame, 0]
        data.qpos[layout.qi] = ref["joint_pos"][frame, list(LAB_TO_MUJOCO)]
        data.qpos[layout.bq : layout.bq + 3] = ref["object_pos_w"][frame]
        data.qpos[layout.bq + 3 : layout.bq + 7] = ref["object_quat_w"][frame]
    mujoco.mj_forward(model, data)
    result = fit_reference_contact(
        model, data, layout, foot=a.foot, max_evaluations=a.max_evaluations
    )
    if digest(a.scene) != a.scene_sha256 or digest(a.reference) != a.reference_sha256:
        raise ValueError("input changed during fit")
    result.update(
        scene_sha256=a.scene_sha256,
        reference_sha256=a.reference_sha256,
        frame=a.frame,
        simulator_only=True,
        research_only_noncommercial=True,
        warning="retrospective single-frame FK, not a dynamic trajectory or policy",
    )
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
