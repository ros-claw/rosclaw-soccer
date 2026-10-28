"""Fit a pinned train reference window without integrating physics."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.temporal_contact_fit import fit_temporal_contact
from rosclaw_soccer.training.omnicontact_reference import LAB_TO_MUJOCO


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("scene", "reference", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("scene-sha256", "reference-sha256"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--center", type=int, required=True)
    p.add_argument("--half-width", type=int, required=True)
    p.add_argument("--max-evaluations", type=int, default=50)
    p.add_argument("--foot", choices=("left", "right"), required=True)
    a = p.parse_args()
    paths = ((a.scene, a.scene_sha256), (a.reference, a.reference_sha256))
    if any(path.is_symlink() or digest(path) != expected for path, expected in paths):
        raise ValueError("pinned scene and train reference required")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    data = mujoco.MjData(model)
    if not 2 <= a.half_width <= 50:
        raise ValueError("bounded window required")
    frames = np.arange(a.center - a.half_width, a.center + a.half_width + 1)
    states = []
    with np.load(a.reference, allow_pickle=False) as ref:
        if frames[0] < 0 or frames[-1] >= len(ref["joint_pos"]):
            raise ValueError("full window must exist")
        for f in frames:
            data.qpos[layout.rq : layout.rq + 3] = ref["body_pos_w"][f, 0]
            data.qpos[layout.rq + 3 : layout.rq + 7] = ref["body_quat_w"][f, 0]
            data.qpos[layout.qi] = ref["joint_pos"][f, list(LAB_TO_MUJOCO)]
            data.qpos[layout.bq : layout.bq + 3] = ref["object_pos_w"][f]
            data.qpos[layout.bq + 3 : layout.bq + 7] = ref["object_quat_w"][f]
            states.append(data.qpos.copy())
    result = fit_temporal_contact(
        model,
        layout,
        np.array(states),
        foot=a.foot,
        event=a.half_width,
        max_evaluations=a.max_evaluations,
    )
    if any(digest(path) != expected for path, expected in paths):
        raise ValueError("input changed during fit")
    result.update(
        scene_sha256=a.scene_sha256,
        reference_sha256=a.reference_sha256,
        frames=frames.tolist(),
        simulator_only=True,
        noncommercial_source=True,
        warning="sampled reference FK only, no rollout, training, or contact force",
    )
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "complete.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("before_gaps_m", "after_gaps_m", "active_leg_offsets_rad")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
