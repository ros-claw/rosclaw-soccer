"""Load only pinned original TRAIN8 motion tensors for the native SIM course."""

import sys
from pathlib import Path


def load_body_references(reference_root, model, layout, plans):
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_GLOB, TRAIN_MOTION_NAMES
    from rosclaw_soccer.rsi.native_course_tracking import TRACKING_BODIES

    sys.path.insert(0, str(reference_root))
    from src.tasks.soccer.mdp.shooter_commands import MultiMotionLoader

    names = [
        model.body(i).name.removeprefix("robot/")
        for i in range(model.nbody)
        if (model.body(i).name or "").startswith("robot/")
    ]
    if len(names) != 30 or names[0] != "pelvis" or names[-1] != "right_wrist_yaw_link":
        raise ValueError("original native30 robot anatomical order required")
    loader = MultiMotionLoader(
        str(Path(reference_root) / "src/assets/soccer/motions/shooter"),
        torch.tensor([names.index(n) for n in TRACKING_BODIES], dtype=torch.long),
        device="cpu",
        motion_glob=TRAIN_MOTION_GLOB,
    )
    if tuple(n + ".npz" for n in loader.motion_names) != TRAIN_MOTION_NAMES:
        raise ValueError("TRAIN8 only; no DEV source references")
    joints = [
        i
        for i in range(model.njnt)
        if (model.joint(i).name or "").startswith("robot/") and model.jnt_type[i] == 3
    ]
    if len(joints) != 29:
        raise ValueError("native29 hinge order required")
    hg = [joints.index(j) for j in layout.joints]
    output = []
    for i, plan in enumerate(plans):
        length = int(loader.file_lengths[i])
        values = {
            field: getattr(loader, attr)[i, :length].numpy().copy()
            for field, attr in (
                ("position", "body_pos_w"),
                ("quaternion", "body_quat_w"),
                ("linear", "body_lin_vel_w"),
                ("angular", "body_ang_vel_w"),
            )
        }
        if not np.array_equal(
            loader.joint_pos[i, :length, hg].numpy(), plan["joint"]
        ) or not np.array_equal(loader.joint_vel[i, :length, hg].numpy(), plan["joint_velocity"]):
            raise ValueError("original full reference HG joint semantics changed")
        for label in ("pelvis", "torso"):
            idx = TRACKING_BODIES.index("torso_link" if label == "torso" else label)
            for field in values:
                if not np.array_equal(values[field][:, idx], plan[label + "_" + field]):
                    raise ValueError("original full reference anchor semantics changed")
        output.append(values)
    return output
