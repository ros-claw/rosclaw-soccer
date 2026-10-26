"""Compare two pinned G1 bodies on candidate reference geometry only."""

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.candidate_motion23_mapping import candidate_map_motion23
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.reference_support_geometry import support_distances
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def compare_candidate_bodies(
    canonical: Any,
    candidate: Any,
    fields: dict[str, np.ndarray],
    *,
    canonical_hash: str,
    candidate_hash: str,
) -> dict[str, Any]:
    """Compare ankle FK and marked foot clearance, never physics or provenance."""
    import mujoco

    if (
        compiled_model_hash(canonical) != canonical_hash
        or compiled_model_hash(candidate) != candidate_hash
    ):
        raise ValueError("both explicitly pinned bodies required")
    floor = int(canonical.geom("floor").id)
    if canonical.geom_bodyid[floor] != 0 or canonical.geom_pos[floor, 2] != 0:
        raise ValueError("world-fixed zero-height canonical plane required")
    canonical_distances, labels = support_distances(
        canonical, fields, expected_model_hash=canonical_hash
    )
    if not labels.any():
        raise ValueError("at least one claimed support label required")
    mapped = candidate_map_motion23(fields["dof"], fields["root_rot"])
    lo = NativeVelocityLayout(canonical, robot_prefix="a/", ball_body_name="soccer_ball")
    names = G1_DDS_JOINT_NAMES[:19] + G1_DDS_JOINT_NAMES[22:26]
    joints = [int(candidate.joint(name).id) for name in names]
    qids = [int(candidate.jnt_qposadr[j]) for j in joints]
    base_joint = int(candidate.body_jntadr[candidate.body("pelvis").id])
    if (
        len(set(qids)) != 23
        or any(candidate.jnt_type[j] != int(mujoco.mjtJoint.mjJNT_HINGE) for j in joints)
        or candidate.jnt_type[base_joint] != int(mujoco.mjtJoint.mjJNT_FREE)
    ):
        raise ValueError("candidate free pelvis and 23 unique named hinge joints required")
    root = int(candidate.jnt_qposadr[base_joint])
    ankle = [int(candidate.body(side + "_ankle_roll_link").id) for side in ("left", "right")]
    feet = [
        [
            g
            for g in range(candidate.ngeom)
            if candidate.geom_bodyid[g] == body
            and candidate.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_SPHERE)
            and candidate.geom_contype[g] != 0
        ]
        for body in ankle
    ]
    if any(len(foot) != 4 for foot in feet):
        raise ValueError("candidate requires four explicit collision spheres per foot")
    d1, d2 = mujoco.MjData(canonical), mujoco.MjData(candidate)
    d1.qpos[lo.bq : lo.bq + 7] = [100, 100, 0.11, 1, 0, 0, 0]
    diffs = np.empty((len(fields["dof"]), 2), dtype=np.float64)
    candidate_gaps = np.empty_like(diffs)
    for i in range(len(diffs)):
        pose = np.r_[fields["root_trans_offset"][i], mapped["root_rot_wxyz"][i]]
        d1.qpos[lo.rq : lo.rq + 7], d2.qpos[root : root + 7] = pose, pose
        d1.qpos[lo.qi], d2.qpos[qids] = mapped["dof29"][i], fields["dof"][i]
        mujoco.mj_kinematics(canonical, d1)
        mujoco.mj_kinematics(candidate, d2)
        for side, name in enumerate(("left", "right")):
            body = int(canonical.body("a/" + name + "_ankle_roll_link").id)
            diffs[i, side] = np.linalg.norm(d1.xpos[body] - d2.xpos[ankle[side]])
            candidate_gaps[i, side] = min(
                float(d2.geom_xpos[g, 2] - candidate.geom_size[g, 0]) for g in feet[side]
            )
    if d1.time != 0 or d2.time != 0:
        raise ValueError("body comparison must never integrate")
    marked = candidate_gaps[labels]
    delta = (canonical_distances - candidate_gaps)[labels]
    return dict(
        frames=len(diffs),
        ankle_samples=diffs.size,
        maximum_ankle_center_difference_m=float(diffs.max()),
        mean_ankle_center_difference_m=float(diffs.mean()),
        claimed_contact_samples=len(marked),
        canonical_minus_candidate_mean_gap_m=float(delta.mean()),
        gap_difference_abs_p95_m=float(np.quantile(np.abs(delta), 0.95)),
        candidate_claimed_contact_mean_gap_m=float(marked.mean()),
        candidate_claimed_contact_min_gap_m=float(marked.min()),
        source_body_provenance_verified=False,
        source_contact_labels_verified=False,
        candidate_mapping_assumed=True,
        integrated_native_steps=0,
        physical_execution_qualified=False,
        hardware_authorized=False,
    )
