"""Build explicitly counterfactual, bounded joint-target imitation labels.

Actual observations/actions are physical collection data; the correction is
not a physically certified teacher. No simulator or hardware is accessed.
"""

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters


def build_imitation_rows(
    raw: dict[str, np.ndarray], params: VelocityPolicyParameters
) -> dict[str, np.ndarray]:
    required = {
        "full_observation": (168,),
        "action": (29,),
        "target": (29,),
        "reference_joint": (29,),
        "measured_joint": (29,),
        "active_before": (),
        "source_id": (),
        "source_phase": (),
    }
    if not required.keys() <= raw.keys():
        raise ValueError("complete actual TRAIN collection fields required")
    frames = len(raw["action"])
    if not 1 <= frames <= 300:
        raise ValueError("bounded whole collection required")
    for key, suffix in required.items():
        value = raw[key]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (frames, 8, *suffix)
            or not np.isfinite(value).all()
        ):
            raise ValueError("finite eight-world aligned collection arrays required")
    active = raw["active_before"]
    if any(
        raw[k].dtype != np.float32 for k in ("full_observation", "action", "reference_joint")
    ) or any(raw[k].dtype.kind not in "iu" for k in ("source_id", "source_phase")):
        raise ValueError(
            "original float32 observations/actions and integer source identifiers required"
        )
    if (
        active.dtype != np.bool_
        or not active[0].all()
        or np.any(np.diff(active.astype(int), axis=0) > 0)
    ):
        raise ValueError("all eight first episodes only; no post-reset sample re-entry")
    source = raw["source_id"]
    phase = raw["source_phase"]
    for world in range(8):
        count = int(active[:, world].sum())
        if not np.all(source[:count, world] == world) or not np.array_equal(
            phase[:count, world], np.arange(count)
        ):
            raise ValueError(
                "eight balanced source identities and consecutive frame0 phases required"
            )
    obs = raw["full_observation"][active].astype(np.float32)
    base = raw["action"][active].astype(np.float32)
    reference = raw["reference_joint"][active].astype(np.float32)
    measured = raw["measured_joint"][active].astype(np.float32)
    scale = np.asarray(params.action_scale, dtype=np.float32)
    offset = np.asarray(params.action_offset, dtype=np.float32)
    if not np.array_equal(obs[:, 98:127], reference) or not np.allclose(
        obs[:, 11:40], measured - np.asarray(params.defaults, dtype=np.float32), rtol=0, atol=1e-6
    ):
        raise ValueError("actual actor reference/current-joint channels must bind the labels")
    base_target = base * scale + offset
    if not np.allclose(base_target, raw["target"][active], rtol=0, atol=1e-7):
        raise ValueError("actual executed original joint-target codec required")
    delta = np.float32(0.25) * np.clip(reference - measured, np.float32(-0.6), np.float32(0.6))
    target_action = base + delta / scale
    desired_target = target_action * scale + offset
    if (
        not np.isfinite(desired_target).all()
        or np.abs(desired_target).max() > 10
        or np.abs(desired_target - base_target).max() > 0.150001
    ):
        raise ValueError("bounded finite counterfactual label target required")
    frame_ids, world_ids = np.nonzero(active)
    return dict(
        observation=obs,
        base_action=base,
        target_action=target_action,
        target_delta_rad=delta,
        source_id=source[active].astype(np.int64),
        control_index=frame_ids.astype(np.int64),
        world_index=world_ids.astype(np.int64),
    )


def localize_kick_window(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Restrict label corrections to a fixed +/-15-control reference swing window.

    Centers use only TRAIN reference hip/knee extension velocity, never ball
    outcomes. Outside the triangular window the executed parent action is the
    exact teacher. This is still a counterfactual label, not a certified kick.
    """
    source = rows["source_id"]
    phase = rows["control_index"]
    observation = rows["observation"]
    if (
        observation.shape != (len(source), 168)
        or source.shape != phase.shape
        or not np.isfinite(observation).all()
        or source.dtype.kind not in "iu"
        or phase.dtype.kind not in "iu"
        or set(source.tolist()) != set(range(8))
    ):
        raise ValueError("all eight finite TRAIN source rows required")
    weight = np.zeros(len(source), dtype=np.float32)
    centers = np.zeros(len(source), dtype=np.int64)
    for world, side in enumerate(
        ("right", "left", "left", "right", "right", "right", "left", "right")
    ):
        ids = np.flatnonzero(source == world)
        if not np.array_equal(phase[ids], np.arange(len(ids))):
            raise ValueError("consecutive first-episode TRAIN phases required")
        hip = G1_DDS_JOINT_NAMES.index(side + "_hip_pitch_joint")
        knee = G1_DDS_JOINT_NAMES.index(side + "_knee_joint")
        velocity = observation[ids, 127:156]
        score = np.maximum(-velocity[:, hip], 0) + np.maximum(-velocity[:, knee], 0)
        if score.max() <= 0:
            raise ValueError("positive reference forward-swing signal required")
        center = int(score.argmax())
        centers[ids] = center
        weight[ids] = np.maximum(0, 1 - np.abs(phase[ids] - center) / 15)
    delta_action = rows["target_action"] - rows["base_action"]
    result = {k: v.copy() for k, v in rows.items()}
    result.update(
        target_action=rows["base_action"] + weight[:, None] * delta_action,
        target_delta_rad=weight[:, None] * rows["target_delta_rad"],
        correction_weight=weight,
        kick_center_control_index=centers,
    )
    return result
