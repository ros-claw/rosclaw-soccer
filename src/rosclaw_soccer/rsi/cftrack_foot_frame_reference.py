"""One explicit SIM CFgen pitch-to-roll reference semantic experiment.

Uses the original full-body motion, not a fitted contact pose. The caller must
qualify the external pure FK source before invoking it. No physics or NN runs.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_contact_reference import _multiply
from rosclaw_soccer.sim.contracts import hash_json


def condition_cftrack_roll_reference(policy: Any) -> dict[str, Any]:
    import mujoco

    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    if (
        policy.task != "kickball"
        or policy.reference_source != "CFgen"
        or policy.counter_step != 0
        or policy.async_stage_thread is not None
        or np.any(policy.obs_history_buffer)
        or np.any(policy.action)
    ):
        raise ValueError("one cold unexecuted CFgen reference required")
    base, quat, joints = policy.ref_base_pos, policy.ref_base_quat, policy.dof_pos
    torso = policy.ref_torso_future_pos
    frames = len(base)
    arrays = [
        (base, (frames, 3)),
        (quat, (frames, 4)),
        (joints, (frames, 29)),
        (torso, (frames, 3)),
    ]
    names = [
        f"ref_{side}_ankle_future_{part}" for side in ("left", "right") for part in ("pos", "quat")
    ]
    for name in names:
        arrays.append((getattr(policy, name), (frames, 4 if name.endswith("quat") else 3)))
    if not 2 <= frames <= 1000 or any(
        type(a) is not np.ndarray
        or a.shape != shape
        or a.dtype.kind != "f"
        or not np.isfinite(a).all()
        for a, shape in arrays
    ):
        raise ValueError("finite complete aligned original body/joint/foot references required")
    for a, _ in arrays:
        if a.shape[-1] == 4 and not np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError("normalized wxyz reference quaternions required")
    fk = policy.kinematics
    if type(fk.model) is not mujoco.MjModel or type(fk.data) is not mujoco.MjData:
        raise ValueError("source-qualified scratch MuJoCo FK resources required")
    model_hash = compiled_model_hash(fk.model)
    # forward() on the pinned author class only writes self.data. Never invoke
    # external copy/pickle hooks, replace the donor's data, or advance its clock.
    scratch = object.__new__(type(fk))
    vars(scratch).update(vars(fk))
    scratch.data = mujoco.MjData(fk.model)
    replacements = {name: np.empty_like(getattr(policy, name)) for name in names}
    for tick in range(frames):
        measured = scratch.forward(joints[tick], base[tick], quat[tick])
        for side in ("left", "right"):
            prefix = f"ref_{side}_ankle_future_"
            pitch = measured[side + "_ankle_pitch_link"]
            old_p, old_q = (
                getattr(policy, prefix + "pos")[tick],
                getattr(policy, prefix + "quat")[tick],
            )
            q = np.asarray(pitch["quat"], np.float64).copy()
            if (
                q.shape != (4,)
                or not np.isfinite(q).all()
                or abs(np.linalg.norm(q) - 1) > 1e-5
                or np.asarray(pitch["pos"]).shape != (3,)
                or not np.isfinite(pitch["pos"]).all()
            ):
                raise ValueError("finite normalized source FK pitch pose required")
            q /= np.linalg.norm(q)
            original = old_q.astype(np.float64)
            original /= np.linalg.norm(original)
            if (
                np.linalg.norm(pitch["pos"] - old_p) > 1e-6
                or 2 * np.arccos(np.clip(abs(q @ original), 0, 1)) > 1e-5
            ):
                raise ValueError(
                    "original pitch references do not match their complete joint/base FK"
                )
            roll = measured[side + "_ankle_roll_link"]
            replacements[prefix + "pos"][tick] = roll["pos"]
            replacements[prefix + "quat"][tick] = roll["quat"]
    metrics = {}
    for side in ("left", "right"):
        prefix = f"ref_{side}_ankle_future_"
        pos, rot = replacements[prefix + "pos"], replacements[prefix + "quat"]
        if (
            not np.isfinite(pos).all()
            or not np.isfinite(rot).all()
            or not np.allclose(np.linalg.norm(rot, axis=1), 1, atol=1e-5, rtol=0)
        ):
            raise ValueError("finite normalized encoded roll reference required")
        offset = pos.astype(np.float64) - getattr(policy, prefix + "pos")
        speed = float(np.linalg.norm(np.diff(offset, axis=0) / 0.02, axis=1).max())
        old = getattr(policy, prefix + "quat").astype(np.float64)
        old /= np.linalg.norm(old, axis=1)[:, None]
        old[:, 1:] *= -1
        unit = rot.astype(np.float64)
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        correction = _multiply(unit, old)
        correction /= np.linalg.norm(correction, axis=1)[:, None]
        angle_speed = float(
            (
                2
                * np.arccos(np.clip(abs(np.sum(correction[1:] * correction[:-1], axis=1)), 0, 1))
                / 0.02
            ).max()
        )
        workspace = float(np.linalg.norm(pos - torso, axis=1).max())
        floor = float(pos[:, 2].min())
        if speed > 3.0 + 1e-8 or angle_speed > 8.0 + 1e-8 or workspace > 1.25 or floor < -0.05:
            raise ValueError(
                "roll reference exceeds unchanged sampled speed/workspace/floor envelope"
            )
        metrics[side] = dict(
            maximum_correction_m=float(np.linalg.norm(offset, axis=1).max()),
            maximum_extra_speed_m_s=speed,
            maximum_extra_angular_speed_rad_s=angle_speed,
            maximum_torso_distance_m=workspace,
            minimum_z_m=floor,
        )
    if compiled_model_hash(fk.model) != model_hash:
        raise ValueError("scratch FK changed the shared immutable model")
    receipt = dict(
        schema="soccer.cftrack_ankle_frame_semantic_probe.v1",
        original_reference_hash=hash_json({name: getattr(policy, name).tolist() for name in names}),
        conditioned_reference_hash=hash_json(
            {name: value.tolist() for name, value in replacements.items()}
        ),
        frames=frames,
        source_frame="ankle_pitch",
        proposed_frame="ankle_roll",
        source_fk_model_hash=model_hash,
        metrics=metrics,
        original_complete_joint_motion_unchanged=True,
        other_references_unchanged=True,
        donor_fk_data_unchanged=True,
        physical_state_modified=False,
        neural_weights_modified=False,
        correct_training_frame_not_proven=True,
        hardware_authorized=False,
    )
    for name, value in replacements.items():
        setattr(policy, name, value)
    return receipt
