"""Review continuous common-PD velocity-to-SONIC native traces, SIM only."""

from types import SimpleNamespace
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.sonic_external_history import equivalent_sonic_action
from rosclaw_soccer.providers.g1.sonic_runup import ISAACLAB_TO_MUJOCO
from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


def review_switched(
    raw: dict[str, np.ndarray],
    neural: dict[str, np.ndarray],
    params: Any,
    tracker: Any,
    world: Any,
    *,
    handoff: int,
) -> dict[str, Any]:
    n = 300 - handoff
    physical_shapes = dict(
        canonical_qpos=(300, 43),
        canonical_qvel=(300, 41),
        joint_target_rad=(300, 29),
        pd_requested_torque_nm=(300, 10, 29),
        torque_nm=(300, 10, 29),
        actual_actuator_force_nm=(300, 10, 29),
        pelvis_z_per_substep_m=(300, 10),
        force_n=(300, 6),
        ball_position_after_step_m=(300, 3),
        target_body_canonical_qpos_after_substep=(300, 10, 43),
        target_body_canonical_qvel_after_substep=(300, 10, 41),
        ball_contact_force_by_body_n=(300, 10, world.nbody),
    )
    if (
        set(raw) != set(physical_shapes)
        or any(
            type(raw[k]) is not np.ndarray
            or raw[k].shape != shape
            or raw[k].dtype != np.float64
            or not np.isfinite(raw[k]).all()
            for k, shape in physical_shapes.items()
        )
        or np.any(raw["ball_contact_force_by_body_n"] < 0)
        or np.any(raw["force_n"] < 0)
    ):
        raise ValueError("all twelve typed native physical fields required")
    shapes = dict(
        encoder_input=(n, 1247),
        encoder_output=(n, 64),
        decoder_input=(n, 994),
        decoder_output=(n, 29),
    )
    if not 30 <= handoff <= 90 or set(neural) != set(shapes):
        raise ValueError("whole explicit two-codec continuous motor trace required")
    for k, shape in shapes.items():
        if (
            neural[k].shape != shape
            or neural[k].dtype != np.float32
            or not np.isfinite(neural[k]).all()
        ):
            raise ValueError("complete finite neural inference record required: " + k)
    q, v = raw["canonical_qpos"], raw["canonical_qvel"]
    nq, nv = (
        raw["target_body_canonical_qpos_after_substep"],
        raw["target_body_canonical_qvel_after_substep"],
    )
    if (
        q.shape != (300, 43)
        or v.shape != (300, 41)
        or nq.shape != (300, 10, 43)
        or nv.shape != (300, 10, 41)
        or any(a.dtype != np.float64 or not np.isfinite(a).all() for a in raw.values())
        or not np.array_equal(q[1:], nq[:-1, -1])
        or not np.array_equal(v[1:], nv[:-1, -1])
    ):
        raise ValueError("complete continuous native canonical trajectory required")
    caps = np.array(G1_HARD_TORQUE_LIMITS)
    kp, kd = np.array(params.stiffness), np.array(params.damping)
    for f in range(300):
        target = raw["joint_target_rad"][f]
        before_q = np.concatenate((q[f, None], nq[f, :-1]))
        before_v = np.concatenate((v[f, None], nv[f, :-1]))
        requested = (target - before_q[:, 7:36]) * kp - before_v[:, 6:35] * kd
        if (
            not np.array_equal(requested, raw["pd_requested_torque_nm"][f])
            or not np.array_equal(np.clip(requested, -caps, caps), raw["torque_nm"][f])
            or np.any(abs(raw["actual_actuator_force_nm"][f]) > caps + 1e-8)
        ):
            raise ValueError("unchanged common fixed PD/torque contract required")
        if f < handoff:
            continue
        local = f - handoff
        expected_encoder = tracker._encoder_observation(
            SimpleNamespace(qpos=q[f], qvel=v[f]), local
        )[0]
        if not np.array_equal(expected_encoder, neural["encoder_input"][local]):
            raise ValueError("actual current-state/source encoder input differs")
        target_expected = (
            tracker.default_angles
            + neural["decoder_output"][local][ISAACLAB_TO_MUJOCO] * tracker._action_scale
        )
        if not np.array_equal(target, target_expected):
            raise ValueError("SONIC ISAAC output cannot be replaced by foreign raw action")
        expected_decoder = np.zeros(994, dtype=np.float32)
        expected_decoder[:64] = neural["encoder_output"][local]
        history = []
        for past in range(f - 10, f):
            action = (
                equivalent_sonic_action(
                    raw["joint_target_rad"][past], tracker.default_angles, tracker._action_scale
                )[0]
                if past < handoff
                else neural["decoder_output"][past - handoff]
            )
            history.append(
                tracker._history_entry(
                    SimpleNamespace(qpos=nq[past, -1], qvel=nv[past, -1]), action
                )
            )
        offsets = (64, 94, 384, 674, 964, 994)
        for field, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            expected_decoder[start:end] = np.asarray(
                [row[field] for row in history], dtype=np.float32
            ).reshape(-1)
        if not np.array_equal(expected_decoder, neural["decoder_input"][local]):
            raise ValueError("every decoder history must derive from actual past body and target")
    score = score_full_entry(world, {k: a[20:] for k, a in raw.items()})
    root = world.body_rootid[world.body("pelvis").id]
    feet = [
        world.body(side + suffix).id
        for side in ("left", "right")
        for suffix in ("_ankle_roll_link", "_ankle_pitch_link")
    ]
    nonfeet = np.flatnonzero((world.body_rootid == root) & ~np.isin(np.arange(world.nbody), feet))
    full_safe = bool(
        nq[:, :, 2].min() >= 0.65
        and (1 - 2 * (nq[:, :, 4] ** 2 + nq[:, :, 5] ** 2)).min() >= 0.9
        and raw["ball_contact_force_by_body_n"][:, :, nonfeet].max() <= 1
    )
    score["whole300_physical_safe"] = full_safe
    score["qualified"] = bool(score["qualified"] and full_safe)
    if not score["qualified"]:
        score["cost"] = None
    return dict(
        score=score,
        handoff_frame=handoff,
        all_common_pd_and_policy_inputs_reconstructed=True,
        body_reset_performed=False,
        gain_switch_performed=False,
        common_fixed_pd_is_velocity_deployment_not_sonic_deployment=True,
        old_sonic_motor_certificate_inherited=False,
        new_optimizer_steps=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
