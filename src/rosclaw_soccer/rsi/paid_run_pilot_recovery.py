"""Reward-independent physical scoring and narrow charged-world recovery."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.paid_canonical_behavior import digest


def contact_quality(result: dict[str, Any]) -> bool:
    p = result["postcontact"]
    return bool(
        not result["body_failed"]
        and not result["forbidden"]
        and p["correct_foot_first"]
        and p["post_contact_quality_observed"]
        and p["goal_progress_1p2s_m"] >= 1
        and abs(p["goal_lateral_1p2s_m"]) <= 0.75
        and not p["other_foot_contact_in_window"]
        and p["maximum_pelvis_retreat_after_contact_m"] <= 0.05
    )


def stand_from_raw(raw: dict[str, np.ndarray]) -> dict[str, Any]:
    from rosclaw_soccer.rsi.paid_balance_recovery import body_feedback

    q, v = raw["qpos"][-1, -1], raw["qvel"][-1, -1]
    feature = body_feedback(q, v)
    healthy = not bool(raw["body_failed"].any())
    upright = float(1 - 2 * (q[4] ** 2 + q[5] ** 2))
    return dict(
        task="static_neural_stand",
        memory_mode="carry",
        controls=len(raw["qpos"]),
        native_steps=len(raw["qpos"]) * 10,
        neural_rows=len(raw["qpos"]),
        healthy=healthy,
        stability_observed=bool(
            healthy
            and q[2] >= 0.72
            and upright >= 0.98
            and np.linalg.norm(feature[4:6]) <= 0.2
            and np.linalg.norm(v[3:6]) <= 0.3
        ),
        terminal_feature=feature.tolist(),
        terminal_upright=upright,
        ball_settled=bool(q[38] <= 0.12 and abs(v[37]) <= 0.1),
        no_motion_padding=True,
        old_exam_result=False,
        optimizer_updates=0,
        hardware_authorized=False,
    )


def recover_first_standing_world(
    root: Path,
    expected: str,
    case: dict[str, Any],
    courses: list[dict[str, float]],
    arms: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, str], dict[str, int]]:
    """Only the observed reward-field fault after an entire first world.

    Refuse arbitrary partial simulation continuation or a completed campaign.
    Data are re-scored, never replayed here and never counted as new physics.
    """
    from rosclaw_soccer.rsi.native_contact_window import native_contact_window

    fp, dp = root / "failed.json", root / "declaration.json"
    if (
        fp.is_symlink()
        or dp.is_symlink()
        or digest(fp) != expected
        or (root / "complete.json").exists()
    ):
        raise ValueError("exact failed predecessor required")
    failure, decl = [json.loads(p.read_text()) for p in (fp, dp)]
    counts = dict(R2_applied=289, R2_shadow=0, R2_stand=100, teacher_IK=0, velocity=0)
    if (
        failure["error"] != "KeyError('terminal_reward')"
        or failure["counts"] != counts
        or failure["actual_native_steps"] != 3890
        or failure["automatic_retry"] is not False
        or decl["protocol"] != "paid.velocity_actual_shadow_then_reference80.v1"
        or decl["case"] != case
        or decl["courses"] != courses
        or decl["arms"] != arms
        or decl["prepare_velocity_controls"] != 50
        or decl["run_controls"] != 100
        or decl["shadow_reference_controls"] != 80
    ):
        raise ValueError("this recovery only admits the complete known first-world scorer fault")
    pins = dict(decl["input_file_hashes"])
    pins.update({str(fp): expected, str(dp): digest(dp)})
    folder = root / "course0-standing_current-repeat0"
    if folder.is_symlink() or not folder.resolve().is_relative_to(root.resolve()):
        raise ValueError("confined actual first-world folder required")
    blobs = {}
    for name in ("initial", "R2-applied", "stand", "physical"):
        p = folder / (name + ".npz")
        if p.is_symlink() or p.stat().st_size > 100000000:
            raise ValueError("bounded ordinary first-world data required")
        with np.load(p, allow_pickle=False) as z:
            blobs[name] = {k: z[k].copy() for k in z.files}
        pins[str(p)] = digest(p)
    for ref in failure["partial_artifacts"].values():
        path = Path(ref["path"])
        if path.is_symlink() or path.parent != root or digest(path) != ref["hash"]:
            raise ValueError("failed prefix/terminal commitments changed")
        pins[str(path)] = ref["hash"]
    raw, nn, stand = blobs["physical"], blobs["R2-applied"], blobs["stand"]
    prefix = Path(failure["partial_artifacts"]["physical_prefix"]["path"])
    with np.load(prefix, allow_pickle=False) as z:
        if any(not np.array_equal(z[k], raw[k]) for k in z.files):
            raise ValueError("physical prefix differs from failed receipt")
    if (
        raw["qpos"].shape != (389, 10, 43)
        or raw["qvel"].shape != (389, 10, 41)
        or not np.array_equal(
            raw["controller"], np.r_[np.ones(289, np.int32), np.full(100, 3, np.int32)]
        )
        or not np.array_equal(nn["reference_frame"], np.arange(289))
        or nn["h_before"][0].any()
        or nn["c_before"][0].any()
        or not np.array_equal(nn["h_before"][1:], nn["h_after"][:-1])
        or not np.array_equal(nn["c_before"][1:], nn["c_after"][:-1])
        or not np.array_equal(nn["proposal"], nn["applied_action"])
        or not np.array_equal(nn["observation"][1:, 125:154], nn["applied_action"][:-1])
        or not np.array_equal(nn["h_after"][-1], stand["NN_h_before"][0])
        or not np.array_equal(nn["c_after"][-1], stand["NN_c_before"][0])
        or not np.array_equal(nn["applied_action"][-1], stand["NN_obs"][0, 125:154])
        or not np.array_equal(raw["qpos"][-100:], stand["qpos"])
    ):
        raise ValueError("complete own-history, actual actions and standing continuation required")
    with np.load(failure["partial_artifacts"]["live_state"]["path"], allow_pickle=False) as z:
        if any(
            not np.array_equal(v, z[k.removeprefix("terminal_")])
            for k, v in stand.items()
            if k.startswith("terminal_")
        ):
            raise ValueError("last live state differs from completed stand")
    if abs(float(stand["terminal_time"]) - float(blobs["initial"]["time"]) - 7.78) > 1e-9:
        raise ValueError("all3890 charged native steps must share one clock")
    forces = raw["forces"].reshape(-1, 3)
    post = native_contact_window(
        force=forces[:, :2],
        canonical_qpos=raw["qpos"].reshape(-1, 43),
        canonical_qvel=raw["qvel"].reshape(-1, 41),
        goal=np.array(case["goal_world"]),
        intended_foot=0 if "left" in case["source"] else 1,
    )
    result = dict(
        body_failed=bool(raw["body_failed"].any()),
        forbidden=bool((forces[:, 2] > 1).any()),
        postcontact=post,
        nonfoot_impulse_Ns=float(forces[:, 2].sum() * 0.002),
    )
    ss = stand_from_raw(stand)
    quality = contact_quality(result)
    first = post["first_foot_contact_substep"]
    if first is not None and not isinstance(first, int):
        raise ValueError("integer measured contact index required")
    outcome = dict(
        course=0,
        arm="standing_current",
        repeat=0,
        quality=quality,
        healthy_success=bool(quality and ss["healthy"] and ss["stability_observed"]),
        result=result,
        stand=ss,
        first_contact_controller=int(raw["controller"][int(first) // 10])
        if first is not None
        else None,
        first_contact_before_kick_end=bool(first is not None and int(first) < 2890),
        total_native_steps=3890,
        actual_world_time_s=float(stand["terminal_time"]),
        R2_shadow_queries=0,
        R2_shadow_proposals_executed=False,
        official_exam_result=False,
        learned_player_claim=False,
        source_execution_head=decl["source_head"],
        recovered_without_new_physics=True,
        recovery_failed_hash=expected,
    )
    return outcome, blobs, pins, counts
