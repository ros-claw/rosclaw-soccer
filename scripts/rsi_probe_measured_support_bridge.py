"""Bounded static support-anchor study for two consumed moving-entry states."""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--actual-review", type=Path, required=True)
    p.add_argument("--actual-review-hash", required=True)
    p.add_argument("--terminal-anchor-review", type=Path)
    p.add_argument("--terminal-anchor-review-hash")
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete = a.source / "complete.json"
    if (
        a.output.exists()
        or complete.is_symlink()
        or a.actual_review.is_symlink()
        or digest(complete) != a.source_hash
        or digest(a.actual_review) != a.actual_review_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused output, exact source/replay and isolated imports required")
    source, review = json.loads(complete.read_text()), json.loads(a.actual_review.read_text())
    terminal_anchor = a.terminal_anchor_review is not None
    if terminal_anchor != (a.terminal_anchor_review_hash is not None):
        raise ValueError("terminal anchor mode requires exact earlier independent review")
    if terminal_anchor:
        if (
            a.terminal_anchor_review.is_symlink()
            or digest(a.terminal_anchor_review) != a.terminal_anchor_review_hash
        ):
            raise ValueError("exact previous independent bridge review required")
        bridge_review = json.loads(a.terminal_anchor_review.read_text())
        if (
            bridge_review["status"] != "MEASURED_SUPPORT_BRIDGE_READBACK_NOT_PHYSICS"
            or bridge_review["source_hash"]
            != "sha256:5e6033c223cfe95c5c4bd6f07443199e6a8b76f492e85bd2ddaf59e2e0bdc078"
            or [r["all_frames_pass"] for r in bridge_review["records"]] != [False, False]
        ):
            raise ValueError("specific prior failed two-bridge review required")
    if (
        source["status"] != "WARM_LEFT_SETTLING_MEASURED_NOT_KICK"
        or source["controls"] != 157
        or review["status"] != "CFTRACK_PRIVATE_FUTURES_REPLAYED_ACTUALLY_NOT_PROMOTED"
        or review["source_hash"] != a.source_hash
        or review["native_steps"] != 6780
    ):
        raise ValueError("complete measured source and actual paired replay required")
    trace = a.source / source["artifact"]["path"]
    if (
        trace.is_symlink()
        or trace.parent != a.source
        or digest(trace) != source["artifact"]["hash"]
    ):
        raise ValueError("exact source physical trajectory required")
    pins = dict(source["input_file_hashes"])
    pins.update(
        {
            str(complete): a.source_hash,
            str(a.actual_review): a.actual_review_hash,
            str(trace): source["artifact"]["hash"],
        }
    )
    if terminal_anchor:
        pins[str(a.terminal_anchor_review)] = a.terminal_anchor_review_hash
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    external_files = [Path(x) for x in pins if x.endswith("/policy/omnicontact/OmniContact.py")]
    if len(scenes) != 1 or len(external_files) != 1:
        raise ValueError("one pinned body and source reference required")
    import mujoco
    import numpy as np

    from rosclaw_soccer.rsi.measured_joint_bridge import sampled_hermite_joint_bridge
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.reference_support_ik import fit_reference_support_pose
    from rosclaw_soccer.rsi.reference_support_ik_audit import audit_support_candidate
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    body_hash = "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    if compiled_model_hash(model) != body_hash:
        raise ValueError("exact native body required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    sys.path.insert(0, str(external_files[0].parents[2]))
    from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ

    with np.load(trace, allow_pickle=False) as f:
        raw = {k: f[k].copy() for k in ("qpos", "qvel")}
    if raw["qpos"].shape != (1570, model.nq) or raw["qvel"].shape != (1570, model.nv):
        raise ValueError("complete reviewed actual state trajectory required")
    floor = model.geom("floor").id
    ankles = [model.body(f"a/{side}_ankle_roll_link").id for side in ("left", "right")]
    feet = [
        [model.geom(f"a/{side}_foot{i}_collision").id for i in range(1, 8)]
        for side in ("left", "right")
    ]
    data = mujoco.MjData(model)

    def measure(root, joints):
        data.qpos[lo.bq : lo.bq + 7] = [100, 100, 0.11, 1, 0, 0, 0]
        data.qpos[lo.rq : lo.rq + 7], data.qpos[lo.qi] = root, joints
        mujoco.mj_forward(model, data)
        gaps = np.array(
            [
                min(float(mujoco.mj_geomDistance(model, data, floor, g, 1.0, None)) for g in foot)
                for foot in feet
            ]
        )
        return gaps, data.xpos[ankles, :2].copy()

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            source_hash=a.source_hash,
            actual_review_hash=a.actual_review_hash,
            input_file_hashes=pins,
            entry_controls=[57] if terminal_anchor else [57, 82],
            controls_per_bridge=20,
            interval_s=0.02,
            max_sampled_joint_speed_rad_s=6.0,
            max_ik_evaluations_per_frame=2000,
            total_max_ik_evaluations=38000 if terminal_anchor else 80000,
            terminal_anchor_adjustment=terminal_anchor,
            terminal_root_xy_adjustment_limit_m=0.08 if terminal_anchor else None,
            terminal_root_forward_not_negative=terminal_anchor,
            terminal_frame_raw_default_without_ik=terminal_anchor,
            provisional_support_gap_m=[-0.002, 0.003],
            support_anchor_limit_m=0.02,
            foot_penetration_limit_m=0.002,
            root_adjustment_limit_m=0.02,
            nonfoot_penetration_limit_m=0.002,
            max_root_step_m=0.006,
            max_joint_step_rad=0.12,
            max_support_ankle_step_m=0.006,
            initial_velocity_mismatch_rad_s=2.0,
            terminal_joint_error_rad=0.02,
            support_is_geometry_proxy_not_contact_truth=True,
            no_integration_or_training=True,
            hardware_authorized=False,
        ),
    )
    results, total_evaluations, unknown_failed_fits = [], 0, 0
    for entry in (57,) if terminal_anchor else (57, 82):
        qpos, qvel = raw["qpos"][entry * 10 - 1].copy(), raw["qvel"][entry * 10 - 1].copy()
        root0, q0, v0 = qpos[lo.rq : lo.rq + 7].copy(), qpos[lo.qi].copy(), qvel[lo.vi].copy()
        initial_gaps, anchor = measure(root0, q0)
        support = (initial_gaps >= -0.002) & (initial_gaps <= 0.003)
        if not support.any():
            write_once(
                a.output / f"entry{entry}-rejected.json",
                dict(entry=entry, reason="NO_PROVISIONAL_SUPPORT"),
            )
            results.append(
                dict(entry=entry, all_frames_pass=False, reason="NO_PROVISIONAL_SUPPORT")
            )
            continue
        bridge = sampled_hermite_joint_bridge(
            q0,
            v0,
            DEFAULT_JOINT_POS_MJ.astype(np.float64),
            model.jnt_range[lo.joints].astype(np.float64),
            controls=20,
            control_interval_s=0.02,
            maximum_sampled_speed_rad_s=6.0,
        )
        root_terminal = root0[:3] + qvel[lo.rv : lo.rv + 3] * 0.2
        terminal_shift = np.zeros(2)
        terminal_forward = None
        if terminal_anchor:
            if support.tolist() != [False, True]:
                raise ValueError("declared single-right-foot support proxy required")
            raw_terminal = root0.copy()
            raw_terminal[:3] = root_terminal
            _, terminal_xy = measure(raw_terminal, bridge["position"][-1])
            terminal_shift = np.mean((anchor - terminal_xy)[support], axis=0)
            if np.linalg.norm(terminal_shift) > 0.08:
                raise ValueError("terminal anchor root adjustment exceeds declared bound")
            root_terminal[:2] += terminal_shift
            goal_direction = np.array([5.0, 0.0]) - qpos[lo.bq : lo.bq + 2]
            goal_direction /= np.linalg.norm(goal_direction)
            terminal_forward = float((root_terminal[:2] - root0[:2]) @ goal_direction)
            if terminal_forward < 0:
                raise ValueError("terminal root would move backward toward goal axis")
        records, q_candidates, roots, ankle_positions = [], [], [], []
        for i, raw_joint in enumerate(bridge["position"]):
            t = i * 0.02
            raw_root = root0.copy()
            if terminal_anchor:
                u = t / 0.4
                raw_root[:3] = (
                    (2 * u**3 - 3 * u**2 + 1) * root0[:3]
                    + (u**3 - 2 * u**2 + u) * 0.4 * qvel[lo.rv : lo.rv + 3]
                    + (-2 * u**3 + 3 * u**2) * root_terminal
                )
            else:
                raw_root[:3] += qvel[lo.rv : lo.rv + 3] * (t - t * t / 0.8)
            fit = None
            error = None
            try:
                if i == 0 or (terminal_anchor and i == 20):
                    candidate_root, candidate_q = raw_root.copy(), raw_joint.copy()
                else:
                    fit = fit_reference_support_pose(
                        model,
                        raw_root,
                        raw_joint,
                        support,
                        expected_model_hash=body_hash,
                        maximum_evaluations=2000,
                        max_ankle_xy_change_m=0.02,
                        ankle_target_xy=anchor,
                    )
                    total_evaluations += fit["solver_evaluations"]
                    candidate_root, candidate_q = fit["root_pose_wxyz"], fit["joints_hg29"]
                audit = audit_support_candidate(
                    model,
                    raw_root,
                    raw_joint,
                    candidate_root,
                    candidate_q,
                    support,
                    expected_model_hash=body_hash,
                )
                gaps, xy = measure(candidate_root, candidate_q)
                anchor_error = float(np.max(np.linalg.norm((xy - anchor)[support], axis=1)))
                good = bool(
                    (
                        fit is None
                        or (
                            fit["reference_geometry_feasible"]
                            and not fit["joint_trust_boundary_active"]
                        )
                    )
                    and abs(audit["root_z_change_m"]) <= 0.02
                    and audit["candidate_minimum_nonfoot_floor_distance_m"] >= -0.002
                    and np.min(gaps) >= -0.002
                    and np.max(np.abs(gaps[support])) <= 0.01
                    and anchor_error <= 0.02
                )
            except (ValueError, RuntimeError) as exc:
                if i > 0 and not (terminal_anchor and i == 20) and fit is None:
                    unknown_failed_fits += 1
                candidate_root, candidate_q = raw_root.copy(), raw_joint.copy()
                gaps, xy = measure(candidate_root, candidate_q)
                audit, anchor_error, good = None, None, False
                error = dict(type=type(exc).__name__, message=str(exc))
            q_candidates.append(candidate_q.copy())
            roots.append(candidate_root.copy())
            ankle_positions.append(xy.copy())
            frame_path = a.output / f"entry{entry}-frame{i:02d}.npz"
            np.savez_compressed(
                frame_path,
                raw_root=raw_root,
                raw_joint=raw_joint,
                candidate_root=candidate_root,
                candidate_joint=candidate_q,
            )
            rec = dict(
                frame=i,
                valid=good,
                error=error,
                foot_gaps_m=gaps.tolist(),
                max_anchor_error_m=anchor_error,
                geometry=audit,
                solver_evaluations=(0 if i == 0 or (terminal_anchor and i == 20) else None)
                if fit is None
                else fit["solver_evaluations"],
                fit_outcome=(
                    None
                    if fit is None
                    else {
                        "solver_success": fit["solver_success"],
                        "reference_geometry_feasible": fit["reference_geometry_feasible"],
                        "joint_trust_boundary_active": fit["joint_trust_boundary_active"],
                        "anchor_error_m": fit["maximum_claimed_ankle_anchor_error_m"],
                    }
                ),
                artifact=dict(path=frame_path.name, hash=digest(frame_path)),
            )
            write_once(a.output / f"entry{entry}-frame{i:02d}.json", rec)
            records.append(rec)
        qs, r, xy = np.stack(q_candidates), np.stack(roots), np.stack(ankle_positions)
        root_step = float(np.max(np.abs(np.diff(r[:, 2]))))
        joint_step = float(np.max(np.abs(np.diff(qs, axis=0))))
        support_step = float(np.max(np.linalg.norm(np.diff(xy[:, support], axis=0), axis=2)))
        initial_error = float(np.max(np.abs((qs[1] - qs[0]) / 0.02 - v0)))
        terminal_error = float(np.max(np.abs(qs[-1] - DEFAULT_JOINT_POS_MJ)))
        passed = bool(
            all(row["valid"] for row in records)
            and root_step <= 0.006
            and joint_step <= 0.12
            and support_step <= 0.006
            and initial_error <= 2.0
            and terminal_error <= 0.02
        )
        path = a.output / f"entry{entry}.npz"
        np.savez_compressed(
            path,
            raw_joint=bridge["position"],
            candidate_joint=qs,
            candidate_root=r,
            candidate_ankle_xy=xy,
            support=support,
            ankle_anchor_xy=anchor,
        )
        case = dict(
            entry=entry,
            source_initial_foot_gaps_m=initial_gaps.tolist(),
            provisional_support=support.tolist(),
            valid_frames=sum(row["valid"] for row in records),
            max_root_step_m=root_step,
            max_joint_step_rad=joint_step,
            max_support_ankle_step_m=support_step,
            initial_velocity_mismatch_rad_s=initial_error,
            terminal_joint_error_rad=terminal_error,
            all_frames_pass=passed,
            terminal_anchor_adjustment=terminal_anchor,
            terminal_root_xy_shift_m=terminal_shift.tolist(),
            terminal_forward_displacement_m=terminal_forward,
            artifact=dict(path=path.name, hash=digest(path)),
        )
        write_once(a.output / f"entry{entry}.json", case)
        results.append(case)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "MEASURED_TERMINAL_ANCHOR_BRIDGE_STATIC_STUDY_NOT_PHYSICS"
                if terminal_anchor
                else "MEASURED_SUPPORT_BRIDGE_STATIC_STUDY_NOT_PHYSICS"
            ),
            source_head=a.expected_head,
            source_hash=a.source_hash,
            actual_review_hash=a.actual_review_hash,
            input_file_hashes=pins,
            records=results,
            known_solver_evaluations=total_evaluations,
            solver_calls_with_unknown_evaluation_count=unknown_failed_fits,
            inferred_support_not_contact_truth=True,
            neural_calls=0,
            native_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
