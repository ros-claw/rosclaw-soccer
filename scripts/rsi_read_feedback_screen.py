"""Read every full-feedback screen branch, including vetoes; no new physics."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.physics.g1_feedback_forecast import G1FeedbackForecast
from rosclaw_soccer.rsi.feedback_impact_score import feedback_impact_score
from rosclaw_soccer.rsi.impact_torque_screen import CAPS
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_feedback_screen_inputs import best_proxy_index, require_scope
from scripts.rsi_impact_torque_authority_screen import equal
from scripts.rsi_read_zero_feedback import verify_branch
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def verify_prediction(a, metadata, side):
    offset = a["offset_nm"]
    if (
        offset.shape != (29,)
        or offset.dtype != np.float64
        or not np.isfinite(offset).all()
        or np.any(abs(offset) > CAPS)
        or offset.tobytes() != np.asarray(metadata["offset_nm"], dtype=np.float64).tobytes()
        or metadata["is_zero"] is not bool(not np.any(offset))
    ):
        raise ValueError("same finite capped recorded action and metadata required")
    allowed = np.zeros(29, dtype=bool)
    if side in (0, 1):
        allowed[6 * side : 6 * side + 6] = True
    if np.any(offset[~allowed] != 0):
        raise ValueError("only the source current causal swing leg may receive an offset")
    for key, shape in (
        ("canonical_qpos_after_substep", (20, 43)),
        ("canonical_qvel_after_substep", (20, 41)),
        ("requested_pd_torque_nm", (20, 29)),
        ("applied_control_nm", (20, 29)),
        ("actual_actuator_force_nm", (20, 29)),
        ("next_navigation_target_rad", (29,)),
        ("next_taskspace_increment_rad", (29,)),
        ("next_motor_delta_rad", (12,)),
        ("next_joint_target_rad", (29,)),
        ("next_canonical_qpos", (43,)),
        ("next_canonical_qvel", (41,)),
    ):
        value = a[key]
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("whole finite physical/next-target arrays required")
    end = a["end_integration_state"]
    if (
        end.ndim != 1
        or end.dtype != np.float64
        or not 41 <= len(end) <= 65536
        or not np.isfinite(end).all()
    ):
        raise ValueError("whole finite final integration state required")
    for name, width in (
        ("encoder_features", 640),
        ("latent_token", 64),
        ("decoder_input", 994),
        ("raw_action_isaac", 29),
        ("joint_target_mujoco_rad", 29),
    ):
        v = a["neural_next_" + name]
        if v.shape != (1, width) or v.dtype != np.float32 or not np.isfinite(v).all():
            raise ValueError("whole finite actual next neural captures required")
    requested = a["requested_pd_torque_nm"].copy()
    requested[:10] += offset
    expected = np.clip(requested, -np.asarray(G1_HARD_TORQUE_LIMITS), G1_HARD_TORQUE_LIMITS)
    equal(a["applied_control_nm"], expected, "first20ms offset then unmodified feedback PD")
    equal(a["actual_actuator_force_nm"], a["applied_control_nm"], "actual canonical motor force")
    equal(
        a["next_canonical_qpos"], a["canonical_qpos_after_substep"][9], "actual branch next body q"
    )
    equal(
        a["next_canonical_qvel"], a["canonical_qvel_after_substep"][9], "actual branch next body v"
    )
    target = a["next_navigation_target_rad"] + a["next_taskspace_increment_rad"]
    target[:12] += a["next_motor_delta_rad"]
    equal(a["next_joint_target_rad"], target, "whole next target decomposition")
    forecast = G1FeedbackForecast(
        metadata["donor_hash"],
        end,
        a,
        {k.removeprefix("next_"): v for k, v in a.items() if k.startswith("next_")},
    )
    score = feedback_impact_score(forecast, offset)
    if hash_json(score) != hash_json(metadata["score"]):
        raise ValueError("raw complete-window proxy and recorded score differ")
    return score


def require_totals(complete, branch_count, nonzero_count, improved):
    if not 12 <= branch_count <= 84:
        raise ValueError("bounded complete branch bank required")
    for key, expected in (
        ("recorded_attempted_steps", 18000),
        ("recorded_integrated_steps", 18000),
        ("forecast_attempted_steps", branch_count * 20),
        ("forecast_integrated_steps", branch_count * 20),
        ("combined_attempted_steps", 18000 + branch_count * 20),
        ("combined_integrated_steps", 18000 + branch_count * 20),
        ("planner_runs_attempted", 60),
        ("current_pairs_completed", 6),
        ("original_residual_decisions_completed", 1800),
        ("original_recurrent_updates_completed", 1620),
        ("branch_target_attempts", branch_count),
        ("branch_target_completed", branch_count),
        ("new_closed_loop_policy_trials", 0),
        ("new_optimizer_updates", 0),
        ("candidate_actions_optimized", nonzero_count),
        ("local_proxy_improved_source_cases", improved),
    ):
        if type(complete[key]) is not int or complete[key] != expected:
            raise ValueError("actual completed screen counts do not match all raw branches")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--complete-file-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
        or args.output.exists()
        or (args.root / "failure.json").exists()
        or digest(args.root / "complete.json") != args.complete_file_hash
    ):
        raise ValueError("clean frozen reader, exact complete bytes and unused output required")
    require_terminal(args.unit)
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }

    def pin(path, expected=None):
        path = Path(path)
        actual = digest(path)
        if (
            path.is_symlink()
            or (expected is not None and actual != expected)
            or (str(path) in pins and pins[str(path)] != actual)
        ):
            raise ValueError("original ordinary source/model/trajectory bytes changed")
        pins[str(path)] = actual

    def sealed(path):
        path = Path(path)
        pin(path)
        d = json.loads(path.read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole original artifact seal differs")
        return d

    def arrays(w):
        pin(w["path"], w["sha256"])
        with np.load(w["path"], allow_pickle=False) as a:
            return dict(a)

    complete, declaration = (sealed(args.root / n) for n in ("complete.json", "declaration.json"))
    cfg = declaration["config"]
    require_scope(cfg)
    if (
        complete["schema"] != "soccer.rsi.feedback_impact_screen_complete.v1"
        or declaration["schema"] != "soccer.rsi.feedback_impact_screen_declaration.v1"
        or complete["declaration_hash"] != declaration["report_hash"]
        or Path(cfg["output_root"]) != args.root
        or len(complete["rows"]) != 6
        or any(
            complete[k] is not True
            for k in (
                "all_six_original_cases_retained",
                "all_zero_baselines_and_repeats_match_original",
            )
        )
        or any(
            complete[k] is not False
            for k in (
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
                "full_m0_gain_proven",
            )
        )
    ):
        raise ValueError("complete non-promoting original six-case screen required")
    for name, h in declaration["input_file_hashes"].items():
        pin(name, h)
    pin(cfg["zero_calibration_readback"], cfg["zero_calibration_readback_file_hash"])
    zero = sealed(cfg["zero_calibration_readback"])
    if (
        zero["report_hash"] != cfg["zero_calibration_readback_hash"]
        or zero["report_hash"] != declaration["zero_feedback_readback_hash"]
    ):
        raise ValueError("same previously qualified whole-zero predecessor required")
    stage = sealed(cfg["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    native_declaration = sealed(stage["effective_cohort_spec"]["producer_declaration"])
    dense = sealed(stage["effective_contact_spec"]["replay_complete"])
    if (
        native["report_hash"] != declaration["native_complete_hash"]
        or len(native["results"]) != 6
        or len(dense["rows"]) != 6
    ):
        raise ValueError("same original six-case native/dense chain required")
    rows, total, nonzero, improved = [], 0, 0, 0
    for i, (row, parent, contact) in enumerate(
        zip(complete["rows"], native["results"], dense["rows"], strict=True)
    ):
        if (
            sealed(args.root / f"case-{i}-complete.json") != row
            or row["index"] != i
            or row["source_receipt_hash"] != parent["receipt_hash"]
            or row["source_receipt_hash"] != contact["source_receipt_hash"]
            or row["original_outcome"] != parent["outcome"]
            or row["job"] != native_declaration["scope"]["jobs"][i]
            or any(
                row[k] is not True
                for k in (
                    "all_original300_residual_ik_and_dynamics_frames_matched",
                    "both_zero_feedback40ms_branches_match_original",
                    "baseline_and_repeated_zero_bits_equal",
                    "zero_next_target_and_neural_capture_matched",
                    "original_controller_unchanged",
                )
            )
            or any(
                row[k] is not False
                for k in (
                    "new_closed_loop_policy_trial",
                    "teacher_qualified",
                    "hardware_authorized",
                    "promotion_authorized",
                    "full_m0_gain_proven",
                    "nonzero_branch_independent_physics_replayed",
                )
            )
        ):
            raise ValueError("every original ordered source including failures required")
        frame = row["retrospective_frame"]
        if (
            type(frame) is not int
            or frame != parent["outcome"]["first_foot_contact_frame"]
            or not 30 <= frame < 298
            or (frame + 1) % 20 == 0
        ):
            raise ValueError("unchanged original diagnostic frame required")
        trace, d = arrays(parent["witnesses"]["trace"]), arrays(contact["arrays"])
        heading = trace["foot_heading_state"][frame, 0]
        side = int(heading[1])
        eligible = bool(side in (0, 1) and heading[2] == 1)
        bank = row["predictions"]
        if (
            row["screen_eligible"] is not eligible
            or not 2 <= len(bank) <= 14
            or (not eligible and len(bank) != 2)
            or row["candidate_actions_including_baseline"] != len(bank) - 1
        ):
            raise ValueError("complete causal eligible/ineligible bank required")
        raw = []
        for j, w in enumerate(bank):
            if (
                w["path"] != str(args.root / f"case-{i}-branch-{j}.npz")
                or w["donor_hash"] != bank[0]["donor_hash"]
            ):
                raise ValueError("same donor and every ordered raw branch required")
            a = arrays(w)
            verify_prediction(a, w, side if eligible else -1)
            raw.append(a)
        if (
            bank[0]["name"] != "baseline"
            or bank[-1]["name"] != "repeat_zero"
            or bank[0]["is_zero"] is not True
            or bank[-1]["is_zero"] is not True
            or len({w["name"] for w in bank}) != len(bank)
        ):
            raise ValueError("first zero baseline and last repeated zero required")
        for a in (raw[0], raw[-1]):
            verify_branch(a, trace, d, frame)
        if set(raw[0]) != set(raw[-1]) or any(
            raw[0][k].dtype != raw[-1][k].dtype
            or raw[0][k].shape != raw[-1][k].shape
            or raw[0][k].tobytes() != raw[-1][k].tobytes()
            for k in raw[0]
        ):
            raise ValueError("all zero repeated branch bytes differ")
        if len({a["offset_nm"].tobytes() for a in raw[:-1]}) != len(raw) - 1:
            raise ValueError("candidate offsets must be deduplicated except final zero repeat")
        best = best_proxy_index(bank)
        base_cost = bank[0]["score"]["proxy_cost"]
        cost = None if best is None else bank[best]["score"]["proxy_cost"]
        gain = base_cost is not None and cost is not None and base_cost - cost > 1e-6
        if (
            row["selected_proxy_index"] != best
            or row["local_proxy_improved_over_baseline"] is not gain
        ):
            raise ValueError("whole stable ranking including baseline and vetoes differs")
        total += len(bank)
        nonzero += len(bank) - 2
        improved += int(gain)
        rows.append(
            dict(
                index=i,
                job=row["job"],
                original_outcome=row["original_outcome"],
                source_receipt_hash=row["source_receipt_hash"],
                all_raw_branch_scores_offsets_and_feedback_targets_checked=True,
                zero_branches_match_original_dense_and_neural_bits=True,
                selected_proxy_index=best,
                selected_name=None if best is None else bank[best]["name"],
                baseline_proxy_cost=base_cost,
                selected_proxy_cost=cost,
                local_proxy_improved=gain,
                rejected_candidates=sum(
                    w["score"]["rejection_reason"] is not None for w in bank[:-1]
                ),
            )
        )
    require_totals(complete, total, nonzero, improved)
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError("source or evidence changed during readback")
    report = dict(
        schema="soccer.rsi.feedback_impact_screen_readback.v1",
        source_head=args.source_head,
        complete_hash=complete["report_hash"],
        rows=rows,
        input_file_hashes=pins,
        all_original_six_including_failures_retained=True,
        raw_branch_count=total,
        nonzero_candidates=nonzero,
        local_proxy_improved_source_cases=improved,
        score_formula_shared_with_producer=True,
        candidate_jacobian_bank_independently_regenerated=False,
        nonzero_branch_independent_physics_replayed=False,
        prospective_causal_trigger_qualified=False,
        full_m0_gain_proven=False,
        new_physical_steps=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(args.output, report)
    print("WHOLE_FEEDBACK_SCREEN_READBACK", report["report_hash"], flush=True)


if __name__ == "__main__":
    main()
