"""Verify every public40 arm and raw MPC bank before recorded-body replay.

No Foundation reexecution, teacher admission, actor update or private Fresh.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_parity import original_arrays, require_heading_parity
from rosclaw_soccer.rsi.postcontact_learning_observation import current_postcontact_features
from rosclaw_soccer.rsi.postcontact_robustness_plan import declare_robustness40
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_public_four_arm import summarize_pairs
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_postcontact_exit_decisions import read_exit_decisions
from scripts.rsi_postcontact_exit_trace import check_exit_trace
from scripts.rsi_public_four_arm_inputs import require_scope
from scripts.rsi_read_local_impact_robustness import verify_original_control_torques
from scripts.rsi_read_postcontact_full_bank import current_named_bank
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def check_complete(c, d, root):
    require_scope(d["config"])
    plan = declare_robustness40(mean_model_hash=d["config"]["mean_model_hash"])
    if (
        d["schema"] != "soccer.rsi.public_four_arm40_declaration.v1"
        or c["schema"] != "soccer.rsi.public_four_arm40_complete.v1"
        or c["declaration_hash"] != d["report_hash"]
        or d["plan"] != plan
        or len(c["rows"]) != 160
        or len(c["pairs"]) != 80
        or Path(d["config"]["output_root"]) != root
        or any(
            type(c.get(k)) is not int or c[k] != v
            for k, v in dict(
                unique_initial_states=40,
                generator_seed_clusters=10,
                new_native_rollouts=160,
                new_native_steps=480000,
                executed_control_foundation_pairs=48000,
                new_optimizer_updates=0,
            ).items()
        )
        or type(c.get("forecast_steps")) is not int
        or not 0 <= c["forecast_steps"] <= 44800
        or any(
            c.get(k) is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "promotion_authorized",
                "hardware_authorized",
                "private_fresh_accessed",
            )
        )
    ):
        raise ValueError("whole predeclared public40 four-arm160 result required")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--complete-file-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if (
        args.output.exists()
        or (args.root / "failure.json").exists()
        or digest(args.root / "complete.json") != args.complete_file_hash
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != args.source_head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("closed exact producer, frozen reviewer and unused output required")
    require_terminal(args.unit)
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }

    def pin(name, expected=None):
        p = Path(name)
        h = digest(p)
        if (
            p.is_symlink()
            or (expected is not None and h != expected)
            or (str(p) in pins and pins[str(p)] != h)
        ):
            raise ValueError("whole immutable public40 inputs required")
        pins[str(p)] = h

    def sealed(name):
        pin(name)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole public40 logical seal required")
        return d

    def arrays(w):
        pin(w["path"], w["sha256"])
        with np.load(w["path"], allow_pickle=False) as archive:
            return dict(archive)

    c, d = sealed(args.root / "complete.json"), sealed(args.root / "declaration.json")
    plan = check_complete(c, d, args.root)
    for n, h in d["input_file_hashes"].items():
        pin(n, h)
    stage = sealed(d["config"]["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    parent = native["results"][0]
    w = parent["witnesses"]
    pin(w["world"]["path"], w["world"]["sha256"])
    pin(w["commitment"]["path"], w["commitment"]["sha256"])
    commitment = json.loads(Path(w["commitment"]["path"]).read_text())
    rows, pairs, forecasts, hashes = [], [], 0, set()
    for job in plan["records"]:
        originals, policy_hash = {}, None
        ci = job["index"]
        for ai, arm in enumerate(job["arms"]):
            row = c["rows"][ci * 4 + ai]
            expected = dict(
                course_index=ci,
                arm_index=ai,
                arm=arm,
                course=job["course"],
                sampling_seed=job["sampling_seed"],
            )
            if (
                row != sealed(args.root / f"course-{ci}-arm-{ai}-complete.json")
                or any(row.get(k) != v for k, v in expected.items())
                or row["arrays"]["path"] != str(args.root / f"course-{ci}-arm-{ai}-trace.npz")
                or row["source_anchor_receipt_hash"] != parent["receipt_hash"]
                or row["new_optimizer_updates"] != 0
                or any(
                    row.get(k) is not False
                    for k in (
                        "teacher_qualified",
                        "training_authorized",
                        "hardware_authorized",
                        "promotion_authorized",
                    )
                )
            ):
                raise ValueError("all ordered declared public40 arm identities required")
            if ai == 0:
                policy_hash = row["original_policy_hash"]
                if policy_hash in hashes:
                    raise ValueError("new per-course sampled motor identity required")
                hashes.add(policy_hash)
            elif row["original_policy_hash"] != policy_hash:
                raise ValueError("same sampled policy for every matched arm required")
            trace = arrays(row["arrays"])
            teacher, heading = arm["postcontact_exit_enabled"], arm["heading_enabled"]
            original_arrays(
                {k: v for k, v in trace.items() if not k.startswith("postcontact_exit_")}
            )
            state = trace.get("foot_heading_state")
            if (
                not isinstance(state, np.ndarray)
                or state.shape != (300, 1, 10)
                or state.dtype != np.float64
                or not np.all(state[:, 0, 3] == int(heading))
            ):
                raise ValueError("actual complete declared OFF/ON heading capture required")
            outcome = label_first_touch(trace).to_dict()
            review = row["causal_review"]
            if (
                outcome != row["outcome"]
                or review["native_integrated_steps"] != 3000
                or review["native_attempted_steps"] != 3000
                or type(review["forecast_integrated_steps"]) is not int
                or review["forecast_integrated_steps"] != review["forecast_attempted_steps"]
            ):
                raise ValueError("whole measured M0 and actual body counters required")
            alignment, features = [], None
            if not teacher:
                verify_original_control_torques(trace)
                if row["control_review"] is not None or review["forecast_integrated_steps"] != 0:
                    raise ValueError("original arm cannot contain forecast/control intervention")
                if heading:
                    require_heading_parity(trace, originals[False][0], frames=30)
                originals[heading] = (trace, row)
            else:
                if review.get("postcontact_exit_owned_updates_reused") is not True:
                    raise ValueError("declared new owned-branch teacher path required")
                control = check_exit_trace(trace, review)
                if control != row["control_review"]:
                    raise ValueError("raw current-contact control differs")
                old, old_row = originals[heading]
                require_prefix(trace, old, control["first_physically_applied_correction_frame"])
                f = control["observed_trigger_frame"]
                bank = None
                if f is not None and trace["postcontact_exit_forecast_steps"][f, 0, 0] > 0:
                    bank = (
                        *current_named_bank(
                            w["world"],
                            trace["canonical_qpos"][f, 0],
                            trace["canonical_qvel"][f, 0],
                            control["measured_contact_side"],
                        ).items(),
                        ("repeat_zero", np.zeros(29)),
                    )
                    features = current_postcontact_features(
                        trace["canonical_qpos"][f, 0],
                        trace["canonical_qvel"][f, 0],
                        trace["joint_target_rad"][f, 0],
                        frame=f,
                        previous_contact_forces=trace["force_n"][f - 1, 0],
                    ).tolist()
                alignment = read_exit_decisions(
                    trace,
                    args.root / f"course-{ci}-arm-{ai}-preaction",
                    expected_bank=bank,
                    sealed=sealed,
                    arrays=arrays,
                )
                pairs.append(
                    dict(
                        course_index=ci,
                        heading_enabled=heading,
                        baseline_hash=old_row["report_hash"],
                        teacher_hash=row["report_hash"],
                        baseline_outcome=old_row["outcome"],
                        teacher_outcome=outcome,
                    )
                )
            forecasts += review["forecast_integrated_steps"]
            rows.append(
                dict(
                    course_index=ci,
                    arm_index=ai,
                    source_case_hash=row["report_hash"],
                    outcome=outcome,
                    selected_execution_alignment=alignment,
                    diagnostic_current_postcontact_features=features,
                    features_are_SIM_privileged_not_approved_actor_input=True,
                )
            )
            print("PUBLIC40_RAW_ARM_VERIFIED", ci, ai, flush=True)
    if (
        pairs != c["pairs"]
        or summarize_pairs(pairs) != c["summary"]
        or forecasts != c["forecast_steps"]
        or any(digest(n) != h for n, h in pins.items())
    ):
        raise ValueError("whole raw bank/summary/source agreement required")
    if args.preflight:
        print("PUBLIC40_WHOLE160_RAW_READY_NO_NEW_PHYSICS", flush=True)
        return
    args.output.mkdir()
    declaration = dict(
        schema="soccer.rsi.public_four_arm40_verification_declaration.v1",
        source_head=args.source_head,
        producer_complete_hash=c["report_hash"],
        input_file_hashes=pins,
        whole_raw_rows=rows,
        physical_step_budget=480000,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(args.output / "declaration.json", declaration)
    results, audit, index = [], None, None
    deadline = time.monotonic() + 7200
    try:
        import mujoco

        from rosclaw_soccer.physics.recorded_early_episode import RecordedEarlyEpisode
        from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController

        for index, row in enumerate(c["rows"]):
            audit = None
            if time.monotonic() >= deadline:
                raise RuntimeError("whole public160 recorded-body deadline exhausted")
            model = mujoco.MjModel.from_binary_path(w["world"]["path"])
            ph = commitment["physics"]
            audit = RecordedEarlyEpisode(
                model,
                world_hash=w["world"]["sha256"],
                kp=np.asarray(ph["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(ph["canonical_pd_kd"], dtype=np.float64),
                course=tuple(row["course"]),
                defaults=np.asarray(G1SonicRunupController.default_angles, dtype=np.float64),
            )
            t = arrays(row["arrays"])
            r = (
                audit.verify_postcontact_exit(t)
                if row["arm"]["postcontact_exit_enabled"]
                else audit.verify_unforced_control(t)
            )
            if r["actual_physical_steps"] != 3000 or any(
                not np.isfinite(e) or e > 1e-8 for e in r["maximum_absolute_errors"].values()
            ):
                raise ValueError("whole finite physical alignment required")
            proof = dict(index=index, source_case_hash=row["report_hash"], physical_review=r)
            proof["report_hash"] = hash_json(proof)
            write_once(args.output / f"body-{index}-complete.json", proof)
            results.append(proof)
            print("PUBLIC40_RECORDED_BODY_VERIFIED", index, flush=True)
        if any(digest(n) != h for n, h in pins.items()):
            raise ValueError("whole public160 source inputs changed during physics")
        result = dict(
            schema="soccer.rsi.public_four_arm40_verification_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=results,
            actual_physical_steps=480000,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            training_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
            independently_recomputed_neural_or_search_qualified=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(args.output / "complete.json", result)
    except BaseException as error:
        write_once(
            args.output / "failure.json",
            dict(
                current_body=index,
                completed_bodies=len(results),
                current_attempted_steps=0 if audit is None else audit.attempted_steps,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
