"""Reintegrate ALL14 closed release bodies; no NN, new context or teacher claim.

Uses the actual postcontact namespace, independently remaps current-pose
Jacobian actions, relabels raw M0, then integrates recorded targets from the
declared initial state. Does not call the causal producer/controller.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_parity import original_arrays
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_postcontact_release_inputs import require_scope
from scripts.rsi_postcontact_release_trace import check_release_trace
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def check_complete(c, d, root):
    require_scope(d["config"])
    if (
        c["schema"] != "soccer.rsi.postcontact_release_authority_complete.v1"
        or d["schema"] != "soccer.rsi.postcontact_release_authority_declaration.v1"
        or c["declaration_hash"] != d["report_hash"]
        or Path(d["config"]["output_root"]) != root
        or len(c["rows"]) != 14
        or any(
            type(c.get(k)) is not int or c[k] != v
            for k, v in dict(
                new_native_steps=42000,
                executed_control_foundation_pairs=4200,
                forecast_steps=0,
                new_optimizer_updates=0,
                physical_contexts=1,
            ).items()
        )
        or any(
            c.get(k) is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or c["both_original_zero_controls_matched"] is not True
        or [r["action_name"] for r in c["rows"]] != d["config"]["action_names"]
        or [r["index"] for r in c["rows"]] != list(range(14))
    ):
        raise ValueError("complete ordered single-context fourteen-body diagnostic required")


def check_row(row, trace, original, *, policy_hash, receipt_hash):
    extra = {k for k in trace if k.startswith("postcontact_release_")}
    if extra != {
        "postcontact_release_requested_offset_nm",
        "postcontact_release_applied_offset_nm",
        "postcontact_release_forecast_steps",
    }:
        raise ValueError("exact three actual release fields required")
    original_arrays({k: v for k, v in trace.items() if k not in extra})
    r = row["causal_review"]
    if (
        row["original_policy_hash"] != policy_hash
        or row["source_receipt_hash"] != receipt_hash
        or row["original_prefix_and_baseline_neural_bits_matched"] is not True
        or r["native_integrated_steps"] != 3000
        or r["forecast_integrated_steps"] != 0
        or any(
            row.get(k) is not False
            for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
        )
        or type(row["new_optimizer_updates"]) is not int
        or row["new_optimizer_updates"] != 0
    ):
        raise ValueError("same complete original motor/receipt and no learning required")
    control = check_release_trace(trace, r, action_name=row["action_name"])
    if control != row["control_review"]:
        raise ValueError("raw causal/action attribution differs from producer")
    require_prefix(trace, original, control["first_physically_applied_correction_frame"])
    outcome = label_first_touch(trace).to_dict()
    if outcome != row["outcome"] or (
        row["action_name"] == "baseline" and outcome != label_first_touch(original).to_dict()
    ):
        raise ValueError("whole unchanged raw M0 outcome required")
    return control, outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--complete-file-hash", required=True)
    parser.add_argument("--output", required=True, type=Path)
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
        raise ValueError("closed complete bytes, unused output and clean frozen reader required")
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
            raise ValueError("same ordinary immutable input/source bytes required")
        pins[str(p)] = h

    def sealed(name):
        pin(name)
        doc = json.loads(Path(name).read_text())
        if doc["report_hash"] != hash_json({k: v for k, v in doc.items() if k != "report_hash"}):
            raise ValueError("whole logical seal required")
        return doc

    def arrays(w):
        pin(w["path"], w["sha256"])
        with np.load(w["path"], allow_pickle=False) as a:
            return dict(a)

    c, d = sealed(args.root / "complete.json"), sealed(args.root / "declaration.json")
    check_complete(c, d, args.root)
    for name, h in d["input_file_hashes"].items():
        pin(name, h)
    cfg = d["config"]
    stage = sealed(cfg["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    parent = native["results"][cfg["original_binding_index"]]
    if parent["receipt_hash"] != d["original_source_receipt_hash"]:
        raise ValueError("same original source receipt required")
    world, cw = parent["witnesses"]["world"], parent["witnesses"]["commitment"]
    pin(world["path"], world["sha256"])
    pin(cw["path"], cw["sha256"])
    commitment = json.loads(Path(cw["path"]).read_text())
    # Native commitment is witness-byte-bound, not a fabricated report_hash.
    pair = sealed(Path(cfg["local_root"]) / f"pair-{cfg['local_context_index']}-complete.json")
    if pair["report_hash"] != d["closed_local_pair_hash"] or pair["job"]["course"] != d["course"]:
        raise ValueError("same declared consumed local course required")
    original = arrays(pair["arms"][0]["arrays"])
    if label_first_touch(original).to_dict() != d["original_outcome"]:
        raise ValueError("original raw source outcome required")
    traces = []
    for i, row in enumerate(c["rows"]):
        if row != sealed(args.root / f"trial-{i}-complete.json") or row["arrays"]["path"] != str(
            args.root / f"trial-{i}-trace.npz"
        ):
            raise ValueError("all fourteen ordered complete trial bytes required")
        trace = arrays(row["arrays"])
        check_row(
            row,
            trace,
            original,
            policy_hash=commitment["motor_policy_hash"],
            receipt_hash=parent["receipt_hash"],
        )
        traces.append(trace)
    successes = [
        r["action_name"] for r in c["rows"] if r["outcome"]["task_pass"] and r["outcome"]["safe"]
    ]
    if (
        c["successful_action_names"] != successes
        or c["consumed_context_rescued"] is not bool(successes)
        or c["unsafe_trials"] != sum(not r["outcome"]["safe"] for r in c["rows"])
    ):
        raise ValueError("whole raw aggregate outcomes required")
    if args.preflight:
        print("POSTCONTACT14_RAW_ADMITTED_NO_NEW_PHYSICS", flush=True)
        return
    args.output.mkdir()
    declaration = dict(
        schema="soccer.rsi.recorded_postcontact14_declaration.v1",
        source_head=args.source_head,
        input_file_hashes=pins,
        producer_complete_hash=c["report_hash"],
        physical_step_budget=42000,
        maximum_runtime_seconds=1800,
        physical_contexts=1,
        recorded_targets_consumed=True,
        new_neural_inferences=0,
        neural_policy_qualification_claimed=False,
    )
    declaration["report_hash"] = hash_json(declaration)
    write_once(args.output / "declaration.json", declaration)
    rows, audit, index = [], None, None
    deadline = time.monotonic() + 1800
    try:
        import mujoco

        from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
        from rosclaw_soccer.physics.recorded_early_episode import RecordedEarlyEpisode
        from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
        from rosclaw_soccer.rsi.impact_torque_screen import impact_torque_candidates

        for index, (row, trace) in enumerate(zip(c["rows"], traces, strict=True)):
            audit = None
            if time.monotonic() >= deadline:
                raise RuntimeError("fixed postcontact14 independent physics deadline exhausted")
            model = mujoco.MjModel.from_binary_path(world["path"])
            ph = commitment["physics"]
            audit = RecordedEarlyEpisode(
                model,
                world_hash=world["sha256"],
                kp=np.asarray(ph["canonical_pd_kp"], dtype=np.float64),
                kd=np.asarray(ph["canonical_pd_kd"], dtype=np.float64),
                course=tuple(d["course"]),
                defaults=np.asarray(G1SonicRunupController.default_angles, dtype=np.float64),
            )
            trigger = row["control_review"]["observed_trigger_frame"]
            if trigger is not None:
                data = mujoco.MjData(model)
                q, v = trace["canonical_qpos"][trigger, 0], trace["canonical_qvel"][trigger, 0]
                data.qpos[:7], data.qpos[audit.qi], data.qpos[audit.bq : audit.bq + 7] = (
                    q[:7],
                    q[7:36],
                    q[36:],
                )
                data.qvel[:6], data.qvel[audit.vi], data.qvel[audit.bv : audit.bv + 6] = (
                    v[:6],
                    v[6:35],
                    v[35:],
                )
                sampler = G1JointTaskKinematicSampler(model, expected_world_hash=world["sha256"])
                bank = dict(
                    impact_torque_candidates(
                        sampler.sample(data), row["control_review"]["measured_contact_side"]
                    )
                )
                available = row["action_name"] in bank
                offset = bank.get(row["action_name"], np.zeros(29))
                if available is not row["causal_review"]["postcontact_release"][
                    "action_available"
                ] or not np.array_equal(
                    offset, trace["postcontact_release_requested_offset_nm"][trigger, 0]
                ):
                    raise ValueError("actual current-pose named Jacobian action differs")
            review = audit.verify_postcontact_release(trace)
            result = dict(
                index=index,
                action_name=row["action_name"],
                source_trial_hash=row["report_hash"],
                current_pose_jacobian_action_rebuilt=True,
                physical_review=review,
                raw_outcome=label_first_touch(trace).to_dict(),
            )
            result["report_hash"] = hash_json(result)
            write_once(args.output / f"trial-{index}-complete.json", result)
            rows.append(result)
            print(
                "POSTCONTACT14_INDEPENDENT_PHYSICS_COMPLETE", index, row["action_name"], flush=True
            )
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("source/input changed during independent physics")
        result = dict(
            schema="soccer.rsi.recorded_postcontact14_complete.v1",
            declaration_hash=declaration["report_hash"],
            rows=rows,
            actual_physical_steps=sum(r["physical_review"]["actual_physical_steps"] for r in rows),
            new_neural_inferences=0,
            new_optimizer_updates=0,
            physical_contexts=1,
            teacher_qualified=False,
            training_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
            independent_neural_or_search_qualified=False,
        )
        if result["actual_physical_steps"] != 42000:
            raise ValueError("whole fourteen independent recorded physical bodies required")
        result["report_hash"] = hash_json(result)
        write_once(args.output / "complete.json", result)
    except BaseException as error:
        if audit is not None:
            with (args.output / "failure-state.npz").open("xb") as stream:
                np.savez_compressed(
                    stream, qpos=audit.data.qpos, qvel=audit.data.qvel, ctrl=audit.data.ctrl
                )
        write_once(
            args.output / "failure.json",
            dict(
                current_trial=index,
                completed_trials=len(rows),
                current_attempted_steps=0 if audit is None else audit.attempted_steps,
                current_integrated_steps=0 if audit is None else audit.integrated_steps,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
