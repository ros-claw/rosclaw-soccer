"""Whole18 consumed postcontact transfer readback; no physics/NN/teacher admission."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.local_impact_perturbation_plan import declare_local_impact_perturbations
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_prospective_impact import require_prefix
from scripts.rsi_postcontact_exit_decisions import read_exit_decisions
from scripts.rsi_postcontact_exit_trace import check_exit_trace
from scripts.rsi_postcontact_release_trace import check_release_trace
from scripts.rsi_postcontact_release_transfer import paired_summary
from scripts.rsi_postcontact_release_transfer_inputs import require_scope
from scripts.rsi_read_local_impact_robustness import verify_original_control_torques
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def check_complete(c, d, root):
    require_scope(d["config"])
    adaptive = d["config"]["profile"] == "postcontact-exit-proxy-transfer18-v1"
    kind = "postcontact_exit_transfer" if adaptive else "postcontact_release_transfer"
    if (
        c["schema"] != f"soccer.rsi.{kind}_complete.v1"
        or d["schema"] != f"soccer.rsi.{kind}_declaration.v1"
        or c["declaration_hash"] != d["report_hash"]
        or Path(d["config"]["output_root"]) != root
        or len(c["rows"]) != 18
        or len(d["jobs"]) != 18
        or [r["index"] for r in c["rows"]] != list(range(18))
        or any(
            type(c[k]) is not int or c[k] != v
            for k, v in dict(
                new_native_steps=54000,
                executed_control_foundation_pairs=5400,
                new_optimizer_updates=0,
                baseline_rollouts_reused=18,
            ).items()
        )
        or type(c["forecast_steps"]) is not int
        or not 0 <= c["forecast_steps"] <= d["config"]["forecast_step_budget"]
        or any(
            c[k] is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("complete whole18 declared consumed transfer profile required")
    summary = paired_summary(c["rows"])
    if adaptive:
        summary["local_action_selection_promising"] = summary.pop("fixed_action_locally_promising")
    if any(c.get(k) != v for k, v in summary.items()):
        raise ValueError("whole raw paired transfer/retention summary required")
    return adaptive, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--complete-file-hash", required=True)
    parser.add_argument("--output", required=True, type=Path)
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
        raise ValueError("closed bytes, clean frozen whole18 reader and unused output required")
    require_terminal(args.unit)
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts")
        for p in folder.rglob("*.py")
    }

    def pin(name, expected=None):
        file = Path(name)
        h = digest(file)
        if (
            file.is_symlink()
            or (expected is not None and h != expected)
            or (str(file) in pins and pins[str(file)] != h)
        ):
            raise ValueError("whole original ordinary transfer source/input bytes required")
        pins[str(file)] = h

    def sealed(name):
        pin(name)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole logical transfer evidence seal required")
        return d

    def arrays(w):
        pin(w["path"], w["sha256"])
        with np.load(w["path"], allow_pickle=False) as a:
            return dict(a)

    c, d = sealed(args.root / "complete.json"), sealed(args.root / "declaration.json")
    adaptive, summary = check_complete(c, d, args.root)
    for name, h in d["input_file_hashes"].items():
        pin(name, h)
    cfg = d["config"]
    stage = sealed(cfg["stage_readback"])
    native = sealed(stage["effective_cohort_spec"]["producer_complete"])
    if len(native["results"]) != 6:
        raise ValueError("all six original bindings required")
    commitments = []
    for parent in native["results"]:
        w = parent["witnesses"]["commitment"]
        pin(w["path"], w["sha256"])
        commitments.append(json.loads(Path(w["path"]).read_text()))
    plan = declare_local_impact_perturbations(tuple(tuple(c["course"]) for c in commitments))
    lc = sealed(Path(cfg["local_root"]) / "complete.json")
    ld = sealed(Path(cfg["local_root"]) / "declaration.json")
    rb = sealed(cfg["local_readback"])
    if (
        lc["report_hash"] != d["local_complete_hash"]
        or lc["declaration_hash"] != ld["report_hash"]
        or rb["complete_hash"] != lc["report_hash"]
        or ld["plan"] != plan
        or len(lc["rows"]) != 12
    ):
        raise ValueError(
            "same whole public local24 raw-readback source and original course plan required"
        )
    rows = []
    forecasts = 0
    for i, row in enumerate(c["rows"]):
        if row != sealed(args.root / f"case-{i}-complete.json") or row["arrays"]["path"] != str(
            args.root / f"case-{i}-trace.npz"
        ):
            raise ValueError("all ordered complete transfer rows and raw files required")
        job = d["jobs"][i]
        if i < 12:
            pair = lc["rows"][i]
            if (
                pair != sealed(Path(cfg["local_root"]) / f"pair-{i}-complete.json")
                or pair["job"] != plan["jobs"][i]
            ):
                raise ValueError("same complete original local paired context required")
            baseline = pair["arms"][0]
            if baseline["arm"] != "original_control":
                raise ValueError("original baseline required")
            source_index = plan["jobs"][i]["source_index"]
            course = plan["jobs"][i]["course"]
            baseline_hash = baseline["report_hash"]
            old = arrays(baseline["arrays"])
        else:
            source_index = i - 12
            parent = native["results"][source_index]
            course = commitments[source_index]["course"]
            baseline_hash = parent["receipt_hash"]
            old = arrays(parent["witnesses"]["trace"])
        verify_original_control_torques(old)
        baseline_outcome = label_first_touch(old).to_dict()
        expected = dict(
            index=i,
            source_index=source_index,
            course=course,
            baseline_hash=baseline_hash,
            baseline_outcome=baseline_outcome,
            partition="CONSUMED_LOCAL_DEVELOPMENT" if i < 12 else "CONSUMED_ORIGINAL_RETENTION",
        )
        parent = native["results"][source_index]
        if (
            job != expected
            or any(row[k] != v for k, v in expected.items())
            or row["original_policy_hash"] != commitments[source_index]["motor_policy_hash"]
            or row["source_receipt_hash"] != parent["receipt_hash"]
            or row["action_name"] != cfg["action_name"]
            or row["original_prefix_and_unmodified_neural_bits_matched"] is not True
            or row["new_optimizer_updates"] != 0
            or any(
                row[k] is not False
                for k in (
                    "teacher_qualified",
                    "training_authorized",
                    "hardware_authorized",
                    "promotion_authorized",
                )
            )
        ):
            raise ValueError("whole ordered original course/policy/baseline identity required")
        trace = arrays(row["arrays"])
        review = row["causal_review"]
        if review["native_integrated_steps"] != 3000 or review["native_attempted_steps"] != 3000:
            raise ValueError("whole native body required")
        control = (
            check_exit_trace(trace, review)
            if adaptive
            else check_release_trace(trace, review, action_name=cfg["action_name"])
        )
        if control != row["control_review"] or row["outcome"] != label_first_touch(trace).to_dict():
            raise ValueError("actual raw postcontact control and unchanged full M0 required")
        require_prefix(trace, old, control["first_physically_applied_correction_frame"])
        alignment = []
        if adaptive:
            expected_bank = None
            f = control["observed_trigger_frame"]
            if f is not None and trace["postcontact_exit_forecast_steps"][f, 0, 0] > 0:
                import mujoco

                from rosclaw_soccer.physics.g1_joint_task_kinematics import (
                    G1JointTaskKinematicSampler,
                )
                from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
                from rosclaw_soccer.rsi.impact_torque_screen import impact_torque_candidates

                w = parent["witnesses"]["world"]
                pin(w["path"], w["sha256"])
                model = mujoco.MjModel.from_binary_path(w["path"])
                data = mujoco.MjData(model)
                joints = [int(model.joint(n).id) for n in G1_DDS_JOINT_NAMES]
                qi = [int(model.jnt_qposadr[j]) for j in joints]
                vi = [int(model.jnt_dofadr[j]) for j in joints]
                ball = int(model.body_jntadr[int(model.body("ball").id)])
                bq, bv = int(model.jnt_qposadr[ball]), int(model.jnt_dofadr[ball])
                q, v = trace["canonical_qpos"][f, 0], trace["canonical_qvel"][f, 0]
                data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7] = q[:7], q[7:36], q[36:]
                data.qvel[:6], data.qvel[vi], data.qvel[bv : bv + 6] = v[:6], v[6:35], v[35:]
                sampler = G1JointTaskKinematicSampler(model, expected_world_hash=w["sha256"])
                expected_bank = (
                    *impact_torque_candidates(
                        sampler.sample(data), control["measured_contact_side"]
                    ),
                    ("repeat_zero", np.zeros(29)),
                )
            alignment = read_exit_decisions(
                trace,
                args.root / f"case-{i}-preaction-evidence",
                expected_bank=expected_bank,
                sealed=sealed,
                arrays=arrays,
            )
        elif review["forecast_integrated_steps"] != 0 or review["forecast_attempted_steps"] != 0:
            raise ValueError("fixed action must not execute forecasts")
        forecasts += review["forecast_integrated_steps"]
        rows.append(
            dict(
                index=i,
                partition=row["partition"],
                source_case_hash=row["report_hash"],
                baseline_outcome=baseline_outcome,
                outcome=row["outcome"],
                raw_control=control,
                selected_execution_alignment=alignment,
            )
        )
    if forecasts != c["forecast_steps"] or any(digest(n) != h for n, h in pins.items()):
        raise ValueError("whole immutable original transfer cohort and actual counters required")
    result = dict(
        schema="soccer.rsi.postcontact_transfer_readback.v1",
        source_head=args.source_head,
        complete_hash=c["report_hash"],
        rows=rows,
        input_file_hashes=pins,
        **summary,
        actual_native_steps=54000,
        actual_forecast_steps=forecasts,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        independent_physical_replay=False,
        independent_neural_implementation=False,
        shared_proxy_and_label_algorithms=True,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print("WHOLE18_POSTCONTACT_TRANSFER_READBACK", result["report_hash"], flush=True)


if __name__ == "__main__":
    main()
